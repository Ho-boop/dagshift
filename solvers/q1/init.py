"""问题1 初解生成：多种核分配 × 多种 Task 切法，统一用参数规格（spec）描述。

spec 例：
  {'gen': 'comp'}                                   计算分量整体 LPT 装箱
  {'gen': 'band', 'level': 'depth', 'balance': 'levels', 'S': 8, 'eps': 0.08, 'tiny': 500}
  {'gen': 'heft', 'lam': 1.0}
再加 Task 切法 'mode'：'phase' / 'phase_alap' / 'band'（可带 'merge'）。

所有生成器先得到 op → 核的分配 core[v]，再用合法 Task 键切 Task（structure.py）：
- 'phase'：phase(v) = max_u phase(u) + [跨核]，Task 数最少；
- 'phase_alap'：同样的 phase 上界内把非关键 op 推迟，减少关键前驱 Task 的长度；
- 'band'：Task = (核, 层带)，按深度切片，控制单个 Task 的工作集（减 spill、减重复读权重）。
这些键沿任意数据边不减、跨核边严格增，因而商图和 Task 顺序都无环。keys_valid 再做一次廉价自检。

核分配：
- band：按层级函数切 S 条层带（S='each' 表示每个层级值一条），层带内按依赖求连通组，
  组按工作量 LPT 分核：负载上限 (1+eps)·层带工作量/k 以内优先放在输入字节最多的核（亲和）；
  工作量小于 tiny 的组（如归约树的小 ADD）不参与均衡，直接按亲和就近放，避免为均衡制造跨核同步。
- heft：HEFT 式列表调度，跨核边代价 lam·1000 + 2·字节/BW。
"""

from collections import defaultdict

from .structure import BW, CROSS_WAIT, phases_from_cores, alap_phases, plan_from_keys


# ---------------------------------------------------------------- 工具
class _DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        p = self.p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                ra, rb = rb, ra
            self.p[ra] = rb


def level_bands(view, S, level='depth', balance='levels'):
    """按层级函数把 op 切成层带，返回 band_of[v]（沿边不减）。

    level: 'depth'（最长路层数）或 'asap'（最早开始时刻）
    balance: 'levels' 按层级值等分；'work' 按累计工作量等分
    S: 层带数；'each' 表示每个不同的层级值单独一条层带
    """
    lv = view.depth if level == 'depth' else view.asap
    values = sorted(set(lv.values()))
    if S == 'each':
        rank = {x: i for i, x in enumerate(values)}
        return {v: rank[lv[v]] for v in view.ops}
    if S <= 1:
        return {v: 0 for v in view.ops}
    if balance == 'levels':
        lo, hi = values[0], values[-1]
        span = max(1, hi - lo + 1)
        return {v: min(S - 1, (lv[v] - lo) * S // span) for v in view.ops}
    work_at = defaultdict(int)
    for v in view.ops:
        work_at[lv[v]] += view.cyc[v]
    total = sum(work_at.values()) or 1
    band_of_value, acc = {}, 0
    for x in values:
        band_of_value[x] = min(S - 1, int(acc * S / total))
        acc += work_at[x]
    return {v: band_of_value[lv[v]] for v in view.ops}


_WSPLIT_DEG = (8, 4, 3, 2)


def _split_wide(view, ops, limit, di=0):
    """把工作量超过 limit 的连通组拆开：组内度数 > d 的 op（扇出/扇入大的枢纽，如多头注意力里连接各头的 MATMUL）
    单独成块，其余边照旧求连通；仍超限的块用更小的 d 递归拆。纯链（d=2 也拆不动）保持原样。"""
    cyc = view.cyc
    if di >= len(_WSPLIT_DEG) or sum(cyc[v] for v in ops) <= limit:
        return [ops]
    d = _WSPLIT_DEG[di]
    inside = set(ops)
    deg = {v: sum(1 for u in view.pred[v] if u in inside) + sum(1 for w in view.succ[v] if w in inside)
           for v in ops}
    hub = {v for v in ops if deg[v] > d}
    if not hub:
        return _split_wide(view, ops, limit, di + 1)
    dsu = _DSU(ops)
    for v in ops:
        if v in hub:
            continue
        for u in view.pred[v]:
            if u in inside and u not in hub:
                dsu.union(u, v)
    parts = defaultdict(list)
    for v in ops:
        parts[dsu.find(v)].append(v)
    out = []
    for p in parts.values():
        out.extend(_split_wide(view, p, limit, di + 1) if len(p) > 1 else [p])
    return out


def band_groups(view, band_of, tinyop=0, wsplit=0, k=1):
    """层带内按依赖求连通组：返回 [(band, [ops...]), ...]（组内 op 按拓扑序，band 取首个 op 的层带）。

    tinyop>0 时，两端都是极小 op（cycles ≤ tinyop，如归约树的小 ADD）的边也合并，跨层带成簇，
    整簇放在同一个核上，避免小 op 在核间来回跳、每跳一次付 1000 周期同步。
    跨层带的簇仍只在一个核上，不影响 Task 键的合法性（同层带的边都在组内）。
    wsplit>0 时，工作量超过 wsplit·层带工作量/k 的宽组按枢纽拆开（_split_wide）。拆开后同层带内会有跨核边，
    'band' 切法因此不合法（build 返回 None），只配 phase 类切法。
    """
    dsu = _DSU(view.ops)
    cyc = view.cyc
    for v in view.ops:
        bv = band_of[v]
        for u in view.pred[v]:
            if band_of[u] == bv or (tinyop and cyc[u] <= tinyop and cyc[v] <= tinyop):
                dsu.union(u, v)
    groups = defaultdict(list)
    for v in view.ops:
        groups[dsu.find(v)].append(v)
    groups = list(groups.values())
    if wsplit:
        band_work = defaultdict(int)
        for v in view.ops:
            band_work[band_of[v]] += cyc[v]
        out = []
        for ops in groups:
            out.extend(_split_wide(view, ops, wsplit * band_work[band_of[ops[0]]] / k))
        groups = out
    return [(band_of[ops[0]], ops) for ops in groups]


def assign_bands_lpt(view, k, band_of, eps=0.08, tiny=0, tinyop=0, affinity=True, wsplit=0):
    """按层带顺序，把每条层带内的连通组 LPT 分到 k 个核，负载上限内优先亲和核。"""
    groups = band_groups(view, band_of, tinyop, wsplit, k)
    by_band = defaultdict(list)
    for b, ops in groups:
        by_band[b].append(ops)
    core = {}
    size = view.t_size
    for b in sorted(by_band):
        gs = by_band[b]
        works = [sum(view.cyc[v] for v in ops) for ops in gs]
        total = sum(works)
        target = total / k * (1 + eps)
        load = [0] * k
        for w, ops in sorted(zip(works, gs), key=lambda x: (-x[0], x[1][0])):
            aff = [0] * k
            if affinity:
                seen = set()
                for v in ops:
                    for t in view.op_in[v]:
                        p = view.t_prod.get(t)
                        if p is not None and t not in seen and p in core:
                            seen.add(t)
                            aff[core[p]] += size[t] + 1
                    for u in view.pred[v]:
                        if u in core:
                            aff[core[u]] += 1
            if w < tiny and max(aff) > 0:
                c = max(range(k), key=lambda c: (aff[c], -load[c], -c))
            else:
                fit = [c for c in range(k) if load[c] + w <= target]
                if fit:
                    c = max(fit, key=lambda c: (aff[c], -load[c], -c))
                else:
                    c = min(range(k), key=lambda c: (load[c], -aff[c], c))
            load[c] += w
            for v in ops:
                core[v] = c
    return core


def component_lpt(view, k):
    core, load = {}, [0] * k
    for comp in view.g.components:
        w = sum(view.cyc[v] for v in comp)
        c = min(range(k), key=lambda c: (load[c], c))
        load[c] += w
        for v in comp:
            core[v] = c
    return core


def heft_assign(view, k, lam=1.0):
    """HEFT 式列表调度得到核分配（不含插入）；跨核边代价 lam·1000 + 2·字节/BW。"""
    size = view.t_size
    edge_bytes = defaultdict(int)
    for t, p in view.t_prod.items():
        for c in view.t_cons.get(t, ()):
            if c != p:
                edge_bytes[(p, c)] += size[t]
    comm = lambda u, v: lam * CROSS_WAIT + 2 * edge_bytes.get((u, v), 0) / BW
    avg = (k - 1) / k
    rank = {}
    for v in reversed(view.ops):
        rank[v] = view.cyc[v] + max((avg * comm(v, w) + rank[w] for w in view.succ[v]), default=0)
    order = sorted(view.ops, key=lambda v: (-rank[v], view.pos[v]))
    avail = [0.0] * k
    aft, core = {}, {}
    for v in order:
        best = None
        for c in range(k):
            ready = avail[c]
            for u in view.pred[v]:
                t = aft[u] + (comm(u, v) if core[u] != c else 0)
                if t > ready:
                    ready = t
            eft = ready + view.cyc[v]
            if best is None or eft < best[0] - 1e-9:
                best = (eft, c)
        eft, c = best
        core[v], aft[v], avail[c] = c, eft, eft
    return core


# ---------------------------------------------------------------- spec → plan
def export_first(view, core, key):
    """在每个 Task（同核同键）内再切一刀：有跨核后继的 op 及其在本 Task 内的祖先排前（子键 0），
    其余排后（子键 1）。这样被别的核等待的前驱 Task 只含“导出”部分，尽早结束。

    合法性：Task 内边 u→v 若 v 子键为 0（是导出的祖先），u 也是祖先，子键 0；故子键沿边不减。
    """
    anc = set()
    for v in reversed(view.ops):
        cv, kv = core[v], key[v]
        for w in view.succ[v]:
            if core[w] != cv or (w in anc and key[w] == kv):
                anc.add(v)
                break
    return {v: (key[v], 0 if v in anc else 1) for v in view.ops}


def keys_for(view, core, band_of, mode, merge=1):
    if mode == 'phase':
        return phases_from_cores(view, core)
    if mode == 'phase_alap':
        return alap_phases(view, core, phases_from_cores(view, core))
    if mode == 'phase_split':
        return export_first(view, core, phases_from_cores(view, core))
    if mode == 'phase_alap_split':
        return export_first(view, core, alap_phases(view, core, phases_from_cores(view, core)))
    if mode == 'band':
        return {v: band_of[v] // merge for v in view.ops}
    if mode == 'phase_band':
        ph = phases_from_cores(view, core)
        return {v: (ph[v], band_of[v] // merge) for v in view.ops}
    raise ValueError(mode)


def keys_valid(view, core, key):
    """廉价自检：同核边键不减、跨核边键严格增。"""
    for v in view.ops:
        for u in view.pred[v]:
            if core[u] == core[v]:
                if key[u] > key[v]:
                    return False
            elif not key[u] < key[v]:
                return False
    return True


def spec_name(spec):
    g = spec['gen']
    if g == 'comp':
        body = 'comp'
    elif g == 'heft':
        body = f"heft[{spec['lam']}]"
    else:
        body = (f"band[{spec['level']},{spec['balance']},S={spec['S']},"
                f"eps={spec.get('eps', 0.08)},tiny={spec.get('tiny', 0)}"
                f"{',op=' + str(spec['tinyop']) if spec.get('tinyop') else ''}"
                f"{',ws=' + str(spec['wsplit']) if spec.get('wsplit') else ''}]")
    mode = spec.get('mode', 'phase')
    if spec.get('merge', 1) != 1:
        mode += f"/{spec['merge']}"
    if spec.get('kc'):
        mode += f"@{spec['kc']}c"
    return body + ':' + mode


class _LRU(dict):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def put(self, key, value):
        if len(self) >= self.cap:
            self.pop(next(iter(self)))
        self[key] = value


class Builder:
    """按 spec 构造方案；核分配与层带按生成器参数缓存（有上限），多种 Task 切法共用。"""

    def __init__(self, view, k, cache=48):
        self.view, self.k = view, k
        self._bands = _LRU(cache)
        self._cores = _LRU(cache)

    def bands(self, spec):
        if spec['gen'] != 'band':
            return None
        key = (spec['level'], spec['balance'], spec['S'])
        if key not in self._bands:
            self._bands.put(key, level_bands(self.view, spec['S'], spec['level'], spec['balance']))
        return self._bands[key]

    def cores(self, spec):
        # kc：只用前 kc 个核（其余核空）。多核分摊的共享权重重复读、spill 超过并行收益时，少用核反而更快
        kc = spec.get('kc') or self.k
        g = spec['gen']
        if g == 'comp':
            key = ('comp', kc)
        elif g == 'heft':
            key = ('heft', spec['lam'], kc)
        else:
            key = ('band', spec['level'], spec['balance'], spec['S'],
                   spec.get('eps', 0.08), spec.get('tiny', 0), spec.get('tinyop', 0), kc,
                   spec.get('wsplit', 0))
        if key not in self._cores:
            if g == 'comp':
                core = component_lpt(self.view, kc)
            elif g == 'heft':
                core = heft_assign(self.view, kc, spec['lam'])
            else:
                core = assign_bands_lpt(self.view, kc, self.bands(spec),
                                        eps=spec.get('eps', 0.08), tiny=spec.get('tiny', 0),
                                        tinyop=spec.get('tinyop', 0), wsplit=spec.get('wsplit', 0))
            self._cores.put(key, core)
        return self._cores[key]

    def build(self, spec):
        """返回 plan；Task 键不合法（如 band 模式合并后跨核边键相等）时返回 None。"""
        view = self.view
        core = self.cores(spec)
        mode = spec.get('mode', 'phase')
        if spec['gen'] == 'comp':
            key = {v: 0 for v in view.ops} if mode != 'phase_alap' else keys_for(view, core, None, 'phase')
        else:
            band_of = self.bands(spec)
            if band_of is None and mode in ('band', 'phase_band'):
                return None
            key = keys_for(view, core, band_of, mode, spec.get('merge', 1))
        if not keys_valid(view, core, key):
            return None
        return plan_from_keys(view, key, core, self.k)


def initial_specs(view, quick=False):
    """初始候选的参数规格列表。"""
    specs = [{'gen': 'comp', 'mode': 'phase'}]
    depth_levels = view.max_depth + 1
    S_list = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    if quick:
        S_list = [2, 4, 8, 16]
    S_list = [S for S in S_list if S <= depth_levels]
    tiny = CROSS_WAIT // 2
    for level in (('depth',) if quick else ('depth', 'asap')):
        for balance in (('levels',) if quick else ('levels', 'work')):
            for S in [1] + S_list:
                base = {'gen': 'band', 'level': level, 'balance': balance, 'S': S,
                        'eps': 0.08, 'tiny': 0}
                specs.append(dict(base, mode='phase'))
                specs.append(dict(base, mode='phase_alap'))
                specs.append(dict(base, mode='phase_split'))
                if S > 1:
                    specs.append(dict(base, mode='band'))
        # 逐层分带 + 小组就近：适合“lane + 每步全局归约”一类结构；tinyop 把归约树整簇放一个核
        for tinyop in (0, 32):
            base = {'gen': 'band', 'level': level, 'balance': 'levels', 'S': 'each',
                    'eps': 0.08, 'tiny': tiny, 'tinyop': tinyop}
            specs.append(dict(base, mode='phase'))
            specs.append(dict(base, mode='phase_alap'))
            specs.append(dict(base, mode='phase_split'))
            specs.append(dict(base, mode='phase_alap_split'))
    for lam in ((1.0, 4.0) if quick else (0.5, 1.0, 2.0, 4.0, 10.0)):
        specs.append({'gen': 'heft', 'lam': lam, 'mode': 'phase'})
        specs.append({'gen': 'heft', 'lam': lam, 'mode': 'phase_split'})
    return specs


_COMBOS = (('depth', 'levels'), ('depth', 'work'), ('asap', 'levels'), ('asap', 'work'))
_COARSE_S = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _band_specs(S, level, balance, modes):
    tiny = CROSS_WAIT // 2
    for mode in modes:
        if S == 1 and mode in ('band', 'phase_band'):
            continue
        yield {'gen': 'band', 'level': level, 'balance': balance, 'S': S, 'eps': 0.08,
               'tiny': 0 if mode in ('band', 'phase', 'phase_alap') else tiny, 'mode': mode}


def _between(lo, hi, cap=40):
    """(lo, hi) 之间的层带数：不多于 cap 个时取全部整数，否则取几何步长。"""
    vals = list(range(lo + 1, hi))
    if len(vals) <= cap:
        return vals
    r = (hi / lo) ** (1 / cap)
    out, x = set(), float(lo)
    while x < hi:
        x *= r
        if lo < round(x) < hi:
            out.add(int(round(x)))
    return sorted(out)


def layer_blocks(view, min_lag=4, thresh=0.35):
    """按深度的每层工作量/op 数剖面求层周期 p（自相关），返回块数 nb = L/p；找不到明显周期返回 None。

    transformer / lane 类图按层重复，层带边界与块边界对齐时 Task 不跨块、跨核同步少；
    粗网格 S 常常错过对齐点（如 nb=10 时 S=50）。滞后 < min_lag 的相关多是奇偶交替，不算。
    """
    L = view.max_depth + 1
    if L < 2 * min_lag:
        return None
    w, c = [0.0] * L, [0.0] * L
    for v in view.ops:
        d = view.depth[v]
        w[d] += view.cyc[v]
        c[d] += 1.0

    def prep(x):
        m = sum(x) / L
        y = [a - m for a in x]
        den = sum(a * a for a in y) / L
        return y, den

    series = [prep(w), prep(c)]
    # 候选滞后：L 不大时全部；L 大时只取 L/nb（nb=2..64）附近
    if L <= 600:
        lags = range(min_lag, L // 2 + 1)
    else:
        lags = sorted({p for nb in range(2, 65) for p in (L // nb - 1, L // nb, L // nb + 1)
                       if min_lag <= p <= L // 2})
    acf = {}
    for p in lags:
        best = 0.0
        n = L - p
        for y, den in series:
            if den > 0:
                best = max(best, sum(y[i] * y[i + p] for i in range(n)) / n / den)
        acf[p] = best
    if not acf:
        return None
    amax = max(acf.values())
    if amax < thresh:
        return None
    p = min(q for q, a in acf.items() if a >= 0.9 * amax)
    nb = L / p
    return nb if nb >= 2 else None


def aligned_S(nb, L):
    """与块数 nb 对齐的层带数（块的整数倍与半块）。"""
    out = []
    for m in (1, 2, 3, 4, 5, 6, 8, 0.5, 1.5, 2.5):
        S = int(round(nb * m))
        if 2 <= S <= min(L, 128) and S not in out:
            out.append(S)
    return out


def sweep_specs(view, scores, k=None, kc_late=False, wide=(1.0,), ws_pos=0, align=True):
    """代理扫描用的规格序列，按优先级由粗到细产出；调用方按时间截断。

    scores：调用方维护的 {spec_name: 代理分}，每产出一个规格后由调用方填入；生成器据此决定在哪里加密。
    层带数 S 与图的层周期（如 transformer 的层数、lane 的步数）对齐与否对结果影响很大，粗网格常常错过对齐点：
    每个 (层级函数, 均衡方式) 先扫粗网格，再在每种 Task 切法代理最好的 S 两侧逐个加密；最后全范围加密。
    给出 k 时，第 0 层之后插入“只用 kc < k 个核”的少量规格（kc = k-1 … 2）；kc_late=True（大图，扫描常被截断）
    时改放在第 1 层之后，不挤占满核规格。
    """
    L = view.max_depth + 1
    tiny = CROSS_WAIT // 2
    # 第 0 层：分量装箱、逐层分带（lane + 全局归约类结构）、HEFT
    yield {'gen': 'comp', 'mode': 'phase'}
    for tinyop in (32, 0):
        for mode in ('phase_split', 'phase', 'phase_alap_split', 'phase_alap'):
            yield {'gen': 'band', 'level': 'depth', 'balance': 'levels', 'S': 'each',
                   'eps': 0.08, 'tiny': tiny, 'tinyop': tinyop, 'mode': mode}
    for lam in (1.0, 4.0):
        for mode in ('phase_split', 'phase'):
            yield {'gen': 'heft', 'lam': lam, 'mode': mode}

    coarse = [S for S in _COARSE_S if S <= L]
    main_modes = ('phase_split', 'band', 'phase', 'phase_band')

    # 第 0.5 层：与层周期对齐的 S（粗网格之外的对齐点优先扫）
    if align:
        nb = layer_blocks(view)
        if nb is not None:
            for S in aligned_S(nb, L):
                yield from _band_specs(S, 'depth', 'levels', main_modes)

    def fewer_cores():
        # 少用核（邻域搜索里 kc 还可再 ±1，S 等参数随规格一起微调）
        for kc in range(k - 1, 1, -1) if k else ():
            yield {'gen': 'comp', 'mode': 'phase', 'kc': kc}
            yield {'gen': 'band', 'level': 'depth', 'balance': 'levels', 'S': 'each',
                   'eps': 0.08, 'tiny': tiny, 'tinyop': 32, 'mode': 'phase_split', 'kc': kc}
            for S in (2, 4, 8, 16, 32, 64):
                if S <= L:
                    for sp in _band_specs(S, 'depth', 'levels', ('phase_split', 'band')):
                        yield dict(sp, kc=kc)

    if not kc_late:
        yield from fewer_cores()

    def wide_split():
        # 宽组拆分（只配 phase 类切法）：层带内宽而连通的块（如多头注意力）按枢纽拆开再 LPT；
        # 胜出的多是粗层带（S=3～16），只扫 S≤32，免得大图上挤占后面的规格
        for ws in wide:
            for S in coarse:
                if S > 32:
                    break
                for sp in _band_specs(S, 'depth', 'levels', ('phase_split', 'phase')):
                    yield dict(sp, wsplit=ws)

    if ws_pos == 0:
        yield from wide_split()

    def refine(level, balance, modes):
        fams = []
        for mode in modes:
            best = None
            for S in coarse:
                for sp in _band_specs(S, level, balance, (mode,)):
                    sc = scores.get(spec_name(sp))
                    if sc is not None and (best is None or sc < best[0]):
                        best = (sc, S)
            if best is not None:
                fams.append((best[0], mode, best[1]))
        fams.sort()
        for _, mode, S in fams:
            i = coarse.index(S)
            lo = coarse[i - 1] if i > 0 else 1
            hi = coarse[i + 1] if i + 1 < len(coarse) else min(L + 1, 2 * S)
            for S2 in _between(lo, hi):
                yield from _band_specs(S2, level, balance, (mode,))

    # 第 1 层：每种 (层级函数, 均衡方式) 粗网格 + 局部加密
    for ci, (level, balance) in enumerate(_COMBOS):
        for S in coarse:
            yield from _band_specs(S, level, balance, main_modes)
        yield from refine(level, balance, main_modes)
        if ws_pos == ci + 1:
            yield from wide_split()
    if kc_late:
        yield from fewer_cores()
    # 第 2 层：全范围加密（调用方跳过已评过的规格名）
    dense = set(range(2, min(L, 40) + 1))
    x = 40.0
    while x <= min(L, 400):
        dense.add(int(round(x)))
        x *= 1.04
    dense = sorted(dense - set(coarse))
    for level, balance in _COMBOS:
        for S in dense:
            yield from _band_specs(S, level, balance, ('phase_split', 'band', 'phase'))
    for S in coarse:
        yield from _band_specs(S, 'depth', 'levels', ('phase_alap', 'phase_alap_split'))

def generate(view, k, *, quick=False):
    """兼容旧接口：产出 (name, plan) 列表（去掉重复方案与不合法键）。"""
    builder = Builder(view, k)
    out, seen = [], set()
    for spec in initial_specs(view, quick):
        plan = builder.build(spec)
        if plan is None:
            continue
        sig = (tuple(map(tuple, plan['core_schedules'])), tuple(plan['node_to_subgraph'].values()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((spec_name(spec), plan))
    return out
