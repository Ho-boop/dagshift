"""关键链引导的块搜索：在 chunk_search 的随机移动之外，按问题2 评估器时间线倒推出的关键链挑移动。

评估器实测（case_071 k=4，见 NOTES.md）：问题2 的 makespan 主要被核内串行 Pipe 的先后次序卡住，
关键链上一半以上的衔接是“就绪了但在同 Pipe 上排在前一个 op 之后”。据此的引导移动：
- 提前（pipe-promote）：链上 b 在同 Pipe 上被 a 卡住 → b 连同其次序晚于 a 的祖先改到 a 的块之前；
- 推后（pipe-demote）：把 a（及其同核的一小片区域）改到 b 的块之后；
- 跨核跳（hop）：链上 o1@c1 → o2@c2，把 o2 附近区域并到 c1（消掉这一跳），或把 o1 及其祖先提前；
- 链核卸载（offload）：关键链时间最多的核上，挑不在链上的一片区域迁到负载最小的核。
关键链只用于挑移动；接受与否仍只看评估器（common.sort_key）。
"""

import math
import random
import time

from common import evaluate
from common.plan import plan_hash

try:  # 评估缓存的内部函数（common/ 由 S3 维护）；接口变了就退化为不复用，结果不变，只是慢一些
    from common.evaluate import _MEMO, _cache_key, _cacheable, _read_disk
except ImportError:
    _MEMO = None

from .chunks import chunk_plan, migrate_rank, random_move
from .crit import critical_chain
from .search import REGION_SIZES, _region

PROMOTE_DELTAS = (1e-6, 0.5, 1.5)


def _consider(arc, name, plan, reuse_full):
    """reuse_full 且未命中评估缓存时，先带完整结果评估一次（写入缓存），再交给 Archive（此时命中进程内缓存），
    返回完整结果供重算关键链，省掉 critical_chain 里的第二次评估；其余情况照常 consider，返回 None。"""
    full = None
    if reuse_full and _MEMO is not None and _cacheable(plan):
        key = _cache_key(arc.g, plan_hash(plan), arc.problem)
        if _MEMO.get(key) is None and _read_disk(key, arc.problem) is None:
            r = evaluate(arc.g, plan, arc.problem, keep_full=True)
            arc.eval_cost = max(arc.eval_cost, r.get('eval_seconds') or 0.0)
            full = r.get('full')
    arc.consider(name, plan)
    return full


def _ancestors_above(g, core_of, rank, v, r, limit, same_core):
    """v 的祖先中次序值 > r 的（BFS，至多 limit 个）；same_core 时只取与 v 同核的。"""
    c = core_of[v]
    out, frontier, seen = [], [v], {v}
    while frontier and len(out) < limit:
        nxt = []
        for u in frontier:
            for w in g.pred[u]:
                if w in seen:
                    continue
                seen.add(w)
                if rank[w] > r and (not same_core or core_of[w] == c):
                    out.append(w)
                    nxt.append(w)
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
        frontier = nxt
    return out


def _guided_move(g, k, core_of, rank, cc, rng):
    """返回 (new_core_updates, new_rank_updates) 或 None。"""
    kinds = []
    if cc['pipe_edges']:
        kinds += ['promote', 'promote', 'demote']
    if cc['hops']:
        kinds += ['hop']
    kinds += ['offload']
    kind = rng.choice(kinds)
    size = rng.choice(REGION_SIZES)
    if kind == 'promote':
        a, b, _ = rng.choice(cc['pipe_edges'])
        r = rank[a] - rng.choice(PROMOTE_DELTAS)
        anc = _ancestors_above(g, core_of, rank, b, r, size * 4, same_core=rng.random() < 0.5)
        return {}, {v: r for v in [b] + anc}
    if kind == 'demote':
        a, b, _ = rng.choice(cc['pipe_edges'])
        r = rank[b] + rng.choice(PROMOTE_DELTAS)
        region = _region(g, core_of, a, rng.choice((1, 1, 4)), rng)
        return {}, {v: r for v in region if rank[v] < r}
    if kind == 'hop':
        o1, o2 = rng.choice(cc['hops'])
        x = rng.random()
        if x < 0.4:
            region = _region(g, core_of, o2, size, rng)
            c1 = core_of[o1]
            return {v: c1 for v in region}, {v: rank[o1] for v in region}
        if x < 0.6:
            region = _region(g, core_of, o1, size, rng)
            c2 = core_of[o2]
            return {v: c2 for v in region}, {v: rank[o2] for v in region}
        r = rank[o1] - rng.choice(PROMOTE_DELTAS)
        anc = _ancestors_above(g, core_of, rank, o1, r, size * 4, same_core=True)
        return {}, {v: r for v in [o1] + anc}
    # offload
    tb = cc['time_by_core']
    src = max(tb, key=lambda c: tb[c])
    on_chain = {o for _, o in cc['ops']}
    ops = [v for v, c in core_of.items() if c == src and v not in on_chain]
    loads = [0] * k
    for v, c in core_of.items():
        loads[c] += g.cycles[v]
    target = min((c for c in range(k) if c != src), key=lambda c: loads[c])
    if not ops:
        return None
    region = [v for v in _region(g, core_of, rng.choice(ops), size, rng) if v not in on_chain]
    if not region:
        return None
    r = migrate_rank(g, core_of, rank, region, target)
    return {v: target for v in region}, {v: r for v in region}


def _tensor_edges(g):
    """(生产 op, 消费 op, 字节)；只取计算 op 之间经 tensor 的边。"""
    return [(p, c, s) for _t, s, p, cs in g.interior_tensors for c in cs if c != p]


def _cut_move(g, k, core_of, rank, edges, rng):
    """减跨核字节的移动（小图上 DDR 常被跨核搬运占满，见 NOTES.md §9）。
    - 按字节加权抽一条跨核边，把一端（及一小片同核区域）并到另一端的核；
    - 或 FM 式：算每个边界 op 迁到各核能省的字节（连到该核的字节 − 连到本核的字节），在增益前 20 名里抽一个迁过去。"""
    cross = [(u, v, s) for u, v, s in edges if core_of[u] != core_of[v]]
    if not cross:
        return None
    if rng.random() < 0.5:
        u, v, _ = rng.choices(cross, weights=[s for _, _, s in cross])[0]
        src, dst = (v, u) if rng.random() < 0.5 else (u, v)
        target = core_of[dst]
        region = _region(g, core_of, src, rng.choice((1, 1, 2, 4)), rng)
    else:
        conn = {}
        for u, v, s in edges:
            cu, cv = core_of[u], core_of[v]
            for a, cb in ((u, cv), (v, cu)):
                d = conn.setdefault(a, {})
                d[cb] = d.get(cb, 0) + s
        cand = []
        for a, d in conn.items():
            own = d.get(core_of[a], 0)
            for c, b in d.items():
                if c != core_of[a] and b > own:
                    cand.append((b - own, a, c))
        if not cand:
            return None
        cand.sort(reverse=True)
        _, src, target = rng.choice(cand[:20])
        region = [src]
    r = migrate_rank(g, core_of, rank, region, target)
    return {v: target for v in region}, {v: r for v in region}


def _input_groups(g):
    """由 COPY_IN 搬入、被 >=2 个计算 op 读取的 tensor：[(字节, 消费 op 列表)]。"""
    return [(s, list(dict.fromkeys(cs))) for _t, s, cs in g.input_tensors if len(set(cs)) >= 2]


def _input_move(g, k, core_of, rank, groups, rng):
    """减输入重复读的移动：问题2 里读同一输入 tensor 的每个核都各自 COPY_IN 一次（NOTES.md §15）。
    按 字节 x (读它的核数 - 1) 加权抽一个 tensor，把少数核上的消费者迁到读它最多的核：
    一半概率迁某个少数核上的全部消费者，一半概率迁一个消费者及一小片同核区域。"""
    cand, w = [], []
    for s, cs in groups:
        m = len({core_of[c] for c in cs})
        if m > 1:
            cand.append(cs)
            w.append(s * (m - 1))
    if not cand:
        return None
    cs = rng.choices(cand, weights=w)[0]
    cnt = {}
    for c in cs:
        cnt[core_of[c]] = cnt.get(core_of[c], 0) + 1
    target = max(cnt, key=lambda c: (cnt[c], rng.random()))
    minority = [c for c in cs if core_of[c] != target]
    if rng.random() < 0.5:
        src = core_of[rng.choice(minority)]
        region = [c for c in minority if core_of[c] == src]
    else:
        region = _region(g, core_of, rng.choice(minority), rng.choice((1, 1, 2, 4)), rng)
    if rng.random() < 0.5:
        r = min(rank[c] for c in cs if core_of[c] == target)
    else:
        r = migrate_rank(g, core_of, rank, region, target)
    return {v: target for v in region}, {v: r for v in region}


def _accept(accept, new_key, cur_key, sideways, it, hist, rng, sa_t0, t_frac):
    """接受准则。strict：严格更优（sideways 时允许相等）；
    lahc：Late Acceptance，新解不差于 L 步前的当前解即接受（hist 为长 L 的环形表）；
    sa：模拟退火，劣化 Δ=新 makespan/当前 makespan−1 以 exp(−Δ/T) 接受，T 从 sa_t0 按已用时间比例线性降到 0。
    不可行解（key[0] != 0）一律不接受。最优解始终由 Archive 记录，接受劣解不会丢最优。"""
    if new_key[0] != 0:
        return False
    if new_key < cur_key or (sideways and new_key == cur_key):
        return True
    if accept == 'lahc':
        return new_key <= hist[it % len(hist)]
    if accept == 'sa':
        t = sa_t0 * max(0.0, 1.0 - t_frac())
        if t <= 0:
            return False
        d = new_key[1] / cur_key[1] - 1.0
        return rng.random() < math.exp(-d / t)
    return False


def crit_search(g, k, arc, core_of, rank, seed=0, max_iters=None, tag='cs', p_rand=0.3,
                sideways=False, crit_factor=3.0, p_cut=0.0, accept='strict', lahc_len=20, sa_t0=0.005,
                reuse_full=True, p_in=0.0):
    """从 (core_of, rank) 出发；每步以 p_rand 概率走 chunk_search 的随机移动，否则走关键链引导移动。
    当前解按 accept 准则移动（默认 strict：评估器严格更优，sideways=True 时允许相等）；严格改进后重算关键链
    （reuse_full 时复用这一步评估的完整结果，否则多一次完整评估；剩余时间不足 crit_factor 次评估时不再重算，
    退化为随机移动）。reuse_full 只省时间，不改变给定评估序列下的搜索轨迹。"""
    if k < 2:
        return
    rng = random.Random(seed)
    core_of, rank = dict(core_of), dict(rank)
    plan = chunk_plan(g, k, core_of, rank)
    full = _consider(arc, f'{tag}_start', plan, reuse_full)
    cur_key = arc.last_key
    cc = (critical_chain(g, plan, arc.problem, full=full)
          if cur_key[0] == 0 and arc.can_afford(crit_factor) else None)
    edges = _tensor_edges(g) if p_cut > 0 else None
    groups = _input_groups(g) if p_in > 0 else None
    hist = [cur_key] * max(1, lahc_len)
    t0 = time.perf_counter()

    def t_frac():
        if arc.deadline is None:
            return it / (max_iters or 1)
        return (time.perf_counter() - t0) / max(1e-9, arc.deadline - t0)

    it = 0
    while arc.can_afford(1.2):
        it += 1
        if max_iters is not None and it > max_iters:
            break
        mv = None
        if groups and rng.random() < p_in:
            mv = _input_move(g, k, core_of, rank, groups, rng)
        elif edges is not None and rng.random() < p_cut:
            mv = _cut_move(g, k, core_of, rank, edges, rng)
        elif cc is not None and rng.random() >= p_rand:
            mv = _guided_move(g, k, core_of, rank, cc, rng)
            if mv is not None:
                cu, ru = mv
                if not cu and all(rank[v] == r for v, r in ru.items()):
                    mv = None
        if mv is None:
            rm = random_move(g, k, core_of, rank, rng)
            if rm is None:
                continue
            region, target, r = rm
            mv = ({v: target for v in region}, {v: r for v in region})
        cu, ru = mv
        new_core, new_rank = dict(core_of), dict(rank)
        new_core.update(cu)
        new_rank.update(ru)
        plan = arc.timed_gen(chunk_plan, g, k, new_core, new_rank)
        full = _consider(arc, f'{tag}_move{it}', plan, reuse_full)
        new_key = arc.last_key
        ok = _accept(accept, new_key, cur_key, sideways, it, hist, rng, sa_t0, t_frac)
        hist[it % len(hist)] = cur_key if not ok else new_key
        if ok:
            # 只在严格改进时重算关键链；接受等值或劣解时沿用旧链挑移动
            improved = new_key < cur_key
            cur_key = new_key
            core_of, rank = new_core, new_rank
            if improved or cc is None:
                cc = (critical_chain(g, plan, arc.problem, full=full)
                      if full is not None or arc.can_afford(crit_factor) else None)
