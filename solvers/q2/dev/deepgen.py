"""深图分核构造（方向 A 的额外起点，开发中，SC）。

规律（离线对比档案/冷启动分核，见 DEEPGEN.md）：深图（最长链层数 >= 200）上，档案方案的分核像“车道”：
同一条纵向列（沿深度的链及其扇出小组）从头到尾留在同一个核，各核分到的列数大致均等，每个深度带内
各核负载都接近均衡；跨核切出的 tensor 字节只有冷启动的约 1/3。冷启动（列表调度逐 op 选最早完成核）
在每层都把宽度重新打散，同一列在核间来回切换。

构造 = 带“深度带均衡”约束的多层图划分（只用输入图，确定性，固定种子）：
1. 按拓扑层把计算 op 分到 B 个深度带（按累计工作量等分）；每个 op 的权重是只在自己那个带上非零的向量。
2. 边权 1/outdeg(生产者)：广播点（扇出大）不拉拢消费者，链和小扇出组被优先合并。
3. 粗化：重边匹配，合并后任一带的权重不超过 该带负载/k 的 1/4；粗到约 40k 个节点。
4. 初分：按首带顺序贪心，先满足各带上限 (1+eps)·带负载/k，再挑连接度最大的核。
5. 逐层展开，每层做标签传播细化（只接受连接度增益为正且不破坏带上限的移动）。
6. 解码：核内顺序用 (层号, 拓扑序)（lay）或固定分核的列表调度开始时间（ls），逐 op 编码。

solve（dev3）：vclock.run_virtual 包 multistart.solve；候选在代理的 q2m.solve / q3m.solve 内部构造
（结构生成器与 k−1 起点之后），追加进 warm。构造耗时按确定性公式计入虚拟时钟（不读墙钟），只从 q2/q3
求解器这一段预算里扣；构造 + 每个候选约 2 次评估的估计成本不超过该段预算的 GATE_FRAC。
结构闸门 eligible(graph, k) 只放行深图中候选可能取胜的图类（见 DEEPGEN.md §6）。

状态（09-26 16:12 S3）：暂停，未接入交付入口 solvers/deliver。dev3 A/B 未完成，见 DEEPGEN.md §7。
"""

import random
from collections import defaultdict

from solvers.q2.encode import per_op_plan
from solvers.q2.listsched import list_schedule

VERSION = 'deepgen-dev4'
LAST_META = None

DEPTH_MIN = 200
# dev4：k=2 不放行【待验证假设】。离线差距组合全在 k=4/5；dev2 A/B 的 p2 胜例在 k=3..5，
# dev3 的 056k2 为 1.047（k=2 两核本就少切边，候选只挤占预算）
MIN_K = 3
SEED = 0
# 离线扫过的全部变体：(带数 B, 带上限松弛 eps, 解码 lay/ls)
ALL_VARIANTS = tuple((B, e, d) for B, e in ((30, 0.3), (30, 0.5), (10, 0.15), (60, 0.3), (60, 0.8))
                     for d in ('lay', 'ls'))
# 求解入口用的变体，按优先级排列（68 个深图组合上逐个贪心加入、使 min(冷启动, 候选) 下降最多的顺序）
# dev3 去掉 lay 解码：q2/q3 求解器会对每个起点再做固定分核的列表调度重排（reseq），lay 候选重排后与 ls 近似重复
VARIANTS = ((30, 0.3, 'ls'), (60, 0.3, 'ls'))
MAX_COMPONENTS = 100    # 计算弱连通分量超过此数不给候选
CP_FRAC = 1.0           # 关键路径 >= CP_FRAC · 最大流水线工作量/k 不给候选
# 构造成本（虚拟秒）= GEN_A · n · 分核次数 + GEN_LS · n · ls 解码次数；n = 计算 op 数。
# 取负载机器上的实测上沿（10 次构造合计 ≤ 4.1e-4 秒/op），见 DEEPGEN.md
GEN_A = 4.0e-5
GEN_LS = 4.0e-5
# 预算闸门：候选按优先级逐个加入，直到 构造 + 每个候选约 2 次评估（起点评估 + 重排序评估）的
# 估计虚拟成本超过 q2/q3 求解器这一段预算的这一比例（dev2 按整个 k 的预算取 0.35；dev3 候选改在结构生成器之后
# 构造，这一段约为整个 k 的一半，取 0.6 保持同量级）
GATE_FRAC = 0.6


def layers(g):
    lay = {}
    for u in g.topo_order:
        lay[u] = max((lay[p] + 1 for p in g.pred[u]), default=0)
    return lay


def depth(g):
    lay = layers(g)
    return max(lay.values(), default=-1) + 1


def band_of(g, lay, B):
    """按层累计工作量把层切成 B 个深度带。"""
    L = max(lay.values()) + 1
    lw = [0.0] * L
    for u in g.compute_ids:
        lw[lay[u]] += g.cycles[u]
    tot, acc, bl = sum(lw) or 1.0, 0.0, []
    for l in range(L):
        bl.append(min(B - 1, int(acc / tot * B)))
        acc += lw[l]
    return {u: bl[lay[u]] for u in g.compute_ids}


def _vadd(a, b):
    r = dict(a)
    for x, w in b.items():
        r[x] = r.get(x, 0.0) + w
    return r


def mlpart(g, k, lay, B=30, eps=0.3, seed=SEED, maxfrac=0.25, passes=6):
    """深度带均衡的多层划分，返回 {op: core}。"""
    rng = random.Random(seed)
    band = band_of(g, lay, B)
    ids = list(g.compute_ids)
    idx = {u: i for i, u in enumerate(ids)}
    vw = [{band[u]: float(g.cycles[u])} for u in ids]
    adj = [defaultdict(float) for _ in ids]
    for u in ids:
        d = len(g.succ[u])
        for v in g.succ[u]:
            w = 1.0 / d
            adj[idx[u]][idx[v]] += w
            adj[idx[v]][idx[u]] += w
    bt = [0.0] * B
    for u in ids:
        bt[band[u]] += g.cycles[u]
    capb = [maxfrac * t / k for t in bt]
    levels = []
    cur_vw, cur_adj = vw, adj
    target = 40 * k
    while len(cur_vw) > target:
        n = len(cur_vw)
        match = [-1] * n
        order = list(range(n))
        rng.shuffle(order)
        order.sort(key=lambda i: sum(cur_vw[i].values()))
        for i in order:
            if match[i] >= 0:
                continue
            best, bw = -1, 0.0
            for j, w in cur_adj[i].items():
                if match[j] >= 0 or j == i or w <= bw:
                    continue
                s = _vadd(cur_vw[i], cur_vw[j])
                if any(s[b] > capb[b] for b in s):
                    continue
                best, bw = j, w
            if best >= 0:
                match[i] = best
                match[best] = i
            else:
                match[i] = i
        cmap = [-1] * n
        m = 0
        for i in range(n):
            if cmap[i] < 0:
                cmap[i] = m
                if match[i] != i:
                    cmap[match[i]] = m
                m += 1
        if m > 0.95 * n:
            break
        nvw = [dict() for _ in range(m)]
        nadj = [defaultdict(float) for _ in range(m)]
        for i in range(n):
            ci = cmap[i]
            for b, w in cur_vw[i].items():
                nvw[ci][b] = nvw[ci].get(b, 0.0) + w
            for j, w in cur_adj[i].items():
                cj = cmap[j]
                if cj != ci:
                    nadj[ci][cj] += w
        levels.append((cur_vw, cur_adj, cmap))
        cur_vw, cur_adj = nvw, nadj
    capv = [(1 + eps) * t / k for t in bt]
    n = len(cur_vw)
    load = [[0.0] * B for _ in range(k)]
    part = [-1] * n
    for i in sorted(range(n), key=lambda i: (min(cur_vw[i]), -sum(cur_vw[i].values()))):
        conn = [0.0] * k
        for j, w in cur_adj[i].items():
            if part[j] >= 0:
                conn[part[j]] += w
        vi = cur_vw[i]
        c = min(range(k), key=lambda c: (sum(max(0.0, load[c][b] + w - capv[b]) for b, w in vi.items()),
                                          -conn[c], sum(load[c][b] for b in vi)))
        part[i] = c
        for b, w in vi.items():
            load[c][b] += w

    def refine(vwl, adjl, part):
        load = [[0.0] * B for _ in range(k)]
        for i, p in enumerate(part):
            for b, w in vwl[i].items():
                load[p][b] += w
        for _ in range(passes):
            moved = 0
            for i in range(len(part)):
                p = part[i]
                conn = [0.0] * k
                for j, w in adjl[i].items():
                    conn[part[j]] += w
                best, bg = p, 0.0
                for c in range(k):
                    if c == p:
                        continue
                    gain = conn[c] - conn[p]
                    if gain <= bg:
                        continue
                    if all(load[c][b] + w <= max(capv[b], load[p][b]) for b, w in vwl[i].items()):
                        best, bg = c, gain
                if best != p:
                    for b, w in vwl[i].items():
                        load[p][b] -= w
                        load[best][b] += w
                    part[i] = best
                    moved += 1
            if not moved:
                break
        return part

    part = refine(cur_vw, cur_adj, part)
    for vwl, adjl, cmap in reversed(levels):
        part = [part[cmap[i]] for i in range(len(vwl))]
        part = refine(vwl, adjl, part)
    return {u: part[idx[u]] for u in ids}


def candidates(graph, k, problem=2, variants=VARIANTS, depth_min=DEPTH_MIN):
    """[(名字, 方案)]：只用输入图；浅图或 k<2 返回 []。"""
    if k < 2:
        return []
    lay = layers(graph)
    if max(lay.values(), default=-1) + 1 < depth_min:
        return []
    order = sorted(graph.compute_ids, key=lambda u: (lay[u], graph.topo_index[u]))
    parts, out = {}, []
    for B, eps, dec in variants:
        if (B, eps) not in parts:
            parts[(B, eps)] = mlpart(graph, k, lay, B, eps)
        core = parts[(B, eps)]
        if dec == 'ls':
            co, sq, _ = list_schedule(graph, k, order='blevel_comm', ddr='none', lam=0.0, fixed=core,
                                      seq_by='start', cache=(problem == 3))
            plan = per_op_plan(k, co, sq)
        else:
            plan = per_op_plan(k, core, order)
        out.append((f'deep_B{B}e{eps}_{dec}', plan))
    return out


def gen_cost(graph, variants=VARIANTS):
    n = len(graph.compute_ids)
    return GEN_A * n * len({(B, e) for B, e, _ in variants}) + GEN_LS * n * sum(d == 'ls' for *_, d in variants)


def gated(graph, problem, budget_s, variants=VARIANTS, frac=GATE_FRAC):
    """按优先级取变体前缀：构造成本 + 每个候选 2 次评估的估计虚拟成本不超过 frac·budget_s。"""
    if budget_s is None:
        return tuple(variants)
    from solvers import vclock
    ev = vclock.eval_cost(len(graph.compute_ids), problem)
    out = ()
    for v in variants:
        nxt = out + (v,)
        if gen_cost(graph, nxt) + 2 * ev * len(nxt) > frac * budget_s:
            break
        out = nxt
    return out


# ---------------------------------------------------------------------------------------------
# 求解入口：multistart + 深图候选（A/B 新臂）

def eligible(graph, k):
    """结构闸门（只看输入图）：深图、至少两条流水线各占 >= 10% 工作量、计算分量不多、关键路径不主导。
    单流水线的极深链（带内无 Cube/Vector 可并行，按带均分只会切断链）、分量上百的图（分量装箱已近最优）、
    关键路径 >= 每核流水线工作量一半的图（makespan 受链下界限制，少切边无收益），候选离线从未取胜。"""
    if k < MIN_K or depth(graph) < DEPTH_MIN:
        return False
    tw = graph.total_work or 1.0
    pw = graph.pipe_work
    if sum(w >= 0.1 * tw for w in pw.values()) < 2:
        return False
    if len(graph.components) > MAX_COMPONENTS:
        return False
    return graph.critical_path < CP_FRAC * max(pw.values()) / k


def _inner(graph, k, problem=2, seed=0, budget_s=None, variants=VARIANTS, **opts):
    """multistart.solve 调 q2/q3 求解器时（结构生成器与 k−1 起点之后）才构造候选并追加进 warm；
    构造成本记入虚拟时钟并从 q2/q3 求解器的预算里扣除，结构生成器拿到的预算与对照臂相同。"""
    from solvers import vclock
    from solvers.q2.dev import multistart
    if not eligible(graph, k):
        return multistart.solve(graph, k, problem, seed=seed, budget_s=budget_s, **opts)
    real = {'q2m': multistart.q2m, 'q3m': multistart.q3m}

    class _Proxy:
        def __init__(self, mod):
            self._mod = mod

        def __getattr__(self, name):
            return getattr(self._mod, name)

        def solve(self, *a, warm=None, budget_s=None, **kw):
            vs = gated(graph, problem, budget_s, variants)
            cands = candidates(graph, k, problem, vs) if vs else []
            clk = vclock.current()
            if cands and clk is not None:
                spent = gen_cost(graph, vs)
                clk.v += spent
                clk.v_charge += spent
                clk._check_cap()
                budget_s = max(1.0, budget_s - spent)
            return self._mod.solve(*a, warm=list(warm or []) + cands, budget_s=budget_s, **kw)

    multistart.q2m, multistart.q3m = _Proxy(real['q2m']), _Proxy(real['q3m'])
    try:
        return multistart.solve(graph, k, problem, seed=seed, budget_s=budget_s, **opts)
    finally:
        multistart.q2m, multistart.q3m = real['q2m'], real['q3m']


def solve(graph, k, problem=2, seed=0, budget_s=None, **opts):
    """与 solvers.vclock.solve 相同（虚拟时钟 + k−1 保底），只多了深图候选。"""
    from solvers import vclock
    global LAST_META
    plan = vclock._guarded('deepgen', graph, k, problem,
                           vclock.run_virtual(_inner, graph, k, problem, seed, budget_s, **opts))
    LAST_META = vclock.LAST_META
    return plan


def solve_multi(graph, k, problem=2, seed=0, budget_s=None, **opts):
    """dev4：交付版 order_postpass.solve_multi（核内多原则重排）套在 deepgen.solve 上。
    order_postpass 不改：调用期间把它引用的 vclock 换成一个只替换 solve 的代理，其余属性（LAST_META、
    eval_cost）仍取 solvers.vclock。deepgen.solve 同样经 vclock._guarded，换入 k−1 方案时 guard 以
    prev_k 开头，solve_multi 的重评判断不变。"""
    from solvers import vclock
    from solvers.q2.dev import order_postpass

    class _VC:
        def __getattr__(self, name):
            return getattr(vclock, name)

    shim = _VC()
    shim.solve = solve
    real = order_postpass.vclock
    order_postpass.vclock = shim
    try:
        plan = order_postpass.solve_multi(graph, k, problem, seed=seed, budget_s=budget_s, **opts)
    finally:
        order_postpass.vclock = real
    global LAST_META
    LAST_META = order_postpass.LAST_META
    return plan


def DETERMINISM_META():
    from solvers import vclock
    from solvers.q2.dev import order_postpass as op
    return dict(vclock.DETERMINISM_META(), deepgen=dict(
        version=VERSION, DEPTH_MIN=DEPTH_MIN, MIN_K=MIN_K, SEED=SEED, VARIANTS=VARIANTS,
        GEN_A=GEN_A, GEN_LS=GEN_LS, GATE_FRAC=GATE_FRAC, MAX_COMPONENTS=MAX_COMPONENTS, CP_FRAC=CP_FRAC),
        order_postpass=dict(version=op.VERSION_MULTI, rules=op.MULTI_RULES, max_evals=op.MAX_EVALS,
                            guard_s=op.GUARD_S))
