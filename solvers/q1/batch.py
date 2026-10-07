"""batch 图候选族：多条同构计算链（同样大小的连通分量）共享同一批权重输入。

问题1 里每个 Task 都要从 DDR 重新读入自己用到的权重，而 DDR 是所有核共享的 60 B/cycle。
把一个分量切成层段、每段再把 n 条链分成 D_s 组，组成 Task = (层段 s, 分量组 j)：
    权重读入量 ≈ D_s * W_s（W_s 是层段 s 的共享权重字节），计算量 n * C_s 分到 D_s 个 Task；
    两者平衡给出 D* = sqrt(60 * n * C_s / W_s)，再乘系数 beta 扫描，截到 [1, dmax]。
层段 s 的组依赖上一层段与它分量有交集的组；因为分量之间没有计算依赖、分量内按拓扑位置切，Task 图一定无环。
核分配用简单时间模型贪心（同核 +100、跨核 +1000），每核按估计开始时间排序。
先按“离优先网格近”的顺序扫描 (层段数, 切法, beta, dmax, 贪心模式)，再从最优配置出发做局部搜索
（切点移动、D_s ±1、拆段、并段），全部候选只以 common.evaluate 为准。
"""

import math
import random
import time

from common import from_groups

B = 60.0
CROSS = 1000
SAME = 100
MODES = ('greedy', 'greedy_bw', 'home')
MIN_SHARE = 0.5     # 同构链的 op 数至少占计算 op 的一半才算 batch 图
SKIP_RATIO = 1.05   # 扫描最优比当前全局最优差 5% 以上：不做局部搜索，把预算还给主流程


def detect(graph):
    """返回 batch 结构信息；没有 >=2 条同大小链或它们之间没有共享输入时返回 None。"""
    comps = graph.components
    if len(comps) < 2:
        return None
    topo = graph.topo_index
    size0 = len(comps[0])
    batch = [sorted(c, key=lambda u: topo[u]) for c in comps if len(c) == size0]
    if len(batch) < 2 or size0 < 2 or len(batch) * size0 < MIN_SHARE * len(graph.compute_ids):
        return None
    extra = [sorted(c, key=lambda u: topo[u]) for c in comps if len(c) != size0]
    cof = graph.component_of
    pos = {}
    for c in batch:
        for j, u in enumerate(c):
            pos[u] = j
    L = size0
    cyc = [graph.cycles[u] for u in batch[0]]
    w = [0] * L
    a_in = [0] * L
    c0 = set(batch[0])
    for t, cons in graph.consumers.items():
        cs = {cof[v] for v in cons if v in cof}
        size = graph.tensors[t].get('size', 0)
        for v in cons:
            if v in c0:
                if len(cs) > 1:
                    w[pos[v]] += size
                else:
                    a_in[pos[v]] += size
    if sum(w) == 0:
        return None
    return {'batch': batch, 'extra': extra, 'L': L, 'cyc': cyc, 'w': w, 'a_in': a_in}


def _chunks(n, d):
    d = max(1, min(n, d))
    out, a = [], 0
    for j in range(d):
        b = a + (n - a) // (d - j)
        out.append((a, b))
        a = b
    return out


def build(graph, k, info, cuts, D, mode='greedy'):
    batch, extra, cyc, w, a_in = info['batch'], info['extra'], info['cyc'], info['w'], info['a_in']
    n = len(batch)
    tasks = []   # (ops, compute, bytes, preds, first comp)
    prev = []
    for s in range(len(cuts) - 1):
        lo, hi = cuts[s], cuts[s + 1]
        C, W, A = sum(cyc[lo:hi]), sum(w[lo:hi]), sum(a_in[lo:hi])
        cur = []
        for (a, b) in _chunks(n, D[s]):
            ops = [batch[i][j] for i in range(a, b) for j in range(lo, hi)]
            preds = [t for (pa, pb, t) in prev if pa < b and a < pb]
            tasks.append((ops, C * (b - a), W + A * (b - a), preds, a))
            cur.append((a, b, len(tasks) - 1))
        prev = cur
    for c in extra:
        tasks.append((c, sum(graph.cycles[u] for u in c), 0, [], 0))
    free = [0.0] * k
    used = [False] * k
    fin = [0.0] * len(tasks)
    start = [0.0] * len(tasks)
    core_of = [0] * len(tasks)
    bw = k if mode == 'greedy_bw' else 1.0
    for t, (ops, C, By, preds, a0) in enumerate(tasks):
        dur = max(C, By / B * bw) + 50
        cand = [a0 * k // n] if mode == 'home' else range(k)
        best = None
        for c in cand:
            st = free[c] + (SAME if used[c] else 0)
            for p in preds:
                st = max(st, fin[p] + (0 if core_of[p] == c else CROSS))
            key = (st + dur, c)
            if best is None or key < best[0]:
                best = (key, c, st)
        _, c, st = best
        core_of[t], start[t], fin[t] = c, st, st + dur
        free[c], used[c] = st + dur, True
    orders = [[] for _ in range(k)]
    for t in sorted(range(len(tasks)), key=lambda t: (start[t], t)):
        orders[core_of[t]].append(t)
    return from_groups([t[0] for t in tasks], core_of, orders=orders, num_cores=k)


def _cuts_uniform(L, S):
    return sorted(set(round(i * L / S) for i in range(S + 1)))


def _cuts_balanced(vals, S):
    L = len(vals)
    tot = sum(vals) or 1
    cuts, acc, nxt = [0], 0, 1
    for j, v in enumerate(vals):
        acc += v
        while nxt < S and acc >= tot * nxt / S:
            if cuts[-1] < j + 1 < L:
                cuts.append(j + 1)
            nxt += 1
    cuts.append(L)
    return sorted(set(cuts))


def _dprofile(info, cuts, beta, dmax):
    n = len(info['batch'])
    D = []
    for s in range(len(cuts) - 1):
        C = sum(info['cyc'][cuts[s]:cuts[s + 1]])
        W = sum(info['w'][cuts[s]:cuts[s + 1]])
        d = math.sqrt(B * n * C / max(W, 1)) * beta
        D.append(int(max(1, min(dmax, round(d)))))
    return D


def _grid(info, k):
    """扫描网格，按各维“优先级序号之和”排序（由粗到细的对角线顺序）。"""
    L, n = info['L'], len(info['batch'])
    S_list = [S for S in (4, 5, 6, 3, 8, 10, 12, 16, 20, 24, 31) if S <= L]
    betas = (1.0, 0.7, 1.4, 0.5, 2.0, 0.35, 3.0, 100.0)
    dmaxs = [d for d in dict.fromkeys((min(n, 2 * k), min(n, k), n))]
    cutf = (('u', lambda S: _cuts_uniform(L, S)),
            ('c', lambda S: _cuts_balanced(info['cyc'], S)),
            ('w', lambda S: _cuts_balanced([x + 1 for x in info['w']], S)))
    items = []
    for i1, S in enumerate(S_list):
        for i2, (cn, cf) in enumerate(cutf):
            for i3, beta in enumerate(betas):
                for i4, dmax in enumerate(dmaxs):
                    for i5, mode in enumerate(MODES):
                        items.append((i1 + i2 + i3 + i4 + i5, (S, cn, cf, beta, dmax, mode)))
    items.sort(key=lambda x: x[0])
    return [it[1] for it in items]


def search(graph, k, info, run, time_left, seed=0, sweep_share=0.4, ref=None):
    """run(name, plan) -> 排序键（common.sort_key）；time_left() -> 剩余秒数（已扣一次评估的估计）；
    ref() -> 当前全局最优键（基线、热启动等），扫描最优差它 SKIP_RATIO 以上就提前返回。
    返回 (最优键, 配置描述)。"""
    t0 = time.perf_counter()
    total = max(0.0, time_left())
    sweep_end = t0 + sweep_share * total
    best, cfg = None, None
    seen = set()
    for (S, cn, cf, beta, dmax, mode) in _grid(info, k):
        if time_left() <= 0 or (best is not None and time.perf_counter() > sweep_end):
            break
        cuts = cf(S)
        D = _dprofile(info, cuts, beta, dmax)
        sig = (tuple(cuts), tuple(D), mode)
        if sig in seen:
            continue
        seen.add(sig)
        key = run(f'batch_{cn}{S}_b{beta}_d{dmax}_{mode}', build(graph, k, info, cuts, D, mode))
        if key is not None and (best is None or key < best):
            best, cfg = key, (list(cuts), D, mode)
    if cfg is None or (ref is not None and best[0] > SKIP_RATIO * ref()[0]):
        return best, cfg
    rng = random.Random(seed * 7919 + k)
    n = len(info['batch'])
    cuts, D, mode = cfg
    cur, curk = (list(cuts), list(D), mode), best
    it = 0
    while time_left() > 0:
        it += 1
        cs, Ds, m = list(cur[0]), list(cur[1]), cur[2]
        S = len(Ds)
        mv = rng.random()
        if mv < 0.35:
            s = rng.randrange(S)
            Ds[s] = max(1, min(n, Ds[s] + rng.choice((-1, 1))))
        elif mv < 0.7 and S > 1:
            i = rng.randrange(1, S)
            v = cs[i] + rng.choice((-4, -2, -1, 1, 2, 4))
            if not cs[i - 1] < v < cs[i + 1]:
                continue
            cs[i] = v
        elif mv < 0.85:
            s = rng.randrange(S)
            if cs[s + 1] - cs[s] < 2:
                continue
            cs.insert(s + 1, rng.randrange(cs[s] + 1, cs[s + 1]))
            Ds.insert(s + 1, Ds[s])
        elif S > 1:
            i = rng.randrange(1, S)
            cs.pop(i)
            Ds.pop(i)
        else:
            continue
        m = rng.choice(('greedy', 'greedy_bw', m, m))
        if (tuple(cs), tuple(Ds), m) in seen:
            continue
        seen.add((tuple(cs), tuple(Ds), m))
        key = run(f'batch_ls{it}', build(graph, k, info, cs, Ds, m))
        if key is not None and key <= curk:
            cur, curk = (cs, Ds, m), key
    return curk, (cur, it, time.perf_counter() - t0)
