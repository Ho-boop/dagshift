"""块表示：op→核 + op→块次序 rank；核内顺序 = 按 (rank, 官方 Step1 位置) 做优先级的全局 Kahn 拓扑序。

来源：问题1 式的方案（每核若干连续子图块）在问题2 评估器里按“子图次序稳定分桶 + 桶内 Step1 序”执行。
把它改成逐 op 子图编码（encode.per_op_plan）、核内顺序仍按 (块次序, Step1 位置)，跨核 COPY_OUT
紧跟生产者发出，不必等到整块的末尾。评估器实测这种顺序明显好于列表调度的模拟开始时间序（见 NOTES.md）。

chunk_search：在块表示上做评估器驱动的邻域搜索。
- 边界迁移：随机取一条跨核边 (p@a → u@b)，把 u 周围同核的一片区域迁到 a 并并入 p 的块（rank 取 p 的），
  或把 p 周围的区域迁到 b 并入 u 的块；
- 负载迁移：计算量最大核上的一片区域迁到计算量最小的核，并入该核上与之相邻（有依赖）的块，没有则放到末块；
- 只接受评估器严格更优（common.sort_key）的候选；遵守 Archive 的截止时间。
"""

import heapq
import random

from .encode import per_op_plan
from .listsched import step1_order
from .search import REGION_SIZES, _cross_edges, _region


def kahn_order(g, prio):
    """全局拓扑序，同时就绪时 prio 小者先。"""
    indeg = {u: len(g.pred[u]) for u in g.compute_ids}
    heap = [(prio[u], u) for u in g.compute_ids if indeg[u] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in g.succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (prio[v], v))
    return out


def plan_to_chunks(plan):
    """方案 → (core_of, rank)：rank = 子图在所在核 core_schedules 里的位置。"""
    core_of, rank = {}, {}
    pos = {}
    for c, order in enumerate(plan['core_schedules']):
        for i, sg in enumerate(order):
            pos[sg] = (c, i)
    for op, sg in plan['node_to_subgraph'].items():
        c, i = pos[sg]
        core_of[int(op)] = c
        rank[int(op)] = float(i)
    return core_of, rank


def plan_to_chunks_exact(g, plan):
    """方案 → (core_of, rank)：rank = 子图在“依赖 + 各核 core_schedules 先后”这张图的一个全局拓扑序里的位置。

    plan_to_chunks 的 rank 只是核内位置，不同核之间不可比：全局 Kahn 按 (rank, Step1) 出队时，一个核上
    rank 大、已就绪的 op 会越过同核 rank 小、还在等跨核前驱的 op，核内顺序因此改变（逐 op 编码的方案
    解码后 makespan 可能大幅变差）。这里的 rank 是全局线性扩展的下标，Kahn 恰好按它出队，
    逐 op 编码的方案解码后核内顺序完全不变。该图有环（方案本身不可行）时返回 None。
    """
    sg_of = {int(op): sg for op, sg in plan['node_to_subgraph'].items()}
    pos, succ = {}, {}
    for c, order in enumerate(plan['core_schedules']):
        for i, sg in enumerate(order):
            pos[sg] = (i, c)
            succ[sg] = set()
            if i:
                succ[order[i - 1]].add(sg)
    for u in g.compute_ids:
        for v in g.succ[u]:
            a, b = sg_of[u], sg_of[v]
            if a != b:
                succ[a].add(b)
    indeg = dict.fromkeys(succ, 0)
    for a in succ:
        for b in succ[a]:
            indeg[b] += 1
    heap = [(pos[s], s) for s in succ if indeg[s] == 0]
    heapq.heapify(heap)
    idx = {}
    while heap:
        _, s = heapq.heappop(heap)
        idx[s] = float(len(idx))
        for b in succ[s]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(heap, (pos[b], b))
    if len(idx) < len(succ):
        return None
    core_of = {u: pos[sg_of[u]][1] for u in sg_of}
    return core_of, {u: idx[sg_of[u]] for u in sg_of}


def chunk_plan(g, k, core_of, rank):
    s1 = {u: i for i, u in enumerate(step1_order(g))}
    seq = kahn_order(g, {u: (rank[u], s1[u]) for u in g.compute_ids})
    return per_op_plan(k, core_of, seq)


def migrate_rank(g, core_of, rank, region, target):
    """区域迁到 target 核时的块次序：并入目标核上与之相邻（有依赖）的最早块，没有则放到末块。"""
    nbr = [rank[w] for v in region for w in list(g.pred[v]) + list(g.succ[v])
           if core_of[w] == target]
    return min(nbr) if nbr else max([rank[w] for w, c in core_of.items() if c == target],
                                    default=0.0)


def random_move(g, k, core_of, rank, rng):
    """chunk_search 的随机移动：返回 (region, target, rank) 或 None。"""
    cross = _cross_edges(g, core_of)
    size = rng.choice(REGION_SIZES)
    if cross and rng.random() < 0.8:
        p, u, _ = rng.choice(cross)
        if rng.random() < 0.5:
            return _region(g, core_of, u, size, rng), core_of[p], rank[p]
        return _region(g, core_of, p, size, rng), core_of[u], rank[u]
    loads = [0] * k
    for v, c in core_of.items():
        loads[c] += g.cycles[v]
    src = max(range(k), key=lambda c: loads[c])
    target = min(range(k), key=lambda c: loads[c])
    ops = [v for v, c in core_of.items() if c == src]
    if src == target or not ops:
        return None
    region = _region(g, core_of, rng.choice(ops), size, rng)
    return region, target, migrate_rank(g, core_of, rank, region, target)


def chunk_search(g, k, arc, core_of, rank, seed=0, max_iters=None, tag='ch'):
    """从 (core_of, rank) 出发的邻域搜索；当前解只在评估器严格更优时移动。"""
    if k < 2:
        return
    rng = random.Random(seed)
    core_of, rank = dict(core_of), dict(rank)
    arc.consider(f'{tag}_start', chunk_plan(g, k, core_of, rank))
    cur_key = arc.last_key
    it = 0
    while arc.can_afford(1.2):
        it += 1
        if max_iters is not None and it > max_iters:
            break
        mv = random_move(g, k, core_of, rank, rng)
        if mv is None:
            continue
        region, target, r = mv
        new_core, new_rank = dict(core_of), dict(rank)
        for v in region:
            new_core[v] = target
            new_rank[v] = r
        plan = arc.timed_gen(chunk_plan, g, k, new_core, new_rank)
        arc.consider(f'{tag}_move{it}', plan)
        if arc.last_key < cur_key:
            cur_key = arc.last_key
            core_of, rank = new_core, new_rank
