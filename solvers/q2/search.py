"""问题2/3 邻域搜索：在 (op→核, 列表调度参数) 表示上做区域迁移，顺序每次由列表调度重新推导。

- 区域迁移：选一条跨核 tensor 边 (生产核 a → 消费核 b)，把消费端（或生产端）周围同核的一小片
  区域（沿同核依赖做 BFS，大小随机取 1/4/16/64）整体迁到对端核，减少跨核搬运与 500 周期同步；
- 负载迁移：从计算量最大的核迁一片区域到计算量最小的核；
- 每个候选先用列表调度的模拟 makespan 预筛（只用来少花评估次数，不用于宣称收益），
  通过预筛的才调用 common.evaluate；只接受评估器严格更优（common.sort_key）的候选；
- 遵守 Archive 的截止时间；已知最优只会被严格更优者替换，不会丢。
"""

import random

from .encode import per_op_plan
from .listsched import list_schedule
from .model import scene_b

REGION_SIZES = (1, 4, 16, 64)


def _region(g, core_of, seed_op, size, rng):
    """从 seed_op 出发，沿同核依赖 BFS 取至多 size 个 op。"""
    c = core_of[seed_op]
    region, frontier, seen = [seed_op], [seed_op], {seed_op}
    while frontier and len(region) < size:
        u = frontier.pop(0)
        nbrs = list(g.pred[u]) + list(g.succ[u])
        rng.shuffle(nbrs)
        for v in nbrs:
            if v not in seen and core_of[v] == c:
                seen.add(v)
                region.append(v)
                frontier.append(v)
                if len(region) >= size:
                    break
    return region


def _cross_edges(g, core_of):
    sb = scene_b(g)
    out = []
    for u in g.compute_ids:
        cu = core_of[u]
        for tid, size, p in sb.ins[u]:
            if core_of[p] != cu:
                out.append((p, u, size))
    return out


def local_search(g, k, arc, seed=0, screen_tol=0.03, max_iters=None):
    if k < 2:
        return
    rng = random.Random(seed)
    cur_key, rep = arc.best_rep
    core_of, params = dict(rep[0]), dict(rep[1])
    _, _, cur_est = arc.timed_gen(list_schedule, g, k, fixed=core_of, **params)
    it = 0
    while arc.can_afford(1.2):
        it += 1
        if max_iters is not None and it > max_iters:
            break
        cross = _cross_edges(g, core_of)
        size = rng.choice(REGION_SIZES)
        if cross and rng.random() < 0.7:
            p, u, _ = rng.choice(cross)
            if rng.random() < 0.5:
                region, target = _region(g, core_of, u, size, rng), core_of[p]
            else:
                region, target = _region(g, core_of, p, size, rng), core_of[u]
        else:
            loads = [0] * k
            for v, c in core_of.items():
                loads[c] += g.cycles[v]
            src = max(range(k), key=lambda c: loads[c])
            target = min(range(k), key=lambda c: loads[c])
            ops = [v for v, c in core_of.items() if c == src]
            if src == target or not ops:
                continue
            region = _region(g, core_of, rng.choice(ops), size, rng)
        new = dict(core_of)
        for v in region:
            new[v] = target
        c2, seq, est = arc.timed_gen(list_schedule, g, k, fixed=new, **params)
        if est > cur_est * (1 + screen_tol):
            continue
        arc.consider(f'ls_move{it}', per_op_plan(k, c2, seq), rep=(c2, params))
        if arc.best_rep[0] < cur_key:
            cur_key = arc.best_rep[0]
            core_of, cur_est = c2, est
