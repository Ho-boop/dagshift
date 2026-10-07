"""保底基线：计算连通分量按总周期贪心装箱（来自 analysis/route_probe.py）。

不是论文主算法，只用于：打通流水线、给最优档案一个保底、作为消融里的“简单负载贪心”。
    variant='bundled'   每核一个子图（分量之间没有依赖，任何核数都合法）
    variant='fine'      每个分量一个子图，按装箱顺序排在各核上
    variant='best'      两者都评估，取 makespan 更小者（problem 决定用哪个评估器）
"""

from common import evaluate, from_groups, sort_key


def component_binpack(graph, k):
    loads = [0] * k
    core_of_comp = []
    for comp in graph.components:
        core = min(range(k), key=lambda c: (loads[c], c))
        loads[core] += sum(graph.cycles[u] for u in comp)
        core_of_comp.append(core)
    return core_of_comp


def solve(graph, k, problem, seed=0, budget_s=None, variant='best'):
    comps = graph.components
    core_of_comp = component_binpack(graph, k)
    fine = from_groups(comps, core_of_comp, num_cores=k)
    per_core = [[] for _ in range(k)]
    for comp, core in zip(comps, core_of_comp):
        per_core[core].extend(comp)
    groups = [ops for ops in per_core if ops]
    cores = [c for c in range(k) if per_core[c]]
    bundled = from_groups(groups, cores, num_cores=k)
    if variant == 'bundled':
        return bundled
    if variant == 'fine':
        return fine
    results = [(sort_key(evaluate(graph, p, problem)), i, p)
               for i, p in enumerate((bundled, fine))]
    return min(results)[2]
