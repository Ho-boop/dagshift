"""方案的廉价结构校验，用于搜索时预筛候选。

检查项与官方入口一致（stub_multicore_cut_and_schedule.derive_multicore_plan 与
evaluation_validation.validate_task_order），但复用 Graph 已建好的计算 DAG，不重复校验原图：
  format       两个字段、键为十进制 op id、值为非负 int
  coverage     恰好覆盖全部非 COPY op
  schedule     每个子图恰好在一个核的列表中出现一次；给定 num_cores 时核数必须相等
  subgraph_dag 收缩后的子图依赖图无环
  core_order   同核子图的列表顺序与依赖方向一致
  task_order   仅问题 1：子图依赖 + 每核相邻 Task 顺序边合起来无环
问题 2/3 还要检查 Step3 执行图 + Pipe FIFO + 跨核 COPY 的全局等待环，这一步必须走
evaluate()；因此问题 2/3 的 report.full_check 为 False。最终可行性一律以评估器为准。
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    ok: bool
    stage: str
    message: str = ''
    full_check: bool = False
    stats: dict = field(default_factory=dict)

    def __bool__(self):
        return self.ok


def _acyclic(nodes, edges):
    succ = {n: set() for n in nodes}
    indeg = dict.fromkeys(nodes, 0)
    for a, b in edges:
        if b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
    ready = deque(n for n, d in indeg.items() if d == 0)
    seen = 0
    while ready:
        n = ready.popleft()
        seen += 1
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return seen == len(indeg)


def subgraph_dependencies(graph, mapping):
    """{(src_sg, dst_sg)}：跨子图的计算依赖对（与官方 dependency_pairs 相同）。"""
    pairs = set()
    for u in graph.compute_ids:
        su = mapping[u]
        for v in graph.succ[u]:
            sv = mapping[v]
            if su != sv:
                pairs.add((su, sv))
    return pairs


def validate_plan(graph, plan, problem=None, num_cores=None):
    fail = lambda stage, msg: ValidationReport(False, stage, msg)

    if not isinstance(plan, dict) or set(plan) != {'node_to_subgraph', 'core_schedules'}:
        return fail('format', 'plan must contain exactly node_to_subgraph and core_schedules')
    raw_map = plan['node_to_subgraph']
    if not isinstance(raw_map, dict):
        return fail('format', 'node_to_subgraph must be an object')
    mapping = {}
    for key, sg in raw_map.items():
        if type(key) is not int and not (isinstance(key, str) and key.isascii() and key.isdecimal()):
            return fail('format', f'node_to_subgraph key {key!r} is not a decimal op id')
        if not isinstance(sg, int) or isinstance(sg, bool) or sg < 0:
            return fail('format', f'subgraph id {sg!r} must be a non-negative int')
        mapping[int(key)] = sg
    if len(mapping) != len(raw_map):
        return fail('format', 'duplicate op ids after int conversion')

    if mapping.keys() != graph.compute_set:
        missing = sorted(graph.compute_set - mapping.keys())[:10]
        extra = sorted(mapping.keys() - graph.compute_set)[:10]
        return fail('coverage', f'must cover exactly the non-COPY ops; missing={missing} extra={extra}')

    schedules = plan['core_schedules']
    if not isinstance(schedules, list) or not schedules or \
            any(not isinstance(o, list) for o in schedules):
        return fail('schedule', 'core_schedules must be a non-empty list of lists')
    if num_cores is not None and len(schedules) != num_cores:
        return fail('schedule', f'expected {num_cores} cores, got {len(schedules)}')
    flat = [sg for order in schedules for sg in order]
    if any(not isinstance(sg, int) or isinstance(sg, bool) for sg in flat):
        return fail('schedule', 'core schedule values must be ints')
    subgraphs = set(mapping.values())
    if len(flat) != len(set(flat)) or set(flat) != subgraphs:
        return fail('schedule', 'every subgraph must appear on exactly one core exactly once')

    pairs = subgraph_dependencies(graph, mapping)
    if not _acyclic(subgraphs, pairs):
        return fail('subgraph_dag', 'contracted subgraph graph contains a cycle')

    core_of = {sg: c for c, order in enumerate(schedules) for sg in order}
    pos = {sg: i for order in schedules for i, sg in enumerate(order)}
    for a, b in pairs:
        if core_of[a] == core_of[b] and pos[a] >= pos[b]:
            return fail('core_order', f'core {core_of[a]}: subgraph {a} must precede {b}')

    if problem == 1:
        order_edges = [(a, b) for order in schedules for a, b in zip(order, order[1:])]
        if not _acyclic(subgraphs, list(pairs) + order_edges):
            return fail('task_order', 'subgraph dependencies + per-core task order form a cycle')

    stats = {'num_subgraphs': len(subgraphs), 'dependency_pairs': len(pairs),
             'num_cores': len(schedules),
             'nonempty_cores': sum(1 for o in schedules if o)}
    return ValidationReport(True, 'ok', '', full_check=(problem == 1), stats=stats)
