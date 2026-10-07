"""Deterministic, graph-derived order candidates for a fixed core assignment.

The candidate generator never reads prior solutions.  A caller supplies a core
assignment made during the current solve; the graph supplies dependencies,
tensor uses, pipe cycles, and the official Step1 order.  All candidates remain
topological and are encoded with one operation per subgraph.  The exact
evaluator, rather than the proxy below, decides whether a candidate is kept.
"""

import heapq

from common import CONFIG, op_to_core

from solvers.q2.encode import per_op_plan
from solvers.q2.listsched import step1_order


def _storage_class(graph, tid):
    pos = graph.tensors[tid].get('pos', 'L1')
    return 'UB' if pos == 'UB' else 'L1'


def _tensor_uses(graph, core_of):
    """Build the local-use view used by the cheap live-memory proxy."""
    inputs = {u: [] for u in graph.compute_ids}
    outputs = {u: [] for u in graph.compute_ids}
    remaining = {}
    for tid, size, producer, consumers in graph.interior_tensors:
        local = [u for u in consumers if core_of[u] == core_of[producer]]
        if not local:
            continue
        key = (tid, core_of[producer])
        remaining[key] = len(local)
        outputs[producer].append((key, size, _storage_class(graph, tid)))
        for u in local:
            inputs[u].append((key, size, _storage_class(graph, tid)))
    for tid, size, consumers in graph.input_tensors:
        by_core = {}
        for u in consumers:
            by_core.setdefault(core_of[u], []).append(u)
        for core, local in by_core.items():
            key = (tid, core)
            remaining[key] = len(local)
            for u in local:
                inputs[u].append((key, size, _storage_class(graph, tid)))
    return inputs, outputs, remaining


def memory_order(graph, core_of, *, critical_weight=0.0, window=12, priority=None):
    """Topological ready-list order with a bounded memory/criticality lookahead.

    The ready list is searched through the first ``window`` Step1 and bottom
    level entries.  This bounds construction to O((V+E) + window*V*log V),
    rather than rescoring every ready node at every step.  The proxy favors
    consuming live tensors before their buffers become old, completing a
    tensor's last local use, and staying under L1/UB capacity.  It deliberately
    does not claim to model Step2 spill or Step3 timing exactly.
    """
    if set(core_of) != set(graph.compute_ids):
        raise ValueError('core assignment must cover every compute operation')
    if window < 1:
        raise ValueError('window must be positive')

    step = {u: i for i, u in enumerate(step1_order(graph))}
    # `priority` (e.g. order_rules.comm_rank) replaces the plain bottom level.
    bottom = graph.bottom_level if priority is None else priority
    max_bottom = max(bottom.values(), default=1)
    count = max(1, len(graph.compute_ids) - 1)
    inputs, outputs, remaining = _tensor_uses(graph, core_of)
    live = set()
    used = [{ 'L1': 0, 'UB': 0 } for _ in range(max(core_of.values()) + 1)]
    caps = CONFIG['capacity']

    indeg = {u: len(graph.pred[u]) for u in graph.compute_ids}
    ready = set()
    by_step, by_crit = [], []

    def add_ready(u):
        ready.add(u)
        heapq.heappush(by_step, (step[u], u))
        heapq.heappush(by_crit, (-bottom[u], step[u], u))

    for u in graph.compute_ids:
        if indeg[u] == 0:
            add_ready(u)

    def pick_window(heap, limit, which):
        out = []
        while heap and len(out) < limit:
            item = heapq.heappop(heap)
            u = item[1] if which == 'step' else item[2]
            if u in ready:
                out.append((item, u))
        for item, _ in out:
            heapq.heappush(heap, item)
        return [u for _, u in out]

    def score(u):
        c = core_of[u]
        delta = {'L1': 0, 'UB': 0}
        reused = 0.0
        for key, size, area in inputs[u]:
            if key in live:
                reused += size / caps[area]
                if remaining[key] == 1:
                    delta[area] -= size
            else:
                # An external input becomes resident at its first local use.
                if key[0] in graph.tensors and key not in output_keys:
                    if remaining[key] > 1:
                        delta[area] += size
        for key, size, area in outputs[u]:
            delta[area] += size
        overflow = sum(max(0, used[c][area] + delta[area] - caps[area]) / caps[area]
                       for area in caps)
        pressure = sum(delta[area] / caps[area] for area in caps)
        return (8.0 * overflow + 2.0 * pressure - 0.5 * reused
                - critical_weight * bottom[u] / max_bottom
                + 0.05 * step[u] / count, step[u], u)

    output_keys = {key for values in outputs.values() for key, _, _ in values}
    order = []
    while ready:
        candidates = set(pick_window(by_step, window, 'step'))
        candidates.update(pick_window(by_crit, window, 'crit'))
        u = min(candidates, key=score)
        ready.remove(u)
        order.append(u)
        c = core_of[u]
        for key, size, area in inputs[u]:
            if key not in live and key not in output_keys:
                live.add(key)
                used[c][area] += size
            remaining[key] -= 1
            if remaining[key] == 0 and key in live:
                live.remove(key)
                used[c][area] -= size
        for key, size, area in outputs[u]:
            if remaining[key] > 0:
                live.add(key)
                used[c][area] += size
        for v in graph.succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                add_ready(v)
    if len(order) != len(graph.compute_ids):
        raise ValueError('compute graph is not acyclic')
    return order


def order_candidates(graph, k, plan):
    """Return three inexpensive order candidates retaining ``plan``'s cores."""
    core_of = op_to_core(plan)
    if set(core_of) != set(graph.compute_ids):
        return []
    step = step1_order(graph)
    return [
        ('order_step1', per_op_plan(k, core_of, step)),
        ('order_memory', per_op_plan(k, core_of, memory_order(graph, core_of))),
        ('order_memory_critical', per_op_plan(
            k, core_of, memory_order(graph, core_of, critical_weight=1.5))),
    ]
