"""Communication-aware order rules for a fixed core assignment (Q2/Q3).

Adds two principles to the tensor-liveness rule of ``order_construct``:

* critical path first, measured with the fixed assignment: a cross-core edge
  costs COPY_OUT + 500-cycle release + COPY_IN, so an op whose output feeds
  another core carries that core's remaining path in its rank;
* cross-core output early: ``timed_order`` list-schedules every (core, pipe)
  FIFO in simulated time, so a remote consumer's wait is visible and the op
  feeding it wins as soon as it can start.

Only the input graph and the caller's core assignment are read.  Every result
is a topological order of all compute ops; the exact evaluator decides whether
it replaces the caller's plan.  Costs are proxies, not evaluator numbers.
"""

import heapq
import math
from collections import defaultdict

from common import CONFIG

from solvers.q2.listsched import step1_order
from solvers.q2.dev.order_construct import _tensor_uses


def _edge_bytes(graph):
    """Bytes moved along each compute edge if its ends sit on different cores."""
    size = defaultdict(int)
    for _tid, nbytes, producer, consumers in graph.interior_tensors:
        for v in consumers:
            size[(producer, v)] += nbytes
    for u, v, nbytes in graph.direct_edges:
        if u in graph.compute_set and v in graph.compute_set:
            size[(u, v)] += nbytes or 0
    return size


def comm_rank(graph, core_of):
    """Upward rank: own cycles + longest path to exit incl. cross-core copies."""
    bw = CONFIG['bandwidth']
    sync = CONFIG['cross_core_copy_delay']
    size = _edge_bytes(graph)
    rank = {}
    for u in reversed(graph.topo_order):
        best = 0
        cu = core_of[u]
        for s in graph.succ[u]:
            r = rank[s]
            if core_of[s] != cu:
                r += sync + 2 * max(1, math.ceil(size.get((u, s), 0) / bw))
            if r > best:
                best = r
        rank[u] = graph.cycles[u] + best
    return rank, size


class _Memory:
    """Live-tensor proxy shared with ``order_construct.memory_order``."""

    def __init__(self, graph, core_of):
        self.inputs, self.outputs, self.remaining = _tensor_uses(graph, core_of)
        self.output_keys = {key for vals in self.outputs.values() for key, _, _ in vals}
        self.live = set()
        self.used = [{'L1': 0, 'UB': 0} for _ in range(max(core_of.values()) + 1)]
        self.caps = CONFIG['capacity']
        self.core_of = core_of

    def score(self, u):
        """(overflow, net live growth - 0.25 * reused live input), capacity units."""
        c = self.core_of[u]
        delta = {'L1': 0, 'UB': 0}
        reused = 0.0
        for key, size, area in self.inputs[u]:
            if key in self.live:
                reused += size / self.caps[area]
                if self.remaining[key] == 1:
                    delta[area] -= size
            elif key not in self.output_keys and self.remaining[key] > 1:
                delta[area] += size
        for _key, size, area in self.outputs[u]:
            delta[area] += size
        over = sum(max(0, self.used[c][a] + delta[a] - self.caps[a]) / self.caps[a]
                   for a in self.caps)
        growth = sum(delta[a] / self.caps[a] for a in self.caps)
        return over, growth - 0.25 * reused

    def overflow(self, u):
        return self.score(u)[0]

    def commit(self, u):
        c = self.core_of[u]
        for key, size, area in self.inputs[u]:
            if key not in self.live and key not in self.output_keys:
                self.live.add(key)
                self.used[c][area] += size
            self.remaining[key] -= 1
            if self.remaining[key] == 0 and key in self.live:
                self.live.remove(key)
                self.used[c][area] -= size
        for key, size, area in self.outputs[u]:
            if self.remaining[key] > 0:
                self.live.add(key)
                self.used[c][area] += size


def timed_order(graph, core_of, *, window=4, rank=None, graded=False):
    """Fixed-assignment list schedule over per-(core, pipe) FIFOs.

    The queue whose next op can start earliest is served first.  Within that
    queue the highest communication-aware rank wins, except that among the top
    ``window`` ranks an op that keeps L1/UB under capacity is preferred
    (``graded``: the one that least grows the live set).
    O((V + E) log V) with at most k * |pipes| queues scanned per step.
    """
    if set(core_of) != set(graph.compute_ids):
        raise ValueError('core assignment must cover every compute operation')
    bw = CONFIG['bandwidth']
    sync = CONFIG['cross_core_copy_delay']
    if rank is None:
        rank, size = comm_rank(graph, core_of)
    else:
        size = _edge_bytes(graph)
    step = {u: i for i, u in enumerate(step1_order(graph))}
    mem = _Memory(graph, core_of)

    indeg = {u: len(graph.pred[u]) for u in graph.compute_ids}
    avail = {}
    finish = {}
    waiting = defaultdict(list)   # queue -> heap (avail, -rank, step, u)
    ready = defaultdict(list)     # queue -> heap (-rank, step, u)
    free = defaultdict(int)       # queue -> pipe free time

    def queue(u):
        return (core_of[u], graph.pipe[u])

    def release(u):
        t = 0
        cu = core_of[u]
        for p in graph.pred[u]:
            f = finish[p]
            if core_of[p] != cu:
                f += sync + 2 * max(1, math.ceil(size.get((p, u), 0) / bw))
            if f > t:
                t = f
        avail[u] = t
        heapq.heappush(waiting[queue(u)], (t, -rank[u], step[u], u))

    for u in graph.compute_ids:
        if indeg[u] == 0:
            release(u)

    order = []
    total = len(graph.compute_ids)
    while len(order) < total:
        best = None
        for q in set(waiting) | set(ready):
            r, w = ready[q], waiting[q]
            if r:
                start, pr = free[q], r[0][0]
                if w and w[0][0] <= start and w[0][1] < pr:
                    pr = w[0][1]
            elif w:
                start, pr = max(free[q], w[0][0]), w[0][1]
            else:
                continue
            key = (start, pr, q)
            if best is None or key < best:
                best = key
        if best is None:
            raise ValueError('compute graph is not acyclic')
        start, _, q = best
        r, w = ready[q], waiting[q]
        while w and w[0][0] <= start:
            _t, nr, st, u = heapq.heappop(w)
            heapq.heappush(r, (nr, st, u))
        picked = []
        while r and len(picked) < window:
            picked.append(heapq.heappop(r))
        if graded:
            # memory-first inside the window: least overflow, then least growth
            choice = min(picked, key=lambda it: (*mem.score(it[2]), it))
        else:
            choice = min(picked, key=lambda it: (mem.overflow(it[2]) > 0, it))
        for it in picked:
            if it is not choice:
                heapq.heappush(r, it)
        u = choice[2]
        mem.commit(u)
        order.append(u)
        finish[u] = start + graph.cycles[u]
        free[q] = finish[u]
        for v in graph.succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                release(v)
    return order
