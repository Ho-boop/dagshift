"""Cold-start order candidate after a full virtual-budget base solve.

The established multistart search keeps its original deterministic virtual
budget and k-to-k track.  One graph-derived fixed-core candidate is generated
and evaluated afterward, with its real wall time charged to the enclosing
case worker.  The extra candidate is a fixed amount of structured work, not
an open-ended local search or an archive lookup.
"""

import time

from common import evaluate, op_to_core, sort_key

from solvers import vclock
from solvers.q2 import main as q2m
from solvers.q2.encode import per_op_plan
from solvers.q2.dev.order_construct import memory_order
from solvers.q2.dev.order_rules import comm_rank, timed_order


VERSION = 'q2q3-order-postpass-dev1'
GEN_S_PER_OP = 8.0e-5
LAST_META = None


def solve(graph, k, problem=2, seed=0, budget_s=None, **opts):
    """Preserve the base search path; submit a better fixed-core order if found."""
    base = vclock.solve(graph, k, problem, seed=seed, budget_s=budget_s, **opts)
    base_meta = dict(vclock.LAST_META or {})
    global LAST_META
    LAST_META = base_meta
    if k < 2 or base_meta.get('capped'):
        return base

    start = time.perf_counter()
    core_of = op_to_core(base)
    candidate = per_op_plan(k, core_of, memory_order(graph, core_of))
    row = evaluate(graph, candidate, problem, cache=False)
    archive = q2m.LAST_ARCHIVE
    base_row = archive.best[3] if archive is not None and archive.best is not None else evaluate(
        graph, base, problem, cache=False)
    extra_s = time.perf_counter() - start
    n = len(graph.compute_ids)
    LAST_META = dict(base_meta,
                     real_s=round(base_meta.get('real_s', 0.0) + extra_s, 3),
                     virtual_s=round(base_meta.get('virtual_s', 0.0)
                                     + GEN_S_PER_OP * n + vclock.eval_cost(n, problem), 3),
                     evals=base_meta.get('evals', 0) + 1)
    return candidate if sort_key(row) < sort_key(base_row) else base


# ---- dev2: a fixed list of order principles, best by the exact evaluator ----

VERSION_MULTI = 'q2q3-order-postpass-dev2'
# (name, virtual seconds per compute op charged for generation).  'memory' is
# the dev1 candidate, so dev2's candidate set contains dev1's.
MULTI_RULES = (('timed_g', 4.0e-5), ('timed', 4.0e-5), ('memory', GEN_S_PER_OP))
MAX_EVALS = 4   # <= one base re-evaluation (only without an archive row) + 3 candidates
# Wall-clock protection only (never the stopping rule): no new candidate is
# started once this k's post-pass has used this many real seconds.
GUARD_S = 90.0


def _generate(name, graph, core_of, cache):
    if name == 'memory':
        return memory_order(graph, core_of)
    if 'rank' not in cache:
        cache['rank'] = comm_rank(graph, core_of)[0]
    if name == 'timed_g':
        return timed_order(graph, core_of, rank=cache['rank'], graded=True)
    if name == 'timed':
        return timed_order(graph, core_of, rank=cache['rank'])
    if name == 'mem_xrank':
        return memory_order(graph, core_of, critical_weight=1.5, priority=cache['rank'])
    raise ValueError(name)


def solve_multi(graph, k, problem=2, seed=0, budget_s=None, **opts):
    """Base solve, then at most MAX_EVALS fixed-core orders; no k->k+1 propagation.

    Candidates come from ``MULTI_RULES`` in a fixed order; a plan identical
    to the base or to an earlier candidate is not re-evaluated.  Any error in
    the post-pass returns the base plan unchanged.
    """
    base = vclock.solve(graph, k, problem, seed=seed, budget_s=budget_s, **opts)
    base_meta = dict(vclock.LAST_META or {})
    global LAST_META
    LAST_META = base_meta
    if k < 2 or base_meta.get('capped'):
        return base

    start = time.perf_counter()
    n = len(graph.compute_ids)
    virtual = 0.0
    evals = 0
    tried = []
    try:
        core_of = op_to_core(base)
        archive = q2m.LAST_ARCHIVE
        # vclock's k-1 guard may return the padded k-1 plan instead of the
        # archive best; then the archive row does not describe ``base``.
        swapped = str(base_meta.get('guard', '')).startswith('prev_k')
        if archive is not None and archive.best is not None and not swapped:
            best_plan, best_row = base, archive.best[3]
        else:
            best_row = evaluate(graph, base, problem, cache=False)
            best_plan = base
            evals += 1
            virtual += vclock.eval_cost(n, problem)
        seen = [base]
        cache = {}
        for name, per_op in MULTI_RULES:
            if evals >= MAX_EVALS:
                break
            if time.perf_counter() - start > GUARD_S:
                tried.append((name, 'guard'))
                break
            candidate = per_op_plan(k, core_of, _generate(name, graph, core_of, cache))
            virtual += per_op * n
            if candidate in seen:
                tried.append((name, 'dup'))
                continue
            seen.append(candidate)
            row = evaluate(graph, candidate, problem, cache=False)
            evals += 1
            virtual += vclock.eval_cost(n, problem)
            tried.append((name, row.get('makespan')))
            if row.get('ok') and sort_key(row) < sort_key(best_row):
                best_plan, best_row = candidate, row
    except Exception as exc:  # post-pass must never lose the base plan
        best_plan = base
        tried.append(('error', repr(exc)[:200]))
    # The wall-clock guard is the only non-deterministic branch; when it fires
    # the row is marked capped (runner rule: deterministic only if uncapped).
    order_guard = any(v == 'guard' for _, v in tried)
    LAST_META = dict(base_meta,
                     capped=bool(base_meta.get('capped')) or order_guard,
                     order_guard=order_guard,
                     real_s=round(base_meta.get('real_s', 0.0) + time.perf_counter() - start, 3),
                     virtual_s=round(base_meta.get('virtual_s', 0.0) + virtual, 3),
                     evals=base_meta.get('evals', 0) + evals,
                     order_tried=tried)
    return best_plan
