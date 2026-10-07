"""交付入口的「追加候选」框架（可选模式；缺省关闭，python -m solvers.deliver 不带 --append 时逐字节同 deliver-v2）。

每个 k：
1. 先原样跑完整 v2（order_postpass.solve_multi），它的预算、随机流、k−1 链（multistart._PREV、vclock._GPREV）
   都不受追加影响——追加步骤不读写这些进程内状态，只拿 v2 的返回方案与其精确评估行；
2. 再按固定顺序生成已登记的追加候选（GENERATORS），逐个精确评估（cache=False，按本问题），每个 k 合计至多
   max_evals 次，不可行的候选不选；
3. 逐 k 按字典序 (makespan, added_copy_bytes)（common.sort_key）在 {v2 结果} ∪ {追加候选} 中取优，相等取 v2。
所以逐例、逐 k 都不劣于同一确定性 v2（v2 结果本身在集合里）。v2 的评估行拿不到时补评一次并计费。

计费：生成与评估按 vclock 的确定性口径（per-op 生成成本、vclock.eval_cost）另记为 append_virtual_s / append_evals，
加进本 k 的 virtual_s / evals；不进 v2 的虚拟时钟段，因而不挤占 v2 或后续 k 的预算。墙钟照实记在 wall_s 里。
k<2、v2 这一 k 已 capped（不再保证可复现）、生成器判定不适用时不追加。

候选登记（名字 -> 生成器）：生成器 f(graph, k, problem) -> (候选列表 [(名字, 方案)], 生成虚拟秒)，不适用时返回 ([], 0)。
- deep：dev4 深图结构构造（solvers.q2.dev.deepgen：层级 + 多层分核 B=30、eps=0.3），两个固定解码：
  ls（列表调度的核内顺序）与 timed_g（同一分核、order_postpass 的 timed_g 核内顺序）。适用条件同 deepgen.eligible。
"""

import time

VERSION = 'deliver-append-v1'
DEFAULT_MAX_EVALS = 2
TIMED_GEN = 4.0e-5      # timed_g 核内顺序的生成成本（虚拟秒/op），与 order_postpass.MULTI_RULES 相同


def _deep(graph, k, problem):
    from solvers.q2.dev import deepgen as dg
    from solvers.q2.dev import order_postpass
    from solvers.q2.encode import per_op_plan
    from solvers.q2.listsched import list_schedule
    if not dg.eligible(graph, k):
        return [], 0.0
    n = len(graph.compute_ids)
    lay = dg.layers(graph)
    B, eps = dg.VARIANTS[0][:2]
    core = dg.mlpart(graph, k, lay, B, eps)
    co, sq, _ = list_schedule(graph, k, order='blevel_comm', ddr='none', lam=0.0, fixed=core,
                              seq_by='start', cache=(problem == 3))
    out = [('deep_ls', per_op_plan(k, co, sq)),
           ('deep_timed_g', per_op_plan(k, core, order_postpass._generate('timed_g', graph, core, {})))]
    return out, (dg.GEN_A + dg.GEN_LS + TIMED_GEN) * n


GENERATORS = {'deep': _deep}


def parse(spec):
    """'deep' 或 'deep,xxx' -> 名字元组；None/''/'none' -> ()。"""
    if spec in (None, '', 'none'):
        return ()
    names = tuple(x.strip() for x in str(spec).split(',') if x.strip())
    bad = [x for x in names if x not in GENERATORS]
    if bad:
        raise SystemExit(f'unknown --append candidate(s) {bad}; known: {sorted(GENERATORS)}')
    return names


def versions(names):
    out = {'append': VERSION}
    if 'deep' in names:
        from solvers.q2.dev import deepgen
        out['deepgen'] = deepgen.VERSION
    return out


def determinism(names, max_evals):
    d = {'version': VERSION, 'names': list(names), 'max_evals': max_evals, 'TIMED_GEN': TIMED_GEN}
    if 'deep' in names:
        from solvers.q2.dev import deepgen
        d['deepgen'] = deepgen.DETERMINISM_META()
    return d


def _row(r):
    return None if r is None else {'ok': bool(r.get('ok')), 'makespan': r.get('makespan'),
                                   'added_copy_bytes': r.get('added_copy_bytes'), 'error': r.get('error_type')}


def _base_v2(graph, k, problem, seed, budget_s):
    """原样跑 order_postpass.solve_multi，同时截获它对返回方案的精确评估行。
    返回 (plan, row 或 None, meta)。截获字典持有方案引用，id 不会被复用。"""
    from solvers.q2 import main as q2m
    from solvers.q2.dev import order_postpass
    seen = {}
    real_eval = order_postpass.evaluate

    def rec(g, plan, prob, **kw):
        r = real_eval(g, plan, prob, **kw)
        seen[id(plan)] = (plan, r)
        return r

    order_postpass.evaluate = rec
    try:
        plan = order_postpass.solve_multi(graph, k, problem, seed=seed, budget_s=budget_s)
    finally:
        order_postpass.evaluate = real_eval
    meta = dict(order_postpass.LAST_META or {})
    hit = seen.get(id(plan))
    row = hit[1] if hit is not None and hit[0] is plan else None
    if row is None:
        # 与 order_postpass 相同的判断：未换入 k−1 方案时，档案最优行就是 base 的评估行
        arc = q2m.LAST_ARCHIVE
        swapped = str(meta.get('guard', '')).startswith('prev_k')
        if arc is not None and arc.best is not None and not swapped and arc.best[2] == plan:
            row = arc.best[3]
    return plan, row, meta


def solve(graph, k, problem, seed, budget_s, names, max_evals=DEFAULT_MAX_EVALS):
    """v2 + 追加候选。返回 (plan, meta)；meta = v2 的 LAST_META 加 append_* 字段。"""
    from common import evaluate, sort_key
    from solvers import vclock
    base, base_row, meta = _base_v2(graph, k, problem, seed, budget_s)
    info = {'append_choice': 'v2', 'append_evals': 0, 'append_virtual_s': 0.0, 'append_tried': []}
    if k < 2 or meta.get('capped') or not names:
        info['append_tried'] = [('skip', 'k<2' if k < 2 else 'capped' if meta.get('capped') else 'none')]
        return base, dict(meta, **info)
    t0 = time.perf_counter()
    n = len(graph.compute_ids)
    virtual, evals, tried = 0.0, 0, []
    best_plan, best_row, choice = base, base_row, 'v2'
    try:
        cands = []
        for nm in names:
            cs, gv = GENERATORS[nm](graph, k, problem)
            virtual += gv
            if not cs:
                tried.append((nm, 'n/a', None))
            cands += cs
        if cands and best_row is None:           # 拿不到 v2 的评估行：补评一次（计费，不占追加名额）
            best_row = evaluate(graph, base, problem, cache=False)
            evals += 1
            virtual += vclock.eval_cost(n, problem)
            tried.append(('v2', 'reeval', _row(best_row)))
        seen, used = [base], 0
        for nm, plan in cands:
            if used >= max_evals:
                tried.append((nm, 'cap', None))
                continue
            if plan in seen:
                tried.append((nm, 'dup', None))
                continue
            seen.append(plan)
            r = evaluate(graph, plan, problem, cache=False)
            used += 1
            evals += 1
            virtual += vclock.eval_cost(n, problem)
            tried.append((nm, 'eval', _row(r)))
            if r.get('ok') and sort_key(r) < sort_key(best_row):
                best_plan, best_row, choice = plan, r, nm
    except Exception as exc:                     # 追加失败绝不丢 v2 方案
        best_plan, choice = base, 'v2'
        tried.append(('error', repr(exc)[:300], None))
    info.update(append_choice=choice, append_evals=evals, append_virtual_s=round(virtual, 3),
                append_real_s=round(time.perf_counter() - t0, 3), append_tried=tried,
                append_base=_row(base_row), append_final=_row(best_row))
    meta = dict(meta, virtual_s=round(meta.get('virtual_s', 0.0) + virtual, 3),
                evals=meta.get('evals', 0) + evals, **info)
    return best_plan, meta


# ---------------------------------------------------------------------------------------------
# 基准钩子：experiments.run_case_budget --solver solvers.deliver.append:solve_deep --deterministic
# runner 只把固定几个 LAST_META 键写进回执；append_* 明细在环境变量 DELIVER_APPEND_LOG 指定目录时
# 另写 append_<pid>.jsonl（旁路日志，不影响方案）。

LAST_META = None


def _log(rec):
    import json
    import os
    d = os.environ.get('DELIVER_APPEND_LOG')
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'append_{os.getpid()}.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(t=round(time.time(), 3), **rec), ensure_ascii=False, default=str) + '\n')


def _hook(graph, k, problem, seed, budget_s, names, max_evals):
    global LAST_META
    plan, meta = solve(graph, k, problem, seed, budget_s, names, max_evals)
    LAST_META = meta
    _log(dict(case=getattr(graph, 'name', None), k=k, problem=problem, names=list(names), max_evals=max_evals,
              **{x: meta.get(x) for x in ('capped', 'guard', 'virtual_s', 'evals', 'append_choice', 'append_evals',
                                          'append_virtual_s', 'append_real_s', 'append_tried', 'append_base',
                                          'append_final')}))
    return plan


def solve_deep(graph, k, problem=2, seed=0, budget_s=None, **opts):
    return _hook(graph, k, problem, seed, budget_s, ('deep',), DEFAULT_MAX_EVALS)


def DETERMINISM_META():
    from solvers import vclock
    from solvers.q2.dev import order_postpass as op
    return dict(vclock.DETERMINISM_META(), append=determinism(('deep',), DEFAULT_MAX_EVALS),
                order_postpass=dict(version=op.VERSION_MULTI, rules=op.MULTI_RULES, max_evals=op.MAX_EVALS,
                                    guard_s=op.GUARD_S))
