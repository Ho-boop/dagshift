"""方向 A（开发中，未合入正式入口）：问题2/问题3 在同一次求解里多源构造起点，再做问题2 的搜索。

动机（已用评估器验证，见 STATUS 09-25 S3 条目）：样例 20 例上，把问题1 档案方案直接按问题2 评估，
比问题2 求解器冷启动 120 s/例 的结果更好（k=5：T/T档案 1.077 对 1.182），两者取优能补回约 2/3 的差距。
说明问题2 自己的构造偏弱，问题1 的结构化生成器（分量装箱、层带与宽组拆分、batch 族）给问题2 的起点更好。
档案不能读（等于查表），所以在同一进程里现场生成：

每个 k 的预算分两段：
1. 结构化构造（约 struct_frac）：调用问题1 求解器的完整流程（保底基线、batch 族、代理扫描、真评估队列、
   参数邻域、组换核/Task 换核），但候选一律用目标问题（2 或 3）的评估器择优。问题1 代码不改：
   只在调用期间把 solvers.q1.main 模块里的 evaluate 换成目标问题的评估（该模块唯一的评估入口是 _Ctx.run）。
   问题1 的代理模型按场景 A 估时，只用于排序；各族用 min(实测/代理) 校准，实测即目标问题的评估。
2. 问题2 求解器（solvers.q2.main.solve；问题3 用 solvers.q3.main 的组合）：以第 1 段最优方案、
   以及同一进程里上一个 k 的最优方案（补一个空核）为起点，做块表示、重排、列表调度组合与关键链搜索。
不读最优档案；k−1→k 的复用只在同一进程内，时间计入本次求解。
"""

import time

from common import evaluate, make_plan

import solvers.q1.main as q1m
from solvers.q2 import main as q2m
from solvers.q3 import main as q3m

VERSION = 'q2q3-multistart-dev1'
STRUCT_FRAC = 0.5
DEFAULT_BUDGET = 60.0
MARGIN_EVALS = 1.5      # 每段预留的余量 = 这么多次单次评估耗时（两段内部的求解器都可能在截止前再发起一次评估）

_PREV = {}      # (graph_hash, problem) -> (k, plan)：同一进程内上一个 k 的最优方案
_EVAL_T = {}    # (graph_hash, problem) -> 实测单次评估秒数（同进程内测一次）


def eval_seconds(graph, k, problem):
    """实测一次评估耗时。用问题2 求解器第一个必评的保底方案（base_bundled），随后求解器评它时命中进程内缓存，不重复计费。
    dev0 在最大图 case_014 上两段各超时约 7 s，累计越过用例截止（A/B abA，09-25）。"""
    key = (graph.graph_hash, problem)
    if key not in _EVAL_T:
        plan = q2m.baseline_candidates(graph, k)[0][1]
        t = time.perf_counter()
        r = evaluate(graph, plan, problem)
        seconds = time.perf_counter() - t
        if r.get('cached'):     # 命中磁盘缓存时测不到真实耗时，按规模估（与问题1 求解器的缺省估计相同）
            seconds = max(seconds, 3e-4 * len(graph.compute_ids))
        _EVAL_T[key] = seconds
    return _EVAL_T[key]


def struct_candidate(graph, k, problem, budget_s, seed=0):
    """问题1 的结构化生成与搜索流程，按目标问题的评估器择优；返回最优方案（可能为 None）。"""
    original = q1m.evaluate
    q1m.evaluate = lambda g, plan, _problem, **kw: evaluate(g, plan, problem, **kw)
    try:
        return q1m.solve(graph, k, problem=1, seed=seed, budget_s=budget_s, warm=False)
    finally:
        q1m.evaluate = original


def solve(graph, k, problem=2, seed=0, budget_s=None, struct_frac=STRUCT_FRAC, chain=True, **opts):
    start = time.perf_counter()
    budget = float(budget_s) if budget_s else DEFAULT_BUDGET
    margin = MARGIN_EVALS * eval_seconds(graph, k, problem) + 0.2
    warm = []
    struct_budget = struct_frac * budget - margin
    if k >= 2 and struct_frac > 0 and struct_budget >= 3 * margin:
        plan = struct_candidate(graph, k, problem, struct_budget, seed)
        if plan is not None:
            warm.append(('struct_q1gen', plan))
    prev = _PREV.get((graph.graph_hash, problem)) if chain else None
    if prev is not None and prev[0] < k:
        pk, pplan = prev
        warm.append((f'prev_k{pk}', make_plan(pplan['node_to_subgraph'],
                                              pplan['core_schedules'] + [[]] * (k - pk))))
    rest = max(1.0, budget - (time.perf_counter() - start) - margin)
    if problem == 3:
        plan = q3m.solve(graph, k, problem=3, seed=seed, budget_s=rest, warm=warm, **opts)
    else:
        plan = q2m.solve(graph, k, problem=problem, seed=seed, budget_s=rest, warm=warm, **opts)
    _PREV[(graph.graph_hash, problem)] = (k, plan)
    return plan


# 消融入口（dev_ab 的臂只能指定 module:function，参数写死在这里）
def solve_f30(graph, k, problem=2, **opts):
    return solve(graph, k, problem, struct_frac=0.3, **opts)


def solve_f70(graph, k, problem=2, **opts):
    return solve(graph, k, problem, struct_frac=0.7, **opts)


def solve_nochain(graph, k, problem=2, **opts):
    return solve(graph, k, problem, chain=False, **opts)
