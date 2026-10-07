"""问题2 solver 入口：保底基线 + 热启动 + 块表示重排 + 列表调度候选组合 + 评估器驱动的邻域搜索。

solve(graph, k, problem=2, seed=0, budget_s=None, **opts) -> plan
    opts:
      portfolio  列表调度参数组合（默认 PORTFOLIO）；'none' 表示不跑组合，但仍做块重排与搜索（消融）
      search     True/False：有剩余预算时是否做邻域搜索（块搜索 + 列表调度搜索，默认 True）
      warm       [(name, plan)] 热启动候选；缺省见 archive_candidates（按当前问题语义重新评估）
      chunks     True/False：热启动/基线方案转块表示（逐 op 编码，核内按 (块次序, Step1 位置)）并做块搜索（默认 True）
      reseq      True/False：热启动方案保留 op→核，用列表调度重排核内顺序后再参评（默认 True）
      crit       True/False：块搜索用关键链引导的移动（critsearch.crit_search），False 为 v2.1 的随机移动（默认 True）
      decode     'exact'/'core'：方案转块表示时 rank 取全局线性扩展下标（默认，逐 op 编码的方案无损）
                 或核内位置（v2.3 及以前，逐 op 编码的方案解码后可能大幅变差，见 NOTES.md §15）
      accept / lahc_len / sa_t0 / reuse_full   透传给 crit_search（默认 strict / 20 / 0.005 / True；消融用，见 NOTES.md §12、§13）
恒不比保底基线差：保底基线的两个变体总是先参评，之后只接受严格更优（common.sort_key）的候选。
问题3 复用本入口（problem=3 时用问题3 评估器择优）。
"""

import time

from common import best_plan, evaluate, from_groups, make_plan, op_to_core, sort_key

from solvers.baseline import component_binpack

from .chunks import chunk_plan, chunk_search, plan_to_chunks, plan_to_chunks_exact
from .cluster import edge_clusters
from .critsearch import crit_search
from .encode import per_op_plan
from .listsched import list_schedule

VERSION = 'q2-ls-v2.4'

_BC = dict(order='blevel_comm', ddr='none', seq_by='start')

# 列表调度参数组合，按贪心覆盖顺序：样例 80 个 (case,k) 上从保底基线出发，逐个加入使“取最好者”的
# 平均加速比增益最大的变体（grid1 + grid4 网格，见 solvers/q2/NOTES.md）。
# 除 list_schedule 的参数外：fixed='comp' 按保底基线的分量装箱固定核；
# cl=(m, w) 用 cluster.edge_clusters 做核亲和簇（簇工作量上限 = 总计算量/(k·m)，w 为边权 'bytes'/'ratio'）。
PORTFOLIO = [
    dict(_BC, lam=8.0),
    dict(_BC, lam=2.0, cl=(4, 'ratio')),
    dict(order='step1', ddr='none', lam=2.0, seq_by='decision'),
    dict(_BC, lam=2.0),
    dict(_BC, lam=2.0, cl=(16, 'ratio')),
    dict(order='step1', ddr='none', lam=2.0, seq_by='start', cl=(4, 'bytes')),
    dict(_BC, lam=2.0, cl=(8, 'ratio')),
    dict(_BC, lam=2.0, cl=(32, 'ratio')),
    dict(order='blevel', ddr='none', lam=0.0, seq_by='start', fixed='comp'),
    # v2 追加（grid5 网格贪心覆盖的前三个）
    dict(_BC, lam=8.0, cl=(16, 'ratio')),
    dict(_BC, lam=8.0, seq_by='decision'),
    dict(_BC, lam=4.0, cl=(4, 'ratio')),
]

RESEQ_PARAMS = dict(_BC, lam=2.0)   # 热启动方案重排核内顺序时的列表调度参数（op→核全部固定）

NO_BUDGET_SEARCH_ITERS = 40     # budget_s=None 时邻域搜索的迭代上限（保证能结束）


def component_assignment(g, k):
    """保底基线的核归属：计算分量按总周期贪心装箱。"""
    cc = component_binpack(g, k)
    return {u: c for comp, c in zip(g.components, cc) for u in comp}


def baseline_candidates(g, k):
    comps = g.components
    cc = component_binpack(g, k)
    fine = from_groups(comps, cc, num_cores=k)
    per_core = [[] for _ in range(k)]
    for comp, c in zip(comps, cc):
        per_core[c].extend(comp)
    cores = [c for c in range(k) if per_core[c]]
    bundled = from_groups([per_core[c] for c in cores], cores, num_cores=k)
    return [('base_bundled', bundled), ('base_fine', fine)]


class Archive:
    """已评估候选里的最优（字典序：makespan，再 added_copy_bytes）；只接受严格更优。"""

    def __init__(self, g, problem, deadline):
        self.g, self.problem, self.deadline = g, problem, deadline
        self.best = None            # (key, name, plan, row, rep)
        self.best_rep = None        # (key, rep)：带 (core_of, 参数) 表示的最优候选，供列表调度邻域搜索起步
        self.best_chunk = None      # (key, (core_of, rank))：块表示的最优候选，供块搜索起步
        self.last_key = None        # 最近一次评估的字典序键
        self.eval_cost = 0.0        # 观测到的单次（未命中缓存）评估耗时上界
        self.gen_cost = 0.0         # 观测到的单次列表调度耗时上界
        self.log = []

    def time_left(self):
        return float('inf') if self.deadline is None else self.deadline - time.perf_counter()

    def can_afford(self, factor=1.0):
        return self.time_left() > factor * self.eval_cost + self.gen_cost

    def timed_gen(self, fn, *args, **kw):
        t = time.perf_counter()
        out = fn(*args, **kw)
        self.gen_cost = max(self.gen_cost, time.perf_counter() - t)
        return out

    def consider(self, name, plan, rep=None, chunk=None):
        r = evaluate(self.g, plan, self.problem)
        if not r.get('cached'):
            self.eval_cost = max(self.eval_cost, r.get('eval_seconds') or 0.0)
        key = sort_key(r)
        self.last_key = key
        self.log.append((name, r.get('makespan'), r.get('ok')))
        if rep is not None and r.get('ok') and (self.best_rep is None or key < self.best_rep[0]):
            self.best_rep = (key, rep)
        if chunk is not None and r.get('ok') and (self.best_chunk is None or key < self.best_chunk[0]):
            self.best_chunk = (key, chunk)
        if self.best is None or key < self.best[0]:
            self.best = (key, name, plan, r, rep)
            return True
        return False


def archive_candidates(g, k, problems=(2, 3, 1), smaller_k=(2, 1)):
    """最优档案里的已知方案，一律按当前问题的语义重新评估，不照搬档案里的评分。

    problems   取这些问题在同一 k 下的最优方案（含本问题：从上一轮最优继续，保证逐轮不变差；
               问题3 的方案按问题2 复评常比问题2 档案更优，但 summarize 不把它收进问题2 档案，所以这里直接取）
    smaller_k  这些问题在 k−1 核下的最优方案补一个空核（允许空核），保证核数增加时不变差
    """
    out = []
    wanted = [(p, k) for p in problems]
    if k > 1:
        wanted += [(p, k - 1) for p in smaller_k]
    for p, kk in wanted:
        plan = best_plan(p, kk, g.name)
        if plan is None or len(plan['core_schedules']) != kk:
            continue
        if kk < k:
            plan = make_plan(plan['node_to_subgraph'], plan['core_schedules'] + [[]] * (k - kk))
        out.append((f'best_q{p}_k{kk}', plan))
    return out


def solve(graph, k, problem=2, seed=0, budget_s=None, portfolio=None, search=True, warm=None,
          reseq=True, chunks=True, crit=True, decode='exact', **opts):
    start = time.perf_counter()
    deadline = None if budget_s is None else start + budget_s
    arc = Archive(graph, problem, deadline)
    starts = []                 # (评估键, 次序, 名字, 方案)：块表示/重排的起点
    for name, plan in baseline_candidates(graph, k):
        arc.consider(name, plan)
        if name == 'base_fine':
            starts.append((arc.last_key, 1, name, plan))
    # 默认热启动：问题2/问题3/问题1 同 k 的最优方案、问题2/问题1 在 k−1 核下的最优方案补空核（均按当前问题重评）；
    # 问题3 入口另传
    warm = archive_candidates(graph, k) if warm is None else [] if warm == 'none' else warm
    for name, plan in warm:
        if arc.can_afford(1.0):
            arc.consider(name, plan)
            starts.append((arc.last_key, 0, name, plan))
    starts.sort(key=lambda s: (s[0], s[1]))
    starts = [(name, plan) for _, _, name, plan in starts]     # 按评估结果从好到差：预算紧时先改造最好的
    # 块表示：热启动方案（和保底基线的细分量方案）保留 op→核与子图次序，改成逐 op 编码，
    # 核内按 (块次序, Step1 位置) 排；也作为块搜索的起点
    for name, plan in (starts if chunks else ()):
        if not arc.can_afford(1.3):
            break
        core_of, rank = plan_to_chunks(plan)
        if set(core_of) != set(graph.compute_ids):
            continue
        exact = plan_to_chunks_exact(graph, plan) if decode == 'exact' else None
        if len(plan['node_to_subgraph']) == len(set(plan['node_to_subgraph'].values())):
            # 已是逐 op 编码：按全局线性扩展取 rank，解码后核内顺序不变（问题2 档案里逐 op 编码的方案实测逐位一致，
            # NOTES.md §15），不必重评，直接作块搜索起点。decode='core' 为 v2.3 及以前的核内位置 rank（有损）
            key = sort_key(evaluate(graph, plan, problem))      # 刚评估过，命中缓存
            if key[0] == 0 and (arc.best_chunk is None or key < arc.best_chunk[0]):
                arc.best_chunk = (key, exact or (core_of, rank))
            continue
        # 多 op 子图：核内位置 rank 让跨核 COPY_OUT 不必等整块结束（见 chunks.py），全局 rank 则保持原顺序；两者都参评
        arc.consider(name + '_chunk', arc.timed_gen(chunk_plan, graph, k, core_of, rank),
                     chunk=(core_of, rank))
        if exact is not None and arc.can_afford(1.3):
            arc.consider(name + '_chunkx', arc.timed_gen(chunk_plan, graph, k, *exact), chunk=exact)
    # 热启动方案保留 op→核，核内顺序改由列表调度推导（逐 op 子图编码），也作为列表调度邻域搜索的起点
    for name, plan in (starts if reseq else ()):
        if name.startswith('base'):
            continue
        if not arc.can_afford(1.3):
            break
        fixed = op_to_core(plan)
        if set(fixed) != set(graph.compute_ids):
            continue
        core_of, seq, est = arc.timed_gen(list_schedule, graph, k, fixed=fixed, seed=seed,
                                          **RESEQ_PARAMS)
        arc.consider(name + '_reseq', per_op_plan(k, core_of, seq), rep=(core_of, dict(RESEQ_PARAMS)))

    variants = PORTFOLIO if portfolio is None else [] if portfolio == 'none' else portfolio
    comp_fixed = None
    for i, v in enumerate(variants):
        if not arc.can_afford(1.3):
            break
        params = dict(v)
        fixed = group = None
        if params.pop('fixed', None) == 'comp':
            if comp_fixed is None:
                comp_fixed = component_assignment(graph, k)
            fixed = comp_fixed
        cl = params.pop('cl', None)
        if cl is not None:
            m, w = cl
            group = edge_clusters(graph, graph.total_work / (k * m), weight=w)
        core_of, seq, est = arc.timed_gen(list_schedule, graph, k, fixed=fixed, group=group,
                                          seed=seed, **params)
        plan = per_op_plan(k, core_of, seq)
        arc.consider(f'ls{i}', plan, rep=(core_of, params))

    if search:
        crit_kw = dict(CRIT_OPTS, **{n: opts[n] for n in CRIT_KEYS if n in opts})
        _search(graph, k, arc, seed, deadline, crit, crit_kw)
    global LAST_ARCHIVE
    LAST_ARCHIVE = arc          # 仅供诊断：最优候选的名字与评估日志（arc.best[1]、arc.log）
    return arc.best[2]


LAST_ARCHIVE = None

SEARCH_SPLIT = 0.6      # 两种表示都有起点时，起点更优的那种搜索先用剩余时间的这一比例


CRIT_OPTS = dict(p_rand=0.3, sideways=False)     # 关键链引导块搜索的参数（等时对比见 NOTES.md §8）
CRIT_KEYS = ('accept', 'lahc_len', 'sa_t0', 'reuse_full')    # 可由 solve 的 opts 覆盖的 crit_search 参数


def _search(graph, k, arc, seed, deadline, crit=True, crit_kw=None):
    """剩余预算内做邻域搜索：块搜索（从 best_chunk）与列表调度搜索（从 best_rep），起点更优者先跑。
    crit=True 时块搜索用关键链引导（critsearch.crit_search），否则用随机移动（chunks.chunk_search）。"""
    from .search import local_search
    runs = []
    if arc.best_chunk is not None:
        runs.append((arc.best_chunk[0], 'chunk'))
    if arc.best_rep is not None:
        runs.append((arc.best_rep[0], 'ls'))
    runs.sort()
    for i, (_, kind) in enumerate(runs):
        if not arc.can_afford(2.0):
            break
        sub = None
        if deadline is not None and i == 0 and len(runs) > 1:
            sub = time.perf_counter() + SEARCH_SPLIT * arc.time_left()
        saved, arc.deadline = arc.deadline, (sub if sub is not None else arc.deadline)
        iters = NO_BUDGET_SEARCH_ITERS if deadline is None else None
        if kind == 'chunk':
            core_of, rank = arc.best_chunk[1]
            if crit:
                crit_search(graph, k, arc, core_of, rank, seed=seed, max_iters=iters,
                            **(CRIT_OPTS if crit_kw is None else crit_kw))
            else:
                chunk_search(graph, k, arc, core_of, rank, seed=seed, max_iters=iters)
        else:
            local_search(graph, k, arc, seed=seed, max_iters=iters)
        arc.deadline = saved
