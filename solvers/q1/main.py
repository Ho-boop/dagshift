"""问题1 solver 入口：保底基线 + 代理稠密扫描 + 预算内真评估 + 参数邻域搜索。

流程（见 solvers/q1/ALGORITHM.md）：
1. 保底：基线两个变体（分量装箱 bundled / fine）都必评，保证不差于保底基线。
   热启动（warm）：再评最优档案里本题 k 核与 k−1 核的方案（k−1 核补一个空核），挡住负载导致的回退。
1b. batch 图（>=2 条同大小的计算链共享权重输入，且占计算 op 一半以上；batch.detect）：
   Task = (层段, 分量组) 的候选族，每段组数按 权重/计算 比取 D* = sqrt(60 n C_s / W_s)；
   先扫描再局部搜索，占约 BATCH_FRAC 的预算；扫描最优比当前最优差 5% 以上就提前把预算还给后面的流程。
2. 代理扫描：init.sweep_specs 由粗到细产出参数规格（逐层分带、HEFT、层带 LPT/亲和 × 各种 Task 切法，
   层带数 S 先粗网格再逐个加密；与层周期对齐的 S（init.layer_blocks，每层剖面自相关）排在粗网格之前；
   另有只用 kc < k 个核的少量规格，以及把层带内宽连通组按枢纽拆开的 wsplit 规格，单列一族），
   Builder 构造方案、ProxyModel 打分；限时约 SWEEP_FRAC 的预算。
   只保留 (代理分, 规格)，评估时再重建方案，控制内存。
3. 真评估队列：代理前几名 → 每个生成器族的代理最优 → 其余按“校准后的代理分”逐个挑；用到约 INIT_FRAC 的预算。
   校准：代理不含 spill，按生成器族记录已评估成员的 min(实测/代理)，乘到该族候选的代理分上。
4. search.spec_search：从当前最优的参数规格出发做邻域爬山，到约 SEARCH_FRAC 的预算。
5. search.group_ils：在真评估最优的层带（或 comp）规格的核分配上做“组换核”迭代局部搜索（代理引导），
   每到代理创新低的局部最优就真评估一次，再随机扰动约 5% 的组继续（group_ls='ls' 退回单次爬山）；
   占剩余时间的 GROUP_SHARE。
5b. search.task_ils：固定当前真评估最优方案（可能来自热启动）的子图划分，只换 Task 所在核的迭代局部搜索，到截止。
6. 剩余预算继续评估队列；返回 common.sort_key 最优（先 makespan，再 added_copy_bytes）。
候选好坏只以 common.evaluate 为准。
"""

import time

from common import evaluate, singlecore_makespan, sort_key, plan_hash, best_plan, make_plan
from solvers.baseline import solve as _baseline

from .structure import View, ProxyModel
from . import init
from . import batch
from .search import spec_search, group_ls, group_ils, task_ils

VERSION = 'q1-v2.9'

DEFAULT_BUDGET = 60.0
SWEEP_FRAC = 0.35         # 代理扫描截止（占预算比例）
SWEEP_MIN_FRAC = 0.15     # 基线评估之后至少留给扫描的预算比例
TOP_PROXY = 4             # 先评代理前几名（可用满整个预算）
INIT_FRAC = 0.55          # 初始队列截止（占预算比例）
SEARCH_FRAC = 0.72        # 参数邻域搜索截止；其后是组换核局部搜索（代理引导）与一次真评估
BIG_OPS = 10000           # 大图：单次评估数秒、扫描常被截断；保持 v2.1 的时间分配（初始队列到 0.65，
BIG_INIT_FRAC = 0.65      # 参数搜索可用到截止），少用核规格放到扫描第 1 层之后，组换核只用参数搜索剩下的时间
GROUP_SHARE = 0.85        # 局部搜索时间里组换核占的比例，其余给 Task 级换核（每步 O(T)，很便宜）
BATCH_FRAC = 0.5          # batch 图：层段 x 分量组候选族（batch.py）用到的预算比例


def archive_candidates(graph, k):
    """热启动：最优档案里本题 k 核与 k−1 核的方案（k−1 核方案补一个空核，仍合法且 makespan 不变）。"""
    out = []
    for kk in (k, k - 1):
        if kk < 1:
            continue
        plan = best_plan(1, kk, graph.name)
        if plan is None or len(plan['core_schedules']) != kk:
            continue
        if kk < k:
            plan = make_plan(plan['node_to_subgraph'], plan['core_schedules'] + [[] for _ in range(k - kk)])
        out.append((f'best_q1_k{kk}', plan))
    return out


_WS_FAM = True            # wsplit 规格单列一族：各自取族内代理最优进队列、各自校准（代理常高估拆开后的方案）


def _family(spec):
    g = spec['gen']
    mode = spec.get('mode', 'phase')
    fam = ('band*' if spec['S'] == 'each' else 'band', mode) if g == 'band' else (g, mode)
    if _WS_FAM and spec.get('wsplit'):
        fam += ('ws',)
    return fam + ('<k',) if spec.get('kc') else fam


def _queue(scored):
    """scored: [(proxy, spec)]，已按 proxy 升序。返回 (先评部分, 其余)：先评 = 代理前几名 + 每个生成器族的代理最优
    （先用满 k 核的族，再少用核的族：大图上评估次数少，少用核的族排后）。"""
    order, seen_fam, used = [], set(), set()
    for i, (_, spec) in enumerate(scored[:TOP_PROXY]):
        order.append(i)
        used.add(i)
        seen_fam.add(_family(spec))
    for fewer in (False, True):
        for i, (_, spec) in enumerate(scored):
            fam = _family(spec)
            if fam not in seen_fam and bool(spec.get('kc')) == fewer:
                seen_fam.add(fam)
                order.append(i)
                used.add(i)
    return [scored[i] for i in order], [scored[i] for i in range(len(scored)) if i not in used]


def _sig(plan):
    return hash((tuple(plan['node_to_subgraph'].values()), tuple(map(tuple, plan['core_schedules']))))


class _Ctx:
    def __init__(self, graph, k, view, proxy, builder, deadline):
        self.graph, self.k, self.view = graph, k, view
        self._proxy, self.builder, self.deadline = proxy, builder, deadline
        self.n = len(graph.compute_ids)
        self.best = None          # (key, name, spec, plan)
        self.best_s = None        # 来自参数规格的真评估最优 (key, spec)：参数邻域搜索的起点与比较基准（热启动方案不算）
        self.best_band = None     # 真评估最优的层带（或 comp）规格 (key, spec)：组换核局部搜索的起点
        self.ls_gens = ('band',)
        self.seen = set()
        self.rate = None          # 单次评估秒数 / op（观测最大值）
        self.log = []
        self.ratio = {}           # 生成器族 -> 已评估成员的 min(实测/代理)，用于校准排序（代理不含 spill）

    def calib(self, spec):
        """代理分的校准系数：该族已评估成员的最小 实测/代理；没有则取所有族的中位数。"""
        if spec is not None and _family(spec) in self.ratio:
            return self.ratio[_family(spec)]
        vals = sorted(self.ratio.values())
        return vals[len(vals) // 2] if vals else 1.0

    def rank(self, spec, px):
        return px * self.calib(spec)

    def build(self, spec):
        return self.builder.build(spec)

    def proxy(self, plan, spec=None):
        """代理分；给出 spec 时乘上该族的校准系数。"""
        try:
            px = self._proxy.evaluate(plan)
        except Exception:
            return float('inf')
        return px if spec is None else self.rank(spec, px)

    def est(self):
        return (self.rate or 3e-4) * self.n

    def time_left(self):
        return self.deadline - time.perf_counter() - self.est()

    def run(self, name, plan, spec=None):
        h = plan_hash(plan)
        if h in self.seen:
            return None
        self.seen.add(h)
        r = evaluate(self.graph, plan, 1)
        if not r.get('cached') and r.get('eval_seconds'):
            e = r['eval_seconds'] / max(1, self.n)
            self.rate = e if self.rate is None else max(self.rate, e)
        key = sort_key(r)
        self.log.append((name, r.get('makespan'), r.get('ok')))
        if spec is not None and r.get('ok') and r.get('makespan'):
            px = self._proxy.evaluate(plan)
            if px > 0:
                fam = _family(spec)
                q = r['makespan'] / px
                self.ratio[fam] = min(q, self.ratio.get(fam, q))
        if self.best is None or key < self.best[0]:
            self.best = (key, name, spec, plan)
        if spec is not None and (self.best_s is None or key < self.best_s[0]):
            self.best_s = (key, spec)
        if spec is not None and spec['gen'] in self.ls_gens and (self.best_band is None or key < self.best_band[0]):
            self.best_band = (key, spec)
        return key

    def best_key(self):
        return self.best_s[0]

    def best_spec(self):
        return self.best_s[1]


def solve(graph, k, problem=1, seed=0, budget_s=None, verbose=False, **opts):
    global _WS_FAM
    t0 = time.perf_counter()
    _WS_FAM = bool(opts.get('ws_fam', True))
    budget = float(budget_s) if budget_s else DEFAULT_BUDGET
    view = View(graph)
    proxy = ProxyModel(view, single_makespan=singlecore_makespan(graph))
    builder = init.Builder(view, k)
    ctx = _Ctx(graph, k, view, proxy, builder, t0 + budget)
    big = ctx.n > BIG_OPS
    if opts.get('ls_comp', True):
        ctx.ls_gens = ('band', 'comp')
    ctx.ils_split = opts.get('ils_split', 0)

    # 1. 保底：基线两个变体都必评
    ctx.run('baseline_bundled', _baseline(graph, k, 1, variant='bundled'), {'gen': 'comp', 'mode': 'phase'})
    ctx.run('baseline_fine', _baseline(graph, k, 1, variant='fine'), None)
    if opts.get('warm', True):
        for name, plan in archive_candidates(graph, k):
            ctx.run(name, plan, None)

    # 1b. batch 图（多条同构链共享权重）：层段 x 分量组候选族，扫描 + 局部搜索，占约 BATCH_FRAC 的预算
    binfo = batch.detect(graph) if opts.get('batch', True) and k >= 2 else None
    if binfo is not None:
        b_end = t0 + budget * opts.get('batch_frac', BATCH_FRAC)
        batch.search(graph, k, binfo, lambda name, plan: ctx.run(name, plan, None),
                     lambda: b_end - time.perf_counter() - ctx.est(), seed=seed, ref=lambda: ctx.best[0])
        # 其余流程按剩余时间重新分配比例（task_ils 会接着在 batch 最优方案上换核）
        t0 = time.perf_counter()
        budget = max(1.0, ctx.deadline - t0)

    # 2. 代理扫描（限时）
    now = time.perf_counter()
    sweep_end = max(t0 + budget * SWEEP_FRAC, now + budget * SWEEP_MIN_FRAC)
    sweep_end = min(sweep_end, ctx.deadline - ctx.est())
    scored, sigs, names, n_specs = [], set(), {}, 0
    for spec in init.sweep_specs(view, names, k, kc_late=big, wide=tuple(opts.get('wide', (1.0,))),
                                 ws_pos=opts.get('ws_pos', 0), align=bool(opts.get('align', True))):
        name = init.spec_name(spec)
        if name in names:
            continue
        if n_specs and time.perf_counter() > sweep_end:
            break
        n_specs += 1
        plan = builder.build(spec)
        if plan is None:
            names[name] = float('inf')
            continue
        px = ctx.proxy(plan)
        names[name] = px
        s = _sig(plan)
        if s in sigs:
            continue
        sigs.add(s)
        scored.append((px, spec))
    scored.sort(key=lambda x: x[0])
    t_sweep = time.perf_counter() - t0

    # 3. 真评估队列
    init_deadline = t0 + budget * (BIG_INIT_FRAC if big else INIT_FRAC)
    head, rest = _queue(scored)

    def run_spec(spec):
        ctx.run(init.spec_name(spec), builder.build(spec), spec)

    def run_rest(deadline):
        # 其余候选按校准后的代理分逐个挑（每评一个，校准系数都可能更新）
        while rest and time.perf_counter() + ctx.est() <= deadline:
            j = min(range(len(rest)), key=lambda i: ctx.rank(rest[i][1], rest[i][0]))
            run_spec(rest.pop(j)[1])

    for i, (px, spec) in enumerate(head):
        if time.perf_counter() + ctx.est() > (ctx.deadline if i < TOP_PROXY else init_deadline):
            rest[:0] = head[i:]
            break
        run_spec(spec)
    run_rest(init_deadline)

    # 4. 参数邻域搜索：从当前最优（若它来自规格）出发
    search_deadline = ctx.deadline if big else t0 + budget * SEARCH_FRAC
    if ctx.best_s is not None:
        spec_search(ctx, ctx.best_s[1], lambda: search_deadline - time.perf_counter() - ctx.est())

    # 5. 组换核局部搜索：从真评估最优的层带规格出发，代理引导；结果真评估一次，更优才采用
    ls = opts.get('group_ls', 'ils')
    tls = opts.get('task_ls', True)
    if ctx.best_band is not None and ls:
        spec = ctx.best_band[1]
        g_deadline = ctx.deadline
        if tls:
            now = time.perf_counter()
            g_deadline = now + opts.get('group_share', GROUP_SHARE) * max(0.0, ctx.deadline - now)
        if ls == 'ils':
            group_ils(ctx, spec, g_deadline, seed=seed, name=init.spec_name(spec) + '+ls')
        else:
            plan = group_ls(ctx, spec, g_deadline - 1.5 * ctx.est(), seed=seed)
            if plan is not None:
                ctx.run(init.spec_name(spec) + '+ls', plan, None)

    # 5b. Task 级换核：固定当前真评估最优方案的子图划分，只换 Task 所在核（代理引导，局部最优真评估）
    if tls:
        task_ils(ctx, ctx.best[3], ctx.deadline, seed=seed, name=ctx.best[1] + '+t',
                 tie=opts.get('task_tie', True))

    # 6. 剩余预算继续按校准后的代理分评估
    run_rest(ctx.deadline)

    if verbose:
        print(f'sweep: {n_specs} specs, {len(scored)} distinct, {t_sweep:.1f}s')
        for row in ctx.log:
            print(row)
        print('best', ctx.best[1], ctx.best[0], f'{time.perf_counter() - t0:.1f}s')
    return ctx.best[3]
