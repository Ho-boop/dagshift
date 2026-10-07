"""确定性预算：虚拟时钟。求解结果只取决于 (图, k, 问题, 种子, 预算)，与机器快慢、负载无关。

动机：09-24 同一份代码在负载下成绩差 15%～20%，原因是 S1/S2 求解器全部按墙钟截止（perf_counter）。
做法（不改求解器代码）：进入 virtual_clock 时，把各求解器模块里的名字 `time` 换成垫片、`evaluate` 换成包装：
  * 垫片的 perf_counter() 返回虚拟时间；每读一次钟推进一个小步长 tick（按调用点所在函数标定，乘以图规模 n），
    其余属性（time.time、time.sleep ...）原样转发；
  * 包装的 evaluate 每次调用推进一个固定评估成本（按问题与 n 标定），与是否命中缓存无关；
    本上下文内同一方案（plan_hash, problem）第二次起按“重复”小成本计。“是否见过”用本上下文自己的集合判断，不看磁盘缓存；
  * 返回值里的 eval_seconds 换成虚拟评估成本、cached 一律为 False，
    因此问题1 _Ctx.rate、问题2 Archive.eval_cost、multistart 的 eval_seconds 都是确定的；
  * 问题1 求解器取的单核基准 singlecore_makespan 每个进程、每张图第一次按一次评估计费；
  * critsearch 直接查磁盘缓存决定要不要 keep_full 评估，上下文内把它的 _read_disk 换成“永远未命中”（只剩进程内缓存，确定）。
真实时间安全上限（cap_s）是唯一的非确定分支：真实耗时超过上限时，虚拟时间一次性跳到很远的将来（粘滞），
所有截止判断立即为假，求解器按各自逻辑尽快返回；触发会记入 EVENTS 并打印到 stderr。

用法
    from solvers import vclock
    with vclock.virtual_clock(graph, budget_s=100, problem=2):
        plan = multistart.solve(graph, k, 2, seed=0, budget_s=100)
    vclock.solve(graph, k, problem, seed, budget_s)   # run_case_budget 可直接用的入口，内部包 multistart.solve

标定与测试（只在 E 核上跑时加 --ecores）：
    python -m solvers.vclock record   --out <dir> --problems 2,3 --per-k 30 --workers 3 --ecores
    python -m solvers.vclock fit      --out <dir>
    python -m solvers.vclock check    --out <dir> --problems 2,3 --per-k 30 --workers 3 --ecores
    python -m solvers.vclock determinism --out <dir> --case case_039 --problem 2 --budget 30 --load 1
    python -m unittest experiments.tests.test_vclock
"""

import contextlib
import hashlib
import json
import math
import sys
import time as _time

VERSION = 'vclock-v2'     # v2：solve/solve_q1/solve_q2main 加 k−1 保底（_guarded）；v1 的 solve 路径不变

# ---------------------------------------------------------------------------------------------
# 常数（标定见 solvers/VCLOCK.md；E 核、3 个并发进程上测得，按中位数 真实/虚拟 ≈ 1 统一缩放）

SCALE = 1.05                        # 全局缩放：换机器后只需重标这一个数
# 单次评估成本 = EVAL_A[p] * n ** EVAL_B[p] 秒（n = 计算 op 数）
EVAL_A = {1: 3.64e-4, 2: 2.01e-4, 3: 2.40e-4}
EVAL_B = {1: 0.907, 2: 1.008, 3: 0.990}
KEEP_FULL_FACTOR = 1.0              # keep_full 评估相对普通评估的成本倍数（实测同 n 下两者相当）
REPEAT_FRAC = 0.025                 # 同一上下文内重复评估同一方案（命中进程内缓存）的成本占比（实测约 3～4 ms/次）
# 每读一次钟的步长 = TICK[(模块, 函数名)] * n 秒；未列出的调用点用 TICK_DEFAULT
# 取值 = 该调用点两次读钟之间（扣除评估）的真实秒数合计 / Σ(读次数·n)，20 样例 × 问题1/2/3 × k=2..5 汇总
TICK = {
    ('q1.search', 'descend'): 3.04e-9,
    ('q1.search', 'task_ils'): 1.23e-8,
    ('q1.search', 'group_ils'): 5.76e-7,
    ('q1.main', 'solve'): 1.21e-7,
    ('q1.main', '<lambda>'): 5.96e-9,
    ('q1.main', 'run_rest'): 3.54e-7,
    ('q1.batch', 'search'): 1.38e-6,
    ('q2.main', 'time_left'): 5.31e-7,
    ('q2.main', 'timed_gen'): 3.10e-6,
    ('q2.main', 'solve'): 1.0e-8,
    ('q2.main', '_search'): 1.0e-8,
    ('q2.critsearch', 'crit_search'): 1.33e-4,
    ('q2.dev.multistart', 'solve'): 4.72e-7,
    ('q2.dev.multistart', 'eval_seconds'): 8.34e-7,
}
TICK_DEFAULT = 2.0e-7
# 计费函数：两次读钟之间真实耗时的大头是问题1 的代理打分（一次约 1 ms，而空转一圈只有几微秒），
# 按读钟次数计费会严重低估，所以这些函数每调用一次按 CHARGE[key] = (A, B) 计 A·size^B 秒。
# size：'view' = 实例的 view.ops 个数；'task' = _TaskState 的 Task 数 + 商图边数。
# 嵌套调用只计最外层（record 也只计最外层的真实耗时，并从读钟间隔里扣除）。
CHARGED = (
    ('solvers.q1.search', '_GroupState', 'score', 'view'),
    ('solvers.q1.search', '_TaskState', 'score', 'task'),
    ('solvers.q1.structure', 'ProxyModel', 'evaluate', 'view'),
    ('solvers.q1.structure', 'ProxyModel', 'task_times', 'view'),
    ('solvers.q1.init', 'Builder', 'build', 'view'),
)
CHARGE = {
    '_GroupState.score': (8.71e-7, 1.119),
    '_TaskState.score': (1.07e-6, 0.583),
    'ProxyModel.evaluate': (1.58e-6, 0.976),
    'ProxyModel.task_times': (1.46e-6, 0.994),
    'Builder.build': (3.89e-6, 0.998),
}
# 单核基准（问题1 求解器开头的 singlecore_makespan）按一次问题1 评估的这么多倍计费（实测约 0.8）
SINGLE_FACTOR = 0.8

CAP_FACTOR = 3.0                    # 真实时间安全上限 = 虚拟预算的这么多倍（solve 里另按案例截止收紧）
JUMP = 1.0e7                        # 上限触发后虚拟时间一次性前跳的秒数
V0 = 1000.0                         # 虚拟时间起点（常数，保证确定）

# 打补丁的模块：读钟的与调用 evaluate 的（grep 核对见 VCLOCK.md）
TIME_MODULES = ('solvers.q1.main', 'solvers.q1.search', 'solvers.q1.batch',
                'solvers.q2.main', 'solvers.q2.critsearch', 'solvers.q2.dev.multistart')
EVAL_MODULES = ('solvers.q1.main', 'solvers.q2.main', 'solvers.q2.critsearch', 'solvers.q2.crit',
                'solvers.q2.dev.multistart', 'solvers.baseline')

EVENTS = []                         # cap 触发等事件（进程内）


def eval_cost(n, problem, keep_full=False):
    c = EVAL_A.get(problem, EVAL_A[2]) * max(1, n) ** EVAL_B.get(problem, EVAL_B[2]) * SCALE
    return c * (KEEP_FULL_FACTOR if keep_full else 1.0)


def tick_cost(n, site):
    return TICK.get(site, TICK_DEFAULT) * max(1, n) * SCALE


def charge_cost(key, size):
    a, b = CHARGE[key]
    return a * max(1, size) ** b * SCALE


def _size(obj, kind):
    """计费规模：按实例缓存（这些对象的规模在生命周期内不变）。"""
    s = obj.__dict__.get('_vclock_size')
    if s is None:
        if kind == 'task':
            preds = getattr(obj, 'preds', None) or []
            s = getattr(obj, 'T', 0) + sum(len(p) for p in preds)
        else:
            s = len(obj.view.ops)
        obj.__dict__['_vclock_size'] = s
    return s


def _import(name):
    import importlib
    return importlib.import_module(name)


def _short(modname):
    return modname[len('solvers.'):] if modname.startswith('solvers.') else modname


class _Clock:
    """一个上下文的状态：虚拟时间、计费、真实上限、（record 模式下）真实耗时记录。"""

    def __init__(self, graph, budget_s, problem, cap_s, mode):
        self.n = len(graph.compute_ids)
        self.graph_name = getattr(graph, 'name', '?')
        self.problem = problem
        self.budget_s = budget_s
        self.mode = mode
        self.v = V0
        self.real0 = _time.perf_counter()
        self.cap_s = cap_s
        self.capped = False
        self.seen = set()
        self.n_reads = 0
        self.n_evals = 0
        self.v_eval = 0.0           # 评估计费合计（虚拟秒）
        self.v_tick = 0.0           # 读钟计费合计
        self.v_charge = 0.0         # 计费函数合计
        self.n_charged = 0
        self._depth = 0             # 计费函数嵌套深度（只计最外层）
        self.charge_rec = {}        # record：key -> [次数, 真实秒, Σsize]
        # record 模式
        self.site_reads = {}        # site -> [读次数, 归到该点的真实间隔秒]
        self.eval_rec = []          # (problem, keep_full, first_seen, 真实秒)
        self._last_real = self.real0
        self._eval_real_since = 0.0

    # --- 真实上限 --------------------------------------------------------------------------
    def _check_cap(self):
        if self.capped or self.cap_s is None:
            return
        real = _time.perf_counter() - self.real0
        if real > self.cap_s:
            self.capped = True
            self.v += JUMP
            ev = dict(event='cap', graph=self.graph_name, problem=self.problem, budget_s=self.budget_s,
                      cap_s=self.cap_s, real_s=round(real, 3), virtual_s=round(self.virtual_elapsed() - JUMP, 3))
            EVENTS.append(ev)
            print(f'[vclock] real-time cap fired: {json.dumps(ev, ensure_ascii=False)}', file=sys.stderr, flush=True)

    def virtual_elapsed(self):
        return self.v - V0

    # --- 读钟 ------------------------------------------------------------------------------
    def read(self, site):
        self.n_reads += 1
        if self.mode == 'record':
            now = _time.perf_counter()
            gap = now - self._last_real - self._eval_real_since
            rec = self.site_reads.setdefault(site, [0, 0.0])
            rec[0] += 1
            rec[1] += max(0.0, gap)
            self._last_real = now
            self._eval_real_since = 0.0
            return now
        dt = tick_cost(self.n, site)
        self.v += dt
        self.v_tick += dt
        self._check_cap()
        return self.v

    # --- 评估 ------------------------------------------------------------------------------
    def charge_eval(self, key, problem, keep_full):
        first = key not in self.seen
        self.seen.add(key)
        c = eval_cost(self.n, problem, keep_full) * (1.0 if first or keep_full else REPEAT_FRAC)
        self.n_evals += 1
        self.v += c
        self.v_eval += c
        self._check_cap()
        return first, eval_cost(self.n, problem)

    def summary(self):
        return dict(n=self.n, graph=self.graph_name, problem=self.problem, budget_s=self.budget_s,
                    virtual_s=round(self.virtual_elapsed() - (JUMP if self.capped else 0.0), 3),
                    real_s=round(_time.perf_counter() - self.real0, 3), capped=self.capped,
                    reads=self.n_reads, evals=self.n_evals, v_eval=round(self.v_eval, 3),
                    v_tick=round(self.v_tick, 3), charged=self.n_charged, v_charge=round(self.v_charge, 3))


class _TimeShim:
    """替代模块里的 time：perf_counter 走虚拟时钟，其余属性转发给真 time。"""

    def __init__(self, clock, modname):
        self._clock = clock
        self._mod = _short(modname)

    def perf_counter(self):
        f = sys._getframe(1)
        return self._clock.read((self._mod, f.f_code.co_name))

    def __getattr__(self, name):
        return getattr(_time, name)


_CURRENT = []       # 活动上下文栈（允许嵌套：内层用自己的时钟）
_SINGLE_DONE = set()


def _plan_key(plan, problem):
    from common.plan import plan_hash
    return (plan_hash(plan), problem)


@contextlib.contextmanager
def virtual_clock(graph, budget_s=None, problem=2, cap_s=None, mode='virtual'):
    """mode='virtual'：虚拟时钟（确定）；mode='record'：用真实时钟，只记录各调用点的真实耗时（标定用）。
    cap_s：真实时间安全上限（秒）；None 时取 CAP_FACTOR * budget_s（budget_s 也为 None 则不设上限）。"""
    if cap_s is None and budget_s:
        cap_s = CAP_FACTOR * float(budget_s)
    clock = _Clock(graph, budget_s, problem, cap_s, mode)
    ev_mod = _import('common.evaluate')
    real_eval = ev_mod.evaluate

    def wrapped_evaluate(g, plan, problem, *args, **kw):
        keep_full = bool(kw.get('keep_full'))
        key = _plan_key(plan, problem)
        if clock.mode == 'record':
            first = key not in clock.seen
            clock.seen.add(key)
            t = _time.perf_counter()
            r = real_eval(g, plan, problem, *args, **kw)
            dt = _time.perf_counter() - t
            clock._eval_real_since += dt
            clock.eval_rec.append((problem, keep_full, first, bool(r.get('cached')), round(dt, 5)))
            clock.n_evals += 1
            return r
        r = real_eval(g, plan, problem, *args, **kw)
        _, cost = clock.charge_eval(key, problem, keep_full)
        r = dict(r)
        r['eval_seconds'] = cost
        r['cached'] = False
        return r

    saved = []

    def patch(mod, name, value):
        saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, value)

    q1m = _import('solvers.q1.main')
    real_single = q1m.singlecore_makespan

    def charged_single(g, *a, **kw):
        key = (getattr(g, 'graph_hash', id(g)))
        first = key not in _SINGLE_DONE
        _SINGLE_DONE.add(key)
        if clock.mode == 'record':
            t = _time.perf_counter()
            out = real_single(g, *a, **kw)
            dt = _time.perf_counter() - t
            clock._eval_real_since += dt
            clock.eval_rec.append((0, False, first, False, round(dt, 5)))
            return out
        if first:
            c = SINGLE_FACTOR * eval_cost(clock.n, 1)
            clock.v += c
            clock.v_eval += c
        return real_single(g, *a, **kw)

    def charged_method(key, kind, real_fn):
        def wrapper(self, *a, **kw):
            if clock._depth:
                return real_fn(self, *a, **kw)
            size = _size(self, kind)
            clock._depth += 1
            try:
                if clock.mode == 'record':
                    t = _time.perf_counter()
                    out = real_fn(self, *a, **kw)
                    dt = _time.perf_counter() - t
                    clock._eval_real_since += dt
                    rec = clock.charge_rec.setdefault(key, [0, 0.0, 0])
                    rec[0] += 1
                    rec[1] += dt
                    rec[2] += size
                    clock.n_charged += 1
                    return out
                out = real_fn(self, *a, **kw)
            finally:
                clock._depth -= 1
            c = charge_cost(key, size)
            clock.v += c
            clock.v_charge += c
            clock.n_charged += 1
            clock._check_cap()
            return out
        wrapper.__wrapped__ = real_fn
        return wrapper

    try:
        for modname, cls_name, meth, kind in CHARGED:
            cls = getattr(_import(modname), cls_name)
            key = f'{cls_name}.{meth}'
            patch(cls, meth, charged_method(key, kind, cls.__dict__[meth]))
        for name in TIME_MODULES:
            patch(_import(name), 'time', _TimeShim(clock, name))
        for name in EVAL_MODULES:
            patch(_import(name), 'evaluate', wrapped_evaluate)
        patch(q1m, 'singlecore_makespan', charged_single)
        cs = _import('solvers.q2.critsearch')
        if getattr(cs, '_MEMO', None) is not None and hasattr(cs, '_read_disk'):
            patch(cs, '_read_disk', lambda *a, **kw: None)
        ms = _import('solvers.q2.dev.multistart')
        patch(ms, '_EVAL_T', _EVAL_T_BY_CASE.setdefault(getattr(graph, 'graph_hash', id(graph)), {}))
        _CURRENT.append(clock)
        yield clock
    finally:
        if _CURRENT and _CURRENT[-1] is clock:
            _CURRENT.pop()
        for mod, name, value in reversed(saved):
            setattr(mod, name, value)


_EVAL_T_BY_CASE = {}     # multistart 的单次评估实测表，按图隔离（虚拟时钟下测得的是虚拟成本，确定）


def current():
    return _CURRENT[-1] if _CURRENT else None


# ---------------------------------------------------------------------------------------------
# run_case_budget 可直接调用的入口：案例级虚拟预算
#
# run_case_budget 给每个 k 的 budget_s = (真实剩余 - 预留) / 剩余核数，含读图等真实耗时，本身不确定。
# 这里在一个案例的第一个 k 把 budget_s * 剩余核数 量化成案例级虚拟预算 V（默认按 60 s 取整，可由环境变量
# VCLOCK_CASE_BUDGET 直接指定），此后每个 k 的虚拟预算 = (V*CASE_USE - 已用虚拟时间) / 剩余核数，
# 只取决于 V 和前面各 k 的虚拟用时，因而确定。真实上限另按本 k 可用的真实时间收紧（见 _real_cap）。

KS = (2, 3, 4, 5)
CASE_QUANTUM = 60.0
CASE_USE = 0.97          # 虚拟预算只用案例预算的这一比例：留出 run_case_budget 的输出预留与读图时间
_CASE = {}               # (graph_hash, problem) -> dict(V, used, last_k)


def _remaining_ks(k):
    return len([x for x in KS if x >= k]) if k in KS else 1


def case_budget_from(budget_s, k):
    import os
    env = os.environ.get('VCLOCK_CASE_BUDGET')
    if env:
        return float(env)
    total = float(budget_s) * _remaining_ks(k)
    q = CASE_QUANTUM if total >= CASE_QUANTUM else 10.0
    # 向下取整、容差 QUANT_TOL 个量子：run_case_budget 给的是扣掉读图与预留后的剩余，120 s 案例约 117 s，
    # 读图耗时随负载波动也不会越过取整边界（边界在 (m-QUANT_TOL)*q，即 120 s 案例的 105 s）
    return max(q, math.floor(total / q + QUANT_TOL) * q)


QUANT_TOL = 0.25


def plan_budget(graph, k, problem, budget_s):
    """返回 (本 k 的虚拟预算, 状态)。同一图同一问题 k 不增时视为新案例。"""
    key = (getattr(graph, 'graph_hash', id(graph)), problem)
    st = _CASE.get(key)
    if st is None or k <= st['last_k']:
        st = dict(V=case_budget_from(budget_s or 60.0, k), used=0.0, last_k=-1)
        _CASE[key] = st
    vk = max(1.0, (st['V'] * CASE_USE - st['used']) / _remaining_ks(k))
    return vk, st


def _real_cap(vk, budget_s, k):
    """本 k 的真实上限：不超过 CAP_FACTOR*vk，也不能吃掉后面各 k 至少一半虚拟预算的真实时间。
    环境变量 VCLOCK_CAP_FACTOR=<x>：可复现优先时用，上限直接取 x*vk，不再按真实剩余时间收紧。"""
    import os
    env = os.environ.get('VCLOCK_CAP_FACTOR')
    if env:
        return max(1.0, float(env) * vk)
    if not budget_s:
        return CAP_FACTOR * vk
    r = _remaining_ks(k)
    real_left = float(budget_s) * r              # run_case_budget 给的 = (真实剩余-预留)/r
    return max(1.0, min(CAP_FACTOR * vk, real_left - 0.5 * vk * (r - 1)))


def run_virtual(inner, graph, k, problem=2, seed=0, budget_s=None, **opts):
    vk, st = plan_budget(graph, k, problem, budget_s)
    cap = _real_cap(vk, budget_s, k)
    n = len(graph.compute_ids)
    # q2.solve 在上限触发后仍会评估保底候选：给它留出几次评估的真实时间
    cap = max(1.0, cap - 4 * eval_cost(n, problem))
    with virtual_clock(graph, vk, problem, cap_s=cap) as clk:
        plan = inner(graph, k, problem, seed=seed, budget_s=vk, **opts)
    s = clk.summary()
    st['used'] += s['virtual_s']
    st['last_k'] = k
    LOG.append(dict(s, k=k, vk=round(vk, 3), cap_s=round(cap, 3), case_V=st['V'], seed=seed))
    global LAST_META
    LAST_META = LOG[-1]
    return plan


LOG = []          # 每次 run_virtual 的摘要（进程内）
LAST_META = None  # 最近一次 run_virtual 的摘要；run_case_budget 读取 capped/vk/virtual_s/real_s 写入 receipt


def DETERMINISM_META():
    """run_case_budget --deterministic 写进 meta.json 的常数与生效的环境变量。"""
    import os
    return dict(version=VERSION, SCALE=SCALE, CAP_FACTOR=CAP_FACTOR, CASE_USE=CASE_USE,
                CASE_QUANTUM=CASE_QUANTUM, QUANT_TOL=QUANT_TOL,
                EVAL_A=EVAL_A, EVAL_B=EVAL_B, REPEAT_FRAC=REPEAT_FRAC, SINGLE_FACTOR=SINGLE_FACTOR,
                TICK_DEFAULT=TICK_DEFAULT, CHARGE=CHARGE,
                VCLOCK_CASE_BUDGET=os.environ.get('VCLOCK_CASE_BUDGET'),
                VCLOCK_CAP_FACTOR=os.environ.get('VCLOCK_CAP_FACTOR'))


# ---------------------------------------------------------------------------------------------
# k−1 保底（vclock-v2）：同一进程、同一图、同一问题、同一入口，缓存上一个 k 返回的方案；本 k 求解后把它补空核，
# 与本 k 方案按字典序（makespan, added_copy_bytes）择优返回。多一个核至少不比少一个核差。
# 这两次评估在虚拟时钟之外、用 common.evaluate 做：不计入虚拟预算，因此各 k 的虚拟预算与求解器内部方案不变，
# 只可能换掉最终返回的方案（真实耗时照常计入案例墙钟，另记 guard_s）。
# 问题2/3 的 multistart 在不截断时本就把 k−1 方案当热启动候选评估，结果应当不变；问题1 求解器原来没有这条链。

_GPREV = {}      # (graph_hash, problem, 入口名) -> (k, plan)：本入口上一个 k 最终返回的方案


def _guarded(entry, graph, k, problem, plan):
    from common import evaluate as _ev, make_plan, sort_key
    key = (getattr(graph, 'graph_hash', id(graph)), problem, entry)
    prev = _GPREV.get(key)
    info, t = 'self', _time.perf_counter()
    if prev is not None and prev[0] < k and plan is not None:
        pk, pplan = prev
        padded = make_plan(pplan['node_to_subgraph'], pplan['core_schedules'] + [[]] * (k - pk))
        if sort_key(_ev(graph, padded, problem)) < sort_key(_ev(graph, plan, problem)):
            plan, info = padded, f'prev_k{pk}'
        else:
            info = f'self>=prev_k{pk}'
    _GPREV[key] = (k, plan)
    global LAST_META
    if LAST_META is not None:
        LAST_META = dict(LAST_META, guard=info, guard_s=round(_time.perf_counter() - t, 3))
        LOG[-1] = LAST_META
    return plan


def solve(graph, k, problem=2, seed=0, budget_s=None, **opts):
    """问题2/3 multistart（S3 dev 版）套虚拟时钟 + k−1 保底。"""
    from solvers.q2.dev import multistart
    return _guarded('solve', graph, k, problem,
                    run_virtual(multistart.solve, graph, k, problem, seed, budget_s, **opts))


def solve_q1(graph, k, problem=1, seed=0, budget_s=None, **opts):
    """问题1 求解器套虚拟时钟（不读档案）+ k−1 保底。"""
    from solvers.q1 import main as q1m
    opts.setdefault('warm', False)
    return _guarded('solve_q1', graph, k, problem,
                    run_virtual(q1m.solve, graph, k, problem, seed, budget_s, **opts))


def solve_q2main(graph, k, problem=2, seed=0, budget_s=None, **opts):
    """问题2 正式入口套虚拟时钟（不读档案）+ k−1 保底。"""
    from solvers.q2 import main as q2m
    opts.setdefault('warm', 'none')
    return _guarded('solve_q2main', graph, k, problem,
                    run_virtual(q2m.solve, graph, k, problem, seed, budget_s, **opts))


# ---------------------------------------------------------------------------------------------
# 「只构造、不搜索」入口（预算曲线的构造点，VCLOCK.md §6）
#
# 问题1：保底基线两个变体 → batch 图再加解析点（batch._grid 第一项：D* 的解析取值，一次评估）→
#        init.sweep_specs 代理扫描 → 真评估代理第一名。返回字典序最优。
#        扫描截断与问题1 求解器相同：虚拟时间到 max(SWEEP_FRAC*qb, 基线后+SWEEP_MIN_FRAC*qb)，qb 为问题1 流程的预算
#        （问题1 入口 qb=本 k 虚拟预算；问题2/3 qb=multistart.STRUCT_FRAC*本 k 虚拟预算，同 multistart 结构段）。
#        不截断的完整扫描在大图上很慢（case_014 k=5 共 687 个规格，约 200 虚拟秒），不是求解器会走的路径。
# 问题2/3：上面的问题1 构造按目标问题评估（同 multistart.struct_candidate 的换评估器做法）+ 同进程 k−1 方案补空核
#        → 问题2 求解器（问题3 用其组合）search=False：保底基线、热启动、块表示、重排、列表调度组合，
#        截止时间与 multistart 相同（本 k 剩余虚拟预算 − 余量），即完整流程去掉邻域搜索段。
#        不限时的版本在 case_014 上每 k 要 270～380 真实秒，触发真实上限，不是求解器会走的路径。
# 两者都没有迭代搜索（没有邻域爬山、ILS、随机扰动）；评估的是固定次序的有限个构造候选。
# 各阶段的当前最优记在 LAST_META['stages']：[阶段名, makespan, added, 累计评估次数, 累计虚拟秒]。

_CPREV = {}      # (graph_hash, problem) -> (k, plan)：construct 入口同进程内上一个 k 的方案


def _stage(stages, name, key):
    clk = current()
    stages.append([name, key[1], key[2], clk.n_evals if clk else None,
                   round(clk.virtual_elapsed(), 3) if clk else None])


def _q1_construct(graph, k, stages, head=False, qb=None):
    """问题1 的构造段；评估走 solvers.q1.main 模块里的 evaluate（调用方可换成目标问题）。返回 (最优键, 方案)。"""
    from solvers.q1 import main as q1m
    from solvers.q1 import init, batch
    from solvers.q1.structure import View, ProxyModel
    view = View(graph)
    proxy = ProxyModel(view, single_makespan=q1m.singlecore_makespan(graph))
    builder = init.Builder(view, k)
    ctx = q1m._Ctx(graph, k, view, proxy, builder, float('inf'))
    ctx.run('baseline_bundled', q1m._baseline(graph, k, 1, variant='bundled'), {'gen': 'comp', 'mode': 'phase'})
    ctx.run('baseline_fine', q1m._baseline(graph, k, 1, variant='fine'), None)
    _stage(stages, 'baseline', ctx.best[0])
    binfo = batch.detect(graph) if k >= 2 else None
    if binfo is not None:
        S, cn, cf, beta, dmax, mode = batch._grid(binfo, k)[0]
        cuts = cf(S)
        ctx.run(f'batch_{cn}{S}_b{beta}_d{dmax}_{mode}',
                batch.build(graph, k, binfo, cuts, batch._dprofile(binfo, cuts, beta, dmax), mode), None)
        _stage(stages, 'batch0', ctx.best[0])
    big = ctx.n > q1m.BIG_OPS
    clk = current()
    t0 = clk.virtual_elapsed() if clk else 0.0
    sweep_end = None
    if qb and clk:
        # 基线/batch 之前的起点按 0 计（问题1 求解器的 t0 是进入 solve 时）
        sweep_end = max(q1m.SWEEP_FRAC * qb, t0 + q1m.SWEEP_MIN_FRAC * qb)
    scored, sigs, names = [], set(), {}
    for spec in init.sweep_specs(view, names, k, kc_late=big, wide=(1.0,), ws_pos=0, align=True):
        if sweep_end is not None and scored and clk.virtual_elapsed() >= sweep_end:
            break
        name = init.spec_name(spec)
        if name in names:
            continue
        plan = builder.build(spec)
        if plan is None:
            names[name] = float('inf')
            continue
        px = ctx.proxy(plan)
        names[name] = px
        s = q1m._sig(plan)
        if s in sigs:
            continue
        sigs.add(s)
        scored.append((px, spec))
    scored.sort(key=lambda x: x[0])
    if scored:
        spec = scored[0][1]
        ctx.run(init.spec_name(spec), builder.build(spec), spec)
    _stage(stages, f'sweep1/{len(scored)}', ctx.best[0])
    if head:
        # 可选：再评估「代理前几名 + 各生成器族代理最优」（问题1 求解器初始队列的先评部分），仍是有限个
        for _, spec in q1m._queue(scored)[0]:
            ctx.run(init.spec_name(spec), builder.build(spec), spec)
        _stage(stages, 'head', ctx.best[0])
    return ctx.best[0], ctx.best[3]


def _construct_q1_inner(graph, k, problem=1, seed=0, budget_s=None, head=False, **opts):
    global _STAGES
    _STAGES = []
    return _q1_construct(graph, k, _STAGES, head=head, qb=budget_s)[1]


def _construct_inner(graph, k, problem=2, seed=0, budget_s=None, head=False, chain=True, **opts):
    from common import evaluate as _ev, make_plan, sort_key
    from solvers.q1 import main as q1m
    from solvers.q2 import main as q2m
    from solvers.q3 import main as q3m
    global _STAGES
    _STAGES = []
    warm = []
    if k >= 2:
        # 与 multistart.struct_candidate 相同：问题1 构造段的候选一律按目标问题评估。
        # 此处 q1m.evaluate 已被 virtual_clock 换成计费版；在它外面再套一层换问题号
        billed = q1m.evaluate
        q1m.evaluate = lambda g, plan, _problem, **kw: billed(g, plan, problem, **kw)
        try:
            from solvers.q2.dev import multistart
            _, plan = _q1_construct(graph, k, _STAGES, head=head,
                                    qb=multistart.STRUCT_FRAC * budget_s if budget_s else None)
        finally:
            q1m.evaluate = billed
        warm.append(('struct_q1gen', plan))
    prev = _CPREV.get((graph.graph_hash, problem)) if chain else None
    if prev is not None and prev[0] < k:
        pk, pplan = prev
        warm.append((f'prev_k{pk}', make_plan(pplan['node_to_subgraph'], pplan['core_schedules'] + [[]] * (k - pk))))
    solver = q3m.solve if problem == 3 else q2m.solve
    kw = dict(problem=problem) if problem != 3 else {}
    rest = None
    if budget_s:
        # 与 multistart.solve 相同：剩余预算减 MARGIN_EVALS 次评估的余量，交给问题2/3 构造段按截止时间取舍阶段
        from solvers.q2.dev import multistart
        clk = current()
        used = clk.virtual_elapsed() if clk else 0.0
        margin = multistart.MARGIN_EVALS * multistart.eval_seconds(graph, k, problem) + 0.2
        rest = max(1.0, budget_s - used - margin)
    plan = solver(graph, k, seed=seed, budget_s=rest, warm=warm, search=False, **kw)
    arc = q2m.LAST_ARCHIVE
    if arc is not None and arc.best is not None:
        _stage(_STAGES, f'q2construct:{arc.best[1]}', arc.best[0])
    _CPREV[(graph.graph_hash, problem)] = (k, plan)
    return plan


_STAGES = []


def _with_stages(plan):
    import json as _json
    global LAST_META
    LAST_META = dict(LAST_META, stages=_json.dumps(_STAGES, separators=(',', ':')))
    LOG[-1] = LAST_META
    return plan


def construct_q1(graph, k, problem=1, seed=0, budget_s=None, **opts):
    """问题1「只构造、不搜索」：保底基线 + batch 解析点 + 完整代理扫描后的第一名。"""
    return _with_stages(run_virtual(_construct_q1_inner, graph, k, problem, seed, budget_s, **opts))


def construct(graph, k, problem=2, seed=0, budget_s=None, **opts):
    """问题2/3「只构造、不搜索」：问题1 构造段（按目标问题评估）+ k−1 链 + 问题2 求解器 search=False。"""
    return _with_stages(run_virtual(_construct_inner, graph, k, problem, seed, budget_s, **opts))


# ---------------------------------------------------------------------------------------------
# 标定 / 检验 CLI

def _set_affinity(pid, mask):
    """pid=None 表示本进程。"""
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.OpenProcess.restype = ctypes.c_void_p
    if pid is None:
        h = k32.GetCurrentProcess()
    else:
        h = k32.OpenProcess(0x0200 | 0x0400, False, pid)   # PROCESS_SET_INFORMATION | QUERY_INFORMATION
    ok = k32.SetProcessAffinityMask(ctypes.c_void_p(h), ctypes.c_size_t(mask))
    if pid is not None:
        k32.CloseHandle(ctypes.c_void_p(h))
    return ok


def _pin_ecores():
    from experiments.dev_ab import e_core_mask
    mask = e_core_mask() or 0xFFFF0000
    _set_affinity(None, mask)
    return mask


def _isolate(dirpath):
    """与 run_case_budget._isolate_worker 相同：私有空评估缓存、空档案、空单核基准。"""
    import importlib
    from pathlib import Path
    from common import paths
    ev = importlib.import_module('common.evaluate')
    iso = Path(dirpath)
    paths.BEST = iso / 'empty_best'
    paths.EVAL_CACHE = iso / 'evalcache'
    paths.BASELINE = iso / 'empty_baseline'
    ev.EVAL_CACHE = paths.EVAL_CACHE
    ev.BASELINE = paths.BASELINE
    ev._MEMO.clear()
    ev._BASELINE = None


def _plan_sha(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]


def _job(args):
    """一个 (case, problem) 的 k=2..5；mode=record 用真实时钟记录，mode=virtual 用虚拟时钟。"""
    case, problem, per_k, mode, outdir, seed, ecores, solver = args
    if ecores:
        _pin_ecores()
    import os
    import tempfile
    from pathlib import Path
    _isolate(tempfile.mkdtemp(prefix=f'vc_{case}_p{problem}_', dir=outdir))
    from common import load_graph, evaluate as real_evaluate
    g = load_graph(case)
    inner = _inner(solver)
    rows = []
    for k in KS:
        t = _time.perf_counter()
        if mode == 'record':
            with virtual_clock(g, per_k, problem, cap_s=None, mode='record') as clk:
                plan = inner(g, k, problem, seed=seed, budget_s=per_k)
            s = clk.summary()
            s['sites'] = {f'{a}.{b}': v for (a, b), v in clk.site_reads.items()}
            s['eval_rec'] = clk.eval_rec
            s['charged_rec'] = clk.charge_rec
        else:
            # 案例虚拟预算直接给定；传给求解器的 budget_s 只用于真实上限，给宽（3 倍），测的是 真实/虚拟 的自然分布
            os.environ['VCLOCK_CASE_BUDGET'] = str(per_k * len(KS))
            plan = {'solvers.vclock:solve': solve, 'q1': solve_q1, 'q2main': solve_q2main}.get(
                solver, solve)(g, k, problem, seed=seed, budget_s=3 * per_k)
            s = dict(LOG[-1])
        s['wall_s'] = round(_time.perf_counter() - t, 3)
        r = real_evaluate(g, plan, problem)
        s.update(case=case, k=k, mode=mode, makespan=r.get('makespan'), ok=r.get('ok'),
                 added=r.get('added_copy_bytes'), plan_sha=_plan_sha(plan), events=list(EVENTS))
        rows.append(s)
    return rows


def _inner(solver):
    if solver in ('q1',):
        from solvers.q1 import main as q1m
        return lambda g, k, p, **kw: q1m.solve(g, k, p, warm=False, **kw)
    if solver in ('q2main',):
        from solvers.q2 import main as q2m
        return lambda g, k, p, **kw: q2m.solve(g, k, p, warm='none', **kw)
    from solvers.q2.dev import multistart
    return multistart.solve


def _cases(spec):
    from pathlib import Path
    if spec == 'sample':
        root = Path(__file__).resolve().parents[1]
        return [ln.split()[0] for ln in (root / 'experiments' / 'sample_cases.txt').read_text(encoding='utf-8').splitlines()
                if ln.strip() and not ln.startswith('#')]
    return spec.split(',')


def _batch(args, mode):
    import multiprocessing as mp
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.ecores:
        _pin_ecores()
    jobs = [(c, p, args.per_k, mode, str(out), args.seed, args.ecores, args.solver)
            for p in map(int, args.problems.split(',')) for c in _cases(args.cases)]
    jobs.sort(key=lambda j: j[0] != 'case_014')      # 最大图先跑
    fn = out / f'{mode}_{args.solver.replace(":", "_").replace(".", "_")}.jsonl'
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool, open(fn, 'a', encoding='utf-8') as f:
        for rows in pool.imap_unordered(_job, jobs):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            print(rows[0]['case'], rows[0]['problem'], [round(r.get('real_s', 0), 1) for r in rows], flush=True)
    print('done', fn, flush=True)


def _fit(args):
    """由 record 数据拟合 EVAL_A/B（首见、未命中缓存的评估）与各调用点 TICK（真实间隔 / (读次数*n)）。"""
    from pathlib import Path
    rows = []
    for fn in Path(args.out).glob('record_*.jsonl'):
        rows += [json.loads(l) for l in fn.read_text(encoding='utf-8').splitlines() if l.strip()]
    ev = {}
    for r in rows:
        for (p, kf, first, cached, dt) in r['eval_rec']:
            if first and not cached:
                ev.setdefault((p, kf), []).append((r['n'], dt))
    res = {'eval': {}, 'tick': {}}
    for (p, kf), pts in sorted(ev.items()):
        xs = [math.log(n) for n, _ in pts]
        ys = [math.log(max(dt, 1e-6)) for _, dt in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = math.exp(my - b * mx)
        res['eval'][f'p{p}_full{int(kf)}'] = dict(A=a, B=b, count=len(pts))
    sites = {}
    for r in rows:
        for site, (cnt, sec) in r['sites'].items():
            s = sites.setdefault(site, [0, 0.0, 0.0])
            s[0] += cnt
            s[1] += sec
            s[2] += cnt * r['n']
    for site, (cnt, sec, cn) in sorted(sites.items(), key=lambda x: -x[1][1]):
        res['tick'][site] = dict(per_op=sec / max(cn, 1), reads=cnt, seconds=round(sec, 2))
    rep = [dict(problem=p, count=0, sec=0.0) for p in (1, 2, 3)]
    for r in rows:
        for (p, kf, first, cached, dt) in r['eval_rec']:
            if not first:
                rep[p - 1]['count'] += 1
                rep[p - 1]['sec'] += dt
    res['repeat'] = rep
    # 计费函数：每个 (运行, key) 一点 (平均 size, 平均每次秒)，按真实秒数加权做对数线性拟合
    ch = {}
    for r in rows:
        for key, (cnt, sec, ssum) in (r.get('charged_rec') or {}).items():
            if cnt:
                ch.setdefault(key, []).append((ssum / cnt, sec / cnt, sec))
    res['charge'] = {}
    for key, pts in sorted(ch.items(), key=lambda x: -sum(p[2] for p in x[1])):
        w = [max(p[2], 1e-6) for p in pts]
        xs = [math.log(max(p[0], 1)) for p in pts]
        ys = [math.log(max(p[1], 1e-8)) for p in pts]
        W = sum(w)
        mx, my = sum(wi * x for wi, x in zip(w, xs)) / W, sum(wi * y for wi, y in zip(w, ys)) / W
        sxx = sum(wi * (x - mx) ** 2 for wi, x in zip(w, xs))
        b = sum(wi * (x - mx) * (y - my) for wi, x, y in zip(w, xs, ys)) / sxx if sxx > 1e-9 else 1.0
        a = math.exp(my - b * mx)
        res['charge'][key] = dict(A=a, B=b, runs=len(pts), seconds=round(sum(p[2] for p in pts), 2))
    # 回放：按当前模块常数给 record 的调用序列计费，真实/虚拟 的分布（评估 + 读钟 + 计费函数）
    ratios = []
    for r in rows:
        n = r['n']
        v = 0.0
        for (p, kf, first, cached, dt) in r['eval_rec']:
            if p == 0:
                v += SINGLE_FACTOR * eval_cost(n, 1) if first else 0.0
            else:
                v += eval_cost(n, p, kf) * (1.0 if first or kf else REPEAT_FRAC)
        for site, (cnt, sec) in r['sites'].items():
            a, b = site.rsplit('.', 1)
            v += cnt * tick_cost(n, (a, b))
        for key, (cnt, sec, ssum) in (r.get('charged_rec') or {}).items():
            if key in CHARGE and cnt:
                v += cnt * charge_cost(key, ssum / cnt)
        if v > 0:
            ratios.append((r['real_s'] / v, r['case'], r['problem'], r['k']))
    ratios.sort()
    if ratios:
        qs = [ratios[min(len(ratios) - 1, int(f * len(ratios)))][0] for f in (0, .1, .5, .9)]
        res['replay'] = dict(runs=len(ratios), min=qs[0], p10=qs[1], median=qs[2], p90=qs[3], max=ratios[-1][0],
                             low=ratios[:3], high=ratios[-5:])
    print(json.dumps(res, indent=1, ensure_ascii=False))
    (Path(args.out) / 'fit.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')


def _determinism(args):
    """同一 case/k/种子：各开一个新进程跑两次，比 plan_sha。
    第 1 次：PYTHONHASHSEED=0，固定在全部能效核上、无额外负载；
    第 2 次：PYTHONHASHSEED=12345，求解进程和 --load 个纯 CPU 空转进程一起挤在 --load-mask 指定的一个能效核上
    （--load 1 时求解只能拿到约一半 CPU，真实耗时约翻倍）。"""
    import os
    import subprocess
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONUTF8='1')
    full_mask = _pin_ecores()
    shas = []
    for i, (hs, load, mask) in enumerate(((0, 0, full_mask), (12345, args.load, int(args.load_mask, 0)))):
        burners = []
        for _ in range(load):
            burners.append(subprocess.Popen([sys.executable, '-c', 'while True: pass'], env=env))
            _set_affinity(burners[-1].pid, mask)
        try:
            e = dict(env, PYTHONHASHSEED=str(hs))
            code = (f'import json,sys; sys.argv=["x"]; from solvers import vclock as v; v._set_affinity(None,{mask}); '
                    f'rows=v._job(({args.case!r},{args.problem},{args.budget},"virtual",{str(out)!r},{args.seed},'
                    f'False,{args.solver!r})); print(json.dumps(rows))')
            p = subprocess.run([sys.executable, '-c', code], cwd=root, env=e, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            if p.returncode:
                print(p.stderr[-3000:])
                sys.exit(1)
            rows = json.loads(p.stdout.strip().splitlines()[-1])
        finally:
            for b in burners:
                b.kill()
        shas.append([(r['k'], r['plan_sha'], r['makespan'], r['real_s'], r['virtual_s'], r['capped']) for r in rows])
        print(f'run {i} hashseed={hs} load={load} mask=0x{mask:X}:', shas[-1], flush=True)
    same = [a[1] == b[1] for a, b in zip(*shas)]
    res = dict(case=args.case, problem=args.problem, budget=args.budget, seed=args.seed, load=args.load,
               runs=shas, identical=all(same))
    print('identical' if all(same) else 'DIFFERENT', same, flush=True)
    with open(out / 'determinism.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(res) + '\n')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('record', 'check'):
        a = sub.add_parser(name)
        a.add_argument('--out', required=True)
        a.add_argument('--problems', default='2')
        a.add_argument('--cases', default='sample')
        a.add_argument('--per-k', type=float, default=30.0)
        a.add_argument('--workers', type=int, default=3)
        a.add_argument('--seed', type=int, default=0)
        a.add_argument('--solver', default='solvers.vclock:solve', help='solvers.vclock:solve | q1 | q2main')
        a.add_argument('--ecores', action='store_true')
    f = sub.add_parser('fit')
    f.add_argument('--out', required=True)
    d = sub.add_parser('determinism')
    d.add_argument('--out', required=True)
    d.add_argument('--case', default='case_039')
    d.add_argument('--problem', type=int, default=2)
    d.add_argument('--budget', type=float, default=30.0, help='每个 k 的预算（案例预算 = 4 倍）')
    d.add_argument('--seed', type=int, default=0)
    d.add_argument('--load', type=int, default=1, help='第二次运行时和求解挤在同一核上的纯 CPU 空转进程数')
    d.add_argument('--load-mask', default='0x10000', help='第二次运行的亲和性掩码（缺省为第一个能效核）')
    d.add_argument('--solver', default='solvers.vclock:solve')
    args = ap.parse_args(argv)
    if args.cmd == 'record':
        _batch(args, 'record')
    elif args.cmd == 'check':
        _batch(args, 'virtual')
    elif args.cmd == 'fit':
        _fit(args)
    else:
        _determinism(args)


if __name__ == '__main__':
    main()
