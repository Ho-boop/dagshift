"""问题1 有限邻域搜索（第一层：生成器参数邻域 spec_search；第二层：组换核局部搜索 group_ls / group_ils；
第三层：固定子图划分的 Task 级换核 task_ils）。

在当前最优候选的参数规格（spec）附近生成邻居：层带数 S、层级函数、均衡方式、负载松弛 eps、
小组阈值 tiny、Task 切法 mode/merge、HEFT 的 lam、用核数 kc。邻居先用代理打分，再按代理顺序真评估；
严格更优（common.sort_key）才替换当前最优，任何时候不丢已知最优。
Task 级换核：问题1 里 Task 的时长与边界字节只由子图划分决定、与所在核无关，所以固定划分后只换核，
每步增量重算代理调度 O(T+E_q)。
"""

import random
import time

from common import make_plan

from .init import spec_name, band_groups, keys_for, keys_valid
from .structure import plan_from_keys, BW, SAME_WAIT, CROSS_WAIT

MODES = ('phase', 'phase_alap', 'phase_split', 'phase_alap_split', 'band')


def _s_neighbors(S, max_levels):
    if S == 'each':
        return [64, 128, 256]
    out = {S - 1, S + 1, S - 2, S + 2, round(S * 1.3), round(S / 1.3), S * 2}
    return sorted(x for x in out if 1 <= x <= max_levels and x != S)


def neighbors(spec, view, k=None):
    g = spec['gen']
    out = []
    if k:
        # 用核数 kc：少用一个 / 多用一个（kc == k 即去掉该键）
        kc = spec.get('kc') or k
        for kc2 in (kc - 1, kc + 1):
            if 2 <= kc2 < k:
                out.append(dict(spec, kc=kc2))
            elif kc2 == k:
                out.append({x: y for x, y in spec.items() if x != 'kc'})
    if g == 'band':
        max_levels = view.max_depth + 1
        for S in _s_neighbors(spec['S'], max_levels):
            out.append(dict(spec, S=S))
        for eps in (0.0, 0.03, 0.15, 0.3):
            if eps != spec.get('eps', 0.08):
                out.append(dict(spec, eps=eps))
        for tiny in (0, 300, 500, 1500):
            if tiny != spec.get('tiny', 0):
                out.append(dict(spec, tiny=tiny))
        for tinyop in (0, 32, 128):
            if tinyop != spec.get('tinyop', 0):
                out.append(dict(spec, tinyop=tinyop))
        for ws in (0, 0.5, 1.0):
            if ws != spec.get('wsplit', 0) and spec.get('mode', 'phase') not in ('band', 'phase_band'):
                out.append({x: y for x, y in spec.items() if x != 'wsplit'} if ws == 0 else dict(spec, wsplit=ws))
        for level in ('depth', 'asap'):
            if level != spec['level']:
                out.append(dict(spec, level=level))
        for balance in ('levels', 'work'):
            if balance != spec['balance'] and spec['S'] != 'each':
                out.append(dict(spec, balance=balance))
        for mode in MODES:
            if mode != spec.get('mode', 'phase'):
                out.append(dict(spec, mode=mode, merge=1))
        if spec.get('mode') == 'band':
            for m in (2, 3):
                out.append(dict(spec, merge=m))
            out.append(dict(spec, mode='phase_band', merge=1))
    elif g == 'heft':
        for f in (0.5, 0.7, 1.4, 2.0):
            out.append(dict(spec, lam=round(spec['lam'] * f, 3)))
        for mode in ('phase', 'phase_alap', 'phase_split', 'phase_alap_split'):
            if mode != spec.get('mode', 'phase'):
                out.append(dict(spec, mode=mode))
    elif g == 'comp':
        for S in (2, 3, 4):
            out.append({'gen': 'band', 'level': 'depth', 'balance': 'levels', 'S': S,
                        'eps': 0.08, 'tiny': 0, 'mode': 'band'})
    return out


def spec_search(ctx, start_spec, time_left, max_rounds=8, per_round=6, max_chunks=2):
    """从 start_spec 出发的参数邻域爬山。

    ctx 需提供：view、build(spec)->plan|None、proxy(plan)->float、
    run(name, plan, spec)->None（真评估并更新最优）、best_key()、best_spec()。
    time_left() 返回剩余秒数与下一次评估预计秒数的差（<=0 即停止）。
    """
    visited = {spec_name(start_spec)}
    current = start_spec
    for _ in range(max_rounds):
        if time_left() <= 0:
            return
        cand = []
        for nb in neighbors(current, ctx.view, ctx.k):
            name = spec_name(nb)
            if name in visited:
                continue
            visited.add(name)
            plan = ctx.build(nb)
            if plan is None:
                continue
            cand.append((ctx.proxy(plan, nb), name, nb, plan))
        if not cand:
            return
        cand.sort(key=lambda x: x[0])
        before = ctx.best_key()
        for i, (px, name, nb, plan) in enumerate(cand[:per_round * max_chunks]):
            if time_left() <= 0:
                break
            ctx.run(name, plan, nb)
            # 按块评估：一块 per_round 个评完后若已改进就进入下一轮
            if (i + 1) % per_round == 0 and ctx.best_key() < before:
                break
        if not ctx.best_key() < before:
            return
        current = ctx.best_spec()

class _GroupState:
    """组换核局部搜索的状态：组划分、组的相邻关系、当前核分配、代理打分。"""

    def __init__(self, ctx, spec):
        b, view = ctx.builder, ctx.view
        self.view, self.k = view, ctx.k
        # comp 规格：整张图一条层带，组 = 连通分量
        self.band_of = b.bands(spec) or {v: 0 for v in view.ops}
        self.mode, self.merge = spec.get('mode', 'phase'), spec.get('merge', 1)
        self.core = dict(b.cores(spec))
        ws = spec.get('wsplit', 0) or getattr(ctx, 'ils_split', 0)
        self.groups = [ops for _, ops in band_groups(view, self.band_of, spec.get('tinyop', 0),
                                                      ws, spec.get('kc') or ctx.k)]
        gid = {}
        for i, ops in enumerate(self.groups):
            for v in ops:
                gid[v] = i
        nbr = [set() for _ in self.groups]
        for v in view.ops:
            for u in view.pred[v]:
                a, c = gid[u], gid[v]
                if a != c:
                    nbr[a].add(c)
                    nbr[c].add(a)
        self.nbr = [sorted(x) for x in nbr]
        self.gcore = [self.core[ops[0]] for ops in self.groups]
        self.proxy = ctx._proxy

    def keys(self):
        return keys_for(self.view, self.core, self.band_of, self.mode, self.merge)

    def score(self):
        key = self.keys()
        if not keys_valid(self.view, self.core, key):
            return float('inf')
        return self.proxy.evaluate_keys(self.core, key)

    def move(self, i, c):
        self.gcore[i] = c
        for v in self.groups[i]:
            self.core[v] = c

    def cands(self, i, rng):
        if rng.random() < 0.2:
            return [c for c in range(self.k) if c != self.gcore[i]]
        return sorted({self.gcore[j] for j in self.nbr[i]} - {self.gcore[i]})

    def descend(self, cur, rng, deadline):
        """首次改进爬山到局部最优（连续 3·组数 步无改进）。返回 (代理分, 是否有接受的步)。"""
        stale, accepted, n = 0, False, len(self.groups)
        while stale < 3 * n and time.perf_counter() < deadline:
            i = rng.randrange(n)
            cands = self.cands(i, rng)
            if not cands:
                stale += 1
                continue
            c2, old = rng.choice(cands), self.gcore[i]
            self.move(i, c2)
            px = self.score()
            if px < cur:
                cur, accepted, stale = px, True, 0
            else:
                self.move(i, old)
                stale += 1
        return cur, accepted

    def plan(self):
        return plan_from_keys(self.view, self.keys(), self.core, self.k)


def group_ls(ctx, spec, deadline, seed=0):
    """第二层：在 spec 的核分配上做“组换核”局部搜索（代理引导），返回改进后的方案或 None。

    组 = 层带内的连通组（init.band_groups）；一步 = 把一个组整体换到相邻组所在的核（20% 概率换到任意核），
    Task 键按 spec 的切法重算（合法性仍由构造保证，不合法的一步直接拒绝）。代理严格变好才接受；
    连续 3·组数 步没有改进即停止。返回的方案由调用方真评估，是否采用只看评估器。
    """
    if ctx.builder.bands(spec) is None:
        return None
    st = _GroupState(ctx, spec)
    if len(st.groups) < 2:
        return None
    _, accepted = st.descend(st.score(), random.Random(seed), deadline)
    return st.plan() if accepted else None


def group_ils(ctx, spec, deadline, seed=0, name='ls'):
    """迭代局部搜索版 group_ls：到局部最优后真评估一次（代理创新低才评），再从代理最优状态随机扰动
    （把约 5% 的组换到随机核）继续爬山，直到截止。每个被评估的局部最优都经 ctx.run，是否采用只看评估器。
    spec 须是层带或 comp 规格。"""
    if spec['gen'] not in ('band', 'comp'):
        return
    st = _GroupState(ctx, spec)
    n = len(st.groups)
    if n < 2:
        return
    rng = random.Random(seed)
    cur = st.score()
    best_px, best_gcore = cur, list(st.gcore)
    rounds = 0
    while time.perf_counter() + 1.5 * ctx.est() < deadline:
        # 留 1.5 倍单次评估时间：大图上一步打分和构造方案本身也要零点几秒，负载高时评估也会变慢
        cur, _ = st.descend(cur, rng, deadline - 1.5 * ctx.est())
        rounds += 1
        if cur < best_px:
            best_px, best_gcore = cur, list(st.gcore)
            if time.perf_counter() + ctx.est() <= deadline:
                ctx.run(f'{name}#{rounds}', st.plan(), None)
        elif cur > best_px:
            for i, c in enumerate(best_gcore):
                if st.gcore[i] != c:
                    st.move(i, c)
            cur = best_px
        # 扰动：约 5% 的组（至少 2 个）换到随机的其他核
        for i in rng.sample(range(n), min(n, max(2, n // 20))):
            st.move(i, rng.choice([c for c in range(st.k) if c != st.gcore[i]]))
        cur = st.score()


class _TaskState:
    """Task 级换核：子图划分固定（问题1里 Task 时长与边界搬运只取决于划分，与核无关），只改 Task 所在核。
    核内顺序按名次排；名次是商图的线性扩展，所以任何换核都合法（商图无环、核内顺序与依赖一致）。
    名次取当前代理调度的开始时刻：重排名次不改变当前方案，只决定换核后插到目标核的哪个位置。
    打分 = (max(代理 makespan, Σ字节/BW), Σ结束时刻)，按字典序比较；第二项用来走出 makespan 的平台。"""

    def __init__(self, ctx, plan, tie=True):
        self.T = 0
        tt = ctx._proxy.task_times(plan)
        if tt is None:
            return
        sg_of, core_of, dur, preds, total_bytes, rank = tt
        order = sorted(dur, key=rank.get)
        idx = {sg: i for i, sg in enumerate(order)}
        T = self.T = len(order)
        self.k, self.tie = ctx.k, tie
        self.task_of = {op: idx[sg] for op, sg in sg_of.items()}
        self.dur = [dur[sg] for sg in order]
        self.preds = [sorted(idx[a] for a in preds[sg]) for sg in order]
        nbr = [set(p) for p in self.preds]
        for b, ps in enumerate(self.preds):
            for a in ps:
                nbr[a].add(b)
        self.nbr = [sorted(x) for x in nbr]
        self.core = [core_of[sg] for sg in order]
        self.order = list(range(T))
        self.floor = total_bytes / BW
        self.start, self.end = [0.0] * T, [0.0] * T

    def score(self):
        dur, preds, core, start, end = self.dur, self.preds, self.core, self.start, self.end
        last = [None] * self.k
        mk = tot = 0.0
        for t in self.order:
            c = core[t]
            p = last[c]
            s = 0.0 if p is None else end[p] + SAME_WAIT
            for a in preds[t]:
                e = end[a] if core[a] == c else end[a] + CROSS_WAIT
                if e > s:
                    s = e
            e = s + dur[t]
            start[t], end[t], last[c] = s, e, t
            tot += e
            if e > mk:
                mk = e
        return (max(mk, self.floor), tot if self.tie else 0.0)

    def rerank(self):
        """名次改为刚打过分的调度的 (开始时刻, 旧名次)：沿商图边开始时刻不减、核内相邻严格增加，故仍是线性扩展且方案不变。"""
        pos = {t: i for i, t in enumerate(self.order)}
        start = self.start
        self.order.sort(key=lambda t: (start[t], pos[t]))

    def move(self, t, c):
        self.core[t] = c

    def cands(self, t, rng):
        if rng.random() < 0.2:
            return [c for c in range(self.k) if c != self.core[t]]
        return sorted({self.core[j] for j in self.nbr[t]} - {self.core[t]})

    def descend(self, cur, rng, deadline):
        stale, accepted, n = 0, False, self.T
        while stale < 3 * n and time.perf_counter() < deadline:
            t = rng.randrange(n)
            cands = self.cands(t, rng)
            if not cands:
                stale += 1
                continue
            c2, old = rng.choice(cands), self.core[t]
            self.move(t, c2)
            sc = self.score()
            if sc < cur:
                cur, accepted, stale = sc, True, 0
                self.rerank()
            else:
                self.move(t, old)
                stale += 1
        return cur, accepted

    def plan(self):
        per_core = [[] for _ in range(self.k)]
        for t in self.order:
            per_core[self.core[t]].append(t)
        return make_plan(self.task_of, per_core)


def task_ils(ctx, plan, deadline, seed=0, name='task', tie=True):
    """第三层：固定 plan 的子图划分，做 Task 级换核的迭代局部搜索（代理引导）。
    一步 = 把一个 Task 换到相邻 Task 所在的核（20% 概率换到任意其他核，含空核），代理严格变好才接受；
    到局部最优且代理创新低时真评估一次（ctx.run），再从代理最优状态把约 5% 的 Task 换到随机核继续，直到截止。"""
    st = _TaskState(ctx, plan, tie=tie)
    if st.T < 2:
        return
    rng = random.Random(seed + 7919)
    cur = st.score()
    st.rerank()
    best = (cur, list(st.core), list(st.order))
    rounds = 0
    while time.perf_counter() + 1.5 * ctx.est() < deadline:
        cur, _ = st.descend(cur, rng, deadline - 1.5 * ctx.est())
        rounds += 1
        if cur < best[0]:
            improved = cur[0] < best[0][0]
            best = (cur, list(st.core), list(st.order))
            if improved and time.perf_counter() + ctx.est() <= deadline:
                ctx.run(f'{name}#{rounds}', st.plan(), None)
        elif cur > best[0]:
            st.core, st.order = list(best[1]), list(best[2])
        for t in rng.sample(range(st.T), min(st.T, max(2, st.T // 20))):
            st.move(t, rng.choice([c for c in range(st.k) if c != st.core[t]]))
        cur = st.score()
        st.rerank()
