"""问题1 求解器的公共结构：图视图、核分配 → Task 的构造、廉价代理模拟。

问题1语义（见 analysis/evaluator_semantics_audit.md §1～§3）：
- 每个子图是一个 Task，各自独立做 Step1/2/3；Task 边界上的 tensor 一律经 DDR：
  生产 Task 写出一次（COPY_OUT），每个消费 Task 各读入一次（COPY_IN），同核也一样。
- 同核 Task 严格按列表顺序串行，后一个 Task 至少等前一个结束 + 100 周期；
  跨核前驱必须整个 Task 完成后再等 1000 周期。
- 全机 DDR 带宽 60 B/cycle 由所有核的 COPY 公平共享。

本模块只提供结构与代理量，不宣称任何收益；候选好坏一律以 common.evaluate 为准。
"""

import math
from collections import defaultdict

from common import CONFIG, make_plan

BW = CONFIG['bandwidth']
SAME_WAIT = CONFIG['same_core_wait']
CROSS_WAIT = CONFIG['cross_core_wait']
L1_CAP = CONFIG['capacity']['L1']
UB_CAP = CONFIG['capacity']['UB']


class View:
    """问题1 求解用的只读视图（基于 common.Graph，不改原图）。"""

    def __init__(self, g):
        self.g = g
        self.ops = list(g.topo_order)          # 计算 op 的确定性拓扑序
        self.pos = g.topo_index
        self.pred = g.pred
        self.succ = g.succ
        self.cyc = g.cycles
        self.pipe = g.pipe
        self.n = len(self.ops)

        depth = {}
        for v in self.ops:
            depth[v] = max((depth[u] + 1 for u in g.pred[v]), default=0)
        self.depth = depth
        self.max_depth = max(depth.values(), default=0)
        self.asap = g.top_level
        self.alap_rank = g.bottom_level

        # tensor 视图：只关心计算 op 读写的 tensor
        ops_set = g.compute_set
        self.t_size = {}
        self.t_prod = {}          # tensor -> 生产它的计算 op（没有则不在表里：图输入）
        self.t_cons = {}          # tensor -> 读它的计算 op 列表
        self.t_forced_out = set()  # 生产 Task 无论如何都要写出的 tensor（图输出/无计算消费者）
        self.op_in = defaultdict(list)
        self.op_out = defaultdict(list)
        for tid, t in g.tensors.items():
            prods = [p for p in g.producers.get(tid, ()) if p in ops_set]
            cons = sorted({c for c in g.consumers.get(tid, ()) if c in ops_set})
            if not prods and not cons:
                continue
            self.t_size[tid] = t['size']
            if prods:
                self.t_prod[tid] = prods[0]
                for p in prods:
                    self.op_out[p].append(tid)
                has_copy_out = any(g.ops[c]['op'] == 'COPY_OUT'
                                   for c in g.consumers.get(tid, ()) if c in g.ops)
                if has_copy_out or not cons:
                    self.t_forced_out.add(tid)
            self.t_cons[tid] = cons
            for c in cons:
                self.op_in[c].append(tid)

        self.total_work = g.total_work
        self.pipe_work = dict(g.pipe_work)

    # ------------------------------------------------------------ 分量等
    def components(self):
        return self.g.components


# ---------------------------------------------------------------- 核分配 → Task
def phases_from_cores(view, core):
    """phase(v) = max_u phase(u) + [core(u) != core(v)]。

    Task = (core, phase)。所有商图边与同核顺序边都使 phase 严格增加，故商图无环、
    Task 顺序无环（问题1的两项校验都满足）。这是给定核分配下 Task 数最少的切法。
    """
    ph = {}
    for v in view.ops:
        cv = core[v]
        p = 0
        for u in view.pred[v]:
            q = ph[u] + (core[u] != cv)
            if q > p:
                p = q
        ph[v] = p
    return ph


def alap_phases(view, core, ph):
    """在不增加最大 phase 的前提下把每个 op 推到尽量晚的 phase（ALAP）。"""
    top = max(ph.values(), default=0)
    late = {}
    for v in reversed(view.ops):
        cv = core[v]
        p = top
        for w in view.succ[v]:
            q = late[w] - (core[w] != cv)
            if q < p:
                p = q
        late[v] = max(p, ph[v])
    return late


def plan_from_keys(view, key, core, k):
    """key[v] 是可比较的 Task 键（同核内按键升序执行）；core[v] 为核。

    调用方负责键的合法性：同核依赖键不减、跨核依赖键严格增加。
    """
    groups = {}
    for v in view.ops:
        groups.setdefault((core[v], key[v]), []).append(v)
    per_core = [[] for _ in range(k)]
    mapping = {}
    for sg, (c, kk) in enumerate(sorted(groups)):
        per_core[c].append((kk, sg))
        for v in groups[(c, kk)]:
            mapping[v] = sg
    orders = [[sg for _, sg in sorted(lst)] for lst in per_core]
    return make_plan(mapping, orders)


def plan_from_cores(view, core, k, alap=False):
    ph = phases_from_cores(view, core)
    if alap:
        ph = alap_phases(view, core, ph)
    return plan_from_keys(view, ph, core, k)


def decode_plan(plan):
    """plan → (sg_of_op{int:int}, core_of_sg, order_pos_of_sg)。"""
    sg_of = {int(op): sg for op, sg in plan['node_to_subgraph'].items()}
    core_of, pos_of = {}, {}
    for c, order in enumerate(plan['core_schedules']):
        for i, sg in enumerate(order):
            core_of[sg] = c
            pos_of[sg] = i
    return sg_of, core_of, pos_of


# ---------------------------------------------------------------- 代理模拟
class ProxyModel:
    """Task 级廉价代理：只用于候选排序，不作为收益证据。

    Task 时长 ≈ rho·Σcycles + gamma·(读入+写出字节)/BW + 首尾搬运延迟；
    按问题1规则（同核 +100、跨核整 Task 完成 +1000）做 Task 级事件推进；
    再与全机 DDR 下界 Σ字节/BW 取大。
    """

    def __init__(self, view, rho=None, gamma=0.3, single_makespan=None):
        self.view = view
        work = max(1, view.total_work)
        if rho is None:
            rho = 1.0
            if single_makespan:
                # 单核实测包含原图搬运；用 (T1 - 原图搬运的一部分)/Σcycles 粗估流水重叠系数
                rho = min(1.0, max(0.5, single_makespan / work))
        self.rho = rho
        self.gamma = gamma

    def task_io(self, ops_set, ops):
        v = self.view
        in_b = out_b = 0
        first_in = last_out = 0
        seen = set()
        for op in ops:
            for t in v.op_in[op]:
                if t in seen:
                    continue
                seen.add(t)
                p = v.t_prod.get(t)
                if p is None or p not in ops_set:
                    in_b += v.t_size[t]
            for t in v.op_out[op]:
                if t in v.t_forced_out or any(c not in ops_set for c in v.t_cons.get(t, ())):
                    out_b += v.t_size[t]
        return in_b, out_b

    def _tensor_table(self):
        if getattr(self, '_tt', None) is None:
            v = self.view
            prods = defaultdict(list)
            for op in v.ops:
                for t in v.op_out[op]:
                    prods[t].append(op)
            self._tt = [(v.t_size[t], v.t_prod.get(t), tuple(v.t_cons.get(t, ())), t in v.t_forced_out,
                         tuple(prods.get(t, ()))) for t in v.t_size]
        return self._tt

    def evaluate(self, plan):
        sg_of, core_of, _ = decode_plan(plan)
        return self._simulate(sg_of, core_of, plan['core_schedules'])

    def evaluate_keys(self, core, key):
        """与 evaluate(plan_from_keys(view, key, core, k)) 相同，但不构造方案（局部搜索用）。"""
        sg_id, sg_of = {}, {}
        for op in self.view.ops:
            t = (core[op], key[op])
            s = sg_id.get(t)
            if s is None:
                s = sg_id[t] = len(sg_id)
            sg_of[op] = s
        core_of, per_core = {}, defaultdict(list)
        for (c, kk), s in sg_id.items():
            core_of[s] = c
            per_core[c].append((kk, s))
        schedules = [[s for _, s in sorted(lst)] for lst in per_core.values()]
        return self._simulate(sg_of, core_of, schedules)

    def task_costs(self, sg_of):
        """子图划分固定时与核无关的部分：(各 Task 代理时长, 商图前驱, 全部边界字节)。"""
        v = self.view
        work = defaultdict(int)
        cyc = v.cyc
        for op in v.ops:
            work[sg_of[op]] += cyc[op]
        io = defaultdict(int)
        # 按 tensor 计边界读写：每个不含生产者的消费 Task 读一次；有 Task 外消费者或必须写出时生产 Task 写一次
        for size, p, cons, forced, prods in self._tensor_table():
            ps = sg_of[p] if p is not None else None
            cs = {sg_of[c] for c in cons}
            cs.discard(ps)
            for s in cs:
                io[s] += size
            if forced or cs:
                for q in prods:
                    io[sg_of[q]] += size
        total_bytes = sum(io.values())
        dur = {sg: self.rho * w + self.gamma * io[sg] / BW for sg, w in work.items()}
        preds = defaultdict(set)
        for op in v.ops:
            a = sg_of[op]
            for w in v.succ[op]:
                b = sg_of[w]
                if a != b:
                    preds[b].add(a)
        return dur, preds, total_bytes

    def _simulate(self, sg_of, core_of, schedules):
        dur, preds, total_bytes = self.task_costs(sg_of)
        members = dur
        # Task 级事件推进：按核顺序 + 商图依赖做最长路
        order_prev = {}
        for order in schedules:
            for a, b in zip(order, order[1:]):
                order_prev[b] = a
        indeg = {sg: len(preds[sg]) + (1 if sg in order_prev else 0) for sg in members}
        succs = defaultdict(list)
        for b, ps in preds.items():
            for a in ps:
                succs[a].append(b)
        for b, a in order_prev.items():
            succs[a].append(b)
        ready = [sg for sg, d in indeg.items() if d == 0]
        end = {}
        while ready:
            sg = ready.pop()
            c = core_of[sg]
            start = 0
            if sg in order_prev:
                start = end[order_prev[sg]] + SAME_WAIT
            for a in preds[sg]:
                if core_of[a] != c:
                    start = max(start, end[a] + CROSS_WAIT)
                else:
                    start = max(start, end[a])
            end[sg] = start + dur[sg]
            for b in succs[sg]:
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
        if len(end) != len(members):
            return float('inf')
        makespan = max(end.values(), default=0)
        return max(makespan, total_bytes / BW)

    def task_times(self, plan):
        """plan 的 Task 级代理：(sg_of, core_of, dur, preds, total_bytes, rank)。
        rank = 按 (代理开始时刻, 推进次序) 排的名次，是商图 ∪ 核内顺序的一个线性扩展；方案不合法返回 None。"""
        sg_of, core_of, _ = decode_plan(plan)
        dur, preds, total_bytes = self.task_costs(sg_of)
        start, last = {}, {}
        for order in plan['core_schedules']:
            for i, sg in enumerate(order):
                last[sg] = order[i - 1] if i else None
        # 按“前驱全部完成”逐个推进（方案合法，商图 ∪ 核内顺序无环）
        indeg, succs = {}, defaultdict(list)
        for sg in dur:
            ps = set(preds[sg])
            if last.get(sg) is not None:
                ps.add(last[sg])
            indeg[sg] = len(ps)
            for a in ps:
                succs[a].append(sg)
        ready = [sg for sg, d in indeg.items() if d == 0]
        end, seq = {}, {}
        while ready:
            sg = ready.pop()
            seq[sg] = len(seq)
            c = core_of[sg]
            s = 0
            if last.get(sg) is not None:
                s = end[last[sg]] + SAME_WAIT
            for a in preds[sg]:
                s = max(s, end[a] + (CROSS_WAIT if core_of[a] != c else 0))
            start[sg], end[sg] = s, s + dur[sg]
            for b in succs[sg]:
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
        if len(end) != len(dur):
            return None
        order = sorted(dur, key=lambda sg: (start[sg], seq[sg]))
        rank = {sg: i for i, sg in enumerate(order)}
        return sg_of, core_of, dur, preds, total_bytes, rank
