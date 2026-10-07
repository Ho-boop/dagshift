"""问题2/3 列表调度器：在近似的场景 B 语义下同时决定 op→核 与各核的执行顺序。

近似模型（只用于生成候选，最终一律由 common.evaluate 评分）：
- 每核 4 条 Pipe（M/V 计算、MTE2 读入、MTE3 写出），每条 Pipe 按调度顺序串行（非插入式），
  与评估器“各 Pipe 按核内序列投影 FIFO 执行”一致；
- 异核内部 tensor：源核 MTE3 上 COPY_OUT，结束 + 500 后目标核 MTE2 上 COPY_IN（同一 tensor 同一
  目标核只搬一次）；图输入在每个消费核首次使用前 COPY_IN 一次；图输出在生产后 COPY_OUT 一次；
- DDR 共享带宽可选两种近似：'none' 各 COPY 独占 60 B/cycle；'serial' 全机 DDR 串行服务。
- 不模拟 Step2 spill 和 Step3 内存依赖（由评估器体现）。
- cache=True（问题3 近似）：同一 tensor 的首次 COPY_IN 完成后写入全局 FIFO（1 MiB，按决策顺序插入、
  最早插入者先驱逐），之后开始的同 tensor COPY_IN 视为命中，时长按 250 B/cycle 计、不占 DDR；
  COPY_OUT 与 500 周期释放不变（审计 §4）。

输出的全局顺序 seq 是计算 DAG 的拓扑序；encode.per_op_plan 把它投影到各核，每个 op 一个子图，
使评估器里各 Pipe 的 FIFO 顺序与这里的模拟顺序一致。
"""

import heapq
import math
import random
from collections import OrderedDict

from common import CONFIG

from .model import BANDWIDTH, DELAY, copy_cycles, scene_b

CACHE_CAPACITY = CONFIG['cache_capacity_bytes']
CACHE_BW = CONFIG['cache_bandwidth_bytes_per_cycle']


def hit_cycles(size):
    return max(1, math.ceil(size / CACHE_BW))

ORDERS = ('blevel', 'blevel_comm', 'step1', 'tlevel')


def _rank_comm(g, sb, k):
    """HEFT 的 upward rank：边上加平均通信代价 (k-1)/k·(2·COPY + 500)。"""
    avg = (k - 1) / k
    rank = {}
    for u in reversed(g.topo_order):
        best = 0.0
        for tid, size, cons in sb.outs[u]:
            cc = avg * (2 * copy_cycles(size) + DELAY)
            for v in cons:
                r = cc + rank[v]
                if r > best:
                    best = r
        for v in g.succ[u]:
            if rank[v] > best:
                best = rank[v]
        rank[u] = g.cycles[u] + best
    return rank


_STEP1 = {}


def step1_order(g):
    """官方 Step1（多源 reverse DFS）在整图上的访问顺序，只保留计算 op：内存友好的单核顺序。"""
    seq = _STEP1.get(g.graph_hash)
    if seq is None:
        from fast_eval.schedule_step1 import step1_schedule
        seq = [u for u in step1_schedule(g.raw) if u in g.compute_set]
        if len(_STEP1) > 8:
            _STEP1.clear()
        _STEP1[g.graph_hash] = seq
    return seq


def decision_order(g, k, order='blevel', seed=0, noise=0.0):
    """列表调度的决策顺序（必须是拓扑序）。noise>0 时对优先级做乘性扰动（多起点）。"""
    if order == 'step1':
        return list(step1_order(g))
    if order == 'tlevel':
        top = g.top_level
        return sorted(g.compute_ids, key=lambda u: (top[u], g.topo_index[u]))
    prio = g.bottom_level if order == 'blevel' else _rank_comm(g, scene_b(g), k)
    if noise > 0:
        rng = random.Random(seed)
        # 乘性扰动后再按拓扑修正：沿拓扑逆序取 max，保证前驱优先级不低于后继
        p = {u: prio[u] * (1.0 + noise * rng.random()) for u in g.compute_ids}
        fixed = {}
        for u in reversed(g.topo_order):
            fixed[u] = max([p[u]] + [fixed[v] + 1e-9 for v in g.succ[u]])
        prio = fixed
    return sorted(g.compute_ids, key=lambda u: (-prio[u], g.topo_index[u]))


def list_schedule(g, k, order='blevel', ddr='none', lam=0.0, fixed=None, seq_by='start',
                  seed=0, noise=0.0, balance=0.0, cache=False, group=None):
    """返回 (core_of, seq, est_makespan)。

    order   决策顺序（ORDERS 之一，或直接给一个拓扑序列表）
    ddr     'none' / 'serial'：DDR 共享带宽的近似方式
    lam     选核代价 = EFT + lam·(本次新增 DDR 字节 / 60)，lam 越大越倾向少跨核
    fixed   {op: core}：给定核归属（可只给一部分 op，其余照常选核），用于邻域搜索和分量预分配
    seq_by  'start'：各核按模拟开始时间排序；'decision'：按决策顺序
    balance 选核代价额外加 balance·(该核已分配计算量)，抑制过度集中
    cache   问题3 近似：模拟只读 FIFO Cache 的命中（见模块说明）
    group   {op: 簇}：同簇 op 绑定到簇内首个被调度 op 所选的核（cluster.edge_clusters）
    """
    sb = scene_b(g)
    dec = order if isinstance(order, list) else decision_order(g, k, order, seed, noise)
    cores = range(k)
    pipe_free = [dict() for _ in cores]
    mte2 = [0.0] * k
    mte3 = [0.0] * k
    ddr_free = 0.0
    serial = ddr == 'serial'
    arrive = {}                     # (tid, core) -> 该核上可用时刻
    core_of, start, finish = {}, {}, {}
    load = [0.0] * k
    cycles, pipe = g.cycles, g.pipe
    ins, gins, fouts = sb.ins, sb.gins, sb.fouts
    cache_at = {}                   # tid -> 写入 Cache 的时刻（首次未命中 COPY_IN 完成）
    fifo, used = OrderedDict(), 0
    hit_saving = 1.0 - BANDWIDTH / CACHE_BW
    group_core = {}

    for u in dec:
        if fixed is not None and u in fixed:
            cand = (fixed[u],)
        elif group is not None and group[u] in group_core:
            cand = (group_core[group[u]],)
        else:
            cand = cores
        best = None
        for c in cand:
            ready = 0.0
            fetch = []              # (release, dur, key, src_core)
            new_bytes = 0
            for tid, size, p in ins[u]:
                pc = core_of[p]
                if pc == c:
                    f = finish[p]
                    if f > ready:
                        ready = f
                    continue
                a = arrive.get((tid, c))
                if a is not None:
                    if a > ready:
                        ready = a
                    continue
                fetch.append((finish[p], copy_cycles(size), tid, pc, size))
                new_bytes += 2 * size
            for tid, size in gins[u]:
                a = arrive.get((tid, c))
                if a is not None:
                    if a > ready:
                        ready = a
                    continue
                fetch.append((0.0, copy_cycles(size), tid, -1, size))
                new_bytes += size
            m2 = mte2[c]
            d = ddr_free
            outs_t = None
            if fetch:
                m3 = {}
                timed = []
                for rel, dur, tid, pc, size in fetch:
                    if pc >= 0:
                        s = max(rel, m3.get(pc, mte3[pc]))
                        if serial:
                            s = max(s, d)
                            d = s + dur
                        m3[pc] = s + dur
                        timed.append((s + dur + DELAY, dur, tid, size))
                    else:
                        timed.append((0.0, dur, tid, size))
                timed.sort()
                arr = []
                for rel, dur, tid, size in timed:
                    s = rel if rel > m2 else m2
                    hit = False
                    if cache:
                        h = cache_at.get(tid)
                        if h is not None and h <= s:
                            hit = True
                            dur = hit_cycles(size)
                            new_bytes -= size * hit_saving
                    if serial and not hit:
                        s = max(s, d)
                        d = s + dur
                    m2 = s + dur
                    arr.append((tid, m2, hit, size))
                    if m2 > ready:
                        ready = m2
                outs_t = (m3, arr)
            pf = pipe_free[c].get(pipe[u], 0.0)
            est = ready if ready > pf else pf
            eft = est + cycles[u]
            cost = eft + lam * new_bytes / BANDWIDTH + balance * load[c]
            key = (cost, load[c], c)
            if best is None or key < best[0]:
                best = (key, c, est, eft, m2, d, outs_t)
        _, c, est, eft, m2, d, outs_t = best
        if outs_t is not None:
            m3, arr = outs_t
            for pc, t in m3.items():
                mte3[pc] = t
            for tid, t, hit, size in arr:
                arrive[(tid, c)] = t
                if cache and not hit and 0 < size <= CACHE_CAPACITY and tid not in cache_at:
                    cache_at[tid] = t
                    fifo[tid] = size
                    used += size
                    while used > CACHE_CAPACITY:
                        old, sz = fifo.popitem(last=False)
                        used -= sz
                        del cache_at[old]
        mte2[c] = m2
        ddr_free = d
        core_of[u] = c
        if group is not None:
            group_core.setdefault(group[u], c)
        start[u] = est
        finish[u] = eft
        pipe_free[c][pipe[u]] = eft
        load[c] += cycles[u]
        for tid, size in fouts[u]:            # 图输出：生产后在本核 MTE3 写回
            dur = copy_cycles(size)
            s = max(eft, mte3[c])
            if serial:
                s = max(s, ddr_free)
                ddr_free = s + dur
            mte3[c] = s + dur

    pos = {u: i for i, u in enumerate(dec)}
    if seq_by == 'decision':
        seq = list(dec)
    else:
        seq = sorted(dec, key=lambda u: (start[u], pos[u]))
    est_ms = max(max(finish.values(), default=0.0), max(mte3, default=0.0))
    return core_of, seq, est_ms
