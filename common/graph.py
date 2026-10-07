"""计算图的只读结构视图：解析一次，所有 solver 共用。

约定
- ``Graph.raw`` 是原始 JSON dict，原样传给评估器，任何代码都不得修改它。
- “计算 op”指非 COPY_IN/COPY_OUT 的 op，也就是方案里要分配子图的节点。
- 计算 DAG（``succ``/``pred``）与官方 ``derive_multicore_plan`` 的收缩规则一致：
  经 tensor 的生产→消费关系、直接 op→op 边，并穿过 COPY 节点保留可达性。
- 共享输入（同一 COPY_IN 输出被多个计算 op 读）不是计算先后关系，单独记录。
"""

import hashlib
import heapq
import json
from collections import defaultdict
from functools import cached_property, lru_cache
from pathlib import Path

from .paths import case_path

COPY_TYPES = frozenset({'COPY_IN', 'COPY_OUT'})


class Graph:
    def __init__(self, raw, name='graph', file_bytes=None):
        self.raw = raw
        self.name = name
        self.graph_hash = hashlib.sha1(
            file_bytes if file_bytes is not None
            else json.dumps(raw, sort_keys=True).encode('utf-8')).hexdigest()

        self.ops = {o['id']: o for o in raw['ops']}
        self.tensors = {t['id']: t for t in raw['tensors']}
        self.compute_ids = sorted(i for i, o in self.ops.items() if o['op'] not in COPY_TYPES)
        self.compute_set = frozenset(self.compute_ids)
        self.cycles = {i: self.ops[i]['cycles'] for i in self.compute_ids}
        self.pipe = {i: self.ops[i]['pipe'] for i in self.compute_ids}

        self.producers = defaultdict(list)   # tensor -> 生产 op（任意 op）
        self.consumers = defaultdict(list)   # tensor -> 消费 op（任意 op）
        self.op_inputs = defaultdict(list)   # op -> 输入 tensor
        self.op_outputs = defaultdict(list)  # op -> 输出 tensor
        self.direct_edges = []               # (src_op, dst_op, data_size) 直接 op→op 边
        for e in raw['edges']:
            u, v = e['source'], e['target']
            if u in self.ops and v in self.tensors:
                self.producers[v].append(u)
                self.op_outputs[u].append(v)
            elif u in self.tensors and v in self.ops:
                self.consumers[u].append(v)
                self.op_inputs[v].append(u)
            elif u in self.ops and v in self.ops:
                self.direct_edges.append((u, v, e.get('data_size', 0)))

        self.succ, self.pred = self._contracted_compute_dag()

    # ------------------------------------------------------------------ DAG
    def _contracted_compute_dag(self):
        full = defaultdict(set)
        for tid, ps in self.producers.items():
            for p in ps:
                for c in self.consumers.get(tid, ()):
                    if p != c:
                        full[p].add(c)
        for u, v, _ in self.direct_edges:
            if u != v:
                full[u].add(v)
        succ = {u: set() for u in self.compute_ids}
        pred = {u: set() for u in self.compute_ids}
        for src in self.compute_ids:
            stack, seen = list(full.get(src, ())), set()
            while stack:
                dst = stack.pop()
                if dst in self.compute_set:
                    if dst != src:
                        succ[src].add(dst)
                        pred[dst].add(src)
                    continue
                if dst in seen:
                    continue
                seen.add(dst)
                stack.extend(full.get(dst, ()))
        return succ, pred

    @cached_property
    def topo_order(self):
        """确定性拓扑序（Kahn，就绪集合按 op id 最小优先）。"""
        indeg = {u: len(self.pred[u]) for u in self.compute_ids}
        heap = [u for u in self.compute_ids if indeg[u] == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            u = heapq.heappop(heap)
            order.append(u)
            for v in self.succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(heap, v)
        if len(order) != len(self.compute_ids):
            raise ValueError(f'{self.name}: compute DAG has a cycle')
        return order

    @cached_property
    def topo_index(self):
        return {u: i for i, u in enumerate(self.topo_order)}

    @cached_property
    def top_level(self):
        """最早开始时刻（只算计算周期，忽略搬运、同步和资源冲突）。"""
        top = {}
        for u in self.topo_order:
            top[u] = max((top[p] + self.cycles[p] for p in self.pred[u]), default=0)
        return top

    @cached_property
    def bottom_level(self):
        """含自身周期的到出口最长计算路径长度，用作关键路径优先级。"""
        bottom = {}
        for u in reversed(self.topo_order):
            bottom[u] = self.cycles[u] + max((bottom[s] for s in self.succ[u]), default=0)
        return bottom

    @cached_property
    def critical_path(self):
        return max(self.bottom_level.values(), default=0)

    @cached_property
    def pipe_work(self):
        """各 Pipe 上计算 op 的周期总和，如 {'PIPE_M': .., 'PIPE_V': ..}。"""
        work = defaultdict(int)
        for u in self.compute_ids:
            work[self.pipe[u]] += self.cycles[u]
        return dict(work)

    @cached_property
    def total_work(self):
        return sum(self.cycles.values())

    def compute_lower_bound(self, k):
        """计算型松弛下界 max{CP, 各 Pipe 工作量/k}；不含搬运，只作参考。"""
        return max([self.critical_path] + [w / k for w in self.pipe_work.values()])

    # ----------------------------------------------------------- components
    @cached_property
    def components(self):
        """计算弱连通分量（只经真实依赖连接），按总周期降序、首 id 升序。"""
        remaining, comps = set(self.compute_ids), []
        for start in self.compute_ids:
            if start not in remaining:
                continue
            remaining.remove(start)
            queue = [start]
            for u in queue:
                for v in self.succ[u] | self.pred[u]:
                    if v in remaining:
                        remaining.remove(v)
                        queue.append(v)
            comps.append(sorted(queue))
        comps.sort(key=lambda c: (-sum(self.cycles[u] for u in c), c[0]))
        return comps

    @cached_property
    def component_of(self):
        return {u: k for k, comp in enumerate(self.components) for u in comp}

    # -------------------------------------------------------------- tensors
    @cached_property
    def interior_tensors(self):
        """由计算 op 生产、被计算 op 消费的 tensor：[(tid, size, producer, consumers)]。"""
        out = []
        for tid, ps in self.producers.items():
            comp_ps = [p for p in ps if p in self.compute_set]
            if not comp_ps:
                continue
            cs = sorted({c for c in self.consumers.get(tid, ()) if c in self.compute_set})
            if cs:
                out.append((tid, self.tensors[tid]['size'], comp_ps[0], cs))
        return out

    @cached_property
    def input_tensors(self):
        """由 COPY_IN 搬入、被计算 op 读取的片上 tensor：[(tid, size, consumers)]。"""
        out = []
        for tid, ps in self.producers.items():
            if not ps or any(p in self.compute_set for p in ps):
                continue
            if all(self.ops[p]['op'] == 'COPY_IN' for p in ps):
                cs = sorted({c for c in self.consumers.get(tid, ()) if c in self.compute_set})
                if cs:
                    out.append((tid, self.tensors[tid]['size'], cs))
        return out

    def boundary_bytes(self, op_group, problem):
        """按 tensor 估算切分新增的边界搬运字节（代理量，不是官方指标）。

        op_group: 计算 op → 分组。问题 1 传子图 id（每子图一个 Task）；
        问题 2/3 传核心 id（同核合并为一个 Task）。规则见 evaluator_semantics_audit.md §2：
        - 问题 1：内部 tensor 有 r 个外部消费组 → s·(1+r)；
        - 问题 2/3：内部 tensor 有 r 个外部消费核 → 2·s·r；
        - 共享输入被 m 个组读取 → s·(m-1)。
        不含 spill、直接 op→op 边和最终输出的特殊边界；最终以官方评估为准。
        """
        interior = inputs = 0
        for _tid, size, producer, consumers in self.interior_tensors:
            home = op_group[producer]
            r = len({op_group[c] for c in consumers} - {home})
            if r:
                interior += size * (1 + r) if problem == 1 else 2 * size * r
        for _tid, size, consumers in self.input_tensors:
            m = len({op_group[c] for c in consumers})
            inputs += size * (m - 1)
        return {'interior_bytes': interior, 'input_dup_bytes': inputs,
                'total_bytes': interior + inputs}

    def summary(self):
        return {
            'case': self.name, 'compute_ops': len(self.compute_ids),
            'components': len(self.components),
            'largest_component_work_fraction':
                sum(self.cycles[u] for u in self.components[0]) / max(1, self.total_work),
            'critical_path': self.critical_path, 'total_work': self.total_work,
            'pipe_work': self.pipe_work, 'direct_edges': len(self.direct_edges),
        }


@lru_cache(maxsize=4)
def load_graph(case):
    """按算例名（如 'case_001'）或 JSON 路径读取并缓存 Graph。"""
    path = case_path(case) if not str(case).endswith('.json') else Path(case)
    data = open(path, 'rb').read()
    name = case if not str(case).endswith('.json') else path.stem
    return Graph(json.loads(data.decode('utf-8')), name=name, file_bytes=data)
