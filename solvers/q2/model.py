"""问题2/3（场景 B）共用的图预处理：按 tensor 的输入/输出视图和 COPY 时长。

只读，按图哈希缓存。规则与 fast_eval/multicore_cut_evaluate_problem_2._build_scene_b_tasks 一致
（见 analysis/evaluator_semantics_audit.md §2）：
- 内部 tensor（计算 op 生产、计算 op 消费）：每个异核消费核生成一对 COPY_OUT(源核 MTE3)/
  COPY_IN(目标核 MTE2)，目标 COPY_IN 最早在源 COPY_OUT 结束 + 500 周期后发射；
- 图输入 tensor（原 COPY_IN 搬入）：每个消费核各 COPY_IN 一次；
- 图输出 tensor：生产核上一次最终 COPY_OUT；
- COPY 时长 max(1, ceil(size / 60))，所有 DDR COPY 共享 60 B/cycle。
100 个正式算例里没有直接 op→op 边、没有多生产者 tensor、收缩 DAG 的每条边都对应一个内部 tensor
（09-23 全量统计），这里仍对直接边做了保守处理：当作零字节的内部 tensor。
"""

import math
from collections import defaultdict

from common import CONFIG

BANDWIDTH = CONFIG['bandwidth']
DELAY = CONFIG['cross_core_copy_delay']
CAPACITY = dict(CONFIG['capacity'])

_CACHE = {}


def copy_cycles(size):
    return max(1, math.ceil(size / BANDWIDTH))


class SceneB:
    """计算 op 的输入/输出 tensor 视图（op 为原图 id）。"""

    def __init__(self, g):
        self.g = g
        self.ops = list(g.compute_ids)
        ins = defaultdict(list)       # op -> [(tid, size, producer)] 内部 tensor 输入
        gins = defaultdict(list)      # op -> [(tid, size)] 图输入 tensor
        outs = defaultdict(list)      # op -> [(tid, size, consumers)] 内部 tensor 输出
        fouts = defaultdict(list)     # op -> [(tid, size)] 图输出（最终 COPY_OUT）
        self.size = {}
        for tid, size, prod, cons in g.interior_tensors:
            self.size[tid] = size
            outs[prod].append((tid, size, tuple(cons)))
            for c in cons:
                ins[c].append((tid, size, prod))
        for tid, size, cons in g.input_tensors:
            self.size[tid] = size
            for c in cons:
                gins[c].append((tid, size))
        for tid, ps in g.producers.items():
            comp = [p for p in ps if p in g.compute_set]
            if not comp:
                continue
            consumers = g.consumers.get(tid, ())
            if any(g.ops[c]['op'] == 'COPY_OUT' for c in consumers if c in g.ops) or \
                    not any(c in g.compute_set for c in consumers):
                size = g.tensors[tid]['size']
                for p in comp:
                    fouts[p].append((tid, size))
        for n, (u, v, size) in enumerate(g.direct_edges):
            if u in g.compute_set and v in g.compute_set:
                tid = ('direct', n)
                self.size[tid] = size
                outs[u].append((tid, size, (v,)))
                ins[v].append((tid, size, u))
        self.ins = {u: sorted(ins[u], key=lambda x: str(x[0])) for u in self.ops}
        self.gins = {u: sorted(gins[u]) for u in self.ops}
        self.outs = {u: outs[u] for u in self.ops}
        self.fouts = {u: fouts[u] for u in self.ops}

    @property
    def total_input_bytes(self):
        return sum(s for tid, s in {(t, s) for u in self.ops for t, s in self.gins[u]})


def scene_b(g):
    sb = _CACHE.get(g.graph_hash)
    if sb is None:
        if len(_CACHE) > 8:
            _CACHE.clear()
        sb = _CACHE[g.graph_hash] = SceneB(g)
    return sb
