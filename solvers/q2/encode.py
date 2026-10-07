"""把 (op→核, 全局拓扑顺序) 编码成官方方案。

场景 B 里同核子图合并成一个 Task，但评估器按 core_schedules 的子图顺序对 Step1 序列做稳定分桶
（桶内保持 Step1 的 reverse-DFS 顺序），再把序列投影成各 Pipe 的 FIFO。因此：
- per_op_plan：每个 op 一个子图，子图编号 = 全局顺序位置，各核按编号升序。核内序列就是 seq 在该核
  上的投影；COPY_IN 落在首个消费者的桶里（紧挨消费者之前），COPY_OUT 落在生产者的桶里（紧跟其后）。
  seq 是计算 DAG 的拓扑序 → 收缩子图 DAG 无环、同核顺序与依赖一致（入口校验必过）；
  Step3 执行图的全局环只能由评估器判定。
- runs_plan：沿 seq 把同核连续段合成一个子图（子图更少；段内顺序交还给 Step1）。
"""

from common import from_groups, make_plan


def per_op_plan(k, core_of, seq):
    mapping = {u: i for i, u in enumerate(seq)}
    schedules = [[] for _ in range(k)]
    for u in seq:
        schedules[core_of[u]].append(mapping[u])
    return make_plan(mapping, schedules)


def runs_plan(k, core_of, seq):
    groups, cores = [], []
    cur, cc = [], None
    for u in seq:
        c = core_of[u]
        if c != cc:
            if cur:
                groups.append(cur)
                cores.append(cc)
            cur, cc = [], c
        cur.append(u)
    if cur:
        groups.append(cur)
        cores.append(cc)
    return from_groups(groups, cores, num_cores=k)


def plan_core_of(plan):
    """方案里的 {op(int): core}。"""
    core_of_sg = {sg: c for c, order in enumerate(plan['core_schedules']) for sg in order}
    return {int(op): core_of_sg[sg] for op, sg in plan['node_to_subgraph'].items()}
