"""方案（Plan）的规范形式、存取和哈希。

方案就是官方要求的 dict，严格只含两个字段：
    {'node_to_subgraph': {'<op id 十进制字符串>': sgid, ...},
     'core_schedules':   [[core0 上的 sgid 顺序], [core1 ...], ...]}
- 键必须是十进制字符串（与 CLI 读入 JSON 后一致），值是非负 int。
- core_schedules 的长度就是核数 k，允许空核；每个 sgid 恰好出现一次。
- 子图顺序是决策的一部分：映射相同但顺序不同的方案是不同候选，哈希也不同。
优化日志、元数据一律另存，不能塞进方案 JSON。
"""

import hashlib
import json
import os
import time
from pathlib import Path

PLAN_FIELDS = ('node_to_subgraph', 'core_schedules')


def make_plan(node_to_subgraph, core_schedules):
    """由 {op: sgid} 和每核 sgid 顺序构造规范方案（键转字符串并按 op id 排序）。"""
    mapping = {str(int(op)): int(sg) for op, sg in
               sorted(node_to_subgraph.items(), key=lambda kv: int(kv[0]))}
    schedules = [[int(sg) for sg in order] for order in core_schedules]
    return {'node_to_subgraph': mapping, 'core_schedules': schedules}


def from_groups(groups, core_of_group, orders=None, num_cores=None):
    """由若干 op 组构造方案。

    groups: [[op, ...], ...]，第 i 组的 sgid 就是 i；
    core_of_group: 每组所在核心；
    orders: 可选，每核的组顺序；缺省按组编号升序（调用方负责依赖合法）。
    """
    k = num_cores if num_cores is not None else max(core_of_group) + 1
    mapping = {op: sg for sg, ops in enumerate(groups) for op in ops}
    if orders is None:
        orders = [[] for _ in range(k)]
        for sg, core in enumerate(core_of_group):
            orders[core].append(sg)
    return make_plan(mapping, orders)


def canonical_json(plan):
    return json.dumps({f: plan[f] for f in PLAN_FIELDS}, sort_keys=False,
                      separators=(',', ':'))


def plan_hash(plan):
    return hashlib.sha1(canonical_json(plan).encode('utf-8')).hexdigest()


def save_plan(plan, path):
    """原子写出方案 JSON（只写两个字段）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps({f: plan[f] for f in PLAN_FIELDS}, ensure_ascii=False),
                   encoding='utf-8')
    for attempt in range(5):   # Windows 上目标文件正被读取时 replace 会短暂失败
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)
    return path


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(plan) != set(PLAN_FIELDS):
        raise ValueError(f'{path}: plan must contain exactly {PLAN_FIELDS}')
    return plan


def plan_stats(plan):
    schedules = plan['core_schedules']
    return {
        'num_cores': len(schedules),
        'nonempty_cores': sum(1 for order in schedules if order),
        'num_subgraphs': len(set(plan['node_to_subgraph'].values())),
        'subgraphs_per_core': [len(order) for order in schedules],
    }


def op_to_core(plan):
    """{op id(int): core}，便于按核统计。"""
    core_of_sg = {sg: c for c, order in enumerate(plan['core_schedules']) for sg in order}
    return {int(op): core_of_sg[sg] for op, sg in plan['node_to_subgraph'].items()}


def best_plan(problem, k, case):
    """最优档案里 (problem, k, case) 的当前最优方案；没有则返回 None。只读。

    用于热启动，例如问题3从问题2最优方案出发、问题2复用问题1的切分。
    """
    from .paths import BEST
    path = BEST / f'q{problem}' / f'k{k}' / f'{case}.json'
    return load_plan(path) if path.exists() else None
