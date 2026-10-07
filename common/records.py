"""结果行格式（JSONL，每次评估一行）与候选比较规则。

每次批量运行写 results/runs/<run_id>/rows.jsonl，方案写在同目录 plans/ 下。
行的字段固定如下（缺失值为 null），summarize 和作图只认这些字段。
"""

import json
import time
from pathlib import Path

ROW_FIELDS = [
    # 身份
    'run_id', 'case', 'problem', 'k', 'solver', 'tag', 'version', 'seed', 'budget_s',
    # 结果（来自 common.evaluate）
    'ok', 'makespan', 'singlecore_makespan', 'speedup',
    'added_copy_bytes', 'partition_added_copy_bytes', 'spill_added_copy_bytes',
    'scheduled_copy_bytes', 'original_graph_copy_bytes', 'cross_task_traffic',
    'hit_rate', 'hit_bytes', 'miss_bytes', 'cache_hits', 'cache_accesses',
    'l1_peak_max', 'ub_peak_max', 'num_subgraphs', 'nonempty_cores',
    'error_type', 'error_message', 'error_cycle',
    # 过程
    'solve_seconds', 'eval_seconds', 'solver_error', 'plan_path', 'plan_hash',
    'eval_version', 'timestamp',
]


def make_row(**values):
    row = {f: values.get(f) for f in ROW_FIELDS}
    row['timestamp'] = row['timestamp'] or time.strftime('%Y-%m-%d %H:%M:%S')
    return row


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding='utf-8') as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def sort_key(row):
    """候选比较：先 makespan，再总额外搬运；不可行永远最差。"""
    if not row.get('ok') or row.get('makespan') is None:
        return (1, float('inf'), float('inf'))
    return (0, row['makespan'], row.get('added_copy_bytes') or 0)


def better(a, b):
    """a 是否严格优于 b（b 可为 None）。"""
    return b is None or sort_key(a) < sort_key(b)
