"""公共接口（S3 总控维护，其他 session 只读）。

    from common import load_graph, evaluate, validate_plan, make_plan, save_plan
    g = load_graph('case_001')
    plan = make_plan({op: sg, ...}, [[sg, ...], ...])
    rep = validate_plan(g, plan, problem=2, num_cores=4)   # 廉价预筛
    r = evaluate(g, plan, problem=2)                       # 官方等价评估（带缓存）
    r['ok'], r['makespan'], r['added_copy_bytes']
"""

import sys

from .paths import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .graph import Graph, load_graph  # noqa: E402
from .plan import (best_plan, canonical_json, from_groups, load_plan,  # noqa: E402
                   make_plan, op_to_core, plan_hash, plan_stats, save_plan)
from .validate import ValidationReport, subgraph_dependencies, validate_plan  # noqa: E402
import os

if os.environ.get('DAGSHIFT_EVALUATOR_ROOT') or os.environ.get('DAGSHIFT_OFFICIAL_PACKAGE'):
    from .evaluate import CONFIG, EVAL_VERSION, evaluate, singlecore_makespan, speedup
else:
    def _external_required(*args, **kwargs):
        raise RuntimeError('This solver needs a user-provided evaluator. Configure DAGSHIFT_EVALUATOR_ROOT; the dependency-free demo is python -m dagshift.demo.')
    evaluate = singlecore_makespan = speedup = _external_required
    CONFIG = None
    EVAL_VERSION = 'external-not-configured'

from .records import ROW_FIELDS, better, make_row, sort_key  # noqa: E402
from . import paths  # noqa: E402

__all__ = [
    'Graph', 'load_graph', 'make_plan', 'from_groups', 'save_plan', 'load_plan',
    'plan_hash', 'plan_stats', 'op_to_core', 'canonical_json', 'best_plan',
    'validate_plan', 'ValidationReport', 'subgraph_dependencies',
    'evaluate', 'singlecore_makespan', 'speedup', 'CONFIG', 'EVAL_VERSION',
    'ROW_FIELDS', 'make_row', 'better', 'sort_key', 'paths',
]
