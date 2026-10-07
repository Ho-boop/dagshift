"""评估适配层：搜索与批量实验一律通过这里调用评估器。

- 内部调用 fast_eval（与官方评估函数逐位一致，见 fast_eval/README.md），固定读取
  data/config.txt，不接受改参数。论文最终数字由 experiments/official_rerun.py 用官方 CLI 复跑。
- 返回扁平的摘要 dict（不含逐 op 时间线）；keep_full=True 时附带完整结果 'full'。
- 评估缓存：进程内 + 磁盘（results/.evalcache）。键 = 图哈希 + 方案哈希(含子图顺序)
  + 问题号 + 配置哈希 + fast_eval 源码哈希，任何一项变化都会自动失效。
- 不可行方案不会被当成低耗时结果：ok=False，makespan=None，并记录错误类型、消息和环。
"""

import hashlib
import json
import os
import time

from .paths import BASELINE, CONFIG_PATH, EVAL_CACHE, ROOT
from .plan import PLAN_FIELDS, plan_hash, plan_stats

# Load a user-provided evaluator outside this repository. It is never redistributed.
from dagshift.external import load_evaluator
fe, evaluator_package = load_evaluator()
if not CONFIG_PATH.is_file():
    raise RuntimeError('Provide DAGSHIFT_CONFIG or DAGSHIFT_OFFICIAL_PACKAGE with data/config.txt; see README.md.')

CONFIG = fe.load_config(CONFIG_PATH)
CONFIG_HASH = hashlib.sha1(CONFIG_PATH.read_bytes()).hexdigest()[:12]
# 只对评估器本身取哈希；verify*.py、__main__.py 等测试/命令行文件变化不应让缓存失效
EVAL_VERSION = hashlib.sha1(b''.join(
    p.read_bytes() for p in sorted(evaluator_package.glob('*.py'))
    if not p.name.startswith('verify') and p.name != '__main__.py')).hexdigest()[:12]

fe.set_quiet(True)

_MEMO = {}
_MEMO_LIMIT = 50000


def _cacheable(plan):
    """plan_hash 只序列化两个字段、且把 tuple 写成 JSON 数组：带多余字段或 tuple 的方案与合法
    方案同哈希，但评估器会拒绝它们。这类方案不读也不写缓存，直接交给评估器。"""
    return (type(plan) is dict and set(plan) == set(PLAN_FIELDS)
            and type(plan['node_to_subgraph']) is dict
            and type(plan['core_schedules']) is list
            and all(type(order) is list for order in plan['core_schedules']))


def _cache_key(graph, phash, problem):
    text = f'{graph.graph_hash}|{phash}|{problem}|{CONFIG_HASH}|{EVAL_VERSION}'
    return hashlib.sha1(text.encode()).hexdigest()


def _cache_path(key, problem):
    return EVAL_CACHE / f'p{problem}' / key[:2] / f'{key}.json'


def _read_disk(key, problem):
    path = _cache_path(key, problem)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _write_disk(key, problem, summary):
    # 缓存写失败不影响评估结果：Windows 上别的进程正读同一文件时 os.replace 会拒绝访问
    path = _cache_path(key, problem)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(summary, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def _summarize(result, problem):
    moved = result.get('data_movement_bytes', {})
    peaks = result.get('memory_peak_by_core', {}) or {}
    cache = result.get('cache_stats') or {}
    return {
        'ok': True,
        'makespan': result['makespan'],
        'added_copy_bytes': moved.get('added_copy_bytes'),
        'partition_added_copy_bytes': moved.get('partition_added_copy_bytes'),
        'spill_added_copy_bytes': moved.get('spill_added_copy_bytes'),
        'scheduled_copy_bytes': moved.get('scheduled_copy_bytes'),
        'original_graph_copy_bytes': moved.get('original_graph_copy_bytes'),
        'cross_task_traffic': result.get('cross_task_traffic'),
        'hit_rate': cache.get('hit_rate') if problem == 3 else None,
        'hit_bytes': cache.get('hit_bytes') if problem == 3 else None,
        'miss_bytes': cache.get('miss_bytes') if problem == 3 else None,
        'cache_hits': cache.get('hits') if problem == 3 else None,
        'cache_accesses': cache.get('accesses') if problem == 3 else None,
        'l1_peak_max': max((p.get('L1', 0) for p in peaks.values()), default=None),
        'ub_peak_max': max((p.get('UB', 0) for p in peaks.values()), default=None),
        'error_type': None, 'error_message': None, 'error_cycle': None,
    }


def evaluate(graph, plan, problem, *, cache=True, keep_full=False):
    """评估一个方案。graph 为 common.Graph，problem 为 1/2/3。

    返回摘要 dict：ok, makespan, added_copy_bytes, partition/spill 分项, hit_rate(仅问题3),
    l1/ub 峰值, 错误信息, eval_seconds, cached 等。
    """
    if problem not in (1, 2, 3):
        raise ValueError('problem must be 1, 2 or 3')
    try:
        phash = plan_hash(plan)
    except Exception:   # 缺字段、不是 dict 等畸形方案：不进缓存，照常交给评估器报错
        phash = None
    cache = cache and phash is not None and _cacheable(plan)
    key = _cache_key(graph, phash, problem) if cache else None
    if cache and not keep_full:
        hit = _MEMO.get(key) or _read_disk(key, problem)
        if hit is not None:
            _MEMO[key] = hit
            return dict(hit, cached=True)

    start = time.perf_counter()
    full = None
    try:
        full = fe.evaluate(graph.raw, plan, problem, CONFIG)
        summary = _summarize(full, problem)
        deterministic = True
    except Exception as error:  # 评估器拒绝方案：记录原因，绝不当作可行结果
        cycle = getattr(error, 'cycle', None)
        summary = {'ok': False, 'makespan': None,
                   'error_type': type(error).__name__,
                   'error_message': str(error)[:500],
                   'error_cycle': repr(cycle[:16])[:2000] if cycle else None}
        deterministic = isinstance(error, (ValueError, RuntimeError))
    try:
        stats = {k: v for k, v in plan_stats(plan).items() if k != 'subgraphs_per_core'}
    except Exception:   # 畸形方案只记录评估器的错误，统计本身不能再抛异常
        stats = {}
    summary.update({'case': graph.name, 'problem': problem, 'plan_hash': phash,
                    'eval_seconds': round(time.perf_counter() - start, 3),
                    **stats, 'eval_version': EVAL_VERSION})
    if cache and deterministic:
        if len(_MEMO) > _MEMO_LIMIT:
            _MEMO.clear()
        _MEMO[key] = summary
        _write_disk(key, problem, summary)
    out = dict(summary, cached=False)
    if keep_full and full is not None:
        out['full'] = full
    return out


# ---------------------------------------------------------------- 单核基准
_BASELINE = None


def _baseline_table():
    global _BASELINE
    if _BASELINE is None:
        path = BASELINE / 'singlecore.json'
        _BASELINE = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return _BASELINE


def singlecore_makespan(graph):
    """官方单核基准（整图一个 Task、无 L2）。优先读 results/baseline/singlecore.json。"""
    row = _baseline_table().get(graph.name)
    if row is not None and row.get('graph_hash') == graph.graph_hash:
        return row['makespan']
    key = _cache_key(graph, 'singlecore', 'single')
    hit = _MEMO.get(key) or _read_disk(key, 'single')
    if hit is None:
        start = time.perf_counter()
        result = fe.singlecore(graph.raw, CONFIG)
        hit = {'makespan': result['makespan'],
               'eval_seconds': round(time.perf_counter() - start, 3)}
        _write_disk(key, 'single', hit)
    _MEMO[key] = hit
    return hit['makespan']


def speedup(graph, makespan):
    return singlecore_makespan(graph) / makespan if makespan else None
