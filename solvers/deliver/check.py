"""核对 deliver 输出：对 <out> 下 _meta 记录的每个方案做三件事并逐行对照。

    python -m solvers.deliver.check <out 目录> [--no-official] [--workers 4]

1. common.validate_plan（格式、覆盖、子图 DAG、核内顺序预筛）；
2. common.evaluate(cache=False)（fast_eval，不读写磁盘评估缓存）；
3. 官方评估 CLI：code/multicore_cut_evaluate_problem_{P}.py <graph> <plan> --config data/config.txt，
   输出（结果、trace、log）全部写到 <out>/_check/ 下，不改、不在官方目录留文件。
makespan、added_copy_bytes 精确相等、问题3 hit_rate 相对误差 ≤1e-9 记 match。
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from solvers import deliver as D

SCRIPT = {1: 'multicore_cut_evaluate_problem_1.py', 2: 'multicore_cut_evaluate_problem_2.py',
          3: 'multicore_cut_evaluate_problem_3.py'}


def _plans(out):
    for meta_path in sorted(Path(out).rglob('*.json')):
        if '_meta' not in meta_path.parts and not meta_path.name.endswith('_deliver_meta.json'):
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if 'rows' not in meta:
            continue
        for r in meta['rows']:
            yield meta['graph'], meta['problem'], r['k'], r['plan']


def official(graph, problem, plan, workdir):
    from common import paths
    workdir.mkdir(parents=True, exist_ok=True)
    res = workdir / 'res.json'
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    cmd = [sys.executable, str(paths.OFFICIAL_CODE / SCRIPT[problem]), str(graph), str(plan),
           '--config', str(paths.CONFIG_PATH), '-o', str(res),
           '--trace-output', str(workdir / 'trace.json'), '--log-output', str(workdir / 'log.txt')]
    t = time.perf_counter()
    proc = subprocess.run(cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL, capture_output=True)
    out = {'rc': proc.returncode, 'seconds': round(time.perf_counter() - t, 2),
           'stderr': proc.stderr.decode('utf-8', 'replace')[-300:]}
    if proc.returncode == 0 and res.exists():
        data = json.loads(res.read_bytes())
        out['makespan'] = data['makespan']
        out['added_copy_bytes'] = (data.get('data_movement_bytes') or {}).get('added_copy_bytes')
        out['hit_rate'] = (data.get('cache_stats') or {}).get('hit_rate')
    for f in ('trace.json', 'log.txt'):          # trace 很大，核对完即删，只留结果 JSON
        (workdir / f).unlink(missing_ok=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m solvers.deliver.check')
    ap.add_argument('out')
    ap.add_argument('--no-official', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args(argv)
    if args.workers < 1 or args.workers > 6:
        raise SystemExit('--workers must be within 1..6')
    D.isolate(tempfile.mkdtemp(prefix='deliver_check_'))
    from common import load_graph, validate_plan
    from common.plan import load_plan
    ev = importlib.import_module('common.evaluate')
    items = list(_plans(args.out))
    rows = []
    for graph_path, problem, k, plan_path in items:
        g = load_graph(graph_path)
        plan = load_plan(plan_path)
        fields_ok = list(json.loads(Path(plan_path).read_text(encoding='utf-8'))) == ['node_to_subgraph',
                                                                                       'core_schedules']
        rep = validate_plan(g, plan, problem=problem, num_cores=k)
        r = ev.evaluate(g, plan, problem, cache=False)
        rows.append({'case': g.name, 'problem': problem, 'k': k, 'plan': plan_path, 'graph': graph_path,
                     'two_fields': fields_ok, 'validate_ok': bool(rep.ok), 'eval_ok': bool(r['ok']),
                     'makespan': r['makespan'], 'added_copy_bytes': r['added_copy_bytes'],
                     'hit_rate': r.get('hit_rate')})
    if not args.no_official:
        base = Path(args.out) / '_check'

        def run(row):
            wd = base / f"q{row['problem']}_k{row['k']}_{row['case']}"
            return official(row['graph'], row['problem'], row['plan'], wd)
        with ThreadPoolExecutor(args.workers) as ex:
            for row, off in zip(rows, ex.map(run, rows)):
                row['official'] = off
                hr_ok = (row['problem'] != 3 or off.get('hit_rate') is None or row['hit_rate'] is None
                         or abs(off['hit_rate'] - row['hit_rate']) <= 1e-9 * max(1.0, abs(off['hit_rate'])))
                row['match'] = (off['rc'] == 0 and off.get('makespan') == row['makespan']
                                and off.get('added_copy_bytes') == row['added_copy_bytes'] and hr_ok)
    bad = 0
    for row in rows:
        ok = row['two_fields'] and row['validate_ok'] and row['eval_ok'] and row.get('match', True)
        bad += not ok
        off = row.get('official', {})
        print(f"q{row['problem']} k={row['k']} {row['case']}: fields={row['two_fields']} "
              f"validate={row['validate_ok']} eval={row['eval_ok']} makespan={row['makespan']} "
              f"copy={row['added_copy_bytes']} official={off.get('makespan', '-')} rc={off.get('rc', '-')} "
              f"match={row.get('match', '-')}", flush=True)
    path = Path(args.out) / '_check' / f'check_{time.strftime("%m%d-%H%M%S")}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(rows)} plans, {bad} problems; wrote {path}', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
