"""命令行入口，用法见 solvers/deliver/README.md。

单个计算图：
    python -m solvers.deliver --problem 2 --graph <计算图.json> --k 3 --out <目录>
    python -m solvers.deliver --problem 2 --graph <计算图.json> --k all --out <目录>     # k=1..5
批量（data 目录全部算例，每个 (算例, 问题) 一个新进程，k=1..5）：
    python -m solvers.deliver --batch --problems 1,2,3 --out <目录> [--data-dir <目录>] [--workers 12]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from solvers import deliver as D

ROOT = Path(__file__).resolve().parents[2]


def _common_args(p):
    p.add_argument('--budget', type=float, default=None,
                   help='一个算例 k=2..5 合计的秒数（virtual 模式即虚拟预算 V）；缺省取环境变量 VCLOCK_CASE_BUDGET，再缺省 120；只请求部分 k 时按份额缩放')
    p.add_argument('--k1-budget', type=float, default=None, help='k=1 的秒数（缺省 budget/4），不占 k=2..5 预算')
    p.add_argument('--seed', type=int, default=D.DEFAULT_SEED)
    p.add_argument('--clock', choices=('wall', 'virtual'), default=D.DEFAULT_CLOCK,
                   help='virtual（缺省）：SD 虚拟时钟（solvers/vclock.py），跨机器同输入同方案；wall：按真实墙钟截止')
    p.add_argument('--cap-factor', type=float, default=None,
                   help='virtual 模式每个 k 的真实上限 = cap_factor × 虚拟预算；缺省取环境变量 VCLOCK_CAP_FACTOR，再缺省 6')
    p.add_argument('--config', default=None, help='可选：与固定 config 做一致性检查，不同则拒绝')
    p.add_argument('--overwrite', action='store_true', help='允许覆盖已存在的方案文件（缺省拒绝）')
    p.add_argument('--append', default=None,
                   help='可选：完整 v2 之后追加的候选（逗号列表，如 deep；见 solvers/deliver/append.py），'
                        '逐 k 字典序取优、不劣于 v2；缺省不追加，与 deliver-v2 逐字节相同')
    p.add_argument('--append-max-evals', type=int, default=None,
                   help='追加候选每个 k 的精确评估上限（缺省 append.DEFAULT_MAX_EVALS=2）')


def _append_names(spec):
    """缺省（不追加）时不导入 append 模块，缺省运行导入的源码与 deliver-v2 相同。"""
    if spec in (None, '', 'none'):
        return ()
    from solvers.deliver import append as A
    return A.parse(spec)


def _append_cmd(args, problem=None):
    cmd = []
    if problem == 1:              # 追加只对问题2/3；批量里的问题1 照常按 v2
        return cmd
    if args.append:
        cmd += ['--append', args.append]
    if args.append_max_evals is not None:
        cmd += ['--append-max-evals', str(args.append_max_evals)]
    return cmd


def _cases(data_dir):
    return sorted(p for p in Path(data_dir).glob('*.json') if not p.name.endswith('_res.json'))


def _job(args, graph, problem, ks):
    tag = f'q{problem}_{graph.stem}'
    logs = Path(args.out) / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'solvers.deliver', '--problem', str(problem), '--graph', str(graph),
           '--k', ','.join(map(str, ks)), '--out', str(args.out), '--layout', 'tree',
           '--budget', repr(args.budget), '--seed', str(args.seed), '--clock', args.clock,
           '--cap-factor', repr(args.cap_factor)]
    if args.k1_budget is not None:
        cmd += ['--k1-budget', repr(args.k1_budget)]
    if args.config:
        cmd += ['--config', args.config]
    if args.overwrite:
        cmd += ['--overwrite']
    cmd += _append_cmd(args, problem)
    t = time.perf_counter()
    with open(logs / f'{tag}.log', 'w', encoding='utf-8') as fh:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONUTF8='1'))
    wall = time.perf_counter() - t
    print(f'{tag} rc={rc} {wall:.1f}s', flush=True)
    return {'case': graph.stem, 'problem': problem, 'rc': rc, 'wall_s': round(wall, 3)}


def _code_check(out):
    """所有 job 的 meta 里记录的源码哈希是否一致（同一文件在不同 job 中哈希不同即视为运行中改了代码）。"""
    seen, n = {}, 0
    for p in sorted((Path(out) / '_meta').glob('q*_*.json')):
        try:
            files = json.loads(p.read_text(encoding='utf-8')).get('code_files') or {}
        except (OSError, ValueError):
            continue
        n += 1
        for f, h in files.items():
            seen.setdefault(f, set()).add(h)
    diff = {f: sorted(hs) for f, hs in seen.items() if len(hs) > 1}
    return {'metas': n, 'files': len(seen), 'versions': 1 + bool(diff), 'differ': diff}


def batch(args):
    if args.workers < 1 or args.workers > 16:
        raise SystemExit('--workers must be within 1..16')
    ks = D.parse_ks(args.k)
    problems = [int(x) for x in args.problems.split(',')]
    cases = _cases(args.data_dir)
    out = Path(args.out)
    jobs, skipped = [], []
    for g in cases:
        for p in problems:
            targets = [out / f'q{p}' / f'k{k}' / D.output_name(g) for k in ks]
            if not args.overwrite and all(t.exists() for t in targets):
                skipped.append(f'q{p}_{g.stem}')
            elif not args.overwrite and any(t.exists() for t in targets):
                raise SystemExit(f'partial outputs exist for q{p} {g.stem}; remove them or pass --overwrite')
            else:
                jobs.append((g, p))
    print(f'{len(cases)} cases x problems {problems} x k={ks}: {len(jobs)} jobs, {len(skipped)} already done',
          flush=True)
    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda j: _job(args, j[0], j[1], ks), jobs))
    summary = {'deliver': D.VERSION, 'data_dir': str(Path(args.data_dir).resolve()), 'problems': problems,
               'ks': ks, 'budget': args.budget, 'k1_budget': args.k1_budget, 'seed': args.seed,
               'clock': args.clock, 'cap_factor': args.cap_factor,
               **({'append': args.append, 'append_max_evals': args.append_max_evals} if args.append else {}),
               'workers': args.workers, 'jobs': res, 'skipped': skipped,
               'failed': [r for r in res if r['rc'] != 0], 'wall_s': round(time.perf_counter() - t, 3)}
    summary['code_check'] = _code_check(out)
    if summary['code_check']['versions'] > 1:
        summary['failed'].append({'code_check': 'source files differ between jobs'})
    path = out / '_meta' / f'batch_{time.strftime("%m%d-%H%M%S")}_{os.getpid()}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'wrote {path}; failed {len(summary["failed"])}', flush=True)
    return 1 if summary['failed'] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m solvers.deliver', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', action='store_true', help='批量模式：data 目录全部算例')
    ap.add_argument('--problem', type=int, choices=(1, 2, 3))
    ap.add_argument('--problems', default='1,2,3', help='批量模式的问题列表')
    ap.add_argument('--graph', help='计算图 JSON 路径（单图模式）')
    ap.add_argument('--data-dir', default=None, help='批量模式的计算图目录（缺省官方附件 data/）')
    ap.add_argument('--k', default=None, help='核数：1..5 之一、逗号列表或 all（单图缺省 all）')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--layout', choices=('flat', 'tree'), default='flat',
                    help='flat：<out>/<case>_multicore_res.json（仅单个 k）；tree：<out>/q<P>/k<k>/…（多个 k 自动用 tree）')
    ap.add_argument('--workers', type=int, default=4, help='批量模式并发进程数（≤16）')
    ap.add_argument('--meta-path', default=None, help=argparse.SUPPRESS)      # 内部用：k=1 子进程的 meta 位置
    _common_args(ap)
    args = ap.parse_args(argv)
    # 显式参数 > 环境变量 > 缺省；解析后的值写进 meta，并原样传给子进程
    if args.budget is None:
        args.budget = float(os.environ.get('VCLOCK_CASE_BUDGET') or D.DEFAULT_BUDGET)
    if args.cap_factor is None:
        args.cap_factor = float(os.environ.get('VCLOCK_CAP_FACTOR') or D.DEFAULT_CAP_FACTOR)

    if args.batch:
        if args.data_dir is None:
            from common import paths
            args.data_dir = str(paths.DATA)
        return batch(args)
    if args.problem is None or args.graph is None:
        ap.error('single-graph mode needs --problem and --graph (or use --batch)')
    ks = D.parse_ks(args.k)
    rc = 0
    if 1 in ks and len(ks) > 1:
        # k=1 放到一个新进程：vclock/multistart 的进程内状态不带过去，--k all 与单独 --k 1 的 k=1 方案相同
        graph = Path(args.graph)
        targets = [Path(args.out) / f'q{args.problem}' / f'k{k}' / D.output_name(graph) for k in ks]
        exist = [str(t) for t in targets if t.exists()]
        if exist and not args.overwrite:
            raise SystemExit('refusing to overwrite existing plan(s): ' + ', '.join(exist))
        cmd = [sys.executable, '-m', 'solvers.deliver', '--problem', str(args.problem), '--graph', str(graph),
               '--k', '1', '--out', str(args.out), '--layout', 'tree',
               '--meta-path', str(Path(args.out) / '_meta' / f'q{args.problem}_{graph.stem}_k1.json'),
               '--budget', repr(args.budget), '--seed', str(args.seed), '--clock', args.clock,
               '--cap-factor', repr(args.cap_factor)]
        if args.k1_budget is not None:
            cmd += ['--k1-budget', repr(args.k1_budget)]
        if args.config:
            cmd += ['--config', args.config]
        if args.overwrite:
            cmd += ['--overwrite']
        cmd += _append_cmd(args, args.problem)      # k=1 不追加（append.solve 在 k<2 直接返回 v2），只为 meta 一致
        rc = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, PYTHONUTF8='1'))
        ks = [k for k in ks if k != 1]
        args.layout = 'tree'
    meta = D.solve_graph(args.graph, args.problem, ks, args.out, budget=args.budget, k1_budget=args.k1_budget,
                         seed=args.seed, clock=args.clock, cap_factor=args.cap_factor, layout=args.layout,
                         overwrite=args.overwrite, config=args.config, meta_path=args.meta_path,
                         append=_append_names(args.append), append_max_evals=args.append_max_evals)
    for r in meta['rows']:
        extra = f" vk={r['vk']} virtual={r['virtual_s']} capped={r['capped']}" if 'capped' in r else ''
        print(f"k={r['k']} wall={r['wall_s']}s budget_s={r['budget_s']}s valid={r['validate_ok']} "
              f"fallback={r['fallback']}{extra} -> {r['plan']}", flush=True)
    print(f"meta: {meta['meta_path']}  total {meta['wall_s']}s", flush=True)
    bad = [r for r in meta['rows'] if r['fallback'] or not r['validate_ok']]
    return 2 if bad else rc


if __name__ == '__main__':
    sys.exit(main())
