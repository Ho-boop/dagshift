"""可复现的提交入口：输入计算图 JSON，按原题附录 B 写出 <case>_multicore_res.json。

    python -m solvers.deliver --problem 2 --graph <计算图.json> --k all --out <目录>
    python -m solvers.deliver --batch --problems 1,2,3 --out <目录>        # data 目录全部算例，k=1..5

只凭输入文件和固定 config 工作：
- 进程启动后先把评估缓存、最优档案、单核基准三处路径指到本次运行的临时空目录（与
  experiments.run_case_budget._isolate_worker 相同），求解器读不到 results/ 下任何历史结果；
- 不按用例编号分支；用例名只用于输出文件名（= 输入文件名去掉 .json）。
- config：求解器内部评估用官方附件 data/config.txt（common.evaluate 启动时读入）。--config 只做一致性检查，
  内容与之不同就拒绝运行，不会拿别的参数求解。

求解器（S3 09-26 16:12 冻结，deliver-v2）：
- 问题1：solvers.vclock.solve_q1（vclock-v2，内部 q1.main.solve，warm=False，带 k−1 保底）
- 问题2/3：solvers.q2.dev.order_postpass.solve_multi（内部 solvers.vclock.solve → multistart，带 k−1 保底；
  之后按固定顺序试至多 3 个固定分核的核内顺序候选 timed_g / timed / memory，按评估器字典序择优）
- 缺省 V=120（--budget，k=2..5 合计虚拟预算）、cap factor 6、seed 0；k=1 预算缺省 V/4
- k=2,3,4,5 在同一进程内升序求解（multistart 用 k-1 的方案作 k 的起点，这是本次运行内的合法起点）；
  k=1 单独计预算，不占 k=2..5 的案例预算，并由 __main__ 放在另一个新进程里求解（vclock 的单核基准计费
  _SINGLE_DONE 与 multistart._PREV 都是进程内状态，同进程先跑 k=2..5 会改变 k=1 的虚拟用时）。

可选追加（--append，缺省关闭；关闭时不导入 append 模块，方案与 deliver-v2 逐字节相同）：
- solvers/deliver/append.py：每个 k 先原样跑完整 v2，再精确评估至多 --append-max-evals（缺省 2）个登记的
  构造候选，按字典序在 {v2 结果, 候选} 中取优，相等取 v2；追加成本另记，不挤占 v2 与后续 k 的预算。

时钟（--clock）：
- virtual（缺省）：solvers.vclock（接口见 solvers/VCLOCK.md §6）。求解器内部看到的时间由评估次数、规模等
  确定性计费得出；VCLOCK_CASE_BUDGET 取本段案例预算，VCLOCK_CAP_FACTOR 取 --cap-factor（缺省 6）。
  同输入、同种子、同预算与 cap factor、同代码版本给出逐字节相同的方案；每个 k 的真实墙钟上限为
  cap_factor × 该 k 虚拟预算，触发时 meta 里 capped=True，该 k 不再保证可复现。
- wall：各 k 预算 = (案例剩余墙钟 - 预留) / 剩余核数，与 run_case_budget 相同。结果随机器快慢变化。
"""

import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

VERSION = 'deliver-v2'
KS_CHAIN = (2, 3, 4, 5)
DEFAULT_BUDGET = 120.0      # 一个 case 的 k=2..5 合计；virtual 模式即虚拟预算 V（S3 冻结 V=120；项目墙钟口径 600 s）
DEFAULT_SEED = 0
DEFAULT_CLOCK = 'virtual'
DEFAULT_CAP_FACTOR = 6.0    # virtual 模式每个 k 的真实上限 = CAP_FACTOR × 该 k 虚拟预算（VCLOCK.md §5.3 推荐 5～8）


def output_name(graph_path):
    """附录 B.5：<case>_multicore_res.json，<case> 为输入计算图文件名去掉 .json。"""
    return f'{Path(graph_path).stem}_multicore_res.json'


def _reserve(case_budget):
    return min(2.0, max(0.1, case_budget * 0.01))


def isolate(state_dir):
    """把评估缓存、最优档案、单核基准指到空的私有目录；必须在任何求解之前调用。"""
    from common import paths
    ev = importlib.import_module('common.evaluate')
    state = Path(state_dir)
    paths.BEST = state / 'empty_best'
    paths.EVAL_CACHE = state / 'evalcache'
    paths.BASELINE = state / 'empty_baseline'
    ev.EVAL_CACHE = paths.EVAL_CACHE
    ev.BASELINE = paths.BASELINE
    ev._MEMO.clear()
    ev._BASELINE = None
    return state


def check_config(config_path):
    """--config 与求解器内部所用的固定 config 内容一致才放行。"""
    if config_path is None:
        return None
    import fast_eval as fe
    ev = importlib.import_module('common.evaluate')
    if fe.load_config(Path(config_path)) != ev.CONFIG:
        raise SystemExit(f'config {config_path} differs from the fixed config '
                         f'{ev.CONFIG_PATH}; the solvers only support the fixed one')
    return hashlib.sha1(Path(config_path).read_bytes()).hexdigest()[:12]


def _solvers(problem, clock, append=(), append_max_evals=None):
    """(调用方式, 版本信息, 取本次 meta)。调用方式签名 f(graph, k, seed, budget_s) -> plan；
    取 meta 的函数返回刚结束那次求解的 dict（virtual 模式），wall 模式返回 None。
    append 非空时（可选模式，见 solvers/deliver/append.py）：问题2/3、virtual 时钟下在完整 v2 之后追加候选；
    缺省 () 时与 deliver-v2 完全相同。"""
    versions = {'deliver': VERSION}
    no_meta = lambda: None
    if append:
        if problem == 1 or clock != 'virtual':
            raise SystemExit('--append only supports problems 2/3 with --clock virtual')
        from solvers.deliver import append as A
        _, versions, _ = _solvers(problem, clock)          # 只取版本信息；v2 本体由 A.solve 原样调用
        versions.update(A.versions(append))
        cap = A.DEFAULT_MAX_EVALS if append_max_evals is None else int(append_max_evals)
        box = {}

        def run(g, k, seed, b):
            box.clear()
            plan, box['meta'] = A.solve(g, k, problem, seed, b, append, cap)
            return plan
        return run, versions, lambda: dict(box.get('meta') or {})
    if problem == 1:
        from solvers.q1 import main as q1m
        versions['q1.main'] = q1m.VERSION
        # k=1 只有一个核，换核类局部搜索无意义（q1.main 的 group_ils 在 k=1 会从空列表里选核而抛异常），
        # 关掉组换核与 Task 换核；其余流程（基线、规格扫描、参数邻域）不变。
        k1 = lambda k: {'group_ls': None, 'task_ls': False} if k < 2 else {}
        if clock == 'virtual':
            from solvers import vclock
            versions['vclock'] = vclock.VERSION
            return ((lambda g, k, seed, b: vclock.solve_q1(g, k, 1, seed=seed, budget_s=b, warm=False,
                                                            **k1(k))),
                    versions, lambda: dict(vclock.LOG[-1]))
        return ((lambda g, k, seed, b: q1m.solve(g, k, 1, seed=seed, budget_s=b, warm=False, **k1(k))),
                versions, no_meta)
    from solvers.q2.dev import multistart
    from solvers.q2 import main as q2m
    from solvers.q3 import main as q3m
    from solvers.q1 import main as q1m
    versions.update({'multistart': multistart.VERSION, 'q2.main': q2m.VERSION,
                     'q3.main': q3m.VERSION, 'q1.main': q1m.VERSION})
    if clock == 'virtual':
        from solvers import vclock
        from solvers.q2.dev import order_postpass
        versions['vclock'] = vclock.VERSION
        versions['order_postpass'] = order_postpass.VERSION_MULTI
        # order_postpass.LAST_META = vclock 的本次 meta（含 guard、vk、cap_s）+ 核内重排的 order_tried、
        # evals、virtual_s（含重排的计费）、capped（含重排的墙钟保护）
        return ((lambda g, k, seed, b: order_postpass.solve_multi(g, k, problem, seed=seed, budget_s=b)),
                versions, lambda: dict(order_postpass.LAST_META or {}))
    # wall 模式只作对照，不含核内重排
    return (lambda g, k, seed, b: multistart.solve(g, k, problem, seed=seed, budget_s=b)), versions, no_meta


def code_files(root=None):
    """本进程实际导入的、位于 A题 目录下的源码文件及其 sha256（冻结核对用）。"""
    root = Path(root or Path(__file__).resolve().parents[2])
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if p.suffix == '.py' and p.exists():
            out[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return dict(sorted(out.items()))


def _fallback(graph, k, problem):
    """求解器抛异常时的保底方案（计算分量装箱，solvers/baseline.py）；meta 里记 fallback=True。"""
    from solvers import baseline
    return baseline.solve(graph, k, problem)


def parse_ks(spec):
    if spec in (None, '', 'all'):
        return [1, *KS_CHAIN]
    ks = sorted({int(x) for x in str(spec).split(',') if x.strip()})
    if not ks or any(k < 1 or k > 5 for k in ks):
        raise SystemExit(f'--k must be within 1..5 or "all", got {spec!r}')
    return ks


def solve_graph(graph_path, problem, ks, out_dir, *, budget=DEFAULT_BUDGET, k1_budget=None,
                seed=DEFAULT_SEED, clock=DEFAULT_CLOCK, cap_factor=DEFAULT_CAP_FACTOR, layout='flat',
                overwrite=False, config=None, meta_path=None, append=(), append_max_evals=None):
    """一个 (计算图, 问题) 的全部 k。返回 meta dict；方案逐 k 写出。

    layout='flat'：<out>/<case>_multicore_res.json（单个 k 时用；多个 k 会互相覆盖，故多个 k 强制 tree）
    layout='tree'：<out>/q<P>/k<k>/<case>_multicore_res.json
    """
    t0 = time.perf_counter()
    graph_path = Path(graph_path).resolve()
    out_dir = Path(out_dir)
    ks = list(ks)
    if len(ks) > 1:
        layout = 'tree'
    chain = [k for k in KS_CHAIN if k in ks]
    rest = [k for k in ks if k not in KS_CHAIN]          # 只有 k=1
    targets = {k: (out_dir / output_name(graph_path) if layout == 'flat'
                   else out_dir / f'q{problem}' / f'k{k}' / output_name(graph_path)) for k in ks}
    if not overwrite:
        exist = [str(p) for p in targets.values() if p.exists()]
        if exist:
            raise SystemExit('refusing to overwrite existing plan(s): ' + ', '.join(exist))

    state = Path(tempfile.mkdtemp(prefix='deliver_'))
    try:
        isolate(state)
        config_hash = check_config(config)
        from common import load_graph, save_plan, validate_plan
        from common.plan import plan_hash
        append = tuple(append or ())
        solve, versions, last_meta = _solvers(problem, clock, append, append_max_evals)
        graph = load_graph(str(graph_path))
        k1_budget = budget / len(KS_CHAIN) if k1_budget is None else float(k1_budget)
        if clock == 'virtual':
            from solvers import vclock
            share = lambda k: vclock._remaining_ks(k)           # vclock 按“k 及其后各 k”均分虚拟预算
        else:
            share = None
        rows = []
        # 两段各自是一个“案例”：先 k=2..5（共享案例预算），再 k=1（k1_budget）。
        # 只请求部分 k 时，案例预算按份额缩放：wall 按请求的 k 数，virtual 按 vclock 的均分口径（首个 k 及其后各 k）。
        segments = []
        if chain:
            n_share = len(chain) if clock == 'wall' else share(chain[0])
            segments.append((chain, budget * n_share / len(KS_CHAIN), t0))
        if rest:
            segments.append((rest, k1_budget, None))
        det_meta = None
        if clock == 'virtual':
            os.environ['VCLOCK_CAP_FACTOR'] = repr(float(cap_factor))
        for seg_ks, seg_budget, seg_start in segments:
            seg_start = seg_start if seg_start is not None else time.perf_counter()
            deadline = seg_start + seg_budget
            if clock == 'virtual':
                os.environ['VCLOCK_CASE_BUDGET'] = repr(float(seg_budget))
                det_meta = vclock.DETERMINISM_META()
            for i, k in enumerate(seg_ks):
                if clock == 'wall':
                    bk = max(1.0, (deadline - time.perf_counter() - _reserve(seg_budget)) / (len(seg_ks) - i))
                else:
                    # 设了 VCLOCK_CAP_FACTOR 后 vclock 不看 budget_s（虚拟预算来自 VCLOCK_CASE_BUDGET，
                    # 真实上限 = cap_factor × vk）；这里只传名义值，不依赖真实时间，保证可复现
                    bk = seg_budget / share(k)
                t = time.perf_counter()
                row = {'k': k, 'budget_s': round(bk, 3), 'fallback': False}
                try:
                    plan = solve(graph, k, seed, bk)
                except Exception:
                    row['fallback'] = True
                    row['error'] = traceback.format_exc()[-2000:]
                    plan = _fallback(graph, k, problem)
                row['wall_s'] = round(time.perf_counter() - t, 3)
                if clock == 'virtual' and not row['fallback']:
                    log = last_meta() or {}
                    row.update(virtual_s=log.get('virtual_s'), vk=log.get('vk'),
                               cap_s=log.get('cap_s'), case_V=log.get('case_V'),
                               capped=bool(log.get('capped')), evals=log.get('evals'),
                               guard=log.get('guard'))
                    if 'order_tried' in log:
                        row['order_tried'] = log['order_tried']
                        row['order_guard'] = bool(log.get('order_guard'))
                    if 'append_choice' in log:
                        for key in ('append_choice', 'append_evals', 'append_virtual_s', 'append_real_s',
                                    'append_tried', 'append_base', 'append_final'):
                            if key in log:
                                row[key] = log[key]
                rep = validate_plan(graph, plan, problem=problem, num_cores=k)
                row['validate_ok'] = bool(rep.ok)
                if not rep.ok:
                    row['validate_msg'] = str(rep)[:500]
                path = save_plan(plan, targets[k])
                row['plan'] = str(path)
                row['plan_sha1'] = plan_hash(plan)
                row['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(row)
        wall = time.perf_counter() - t0
        code = code_files()
    finally:
        shutil.rmtree(state, ignore_errors=True)

    meta = {
        'deliver': VERSION, 'versions': versions, 'problem': problem, 'ks': ks,
        'graph': str(graph_path), 'graph_sha1': graph.graph_hash, 'case': graph_path.stem,
        'seed': seed, 'clock': clock, 'budget_k2to5_s': budget, 'k1_budget_s': k1_budget,
        'cap_factor': float(cap_factor) if clock == 'virtual' else None,
        'vclock_determinism': det_meta,          # 最后一段生效的常数与 VCLOCK_CASE_BUDGET / VCLOCK_CAP_FACTOR
        'config_checked': config_hash, 'rows': rows, 'wall_s': round(wall, 3),
        'code_files': code,                      # 实际导入的源码 sha256[:16]；批量模式据此核对全程代码未变
        'machine': platform.platform(), 'cpu_count': os.cpu_count(),
        'python': sys.version.split()[0], 'pid': os.getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - wall)),
    }
    if append:
        from solvers.deliver import append as A
        meta['append'] = A.determinism(append, A.DEFAULT_MAX_EVALS if append_max_evals is None
                                       else int(append_max_evals))
    if meta_path is None:
        meta_path = (out_dir / f'{graph_path.stem}_deliver_meta.json' if layout == 'flat'
                     else out_dir / '_meta' / f'q{problem}_{graph_path.stem}.json')
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists() and not overwrite:
        meta_path = meta_path.with_name(meta_path.stem + f'.{os.getpid()}.json')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    meta['meta_path'] = str(meta_path)
    return meta
