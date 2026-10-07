"""关键链：从问题2 评估器的完整时间线倒推“卡住 makespan 的那条链”，只用来挑搜索移动（不用于打分）。

倒推规则（PIPE_SLOTS = 1，每条 Pipe 串行；op 在所有前提满足的那一刻发射）：
从最晚结束的 op 出发，在它的开始时刻 s 找“恰好在 s 满足”的前提——
  跨核：对应 COPY_OUT 结束 + 跨核同步 = s（ext）；
  数据/内存复用依赖：前驱结束 = s（dep）；
  同 Pipe 前一个 op 结束 = s（pipe，资源受限）；
依次优先 ext、dep、pipe，沿着走到时刻 0。
返回链上的计算 op 序列（带核号与到达方式），以及跨核跳 (p@a → u@b) 的计算 op 对。
"""

from common import evaluate
from common.evaluate import CONFIG

from fast_eval import multicore_cut_evaluate_problem_2 as _fe2
from fast_eval import multicore_cut_evaluate_problem_3 as _fe3


def critical_chain(g, plan, problem=2, full=None):
    """problem = 2 或 3：用对应评估器的时间线（两者 Task 构造相同，问题3 的 COPY_IN 耗时含 Cache 命中）。
    full：同一方案已有的完整评估结果（evaluate(..., keep_full=True)['full']），给了就不再评估。"""
    if full is None:
        full = evaluate(g, plan, problem, keep_full=True).get('full')
    if full is None:
        return None
    fe = _fe3 if problem == 3 else _fe2
    tasks, links = fe._build_scene_b_tasks(g.raw, plan, CONFIG['bandwidth'], CONFIG['capacity'])[:2]
    delay = CONFIG['cross_core_copy_delay']
    start, end = {}, {}
    for tl in full['per_core_timeline']:
        c = tl['core_id']
        for e in tl['ops']:
            start[(c, e['op_id'])] = e['start']
            end[(c, e['op_id'])] = e['end']
    pipe_prev = {}
    for c, task in tasks.items():
        for order in task['pipe_ops'].values():
            for a, b in zip(order, order[1:]):
                pipe_prev[(c, b)] = (c, a)
    ext_pred = {}
    for ln in links:
        ext_pred.setdefault((ln['target_core'], ln['target_copy_in_id']), []).append(
            (ln['source_core'], ln['source_copy_out_id']))
    compute = set(g.compute_ids)
    x = max(end, key=lambda it: (end[it], it))
    chain = []                  # 倒序：(core, op_id, how)
    seen = set()
    while x is not None and x not in seen:
        seen.add(x)
        c, o = x
        s = start[x]
        nxt, how = None, None
        for src in ext_pred.get(x, ()):
            if end[src] + delay == s:
                nxt, how = src, 'ext'
                break
        if nxt is None:
            for p in tasks[c]['op_preds'][o]:
                if end[(c, p)] == s:
                    nxt, how = (c, p), 'dep'
                    break
        if nxt is None:
            pp = pipe_prev.get(x)
            if pp is not None and end[pp] == s:
                nxt, how = pp, 'pipe'
        chain.append((c, o, how))
        if s == 0:
            break
        x = nxt
    chain.reverse()
    is_comp = [o in compute and tasks[c]['op_by_id'][o]['op'] not in ('COPY_IN', 'COPY_OUT')
               for c, o, _ in chain]
    comp_chain = [(c, o) for (c, o, _), f in zip(chain, is_comp) if f]
    hops = []
    for (c1, o1), (c2, o2) in zip(comp_chain, comp_chain[1:]):
        if c1 != c2:
            hops.append((o1, o2))
    # 句柄：链上每个结点对应的计算 op（COPY 结点取同核链上最近的计算 op，先向后找再向前找）
    n = len(chain)
    handle = [None] * n
    for i in range(n):
        c = chain[i][0]
        j = i
        while j < n and chain[j][0] == c and handle[i] is None:
            if is_comp[j]:
                handle[i] = chain[j][1]
            j += 1
        j = i - 1
        while j >= 0 and chain[j][0] == c and handle[i] is None:
            if is_comp[j]:
                handle[i] = chain[j][1]
            j -= 1
    pipe_edges = []             # (a, b, core)：b 在同 Pipe 上排在 a 之后而被 a 卡住
    for i in range(1, n):
        if chain[i][2] == 'pipe' and handle[i] is not None and handle[i - 1] is not None \
                and handle[i] != handle[i - 1]:
            pipe_edges.append((handle[i - 1], handle[i], chain[i][0]))
    time_by_core = {}
    for c, o, _ in chain:
        time_by_core[c] = time_by_core.get(c, 0) + end[(c, o)] - start[(c, o)]
    return {'makespan': full['makespan'], 'chain': chain, 'ops': comp_chain, 'hops': hops,
            'kinds': [h for _, _, h in chain], 'pipe_edges': pipe_edges,
            'time_by_core': time_by_core}
