"""Run a dependency-free, synthetic graph-partitioning demonstration."""

import argparse
import html
import json
from pathlib import Path

from common import Graph, op_to_core, plan_hash, save_plan, validate_plan
from solvers.baseline import solve


def synthetic_graph():
    """Four independent compute chains; hand-authored, no contest dataset."""
    ops, edges, tensors = [], [], []
    for component, costs in enumerate(((80, 60, 40), (55, 45), (40, 30, 20), (35, 25))):
        ids = []
        for index, cycles in enumerate(costs):
            op_id = component * 10 + index
            ids.append(op_id)
            ops.append({'id': op_id, 'op': 'MATMUL' if index % 2 == 0 else 'ADD',
                        'pipe': 'PIPE_M' if index % 2 == 0 else 'PIPE_V', 'cycles': cycles})
        for index, (left, right) in enumerate(zip(ids, ids[1:])):
            tensor_id = 1000 + component * 10 + index
            tensors.append({'id': tensor_id, 'size': 64, 'pos': 'UB'})
            edges.extend([{'source': left, 'target': tensor_id},
                          {'source': tensor_id, 'target': right}])
    return {'ops': ops, 'tensors': tensors, 'edges': edges}


def partition(raw, cores=3):
    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer')
    graph = Graph(raw, name='synthetic')
    # Materialize the order: reject cyclic compute graphs before partitioning.
    _ = graph.topo_order
    if not graph.compute_ids:
        raise ValueError('the demo needs at least one compute operation')
    plan = solve(graph, cores, problem=1, variant='bundled')
    validation = validate_plan(graph, plan, problem=1, num_cores=cores)
    if not validation.ok:
        raise ValueError(f'{validation.stage}: {validation.message}')
    assignment = op_to_core(plan)
    work = [sum(graph.cycles[op] for op in graph.compute_ids if assignment[op] == core)
            for core in range(cores)]
    report = {'mode': 'synthetic-structural-demo', 'compute_ops': len(graph.compute_ids),
              'components': len(graph.components), 'cores': cores,
              'per_core_compute_work': work, 'largest_core_compute_work': max(work),
              'compute_critical_path': graph.critical_path, 'plan_hash': plan_hash(plan),
              'validation': 'format, coverage, subgraph DAG and task-order DAG passed',
              'metric_scope': 'Compute-work proxy only; no DDR, spill, pipe scheduling or L2 simulation. Not official makespan.'}
    return plan, report


def render_report(report):
    maximum = max(report['per_core_compute_work']) or 1
    rows = ''.join(f'<div class="row"><span>Core {i}</span><div class="track">'
                   f'<div class="bar" style="width:{100 * work / maximum:.2f}%">{work}</div>'
                   '</div></div>' for i, work in enumerate(report['per_core_compute_work']))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DAGShift Demo</title>
<style>body{{font:16px system-ui;background:#edf2f7;color:#203047;max-width:900px;margin:60px auto;padding:24px}}
article{{background:white;border-radius:18px;padding:38px;box-shadow:0 12px 40px #20304716}}
h1{{font-size:40px;margin:0}}.label{{color:#577690;letter-spacing:2px}}.row{{display:flex;align-items:center;gap:20px;margin:18px 0}}
.row span{{width:65px}}.track{{flex:1;background:#edf2f7;border-radius:8px}}.bar{{background:#416f96;color:white;padding:12px 0;box-sizing:border-box;border-radius:8px;text-align:center}}
code{{overflow-wrap:anywhere}}p{{line-height:1.6}}.note{{background:#f3f6f9;padding:16px;border-radius:8px}}</style>
<article><div class="label">GRAPH PARTITIONING · PORTABLE DEMO</div><h1>DAGShift</h1>
<p>{report['compute_ops']} compute operations · {report['components']} independent components · {report['cores']} cores.</p>
<h2>Compute work assigned to each core</h2>{rows}
<p class="note">{html.escape(report['metric_scope'])}</p>
<p>Validation: {html.escape(report['validation'])}</p>
<p>Deterministic plan hash: <code>{report['plan_hash']}</code></p></article></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores', type=int, default=3)
    parser.add_argument('--graph', type=Path, help='Optional graph JSON using the documented schema')
    parser.add_argument('--out', type=Path, default=Path('demo-output'))
    args = parser.parse_args(argv)
    raw = json.loads(args.graph.read_text(encoding='utf-8')) if args.graph else synthetic_graph()
    plan, report = partition(raw, args.cores)
    args.out.mkdir(parents=True, exist_ok=True)
    save_plan(plan, args.out / 'plan.json')
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.out / 'report.html').write_text(render_report(report), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Wrote {args.out / "plan.json"} and {args.out / "report.html"}')


if __name__ == '__main__':
    main()
