import copy
import json
import tempfile
import unittest
from pathlib import Path

from common import Graph, validate_plan
from dagshift.demo import main, partition, synthetic_graph


class DemoTests(unittest.TestCase):
    def test_coverage_core_count_and_deterministic_assignment(self):
        raw = synthetic_graph()
        for cores in (1, 2, 3, 5, 12):
            with self.subTest(cores=cores):
                plan, report = partition(raw, cores)
                self.assertEqual(len(plan['core_schedules']), cores)
                self.assertEqual(set(plan['node_to_subgraph']), {str(op['id']) for op in raw['ops']})
                self.assertTrue(validate_plan(Graph(raw), plan, problem=1, num_cores=cores).ok)
                self.assertEqual((plan, report), partition(copy.deepcopy(raw), cores))

    def test_known_work_balance_is_computation_proxy(self):
        _, report = partition(synthetic_graph(), 3)
        self.assertEqual(report['per_core_compute_work'], [180, 100, 150])
        self.assertEqual(report['compute_critical_path'], 180)
        self.assertNotIn('makespan', report)

    def test_does_not_mutate_input(self):
        raw = synthetic_graph()
        before = copy.deepcopy(raw)
        partition(raw, 3)
        self.assertEqual(raw, before)

    def test_invalid_core_count(self):
        for count in (0, -1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                partition(synthetic_graph(), count)

    def test_rejects_cycle_and_empty_compute_graph(self):
        raw = synthetic_graph()
        raw['edges'].append({'source': 2, 'target': 0})
        with self.assertRaises(ValueError):
            partition(raw, 3)
        with self.assertRaises(ValueError):
            partition({'ops': [], 'tensors': [], 'edges': []}, 3)

    def test_cli_emits_separate_plan_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            main(['--cores', '3', '--out', str(output)])
            plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
            self.assertEqual(set(plan), {'node_to_subgraph', 'core_schedules'})
            self.assertIn('Not official makespan', (output / 'report.html').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((output / 'report.json').read_text())['cores'], 3)


if __name__ == '__main__':
    unittest.main()
