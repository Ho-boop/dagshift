"""Load user-supplied evaluators without copying their source into this project."""

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path


def load_evaluator():
    accelerated = os.environ.get('DAGSHIFT_EVALUATOR_ROOT')
    if accelerated:
        directory = Path(accelerated).resolve() / 'fast_eval'
        initializer = directory / '__init__.py'
        if not initializer.is_file():
            raise RuntimeError('DAGSHIFT_EVALUATOR_ROOT must contain fast_eval/__init__.py.')
        spec = importlib.util.spec_from_file_location('fast_eval', initializer,
                                                    submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['fast_eval'] = module
        spec.loader.exec_module(module)
        return module, directory

    package = os.environ.get('DAGSHIFT_OFFICIAL_PACKAGE')
    if not package:
        raise RuntimeError('Provide DAGSHIFT_OFFICIAL_PACKAGE or DAGSHIFT_EVALUATOR_ROOT; see README.md.')
    directory = Path(package).resolve() / 'code'
    if not (directory / 'multicore_cut_evaluate_problem_1.py').is_file():
        raise RuntimeError('DAGSHIFT_OFFICIAL_PACKAGE must contain the original code/ and data/ directories.')
    # Official scripts import their sibling files with flat module names.
    sys.path.append(str(directory))
    names = ('evaluation_validation', 'multicore_cut_evaluate_problem_1',
             'multicore_cut_evaluate_problem_2', 'multicore_cut_evaluate_problem_3',
             'schedule_step1')
    loaded = {name: importlib.import_module(name) for name in names}
    module = types.ModuleType('fast_eval')
    module.__path__ = []
    module.__file__ = str(directory / 'multicore_cut_evaluate_problem_1.py')
    sys.modules['fast_eval'] = module
    for name, child in loaded.items():
        setattr(module, name, child)
        sys.modules[f'fast_eval.{name}'] = child
    first = loaded['multicore_cut_evaluate_problem_1']
    second = loaded['multicore_cut_evaluate_problem_2']
    third = loaded['multicore_cut_evaluate_problem_3']

    def load_config(path):
        path = str(path)
        base = loaded['evaluation_validation'].read_evaluation_config(path)
        a, b, c = first.read_scene_a_config(path), second.read_scene_b_config(path), third.read_cache_config(path)
        return {'bandwidth': base['bandwidth'], 'capacity': base['capacity'],
                'cross_core_wait': a['task_cross_core_wait_cycles'],
                'same_core_wait': a['task_same_core_wait_cycles'],
                'cross_core_copy_delay': b['cross_core_copy_delay_cycles'],
                'cache_capacity_bytes': c['cache_capacity_bytes'],
                'cache_bandwidth_bytes_per_cycle': c['cache_bandwidth_bytes_per_cycle']}

    def evaluate(graph, plan, problem, config):
        parameters = dict(bandwidth=config['bandwidth'], capacity=config['capacity'])
        if problem == 1:
            return first.evaluate_scene_a(graph, plan, **parameters,
                                          cross_core_wait=config['cross_core_wait'], same_core_wait=config['same_core_wait'])
        parameters['cross_core_copy_delay'] = config['cross_core_copy_delay']
        if problem == 2:
            return second.evaluate_scene_b(graph, plan, **parameters)
        if problem == 3:
            return third.evaluate_problem_3(graph, plan, **parameters,
                                           cache_capacity_bytes=config['cache_capacity_bytes'],
                                           cache_bandwidth_bytes_per_cycle=config['cache_bandwidth_bytes_per_cycle'])
        raise ValueError('problem must be 1, 2, or 3')

    def singlecore(graph, config):
        eligible = sorted(op['id'] for op in graph['ops'] if op['op'] not in ('COPY_IN', 'COPY_OUT'))
        plan = {'node_to_subgraph': {str(op): 0 for op in eligible},
                'core_schedules': [[0] if eligible else []]}
        return evaluate(graph, plan, 1, config)

    module.load_config = load_config
    module.evaluate = evaluate
    module.singlecore = singlecore
    module.set_quiet = lambda enabled=True: None  # Original diagnostic behavior is preserved.
    return module, directory
