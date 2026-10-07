"""Paths for the public release; external contest assets are never bundled."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(os.environ.get('DAGSHIFT_OFFICIAL_PACKAGE', str(ROOT / 'external')))
DATA = Path(os.environ.get('DAGSHIFT_DATA_DIR', str(PACKAGE / 'data')))
OFFICIAL_CODE = PACKAGE / 'code'
CONFIG_PATH = Path(os.environ.get('DAGSHIFT_CONFIG', str(DATA / 'config.txt')))
RESULTS = ROOT / 'results'
RUNS = RESULTS / 'runs'
BEST = RESULTS / 'best'
BASELINE = RESULTS / 'baseline'
OFFICIAL = RESULTS / 'official'
EVAL_CACHE = RESULTS / '.evalcache'
PAPER = ROOT / 'private-paper'
SAMPLE_CASES = ROOT / 'examples/sample_cases.txt'

def list_cases():
    return sorted(p.stem for p in DATA.glob('case_*.json') if p.stem[5:].isdigit())

def sample_cases():
    return [s.split()[0] for s in SAMPLE_CASES.read_text(encoding='utf-8').splitlines()
            if s.strip() and not s.lstrip().startswith('#')]

def resolve_cases(spec):
    if spec in (None, '', 'all'):
        return list_cases()
    if spec == 'sample':
        return sample_cases()
    return [s.strip() for s in spec.split(',') if s.strip()]

def case_path(case):
    return DATA / f'{case}.json'
