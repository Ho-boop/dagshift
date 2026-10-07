# Public release validation · 2026-10-07

The public folder is a separate staging copy. The original competition workbench and its results were not modified. `source-manifest.json` records SHA-256 hashes of the 38 original modules selected from the formal delivery import closure before public-adapter changes.

Release changes are limited to external dependency/path loading in `common/__init__.py`, `common/paths.py`, `common/evaluate.py`, new `dagshift/` modules, public documentation, a synthetic input, tests and CI. The optimization logic is retained.

Local checks used the original project's required native Python 3.14 with `PYTHONUTF8=1`:

| Check | Result |
|---|---|
| Dependency-free tests | 6 passed: coverage/core count and determinism, expected work balance, input unchanged, invalid core count, cyclic/empty compute graph rejection, separate CLI outputs |
| Synthetic input through original official external evaluator | Problems 1, 2 and 3 each accepted the baseline plan |
| Full delivery on the 10-op synthetic input, k=2 | Problems 1, 2 and 3 each produced valid outputs without fallback or real-time cap; Q2/Q3 enabled deep append; approximately 1.4, 0.6 and 0.4 s local end-to-end |

The external evaluator was loaded from the user's local private package solely for the small checks above; it is not part of this release. These checks establish import wiring and small-input execution, not competition-wide performance or cross-machine reproducibility. GitHub CI is configured for dependency-free tests on Windows and Linux; hosted CI results are only established after the workflow runs.
