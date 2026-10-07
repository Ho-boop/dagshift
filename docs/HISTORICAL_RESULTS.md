# Historical experiment scope

Source: local delivery v3 aggregate tables from September 2026, algorithm snapshot `0ffae91`. The public release does not redistribute the official input graphs, original problem statement, evaluator source, run logs, solution archives, or unpublished paper.

The aggregated values in `historical_summary.csv` were read from the frozen delivery tables, not recomputed by this release. Q1 uses delivery v2; Q2/Q3 use the v3 deep-graph append candidate, choosing the better result after the full v2 path. The default CLI without `--append deep` remains v2.

Timing scope: one input graph and one problem, sequential generation of k=2,3,4,5 in one process. k=1 is separate. Internal search evaluations, graph loading, preprocessing and output are included; subsequent validation is separate. Hardware: Intel i9-14900HX, 32 logical threads, 64 GB; Python 3.14.2, Windows 11. Historical blocks used parallel processes, up to 16; the timings are not serial idle-machine measurements.

The 600 s limit is this project's case-level engineering criterion. Virtual budget V=120 and cap factor 6 are algorithm execution settings and should not be described as a 120 s wall-clock guarantee. All three problem sets had 100/100 cases within the project criterion and zero capped rows in the historical summary.

Mean speedup is the arithmetic mean of fixed no-Cache single-core baseline time divided by multicore makespan across 100 cases. Q3 values use that same denominator for cross-problem comparison; they are not the isolated L2 hardware benefit. The Q3 same-plan no-L2/Cache time ratio has separate historical means approximately 1.0022, 1.0079, 1.0236 and 1.0406 for k=2,3,4,5.

Validation evidence from the original workbench:

- The accelerated evaluator matched entire original result objects on 100 inputs × 13 plan variants.
- All 1500 delivered plan hashes were checked against the frozen result table.
- A clean, reduced delivery package regenerated 20 selected plans byte-for-byte and matched the original official CLI on those 20 outputs.
- A complete original-official CLI rerun over all 1500 delivery outputs is not claimed.

The release adapter supports original official evaluators and separately supplied compatible accelerated evaluators. Switching evaluator implementations changes wall-clock performance; the public synthetic demo and its compute-work proxy cannot establish these historical competition measurements.
