# Benchmarks and showcase

Three things you can run. None of them train anything; two need no network.

## 1. Regression tests for the Aug 30 pipeline fixes

```bash
venv/bin/python tests/test_pipeline_fixes.py     # 49 checks, offline
venv/bin/python tests/test_parsers.py            # 38 checks, offline
```

Every test maps to a defect that previously shipped silently. The
API-dependent paths substitute the transport function, so the real
validation logic runs without a network call.

## 2. Knowledge-graph benchmark

```bash
venv/bin/python benchmarks/report_graph.py --neo4j-password capstone123
venv/bin/python benchmarks/report_graph.py --offline   # no database needed
```

Prints entity/relation counts, per-meeting breakdown, type distributions,
and cross-meeting linkage, then re-verifies from the live graph every
property the fixes were meant to establish.

## 3. Continual-learning results

```bash
venv/bin/python benchmarks/report_cl.py
```

ACC / BWT across 3 seeds x 2 tracks, the R matrices, and an explicit check
of the forgetting condition `R[K-1][j] < R[j][j]`. Reads `results/` only.

Selects the newest run per (track, seed) at the 60-epoch cap, which is the
corrected seen-through-t train-mask set. The earlier runs still on disk used
a current-task mask and are superseded.

**This covers the naive regime only.** Replay and FKD are not implemented,
so it is 1 of the 3 regimes the research question needs.
