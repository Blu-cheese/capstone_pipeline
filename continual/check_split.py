"""
Task-split reconciliation guard (PHASE5_SPEC 0.2).

Asserts, for every seed, that the frozen split is internally consistent and
matches what the harness will assume. Exit nonzero on any failure.

This exists because the kept-class count has already changed twice (20 stale
meeting taxonomy -> 18 pre-multi-label-fix -> 17 now) and a stale count
anywhere downstream silently corrupts the student head. THIS SCRIPT is the
source of truth for N_CLASSES, not any figure in the docs.

Usage:
    venv/bin/python -m continual.check_split
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.task_split import (
    load_split, class_index, task_of_class, kept_relations,
    MIN_CLASSES_PER_TASK, MIN_TASK_SHARE,
)

EXPECTED_N_CLASSES = 17
SEEDS = config.SEEDS  # [1234, 1, 2]


def check_seed(seed: int) -> list:
    """Return a list of failure strings for one seed (empty = pass)."""
    failures = []
    split = load_split(seed)
    ci = class_index(split)

    # Task class sets, from the frozen relation->task mapping.
    sets = {}
    for rel, t in split["relation_to_task"].items():
        sets.setdefault(t, set()).add(rel)

    # 1. Pairwise disjoint (guaranteed by dict structure, but assert the
    #    task-index domain is exactly 0..K-1 with no gaps).
    k = split["k"]
    if sorted(sets) != list(range(k)):
        failures.append(f"task indices {sorted(sets)} != 0..{k-1}")
    tasks = [sets.get(t, set()) for t in range(k)]
    for a in range(k):
        for b in range(a + 1, k):
            overlap = tasks[a] & tasks[b]
            if overlap:
                failures.append(f"tasks {a} and {b} share classes: {sorted(overlap)}")

    # 2. Union == 17 == domain of class_index == N_CLASSES.
    union = set().union(*tasks) if tasks else set()
    if len(union) != EXPECTED_N_CLASSES:
        failures.append(f"union has {len(union)} classes, expected {EXPECTED_N_CLASSES}")
    if union != set(ci):
        failures.append("union of task sets != domain of class_index()")
    if set(ci.values()) != set(range(len(ci))):
        failures.append("class_index() values are not a contiguous 0..N-1 range")
    if kept_relations(split) != sorted(ci, key=ci.get):
        failures.append("kept_relations() order disagrees with class_index()")

    # 3. Per-task constraints from Phase 3.
    for t in split["tasks"]:
        if t["n_classes"] < MIN_CLASSES_PER_TASK:
            failures.append(f"task {t['task']} has {t['n_classes']} classes "
                            f"(< {MIN_CLASSES_PER_TASK})")
        for track in ("teacher_share", "gold_share"):
            if t[track] < MIN_TASK_SHARE:
                failures.append(f"task {t['task']} {track}={t[track]:.3f} "
                                f"(< {MIN_TASK_SHARE})")

    # 4. task_of_class covers every head index exactly once.
    toc = task_of_class(split)
    if sorted(toc) != list(range(len(ci))):
        failures.append("task_of_class() does not cover head indices 0..N-1")

    return failures


def main() -> int:
    any_fail = False
    for seed in SEEDS:
        failures = check_seed(seed)
        if failures:
            any_fail = True
            print(f"FAIL  seed {seed}:")
            for f in failures:
                print(f"      - {f}")
        else:
            split = load_split(seed)
            sizes = [t["n_classes"] for t in split["tasks"]]
            print(f"PASS  seed {seed}: {EXPECTED_N_CLASSES} classes, "
                  f"disjoint tasks {sizes}, shares ok on both tracks")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
