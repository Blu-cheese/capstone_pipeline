"""
Task split for continual learning (METHOD_A_SPEC Phase 3).

Partitions the relation label space into K sequential tasks. The student
trains on task 0, then task 1, then task 2, and is evaluated on every task
seen so far after each stage - so the split defines the entire experiment.

Two constraints make this non-trivial here:

1. TWO LABEL TRACKS. Per the dual-track decision, the identical harness runs
   on teacher labels and on gold labels. The split must therefore be the SAME
   partition for both, and balanced in BOTH simultaneously - a split that is
   even under teacher counts can be badly skewed under gold counts, because
   the teacher's error is systematic (e.g. per:other_family is 152 teacher /
   26 gold, per:positive_impression is 50 teacher / 146 gold).

2. THIN CLASSES ARE DROPPED, NOT MERGED. 25 of 37 classes fall under 20
   examples in at least one track. Merging them would mean folding
   semantically unrelated relations together (per:pet into per:friends),
   which produces a meaningless label. They are dropped and logged.

   per:acquaintance is dropped despite having 472 teacher examples: gold uses
   it 9 times. It is the teacher's uncertainty sink (see the Phase 1 report),
   not a learnable relation, and keeping it would make the two tracks
   incomparable.

The split is seeded and frozen to disk. Re-running with the same seed
reproduces it byte-identically; every regime at a given seed sees the same
task sequence, or the regime comparison means nothing.

Usage:
    venv/bin/python -m continual.task_split --seed 1234
    venv/bin/python -m continual.task_split --seed 1234 --show
"""

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.dialogre_parser import DIALOGRE_RELATIONS

SPLIT_DIR = Path("data/task_splits")

# V2 3.3 threshold sensitivity, measured with `unanswerable` already excluded:
#     thr=20 -> 11 classes,  821 teacher ex   (too few classes for a CL problem)
#     thr=15 -> 18 classes,  957 teacher ex   <- chosen
#     thr=10 -> 20 classes,  986 teacher ex   (per:siblings falls to 10 teacher)
# 15 lands in V2's stated 16-18 preference band and gives 6 classes per task at
# K=3. Threshold 10 buys two more classes at the cost of a 10-example class,
# and macro-F1 over classes that thin is mostly noise.
MIN_EXAMPLES_PER_CLASS = 15
MIN_TASK_SHARE = 0.15         # Phase 3 acceptance: no task under 15% of examples
DEFAULT_K = 3

# V2 3.1: `unanswerable` is DialogRE's null/reject class - the direct analogue
# of TACRED's `no_relation`, which Zhao et al. remove before splitting into
# tasks. It is not a relation: training part of a task to predict "nothing
# holds here" behaves differently under forgetting than a real relation, and
# breaks comparability with published CRE numbers. Excluded before thresholding.
EXCLUDED_CLASSES = {"unanswerable"}

# Balancing on example counts alone admits degenerate splits: `unanswerable`
# holds 43% of teacher examples, so the optimiser will happily give it a task
# of its own. A single-class task is not a classification task - the student
# sees one label for that entire stage and collapses onto it, and R[t][t] is
# meaningless. Require enough classes per task for the stage to be learnable.
MIN_CLASSES_PER_TASK = 3


def viable_classes(
    y_teacher: np.ndarray,
    y_gold: np.ndarray,
    min_examples: int = MIN_EXAMPLES_PER_CLASS,
) -> Tuple[List[int], List[dict]]:
    """
    Classes with >= min_examples in BOTH tracks.

    Returns (kept_indices, drop_log) where drop_log records why each dropped
    class was dropped - Phase 3 requires the decision be explicit.
    """
    tc, gc = Counter(y_teacher.tolist()), Counter(y_gold.tolist())

    kept, dropped = [], []
    for i, name in enumerate(DIALOGRE_RELATIONS):
        t, g = tc.get(i, 0), gc.get(i, 0)
        if name in EXCLUDED_CLASSES:
            dropped.append({"relation": name, "teacher": t, "gold": g,
                            "reason": "null/reject class, excluded per V2 3.1 "
                                      "(Zhao et al. remove no_relation before splitting)"})
        elif t >= min_examples and g >= min_examples:
            kept.append(i)
        elif t == 0 and g == 0:
            dropped.append({"relation": name, "teacher": t, "gold": g,
                            "reason": "absent from both tracks"})
        else:
            which = []
            if t < min_examples:
                which.append(f"teacher={t}")
            if g < min_examples:
                which.append(f"gold={g}")
            dropped.append({"relation": name, "teacher": t, "gold": g,
                            "reason": f"under {min_examples} ({', '.join(which)})"})
    return kept, dropped


def _score(assignment: Tuple[int, ...], counts: List[Tuple[int, int]],
           k: int) -> Optional[float]:
    """
    Imbalance score for one assignment, or None if it violates the
    minimum-share constraint in either track.

    Score is the largest deviation from an even share across both tracks;
    lower is better. Scoring both tracks jointly is the point - optimising
    teacher balance alone produces gold tasks that starve.
    """
    t_tot = [0] * k
    g_tot = [0] * k
    n_classes = [0] * k
    for task, (t, g) in zip(assignment, counts):
        t_tot[task] += t
        g_tot[task] += g
        n_classes[task] += 1

    if min(n_classes) < MIN_CLASSES_PER_TASK:
        return None

    t_sum, g_sum = sum(t_tot), sum(g_tot)
    if t_sum == 0 or g_sum == 0:
        return None

    worst = 0.0
    for totals, total in ((t_tot, t_sum), (g_tot, g_sum)):
        for v in totals:
            share = v / total
            if share < MIN_TASK_SHARE:
                return None
            worst = max(worst, abs(share - 1.0 / k))
    return worst


def build_split(
    y_teacher: np.ndarray,
    y_gold: np.ndarray,
    seed: int = 1234,
    k: int = DEFAULT_K,
) -> Dict:
    """
    Search for a class->task assignment balanced in both label tracks.

    With ~12 viable classes and k=3 the space is 3^12 = 531k assignments, so
    it is enumerated exhaustively rather than approximated greedily. Among
    assignments within a small tolerance of the best score, the seed picks
    one - that is what makes different seeds give genuinely different task
    sequences for Phase 7's multi-seed sweep, while staying balanced.
    """
    kept, drop_log = viable_classes(y_teacher, y_gold)
    if len(kept) < k:
        raise ValueError(f"only {len(kept)} viable classes for k={k} tasks")

    tc, gc = Counter(y_teacher.tolist()), Counter(y_gold.tolist())
    counts = [(tc.get(i, 0), gc.get(i, 0)) for i in kept]

    # Seeded random-restart hill climbing.
    #
    # Exhaustive enumeration was fine at 12 classes (3^12 = 531k) but is not
    # at 18 (3^18 = 387M). Local search finds balanced partitions in
    # milliseconds: from a random assignment, repeatedly move the single class
    # whose reassignment most improves balance, until no move helps. Many
    # seeded restarts make it robust, and because the restarts are drawn from
    # the caller's seed, different seeds still yield different task sequences.
    rng = random.Random(seed)
    n = len(kept)
    n_restarts = 200

    def climb(start: List[int]) -> Tuple[Optional[float], List[int]]:
        cur = list(start)
        cur_score = _score(tuple(cur), counts, k)
        improved = True
        while improved:
            improved = False
            best_move, best_move_score = None, cur_score
            for ci in range(n):
                original = cur[ci]
                for t in range(k):
                    if t == original:
                        continue
                    cur[ci] = t
                    s = _score(tuple(cur), counts, k)
                    if s is not None and (best_move_score is None or s < best_move_score):
                        best_move, best_move_score = (ci, t), s
                cur[ci] = original
            if best_move is not None:
                cur[best_move[0]] = best_move[1]
                cur_score, improved = best_move_score, True
        return cur_score, cur

    best_score, best_assignment = None, None
    for _ in range(n_restarts):
        start = [rng.randrange(k) for _ in range(n)]
        score, assignment = climb(start)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score, best_assignment = score, assignment

    if best_assignment is None:
        raise ValueError(
            f"no assignment satisfies the {MIN_TASK_SHARE:.0%} minimum-share and "
            f">={MIN_CLASSES_PER_TASK}-classes-per-task constraints with k={k}"
        )

    best = best_score
    # Canonicalise task numbering so the partition, not the labelling, is what
    # the seed varies: relabel tasks by first-appearing class index.
    order, remap = [], {}
    for t in best_assignment:
        if t not in remap:
            remap[t] = len(order)
            order.append(t)
    chosen = tuple(remap[t] for t in best_assignment)
    near_best = [chosen]

    relation_to_task = {DIALOGRE_RELATIONS[c]: int(t)
                        for c, t in zip(kept, chosen)}

    tasks = []
    for t in range(k):
        members = [DIALOGRE_RELATIONS[c] for c, a in zip(kept, chosen) if a == t]
        tt = sum(tc.get(c, 0) for c, a in zip(kept, chosen) if a == t)
        gt = sum(gc.get(c, 0) for c, a in zip(kept, chosen) if a == t)
        tasks.append({"task": t, "relations": members, "n_classes": len(members),
                      "teacher_examples": tt, "gold_examples": gt})

    t_sum = sum(x["teacher_examples"] for x in tasks)
    g_sum = sum(x["gold_examples"] for x in tasks)
    for x in tasks:
        x["teacher_share"] = round(x["teacher_examples"] / t_sum, 4)
        x["gold_share"] = round(x["gold_examples"] / g_sum, 4)

    return {
        "seed": seed,
        "k": k,
        "min_examples_per_class": MIN_EXAMPLES_PER_CLASS,
        "min_task_share": MIN_TASK_SHARE,
        "label_space": DIALOGRE_RELATIONS,
        "kept_class_indices": kept,
        "relation_to_task": relation_to_task,
        "tasks": tasks,
        "dropped_classes": drop_log,
        "totals": {
            "teacher_examples_kept": t_sum,
            "gold_examples_kept": g_sum,
            "teacher_examples_dropped": int(len(y_teacher) - t_sum),
            "gold_examples_dropped": int(len(y_gold) - g_sum),
        },
        "imbalance_score": round(best, 4),
        "candidates_within_tolerance": len(near_best),
    }


def save_split(split: Dict) -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"split_seed{split['seed']}.json"
    path.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_split(seed: int = 1234) -> Dict:
    path = SPLIT_DIR / f"split_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m continual.task_split --seed {seed}`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def kept_relations(split: Dict) -> List[str]:
    """
    The split's kept relations in canonical order.

    Sorted by position in DIALOGRE_RELATIONS rather than alphabetically, so
    the ordering is tied to the frozen label space and cannot drift if a
    relation is ever renamed.
    """
    return sorted(split["relation_to_task"], key=DIALOGRE_RELATIONS.index)


def class_index(split: Dict) -> Dict[str, int]:
    """
    relation name -> student head index (0 .. N_CLASSES-1).

    The student's head is N_CLASSES wide, not 37: dropped classes have no
    output unit at all. This is the single source of truth for that mapping -
    read N_CLASSES as len(class_index(split)), never hard-code it.
    """
    return {r: i for i, r in enumerate(kept_relations(split))}


def task_of_class(split: Dict) -> Dict[int, int]:
    """student head index -> task index, for logit masking in the harness."""
    ci = class_index(split)
    return {ci[r]: t for r, t in split["relation_to_task"].items()}


def describe(split: Dict) -> None:
    print(f"\n{'='*72}")
    print(f"TASK SPLIT  seed={split['seed']}  k={split['k']}")
    print(f"{'='*72}")
    for t in split["tasks"]:
        print(f"\nTask {t['task']}  "
              f"teacher {t['teacher_examples']:5d} ({t['teacher_share']:.1%})   "
              f"gold {t['gold_examples']:5d} ({t['gold_share']:.1%})")
        for r in t["relations"]:
            print(f"    {r}")

    tot = split["totals"]
    print(f"\nkept   : {tot['teacher_examples_kept']} teacher / "
          f"{tot['gold_examples_kept']} gold")
    print(f"dropped: {tot['teacher_examples_dropped']} teacher / "
          f"{tot['gold_examples_dropped']} gold "
          f"({len(split['dropped_classes'])} classes)")
    print(f"imbalance score: {split['imbalance_score']} "
          f"(max deviation from even share, across both tracks)")

    print(f"\nDropped classes (Phase 3 requires this be explicit):")
    for d in split["dropped_classes"]:
        print(f"    {d['relation']:32s} teacher={d['teacher']:4d} "
              f"gold={d['gold']:4d}   {d['reason']}")


def main():
    ap = argparse.ArgumentParser(description="Build and freeze the task split")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--corpus", default="dialogre_train")
    ap.add_argument("--show", action="store_true",
                    help="load and print the frozen split without rebuilding")
    args = ap.parse_args()

    if args.show:
        describe(load_split(args.seed))
        return

    from continual.features import build_features
    d = build_features(args.corpus)

    split = build_split(d["y_teacher"], d["y_gold"], seed=args.seed, k=args.k)
    path = save_split(split)
    describe(split)
    print(f"\nfrozen to {path}")


if __name__ == "__main__":
    main()
