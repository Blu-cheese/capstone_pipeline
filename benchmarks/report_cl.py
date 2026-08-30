#!/usr/bin/env python3
"""
Continual-learning results report. Run from capstone_pipeline:

    venv/bin/python benchmarks/report_cl.py

Reads the run directories already on disk and prints the ACC / BWT table,
the R matrices, and — most importantly — an explicit check of whether
forgetting actually occurred, i.e. R[K-1][j] < R[j][j] for every earlier
task j.

CLAUDE.md §11 requires a human to verify that condition personally, because
"a broken harness still produces a plausible-looking table". This script does
not replace that judgement; it puts the numbers where they can be judged.

No training is run and no network is used. Reads only results/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# Runs before ~15:00 on 2026-08-29 used a current-task loss mask. Zhao et al.
# Eq. 2 fixes the softmax over ALL seen relations for every loss, so those
# runs are a deviation and are superseded. They are still on disk; the
# selection below keeps only the newest run per (track, seed) at the standard
# 60-epoch cap, which is exactly the corrected set.
EPOCH_CAP = 60


def load_runs():
    """Return {(track, seed): payload} for the canonical naive runs."""
    runs = {}
    for d in sorted(RESULTS.glob("*_naive_*")):
        rm, cfg = d / "R_matrix.json", d / "config.json"
        if not (rm.exists() and cfg.exists()):
            continue
        try:
            r = json.loads(rm.read_text(encoding="utf-8"))
            c = json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if r.get("ACC_f1") is None or c.get("max_epochs_per_task") != EPOCH_CAP:
            continue
        track = d.name.split("_naive_")[0]
        seed = int(d.name.split("_naive_")[1].split("_")[0])
        # Directory names carry a timestamp suffix, so "latest wins" is just
        # max by name.
        prev = runs.get((track, seed))
        if prev is None or d.name > prev["dir"]:
            runs[(track, seed)] = {"dir": d.name, "R": r, "cfg": c}
    return runs


def load_joint():
    out = {}
    for f in RESULTS.glob("joint_baseline_seed*.json"):
        if "cap" in f.name:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d.get("seed")] = d
    return out


def fmt(v, width=7):
    return " " * width if v is None else f"{v:>{width}.3f}"


def show_matrix(name, rows):
    k = len(rows)
    print(f"    {name}")
    print("        " + "".join(f"  task{j}" for j in range(k)))
    for t in range(k):
        cells = "".join(fmt(rows[t][j]) for j in range(k))
        print(f"      T{t} {cells}")


def check_forgetting(rows):
    """
    The condition a human is asked to verify: after training all K tasks,
    performance on each earlier task should be BELOW where it stood when
    that task was current.
    """
    k = len(rows)
    final = rows[k - 1]
    verdicts = []
    for j in range(k - 1):
        after, peak = final[j], rows[j][j]
        if after is None or peak is None:
            verdicts.append((j, None, None, None))
            continue
        verdicts.append((j, peak, after, after < peak))
    return verdicts


def main():
    runs, joint = load_runs(), load_joint()
    if not runs:
        print("No usable run directories found under results/.")
        return 1

    print("=" * 74)
    print("CONTINUAL LEARNING — NAIVE REGIME".center(74))
    print("=" * 74)
    print("\nEvaluation is against GOLD dev/test for BOTH tracks. The tracks")
    print("differ in exactly one variable: the source of the TRAINING labels.")
    print("Evaluating the teacher-track student against teacher labels would")
    print("measure fidelity to Gemini, not accuracy.\n")

    tracks = sorted({t for t, _ in runs})
    seeds = sorted({s for _, s in runs})

    print("-" * 74)
    print(f"{'track':<9}{'seed':>6}{'ACC_f1':>10}{'BWT_f1':>10}"
          f"{'ACC_acc':>10}{'BWT_acc':>10}{'joint_f1':>11}")
    print("-" * 74)
    summary = {}
    for track in tracks:
        accs, bwts = [], []
        for seed in seeds:
            r = runs.get((track, seed))
            if not r:
                continue
            R = r["R"]
            jb = R.get("joint_baseline_ref") or {}
            jf1 = jb.get("test_macro_f1") or jb.get("macro_f1")
            accs.append(R["ACC_f1"]); bwts.append(R["BWT_f1"])
            print(f"{track:<9}{seed:>6}{R['ACC_f1']:>10.3f}{R['BWT_f1']:>+10.3f}"
                  f"{R['ACC_acc']:>10.3f}{R['BWT_acc']:>+10.3f}"
                  f"{(f'{jf1:.3f}' if jf1 else '—'):>11}")
        if accs:
            summary[track] = (sum(accs) / len(accs), sum(bwts) / len(bwts))
            print(f"{track:<9}{'mean':>6}{summary[track][0]:>10.3f}"
                  f"{summary[track][1]:>+10.3f}")
        print("-" * 74)

    print("\nSeeds produce different TASK ORDERINGS, not just different inits.")
    print("Task order affects forgetting, so averaging over orderings is the")
    print("honest measure (Zhao et al. average over 5; RP-CRE set the convention).")

    # ---- the check CLAUDE.md §11 asks a human to make -------------------
    print("\n" + "=" * 74)
    print("FORGETTING CHECK — is R[K-1][j] < R[j][j] ?".center(74))
    print("=" * 74)
    print("A harness bug produces a plausible table. This is the condition")
    print("that distinguishes real forgetting from a broken measurement.\n")

    all_ok = True
    for (track, seed), r in sorted(runs.items()):
        rows = r["R"]["R_f1"]
        print(f"  {track} / seed {seed}")
        show_matrix("R_f1 (rows = after training task T, cols = eval on task j)", rows)
        for j, peak, after, ok in check_forgetting(rows):
            if ok is None:
                print(f"      task {j}: INCOMPLETE")
                all_ok = False
            else:
                mark = "forgot" if ok else "*** NO FORGETTING ***"
                print(f"      task {j}: {peak:.3f} while current -> "
                      f"{after:.3f} at end   {mark}")
                all_ok &= ok
        warns = r["R"].get("diagonal_warnings") or []
        for w in warns:
            print(f"      [warn] {w}")
        print()

    print("=" * 74)
    print(f"VERDICT: forgetting is visible in every run: {all_ok}")
    print("=" * 74)

    # ---- the caveat that must travel with these numbers -----------------
    print("""
READ THIS BEFORE QUOTING THE TABLE

Under the class-incremental train mask, naive forgetting is TOTAL: the
final-row off-diagonals are ~0 everywhere. BWT of -0.32 to -0.61 is not a
partial-decay measurement, it is a floor.

The majority class poisons whatever task it occupies, and it occupies the
same task at every seed. The example-count balance constraint forces
per:alternate_names (408 of 1,144 test examples) into a 4-class task with
the three thinnest companions. That task's diagonal collapses to the exact
constant-majority predictor (f1 0.238, acc 0.909) in 5 of 6 seed x track
cells. Loss falls while dev is pinned: imbalance, not an optimisation
failure.

So regime comparisons in Phase 6 must discriminate on OFF-DIAGONAL
retention. The alt-task diagonal is a constant across the sweep, and
"retaining task 1" there mostly means retaining the majority predictor.
Report this as a property of imbalanced CRE, not as a bug.
""")

    print("STATUS: naive only. replay and FKD are not implemented")
    print("(continual/regimes.py does not exist), so this is 1 of the 3")
    print("regimes the research question needs.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
