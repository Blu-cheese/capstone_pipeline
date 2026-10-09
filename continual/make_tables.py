"""
Phase 7 tables and figures (PHASE6_7_SPEC §4.2-4.4, v4), generated only from
saved run files. No hand edits.

    venv/bin/python -m continual.make_tables

Runs are selected from metrics.json metadata, never from paths: per schedule,
every (track, regime, seed) must have exactly one clean run at one git hash
(default: the single hash shared by that schedule's clean runs; --hash-primary
/ --hash-fixed to choose when there are several).

Writes results/tables/{T1..T6, D1, prereg_outcomes}.{md,csv} and
results/figures/*.png.

Definitions, fixed here BEFORE the sweep ran, so that the open wording in
PREREG lines 4, 6 and 8 is measured by a rule chosen without seeing results:

  forgetting_j   R_f1[K-1][j] - R_f1[j][j] for old tasks j < K-1 (negative =
                 forgetting). Non-degenerate: j is not the task holding
                 per:alternate_names. Seeds 1234 and 1 have one non-degenerate
                 old task (task 0); seed 2 has two (tasks 0 and 1).
  T2             accuracy over the pooled test examples of all seen tasks after
                 task t (sample weighted, as in Zhao Table 2).
  T4             per-class F1 over the pooled seen test set after task t, at the
                 task where the class is learned and at the end; drop = first -
                 final. A PAIR's drop is the drop of its earlier-learned member
                 (the one exposed to its analogue arriving later). Cross-task =
                 members in different tasks for that seed.
  PREREG 4       per track and seed, mean pair drop over that seed's cross-task
                 pairs; "FKD's drop smaller than replay's by more than 0.10"
                 when mean_drop(replay) - mean_drop(fkd) > 0.10.
  T5 broad class argmax over the 17 kept classes of teacher train count / gold
                 train count. Share after task t = fraction of the pooled seen
                 test set predicted as that class.
  PREREG 6       "falls" when share(task where learned) - share(final) > 0.03
                 on at least 2 of 3 seeds (the PREREG threshold rule).
  PREREG 8       "forgets less" when mean non-degenerate forgetting(kd) -
                 mean non-degenerate forgetting(replay) > 0.03 on >= 2 of 3 seeds.
  "real" diff    |difference| > 0.03 on at least 2 of 3 seeds, same sign.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.features import build_features
from continual.student import macro_f1
from continual.task_split import load_split, class_index
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS

TRACKS = ["gold", "teacher"]
SEEDS = config.SEEDS
PRIMARY = ["naive", "replay", "fkd"]
FIXED = ["replay", "fkd", "kd"]
THRESH = 0.03
DEGENERATE_CLASS = "per:alternate_names"
T4_CLASSES = ["per:positive_impression", "per:negative_impression", "per:friends",
              "per:girl/boyfriend", "per:roommate"]
T4_PAIRS = [("per:positive_impression", "per:negative_impression"),
            ("per:friends", "per:girl/boyfriend"),
            ("per:friends", "per:roommate"),
            ("per:girl/boyfriend", "per:negative_impression")]    # within-task reference

TABLE_DIR = config.RESULTS_DIR / "tables"
FIG_DIR = config.RESULTS_DIR / "figures"


# ------------------------------------------------------------------ io

class Run:
    def __init__(self, d: Path, meta: Dict, metrics: Dict):
        self.dir, self.meta, self.m = d, meta, metrics
        self._r = self._p = None

    @property
    def R(self) -> Dict:
        if self._r is None:
            self._r = json.loads((self.dir / "R_matrix.json").read_text(encoding="utf-8"))
        return self._r

    @property
    def P(self):
        if self._p is None:
            self._p = np.load(self.dir / "predictions.npz")
        return self._p

    def json(self, name: str) -> Dict:
        return json.loads((self.dir / name).read_text(encoding="utf-8"))


def select_runs(schedule: str, regimes: List[str], hash_: Optional[str]) -> Tuple[Dict, str]:
    clean = []
    for p in sorted(config.RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        meta = m.get("meta", {})
        if meta.get("schedule") == schedule and meta.get("dirty") is False:
            clean.append(Run(p.parent, meta, m))
    hashes = sorted({r.meta["git_hash"] for r in clean})
    if hash_ is None:
        if len(hashes) != 1:
            sys.exit(f"{schedule}: runs at {hashes}; choose one with --hash-{schedule}")
        hash_ = hashes[0]
    chosen: Dict[Tuple, Run] = {}
    for r in clean:
        if r.meta["git_hash"] != hash_:
            continue
        key = (r.meta["track"], r.meta["regime"], r.meta["seed"])
        if key in chosen:
            sys.exit(f"{schedule}: two runs for {key} at {hash_}")
        chosen[key] = r
    missing = [(t, g, s) for t in TRACKS for g in regimes for s in SEEDS
               if (t, g, s) not in chosen]
    if missing:
        sys.exit(f"{schedule}: missing runs at {hash_}: {missing}")
    return chosen, hash_


class SplitInfo:
    def __init__(self, seed: int):
        split = load_split(seed)
        self.ci = class_index(split)
        self.names = {i: r for r, i in self.ci.items()}
        self.task_of = {self.ci[r]: t for r, t in split["relation_to_task"].items()}
        self.K = split["k"]
        self.task_classes = [sorted(c for c, t in self.task_of.items() if t == k)
                             for k in range(self.K)]
        self.degenerate = self.task_of[self.ci[DEGENERATE_CLASS]]

    def seen(self, t: int) -> List[int]:
        return sorted(c for c, k in self.task_of.items() if k <= t)


def write_table(name: str, title: str, header: List[str], rows: List[List],
                notes: List[str], csv_rows: Optional[List[Dict]] = None) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    md = [f"# {title}", "", "Generated by `continual/make_tables.py` from saved run files. "
          "Do not edit by hand.", ""]
    md += [f"| {' | '.join(header)} |", f"|{'|'.join('---' for _ in header)}|"]
    md += [f"| {' | '.join(fmt(v) for v in row)} |" for row in rows]
    md += [""] + [f"- {n}" for n in notes]
    (TABLE_DIR / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out = csv_rows if csv_rows is not None else [dict(zip(header, r)) for r in rows]
    fields = list(dict.fromkeys(k for row in out for k in row))   # union, first-seen order
    with open(TABLE_DIR / f"{name}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)


def f(v: Optional[float], signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def real_difference(diffs: List[float]) -> bool:
    """|diff| > THRESH on >= 2 of 3 seeds with the same sign."""
    pos = sum(d > THRESH for d in diffs)
    neg = sum(d < -THRESH for d in diffs)
    return pos >= 2 or neg >= 2


# ------------------------------------------------------------- derived

def per_class_f1_seen(run: Run, info: SplitInfo, t: int) -> Dict[int, float]:
    y = run.P["test_gold"]
    seen = info.seen(t)
    idx = np.flatnonzero(np.isin(y, seen))
    _, pc = macro_f1(y[idx], run.P["test_pred"][t][idx], len(info.names))
    return pc


def class_first_final(run: Run, info: SplitInfo, cls: str) -> Tuple[int, float, float]:
    c = info.ci[cls]
    t0 = info.task_of[c]
    first = per_class_f1_seen(run, info, t0).get(c, 0.0)
    final = per_class_f1_seen(run, info, info.K - 1).get(c, 0.0)
    return t0, first, final


def pair_drop(run: Run, info: SplitInfo, a: str, b: str) -> Dict:
    ta, fa, ea = class_first_final(run, info, a)
    tb, fb, eb = class_first_final(run, info, b)
    earlier = a if ta < tb else b if tb < ta else None
    drop = None
    if earlier is not None:
        drop = (fa - ea) if earlier == a else (fb - eb)
    return {"cross_task": ta != tb, "task_a": ta, "task_b": tb,
            "earlier": earlier, "drop_earlier": drop}


def broad_class() -> Tuple[str, Dict[str, Tuple[int, int]]]:
    feats = build_features(config.CORPUS)
    names = list(class_index(load_split(SEEDS[0])))
    tc, gc = Counter(feats["y_teacher"].tolist()), Counter(feats["y_gold"].tolist())
    counts = {n: (tc.get(DIALOGRE_RELATIONS.index(n), 0), gc.get(DIALOGRE_RELATIONS.index(n), 0))
              for n in names}
    return max(names, key=lambda n: counts[n][0] / counts[n][1]), counts


def acc_after_each_task(run: Run) -> List[float]:
    F = run.R["R_f1"]
    return [float(np.mean([F[t][j] for j in range(t + 1)])) for t in range(len(F))]


# --------------------------------------------------------------- tables

def table_T1(P: Dict, infos: Dict[int, SplitInfo], metric: str) -> None:
    """T1 (metric='f1', primary) and T1b (metric='acc', secondary)."""
    sfx = "f1" if metric == "f1" else "acc"
    header = ["track", "regime"]
    for s in SEEDS:
        header += [f"{s} ACC", f"{s} BWT", f"{s} forget t0", f"{s} forget t1"]
    deg_row = ["degenerate task", ""]
    for s in SEEDS:
        deg_row += ["", "", "†" if infos[s].degenerate == 0 else "",
                    "†" if infos[s].degenerate == 1 else ""]
    rows, csv_rows = [deg_row], []
    for tr in TRACKS:
        for rg in PRIMARY:
            row = [tr, rg]
            for s in SEEDS:
                m = P[(tr, rg, s)].m
                fg = m[f"forgetting_{sfx}"]
                row += [f(m[f"ACC_{sfx}"]), f(m[f"BWT_{sfx}"], True),
                        f(fg["0"], True) + (" †" if infos[s].degenerate == 0 else ""),
                        f(fg["1"], True) + (" †" if infos[s].degenerate == 1 else "")]
                csv_rows.append({"track": tr, "regime": rg, "seed": s,
                                 "degenerate_task": infos[s].degenerate,
                                 f"ACC_{sfx}": m[f"ACC_{sfx}"], f"BWT_{sfx}": m[f"BWT_{sfx}"],
                                 f"forgetting_t0_{sfx}": fg["0"], f"forgetting_t1_{sfx}": fg["1"],
                                 f"mean_forgetting_nondegenerate_{sfx}":
                                     m[f"mean_forgetting_nondegenerate_{sfx}"],
                                 "run_dir": P[(tr, rg, s)].dir.name})
            rows.append(row)
    deg = ", ".join(f"seed {s}: task {infos[s].degenerate}" for s in SEEDS)
    name, title = (("T1_main", "T1 — ACC_f1, BWT_f1 and per-task forgetting (primary)")
                   if metric == "f1" else
                   ("T1b_accuracy", "T1b — ACC_acc, BWT_acc and per-task forgetting (secondary)"))
    write_table(name, title, header, rows, [
        f"Degenerate task (holds {DEGENERATE_CLASS}), marked †: {deg}. Tasks numbered from 0.",
        "forget tj = R[K-1][j] - R[j][j]; negative = forgetting. Forgetting conclusions come "
        "from the non-degenerate tasks (PREREG 9).",
        "BWT is the mean of the forget columns. It is not read as forgetting when the "
        "degenerate task isn't last (seeds 1234 and 1).",
        "No cross-seed mean ± std: the degenerate task moves between seeds.",
        "ACC/BWT definitions unchanged (continual/metrics.py)."], csv_rows)


def table_T2(P: Dict, infos: Dict[int, SplitInfo]) -> None:
    header = ["track", "regime"] + [f"{s} after t{t}" for s in SEEDS for t in range(3)] + \
             [f"{s} avg" for s in SEEDS]
    rows, csv_rows = [], []
    for tr in TRACKS:
        for rg in PRIMARY:
            obs = {s: P[(tr, rg, s)].m["observed_accuracy_after_task"] for s in SEEDS}
            avg = {s: P[(tr, rg, s)].m["average_observed_accuracy"] for s in SEEDS}
            rows.append([tr, rg] + [f(obs[s][t]) for s in SEEDS for t in range(3)] +
                        [f(avg[s]) for s in SEEDS])
            for s in SEEDS:
                csv_rows.append({"track": tr, "regime": rg, "seed": s,
                                 **{f"after_task_{t}": obs[s][t] for t in range(3)},
                                 "average": avg[s]})
    write_table("T2_observed_accuracy", "T2 — accuracy on all observed relations after each task "
                "(Zhao Table 2 style)", header, rows, [
                    "Accuracy over the pooled test examples of every seen task (sample weighted).",
                    "avg = mean over the three stages (Zhao's 'average accuracy')."], csv_rows)


def table_T3(P: Dict) -> None:
    joint = {s: json.loads((config.RESULTS_DIR / f"joint_baseline_seed{s}.json")
                           .read_text(encoding="utf-8"))["tracks"] for s in SEEDS}
    header = ["track", "regime"] + [str(s) for s in SEEDS]
    rows, csv_rows = [], []
    for tr in TRACKS:
        for rg in PRIMARY:
            v = {s: P[(tr, rg, s)].m["final_macro_f1_all"] for s in SEEDS}
            rows.append([tr, rg] + [f(v[s]) for s in SEEDS])
            csv_rows += [{"track": tr, "regime": rg, "seed": s, "final_macro_f1_all": v[s]}
                         for s in SEEDS]
        rows.append([tr, "joint (upper bound)"] +
                    [f(joint[s][tr]["test_macro_f1"]) for s in SEEDS])
        csv_rows += [{"track": tr, "regime": "joint", "seed": s,
                      "final_macro_f1_all": joint[s][tr]["test_macro_f1"]} for s in SEEDS]
    write_table("T3_final_macro_f1", "T3 — final macro-F1 over all 17 classes, with joint "
                "upper bounds", header, rows, [
                    "Final model, full test set (1,144 examples), seen mask = all 17 classes.",
                    "Joint = joint-training baseline for the same seed (results/joint_baseline_seed*.json)."],
                csv_rows)


def table_T4(P: Dict, infos: Dict[int, SplitInfo]) -> Dict:
    """Per-class first/final/drop, and pair drops. Returns pair drops for PREREG 4."""
    header = ["track", "seed", "regime", "class", "learned at", "F1 first", "F1 final", "drop"]
    rows, csv_rows = [], []
    for tr in TRACKS:
        for s in SEEDS:
            for rg in PRIMARY:
                for cls in T4_CLASSES:
                    t0, first, final = class_first_final(P[(tr, rg, s)], infos[s], cls)
                    rows.append([tr, s, rg, cls, f"task {t0}", f(first), f(final),
                                 f(first - final, True)])
                    csv_rows.append({"track": tr, "seed": s, "regime": rg, "class": cls,
                                     "learned_task": t0, "f1_first": round(first, 6),
                                     "f1_final": round(final, 6),
                                     "drop": round(first - final, 6)})
    write_table("T4_analogous_classes", "T4a — per-class F1 for the analogous-pair classes",
                header, rows, [
                    "F1 over the pooled seen test set, at the task where the class is learned "
                    "and after the last task. drop = first - final (positive = lost F1)."],
                csv_rows)

    header = ["track", "seed", "regime", "pair", "tasks", "cross-task", "earlier member",
              "drop of earlier"]
    rows, csv_rows = [], []
    drops: Dict[Tuple, List[float]] = defaultdict(list)
    for tr in TRACKS:
        for s in SEEDS:
            for rg in PRIMARY:
                for a, b in T4_PAIRS:
                    d = pair_drop(P[(tr, rg, s)], infos[s], a, b)
                    ref = (a, b) == T4_PAIRS[-1]
                    rows.append([tr, s, rg, f"{a} ↔ {b}" + (" (reference)" if ref else ""),
                                 f"{d['task_a']}/{d['task_b']}", "yes" if d["cross_task"] else "no",
                                 d["earlier"] or "—", f(d["drop_earlier"], True)])
                    csv_rows.append({"track": tr, "seed": s, "regime": rg, "a": a, "b": b,
                                     "task_a": d["task_a"], "task_b": d["task_b"],
                                     "cross_task": d["cross_task"], "earlier": d["earlier"],
                                     "drop_earlier": None if d["drop_earlier"] is None
                                     else round(d["drop_earlier"], 6)})
                    if d["cross_task"]:
                        drops[(tr, s, rg)].append(d["drop_earlier"])
    write_table("T4_analogous_pairs", "T4b — analogous pairs: drop of the earlier-learned member",
                header, rows, [
                    "A pair's drop is the drop of its earlier-learned member, the one exposed to "
                    "its analogue arriving later. Within-task pairs have no earlier member.",
                    "per:girl/boyfriend ↔ per:negative_impression is within-task on every seed "
                    "(reference row)."], csv_rows)
    return {k: float(np.mean(v)) for k, v in drops.items()}


def table_T5(P: Dict, infos: Dict[int, SplitInfo]) -> Dict:
    """Teacher-track buffer purity and broad-class share. Returns share data for PREREG 6."""
    broad, counts = broad_class()
    feats = build_features(config.CORPUS)
    header = ["seed", "task", "memory selected at task", "of which teacher ≠ gold",
              "purity (selected)", "cumulative memory", "cumulative purity"]
    rows, csv_rows = [], []
    for s in SEEDS:
        info = infos[s]
        mem = {rg: P[("teacher", rg, s)].json("memory_indices.json")["classes"]
               for rg in ("replay", "fkd")}
        assert {k: v["train_rows"] for k, v in mem["replay"].items()} == \
               {k: v["train_rows"] for k, v in mem["fkd"].items()}, "memory differs"
        cum_n = cum_bad = 0
        for t in range(info.K):
            rows_t = [fr for name, v in mem["replay"].items()
                      if info.task_of[info.ci[name]] == t for fr in v["feature_rows"]]
            bad = int(sum(feats["y_teacher"][r] != feats["y_gold"][r] for r in rows_t))
            cum_n, cum_bad = cum_n + len(rows_t), cum_bad + bad
            rows.append([s, t, len(rows_t), bad, f(1 - bad / len(rows_t)), cum_n,
                         f(1 - cum_bad / cum_n)])
            csv_rows.append({"seed": s, "task": t, "selected": len(rows_t), "mismatched": bad,
                             "purity_selected": round(1 - bad / len(rows_t), 6),
                             "cumulative": cum_n,
                             "purity_cumulative": round(1 - cum_bad / cum_n, 6)})
    write_table("T5a_buffer_purity", "T5a — teacher track: memory buffer purity", header, rows, [
        "Purity = fraction of memory samples whose teacher label equals the gold label "
        "(gold outside the 17 kept classes counts as a mismatch).",
        "Replay and FKD select identical memories (asserted)."], csv_rows)

    header = ["seed", "regime", "learned at"] + [f"share after t{t}" for t in range(3)] + \
             [f"gold share t{t}" for t in range(3)]
    rows, csv_rows, shares = [], [], {}
    for s in SEEDS:
        info = infos[s]
        c = info.ci[broad]
        for rg in PRIMARY:
            run = P[("teacher", rg, s)]
            y = run.P["test_gold"]
            sh, gsh = [], []
            for t in range(info.K):
                idx = np.flatnonzero(np.isin(y, info.seen(t)))
                sh.append(float(np.mean(run.P["test_pred"][t][idx] == c)))
                gsh.append(float(np.mean(y[idx] == c)))
            shares[(s, rg)] = (info.task_of[c], sh)
            rows.append([s, rg, f"task {info.task_of[c]}"] + [f(v) for v in sh] +
                        [f(v) for v in gsh])
            csv_rows.append({"seed": s, "regime": rg, "learned_task": info.task_of[c],
                             **{f"share_t{t}": round(sh[t], 6) for t in range(3)},
                             **{f"gold_share_t{t}": round(gsh[t], 6) for t in range(3)}})
    tcount, gcount = counts[broad]
    write_table("T5b_broad_class_share", f"T5b — teacher track: share of predictions to the broad "
                f"class ({broad})", header, rows, [
                    f"Broad class = largest teacher/gold train-count ratio among the 17 kept "
                    f"classes: {broad}, {tcount} teacher vs {gcount} gold training labels.",
                    "Share = fraction of the pooled seen test set predicted as the broad class; "
                    "gold share = fraction whose gold label is the broad class."], csv_rows)
    return {"broad": broad, "shares": shares}


def diag_summary(run: Run) -> Dict:
    g = run.json("fkd_diagnostics.json")["tasks"]
    ent = np.mean([d["s"]["mean_row_entropy"] / d["s"]["uniform_entropy"] for d in g])
    smax = np.mean([d["s"]["mean_row_max"] / d["s"]["uniform_max"] for d in g])
    ratio_first = np.mean([d["epochs"][0]["lam_L_fkd"] / d["epochs"][0]["L_cls"] for d in g])
    ratio_last = np.mean([d["epochs"][-1]["lam_L_fkd"] / d["epochs"][-1]["L_cls"] for d in g])
    foc = [d["epochs"][0].get("focal_mean") for d in g]
    foc_last = [d["epochs"][-1].get("focal_mean") for d in g]
    return {"s_entropy_frac": float(ent), "s_max_over_uniform": float(smax),
            "lamL_over_Lcls_first": float(ratio_first), "lamL_over_Lcls_last": float(ratio_last),
            "focal_first": None if foc[0] is None else float(np.mean(foc)),
            "focal_last": None if foc_last[0] is None else float(np.mean(foc_last)),
            "weighting": g[0]["weighting"]}


def table_D1(P: Dict) -> None:
    header = ["track", "seed", "s entropy / uniform", "s max / uniform", "focal first→last",
              "λ·L_fkd / L_cls first→last"]
    rows, csv_rows = [], []
    for tr in TRACKS:
        for s in SEEDS:
            d = diag_summary(P[(tr, "fkd", s)])
            rows.append([tr, s, f"{d['s_entropy_frac']:.1%}", f"{d['s_max_over_uniform']:.2f}×",
                         f"{d['focal_first']:.2f} → {d['focal_last']:.2f}",
                         f"{d['lamL_over_Lcls_first']:.1%} → {d['lamL_over_Lcls_last']:.1%}"])
            csv_rows.append({"track": tr, "seed": s, **d})
    write_table("D1_fkd_diagnostics", "D1 — FKD diagnostics, primary runs (§3.7)", header, rows, [
        "Averaged over the replay stages (tasks 1 and 2). s entropy as a fraction of the uniform "
        "maximum; s max as a multiple of 1/|OLD|.",
        "Per-epoch values and the analogous-pair s are in each run's fkd_diagnostics.json."],
        csv_rows)


def table_T6(Fx: Dict, infos: Dict[int, SplitInfo]) -> None:
    header = ["track", "seed", "regime", "ACC_f1", "forget (non-degenerate)", "mean forget",
              "s entropy / uniform", "focal first→last", "λ·L / L_cls first→last"]
    rows, csv_rows = [], []
    for tr in TRACKS:
        for s in SEEDS:
            info = infos[s]
            nondeg = [j for j in range(info.K - 1) if j != info.degenerate]
            for rg in FIXED:
                run = Fx[(tr, rg, s)]
                m = run.m
                fg = ", ".join(f"t{j} {f(m['forgetting_f1'][str(j)], True)}" for j in nondeg)
                d = diag_summary(run) if rg in ("fkd", "kd") else None
                rows.append([tr, s, rg, f(m["ACC_f1"]), fg,
                             f(m["mean_forgetting_nondegenerate_f1"], True),
                             "—" if d is None else (f"{d['s_entropy_frac']:.1%}"
                                                    + (" (unused)" if rg == "kd" else "")),
                             "—" if d is None or d["focal_first"] is None else
                             f"{d['focal_first']:.2f} → {d['focal_last']:.2f}",
                             "—" if d is None else
                             f"{d['lamL_over_Lcls_first']:.1%} → {d['lamL_over_Lcls_last']:.1%}"])
                csv_rows.append({"track": tr, "seed": s, "regime": rg, "ACC_f1": m["ACC_f1"],
                                 "BWT_f1": m["BWT_f1"],
                                 **{f"forgetting_t{j}_f1": m["forgetting_f1"][str(j)]
                                    for j in range(info.K - 1)},
                                 "nondegenerate_tasks": "/".join(map(str, nondeg)),
                                 "mean_forgetting_nondegenerate_f1":
                                     m["mean_forgetting_nondegenerate_f1"],
                                 **({} if d is None else d), "run_dir": run.dir.name})
    write_table("T6_fixed_schedule", "T6 — fixed 10-epoch replay stage (declared ablation, §4.4)",
                header, rows, [
                    "Replay stage runs exactly 10 epochs, no dev-based selection, final weights "
                    "kept. Everything else as in the primary runs.",
                    "kd = fkd with w ≡ 1 (no s, no focal factor); its s is computed but unused.",
                    "Forgetting only on non-degenerate old tasks (PREREG 9)."], csv_rows)


# ---------------------------------------------------------------- PREREG

def prereg(P: Dict, Fx: Dict, pair_drops: Dict, t5: Dict, infos: Dict) -> None:
    out = []

    def add(line, prediction, outcome, verdict):
        out.append({"line": line, "prediction": prediction, "outcome": outcome,
                    "verdict": verdict})

    joint = {s: json.loads((config.RESULTS_DIR / f"joint_baseline_seed{s}.json")
                           .read_text(encoding="utf-8"))["tracks"] for s in SEEDS}
    ratio = {s: joint[s]["teacher"]["test_macro_f1"] / joint[s]["gold"]["test_macro_f1"]
             for s in SEEDS}
    dev = {s: abs(ratio[s] - ratio[1234]) for s in (1, 2)}
    add(0, "teacher/gold joint ratio within 0.05 of seed 1234's on seeds 1 and 2",
        ", ".join(f"seed {s}: {ratio[s]:.3f} (Δ {dev[s]:.3f})" for s in (1, 2)),
        "holds" if all(v <= 0.05 for v in dev.values()) else "fails")

    beats = {(tr, s): P[(tr, "replay", s)].m["ACC_f1"] > P[(tr, "naive", s)].m["ACC_f1"]
             for tr in TRACKS for s in SEEDS}
    add(1, "replay beats naive on final ACC_f1 in all 6 track × seed runs",
        f"{sum(beats.values())} of 6", "holds" if all(beats.values()) else "fails")

    diffs = {tr: [P[(tr, "fkd", s)].m["ACC_f1"] - P[(tr, "replay", s)].m["ACC_f1"]
                  for s in SEEDS] for tr in TRACKS}
    for line, tr in ((2, "gold"), (3, "teacher")):
        inside = [abs(d) <= THRESH for d in diffs[tr]]
        add(line, f"{tr}: FKD − replay final ACC_f1 within ±0.03 on every seed",
            ", ".join(f"{s}: {d:+.3f}" for s, d in zip(SEEDS, diffs[tr])) +
            f" — real difference by the threshold rule: {'yes' if real_difference(diffs[tr]) else 'no'}",
            "holds" if all(inside) else f"fails ({3 - sum(inside)} of 3 seeds outside)")

    counts = {}
    for tr in TRACKS:
        counts[tr] = sum(pair_drops[(tr, s, "replay")] - pair_drops[(tr, s, "fkd")] > 0.10
                         for s in SEEDS)
    add(4, "FKD's cross-task pair drop smaller than replay's by > 0.10 on at most 1 of 3 seeds, "
        "both tracks",
        "; ".join(f"{tr}: {counts[tr]} of 3 (" + ", ".join(
            f"{s}: {pair_drops[(tr, s, 'replay')] - pair_drops[(tr, s, 'fkd')]:+.3f}"
            for s in SEEDS) + ")" for tr in TRACKS),
        "holds" if all(c <= 1 for c in counts.values()) else "fails")

    real = {tr: real_difference(diffs[tr]) for tr in TRACKS}
    if not any(real.values()):
        v5 = "holds (no real gap in either track)"
    else:
        mg = {tr: float(np.mean(np.abs(diffs[tr]))) for tr in TRACKS}
        v5 = ("main clause fails; conditional clause " +
              ("holds" if mg["teacher"] < mg["gold"] else "fails") +
              f" (mean |gap| teacher {mg['teacher']:.3f} vs gold {mg['gold']:.3f})")
    add(5, "FKD − replay gap the same in both tracks (both within threshold); if a gap appears, "
        "smaller in teacher", f"real gap — gold: {real['gold']}, teacher: {real['teacher']}", v5)

    falls = []
    parts = []
    for s in SEEDS:
        t0, sh = t5["shares"][(s, "replay")]
        drop = sh[t0] - sh[-1] if t0 < 2 else None
        falls.append(drop is not None and drop > THRESH)
        parts.append(f"{s}: learned t{t0}, {sh[t0]:.3f} → {sh[-1]:.3f}")
    add(6, f"teacher, replay: broad-class ({t5['broad']}) share falls after the task where it is "
        "learned", "; ".join(parts), "holds" if sum(falls) >= 2 else "fails")

    fdiffs = {tr: [Fx[(tr, "fkd", s)].m["ACC_f1"] - Fx[(tr, "replay", s)].m["ACC_f1"]
                   for s in SEEDS] for tr in TRACKS}
    ok7 = {tr: sum(abs(d) <= THRESH for d in fdiffs[tr]) >= 2 for tr in TRACKS}
    add(7, "fixed schedule: FKD − replay final ACC_f1 within ±0.03 on ≥ 2 of 3 seeds, both tracks",
        "; ".join(f"{tr}: " + ", ".join(f"{s}: {d:+.3f}" for s, d in zip(SEEDS, fdiffs[tr]))
                  for tr in TRACKS),
        "holds" if all(ok7.values()) else "fails")

    kd = {tr: [Fx[(tr, "kd", s)].m["mean_forgetting_nondegenerate_f1"] -
               Fx[(tr, "replay", s)].m["mean_forgetting_nondegenerate_f1"] for s in SEEDS]
          for tr in TRACKS}
    ok8 = {tr: sum(d > THRESH for d in kd[tr]) >= 2 for tr in TRACKS}
    add(8, "fixed schedule: KD forgets less than replay on non-degenerate tasks (> 0.03, ≥ 2 of 3 "
        "seeds), both tracks",
        "; ".join(f"{tr}: " + ", ".join(f"{s}: {d:+.3f}" for s, d in zip(SEEDS, kd[tr]))
                  for tr in TRACKS) + " (kd − replay mean forgetting; positive = kd forgets less)",
        "holds" if all(ok8.values()) else
        ("fails" if not any(ok8.values()) else
         "partly: " + ", ".join(f"{tr} {'holds' if ok8[tr] else 'fails'}" for tr in TRACKS)))

    not_last = [s for s in SEEDS if infos[s].degenerate != infos[s].K - 1]
    add(9, "reporting rule: forgetting from non-degenerate tasks; BWT_f1 not read as forgetting "
        "when the degenerate task isn't last",
        f"applied in T1/T6; degenerate task not last on seeds {not_last}", "applied")

    header = ["line", "prediction", "outcome", "verdict"]
    write_table("prereg_outcomes", "PREREG lines 0–9 against outcomes", header,
                [[o[k] for k in header] for o in out], [
                    "Computed by make_tables.py with the definitions in its module docstring, "
                    "fixed before the sweep ran.",
                    "Seeds 1234 / 1 / 2. ACC_f1 and BWT_f1 primary (PREREG metric clarification)."],
                out)


# --------------------------------------------------------------- figures

INK, INK2, MUTED, GRID, BASE, SURFACE = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                         "#c3c2b7", "#fcfcfb")
REGIME_STYLE = {"naive": ("#2a78d6", "o"), "replay": ("#eb6834", "s"), "fkd": ("#1baf7a", "^")}
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
             "#184f95", "#0d366b"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_acc(P: Dict, infos: Dict[int, SplitInfo]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True, facecolor=SURFACE)
    for r, tr in enumerate(TRACKS):
        for c, s in enumerate(SEEDS):
            ax = axes[r][c]
            _style(ax)
            deg = infos[s].degenerate
            ax.axvspan(deg - 0.4, deg + 0.4, color=GRID, alpha=0.6, linewidth=0)
            ends = []
            for rg in PRIMARY:
                col, mk = REGIME_STYLE[rg]
                ys = acc_after_each_task(P[(tr, rg, s)])
                ax.plot(range(3), ys, color=col, linewidth=2, marker=mk, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=rg, zorder=3)
                ends.append((ys[-1], rg))
            # Direct end labels; regimes ending within 0.03 share one label so
            # coinciding lines (replay and fkd often do) don't overprint.
            groups: List[List] = []
            for y_end, rg in sorted(ends):
                if groups and y_end - groups[-1][-1][0] < 0.03:
                    groups[-1].append((y_end, rg))
                else:
                    groups.append([(y_end, rg)])
            for g in groups:
                y_mid = float(np.mean([v for v, _ in g]))
                label = " = ".join(rg for _, rg in g) + f" {y_mid:.2f}"
                ax.annotate(label, (2, y_mid), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"task {t}" + ("\n(degenerate)" if t == deg else "")
                                for t in range(3)])
            ax.set_xlim(-0.4, 3.3)
            ax.set_ylim(0, 0.75)
            ax.set_title(f"{tr} · seed {s}", fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel("ACC_f1 after task t", fontsize=9, color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("ACC_f1 after each task (mean of R row t)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "acc_after_each_task.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_confusion(P: Dict, infos: Dict[int, SplitInfo]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    s, info = 1234, infos[1234]
    short = lambda n: n.split(":", 1)[1]
    for tr in TRACKS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 13), facecolor=SURFACE,
                                 layout="constrained",
                                 gridspec_kw={"width_ratios": [7, 17]})
        for r, rg in enumerate(("replay", "fkd")):
            run = P[(tr, rg, s)]
            y = run.P["test_gold"]
            for c, t in enumerate((0, info.K - 1)):
                ax = axes[r][c]
                cls = info.seen(t)
                idx = np.flatnonzero(np.isin(y, cls))
                pred = run.P["test_pred"][t][idx]
                pos = {k: i for i, k in enumerate(cls)}
                M = np.zeros((len(cls), len(cls)))
                for g, p in zip(y[idx], pred):
                    M[pos[g], pos[p]] += 1
                M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
                im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
                ax.set_xticks(range(len(cls)))
                ax.set_yticks(range(len(cls)))
                ax.set_xticklabels([short(info.names[k]) for k in cls], rotation=90, fontsize=7,
                                   color=INK2)
                ax.set_yticklabels([short(info.names[k]) for k in cls], fontsize=7, color=INK2)
                ax.set_title(f"{rg} · after task {t}", fontsize=10, color=INK, loc="left")
                ax.set_xlabel("predicted", fontsize=8, color=MUTED)
                ax.set_ylabel("gold", fontsize=8, color=MUTED)
                for sp in ax.spines.values():
                    sp.set_color(BASE)
        cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
        cb.set_label("row-normalised (recall per gold class)", fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        fig.suptitle(f"Confusion matrices, {tr} track, seed 1234 (pooled seen test set)",
                     x=0.01, ha="left", fontsize=12, color=INK)
        fig.savefig(FIG_DIR / f"confusion_seed1234_{tr}.png", dpi=150, facecolor=SURFACE,
                    bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Phase 7 tables and figures")
    ap.add_argument("--hash-primary", default=None)
    ap.add_argument("--hash-fixed", default=None)
    args = ap.parse_args()

    infos = {s: SplitInfo(s) for s in SEEDS}
    P, hp = select_runs("primary", PRIMARY, args.hash_primary)
    print(f"primary: {len(P)} runs at {hp}")
    table_T1(P, infos, "f1")
    table_T1(P, infos, "acc")
    table_T2(P, infos)
    table_T3(P)
    pair_drops = table_T4(P, infos)
    t5 = table_T5(P, infos)
    table_D1(P)
    figure_acc(P, infos)
    figure_confusion(P, infos)

    Fx, hf = select_runs("fixed", FIXED, args.hash_fixed)
    print(f"fixed:   {len(Fx)} runs at {hf}")
    table_T6(Fx, infos)
    prereg(P, Fx, pair_drops, t5, infos)
    print(f"tables -> {TABLE_DIR}   figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
