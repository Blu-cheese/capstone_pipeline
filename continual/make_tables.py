"""
Phase 7 tables and figures (PHASE6_7_SPEC §4.2-4.4, v4), generated only from
saved run files. No hand edits.

    venv/bin/python -m continual.make_tables

Runs are selected from metrics.json metadata, never from paths: per schedule,
every (track, regime, seed) must have exactly one clean run at one git hash
(default: the single hash shared by that schedule's clean runs; --hash-primary
/ --hash-fixed to choose when there are several).

Writes results/tables/{T1..T6, D1, prereg_outcomes}.{md,csv} and
results/figures/*.png, then the RQ3 tables (T7 to T11, prereg_outcomes_rq3)
and figures (rq3_*.png); the RQ3 definitions sit with the RQ3 code below.
--only phase7 / --only rq3 builds one half.

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
from continual.conditions import (
    DRAWN_TRACKS, REPLICATES, RQ3_REGIMES, RQ3_TRACKS, head_labels, load_condition, train_rows,
)
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


def select_runs(schedule: str, regimes: List[str], hash_: Optional[str],
                tracks: List[str] = TRACKS) -> Tuple[Dict, str]:
    """
    Phase 7 run selection. Filters on schedule AND track set (RQ3 Gate 1
    guard): RQ3 runs use the primary schedule too, at another hash, and would
    otherwise make the one-hash rule below refuse to build the Phase 7 tables.
    """
    clean = []
    for p in sorted(config.RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        meta = m.get("meta", {})
        if (meta.get("schedule") == schedule and meta.get("dirty") is False
                and meta.get("track") in tracks):
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

def phase7(hash_primary: Optional[str] = None, hash_fixed: Optional[str] = None) -> None:
    infos = {s: SplitInfo(s) for s in SEEDS}
    P, hp = select_runs("primary", PRIMARY, hash_primary)
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

    Fx, hf = select_runs("fixed", FIXED, hash_fixed)
    print(f"fixed:   {len(Fx)} runs at {hf}")
    table_T6(Fx, infos)
    prereg(P, Fx, pair_drops, t5, infos)
    print(f"tables -> {TABLE_DIR}   figures -> {FIG_DIR}")


# =================================================================== RQ3
#
# RQ3_SPEC §2.5 with the Gate 1 amendments: T7 to T11, figures F1 and F2, and
# the PREREG 10-16 scorecard. Reads saved files only: results/conditions/*.json
# and the run dirs whose metrics.json meta.spec is "RQ3_SPEC".
#
# Definitions, fixed BEFORE the RQ3 sweep so that PREREG lines 10 to 16 are
# measured by rules chosen without seeing results:
#
#   runs        clean RQ3 runs at one git hash. Replicate 0 must be complete (54
#               runs); replicates 1 and 2 are either complete or absent.
#   F(c,g,s,r)  final 17-class test macro-F1 (metrics.json final_macro_f1_all)
#               for condition c, regime g, seed s, replicate r.
#   S(c,g,s,r)  broad-class share: the fraction of the 1,144 test examples the
#               final model predicts as per:other_family (the T5 broad class).
#   gold share  the fraction of test examples whose gold label is that class.
#   references  full gold / teacher joint = joint_baseline_seed{s}.json; they
#               have no replicates.
#   Δ_replay    F(c,replay) − F(c,joint). Δ_bal = F(c,joint_balanced) − F(c,joint).
#   Δ_share     S(c,replay) − S(c,joint), and S(c,joint_balanced) − S(c,joint).
#               Every Δ is paired within a replicate (same init seed and draw).
#   real cell   |D(s, r=0)| > 0.03 and, if replicates ran, D(s, r) has a single
#               sign over r = 0, 1, 2: the replicate spread doesn't cross zero
#               (PREREG 16). Shares use the same 0.03, read as 3 points.
#   direction   real-positive / real-negative when at least 2 of 3 seeds are real
#               cells of that sign; otherwise "not real" (the 2-of-3 rule).
#   line 10     (a) D = F(gold_I,joint) − F(gold ref): real iff direction is not
#               "not real"; scored against line10_gold_I_real. (b) Point
#               predictions for that D and for F(gold_sub,joint) − F(gold ref):
#               each holds if |D(s,0) − predicted| <= 0.03 on >= 2 of 3 seeds.
#               (c) ratio F(teacher_I,joint) / F(gold_I,joint), replicate 0: holds
#               if within 0.05 of line10_ratio on >= 2 of 3 seeds (line 0's
#               tolerance). The line holds if (a), both (b) and (c) hold.
#   lines 11-12 D = F(teacher_I,joint) − F(rand_I or perm_I, joint), paired by
#               replicate. Observed: lower = real-negative, higher =
#               real-positive, same = not real. Holds if it equals the prediction.
#   line 13     Δ_replay direction per condition. Clauses: teacher_I and perm_I
#               real-positive; rand_I and gold_I "not real" (line13_controls =
#               not_real) or "not real-positive" (= not_real_positive).
#               Hypothesis outcome: holds (all four clauses), fails (none), or
#               partly (the set of conditions whose clause holds). The prediction
#               holds if it names the observed outcome, and for "partly" the same set.
#   line 14     teacher_I. A seed counts if its Δ_replay cell is real and positive
#               and Δ_bal / Δ_replay >= line14_pct / 100 in every replicate run
#               (ratio compared at 6 decimals, the precision metrics are stored
#               at). Holds if >= 2 of 3 seeds count. If Δ_replay is real-positive on
#               fewer than 2 seeds, the line fails: there is no replay gain to
#               recover.
#   line 15     (a) S(c,joint) > 2 × gold share in every replicate, for teacher_I
#               and perm_I, and <= 2 × gold share in every replicate for rand_I,
#               each on >= 2 of 3 seeds. (b) S(c,joint) − S(c,replay) >
#               line15_pp / 100 in every replicate, for teacher_I and perm_I, each
#               on >= 2 of 3 seeds. The line holds if (a) and (b) hold.
#   line 16     reporting rule: "applied", stating whether replicates ran.
#   framing     every branch of the PREREG map whose condition is met:
#               13 holds; 13 does not hold and rand_I's Δ_replay is real-positive;
#               14 holds; line 12 observed "same" (class-conditional structure)
#               or not (instance dependence matters too); Δ_share(replay) real
#               for teacher_I and perm_I while Δ_replay is "not real" for both.

RQ3_TAG = "RQ3_SPEC"
FIG_ORDER = ["gold_I", "gold_sub", "rand_I", "perm_I", "teacher_I"]   # clean -> random -> structured
CL_CONDS = ["teacher_I", "perm_I", "rand_I", "gold_I"]
BROAD = "per:other_family"
POINT_TOL, RATIO_TOL = 0.03, 0.05
RQ3_JOINT = ("joint", "joint_balanced")
RQ3_STYLE = {"joint": ("#6250d6", "D"), "joint_balanced": ("#1baf7a", "v"),
             "replay": REGIME_STYLE["replay"]}
SEED_MARKER = {1234: "o", 1: "s", 2: "^"}


def real_sign(vals: List[float], reps_ran: bool) -> int:
    """+1 / -1 for a real cell, 0 otherwise (PREREG 16 when replicates ran)."""
    d0 = vals[0]
    if abs(d0) <= THRESH:
        return 0
    if reps_ran and not (min(vals) > 0 or max(vals) < 0):
        return 0
    return 1 if d0 > 0 else -1


def direction(signs: List[int]) -> str:
    if sum(x > 0 for x in signs) >= 2:
        return "real-positive"
    if sum(x < 0 for x in signs) >= 2:
        return "real-negative"
    return "not real"


def select_rq3_runs(hash_: Optional[str] = None):
    clean = []
    for p in sorted(config.RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        meta = m.get("meta", {})
        if meta.get("spec") == RQ3_TAG and meta.get("dirty") is False:
            clean.append(Run(p.parent, meta, m))
    if not clean:
        return None, None, []
    hashes = sorted({r.meta["git_hash"] for r in clean})
    if hash_ is None:
        if len(hashes) != 1:
            sys.exit(f"rq3: runs at {hashes}; choose one with --hash-rq3")
        hash_ = hashes[0]
    chosen: Dict[Tuple, Run] = {}
    for r in clean:
        if r.meta["git_hash"] != hash_:
            continue
        key = (r.meta["track"], r.meta["regime"], r.meta["seed"], r.meta["replicate"])
        if key in chosen:
            sys.exit(f"rq3: two runs for {key} at {hash_}")
        chosen[key] = r
    reps = sorted({k[3] for k in chosen})
    if reps not in ([0], list(REPLICATES)):
        sys.exit(f"rq3: replicates present {reps}; need [0] or {list(REPLICATES)}")
    grid = [(t, g, s) for t in RQ3_TRACKS for g in RQ3_REGIMES[t] for s in SEEDS]
    missing = [(t, g, s, r) for r in reps for (t, g, s) in grid if (t, g, s, r) not in chosen]
    if missing:
        sys.exit(f"rq3: missing runs at {hash_}: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    return chosen, hash_, reps


class RQ3:
    """Accessors over the selected RQ3 runs, per (condition, regime, seed, replicate)."""

    def __init__(self, runs: Dict, reps: List[int], infos: Dict[int, SplitInfo]):
        self.runs, self.reps, self.infos = runs, reps, infos
        self.ran = len(reps) > 1
        self.ref = {s: json.loads((config.RESULTS_DIR / f"joint_baseline_seed{s}.json")
                                  .read_text(encoding="utf-8"))["tracks"] for s in SEEDS}
        any_run = runs[(RQ3_TRACKS[0], "joint", SEEDS[0], 0)]
        y = any_run.P["test_gold"]
        self.gold_share = {s: float(np.mean(y == infos[s].ci[BROAD])) for s in SEEDS}

    def F(self, c, g, s, r):
        return self.runs[(c, g, s, r)].m["final_macro_f1_all"]

    def acc(self, c, g, s, r):
        m = self.runs[(c, g, s, r)].m
        return m["final_accuracy"] if g in RQ3_JOINT else m["observed_accuracy_after_task"][-1]

    def S(self, c, g, s, r):
        run = self.runs[(c, g, s, r)]
        return float(np.mean(run.P["test_pred"][-1] == self.infos[s].ci[BROAD]))

    def vals(self, fn, c, g, s):
        return [fn(c, g, s, r) for r in self.reps]

    def delta(self, fn, c, g_a, g_b, s):
        """fn(c, g_a) − fn(c, g_b), paired by replicate."""
        return [fn(c, g_a, s, r) - fn(c, g_b, s, r) for r in self.reps]

    def d_replay(self, c, s):
        return self.delta(self.F, c, "replay", "joint", s)

    def d_bal(self, c, s):
        return self.delta(self.F, c, "joint_balanced", "joint", s)

    def d_share_replay(self, c, s):
        return self.delta(self.S, c, "replay", "joint", s)

    def d_share_bal(self, c, s):
        return self.delta(self.S, c, "joint_balanced", "joint", s)

    def sign(self, vals):
        return real_sign(vals, self.ran)

    def dirn(self, fn, c):
        return direction([self.sign(fn(c, s)) for s in SEEDS])


def fv(vals: List[float], signed: bool = False, pct: bool = False) -> str:
    """Replicate-0 value, with the replicate range in brackets when replicates ran."""
    k = 100.0 if pct else 1.0
    one = (lambda v: f"{k * v:+.1f}" if signed else f"{k * v:.1f}") if pct else \
          (lambda v: f(v, signed))
    s = one(vals[0])
    if len(vals) > 1:
        s += f" [{one(min(vals))}, {one(max(vals))}]"
    return s


def _confusion(g: np.ndarray, t: np.ndarray, n: int) -> np.ndarray:
    C = np.zeros((n, n), dtype=int)
    np.add.at(C, (g, t), 1)
    return C


def table_T7(infos: Dict[int, SplitInfo]) -> None:
    feats = build_features(config.CORPUS)
    header = ["condition", "seed", "draw", "rows", "noise vs gold", f"{_short(BROAD)} rows",
              "task 0", "task 1", "task 2", "classes < 15 rows", "C = teacher_I's"]
    rows, csv_rows = [], []

    def per_task(info, y):
        return [int(np.isin(y, info.task_classes[k]).sum()) for k in range(info.K)]

    def add(name, s, draw, y, gold, c_same=""):
        info = infos[s]
        pt = per_task(info, y)
        cc = Counter(y.tolist())
        under = [f"{_short(info.names[c])} {cc.get(c, 0)}" for c in sorted(info.names)
                 if cc.get(c, 0) < 15]
        noise = float(np.mean(y != gold))
        nb = int((y == info.ci[BROAD]).sum())
        rows.append([name, s, draw, len(y), f"{noise:.4f}", nb] +
                    [f"{v}" + (" †" if k == info.degenerate else "") for k, v in enumerate(pt)] +
                    [", ".join(under) or "—", c_same])
        csv_rows.append({"condition": name, "seed": s, "draw_seed": draw, "rows": len(y),
                         "noise_rate_vs_gold": round(noise, 6), "broad_class_rows": nb,
                         **{f"task_{k}": v for k, v in enumerate(pt)},
                         "degenerate_task": info.degenerate,
                         "classes_under_15": "; ".join(under), "C_equals_teacher_I": c_same})

    for s in SEEDS:
        ci = infos[s].ci
        for tr in ("gold", "teacher"):
            ids, y = train_rows(tr, feats, ci)
            add(f"{tr} (full, reference)", s, "—", y, head_labels(feats["y_gold"], ci)[ids])
        for tr in RQ3_TRACKS:
            for r in (REPLICATES if tr in DRAWN_TRACKS else (0,)):
                p = load_condition(tr, s, r)
                y, gold = np.array(p["labels"]), np.array(p["gold"])
                c_same = ""
                if tr == "perm_I":
                    ref = load_condition("teacher_I", s, 0)
                    assert p["row_ids"] == ref["row_ids"]
                    same = np.array_equal(_confusion(gold, y, len(ci)),
                                          _confusion(np.array(ref["gold"]),
                                                     np.array(ref["labels"]), len(ci)))
                    assert same, f"perm_I seed {s} r{r}: C differs from teacher_I's"
                    c_same = "yes (asserted)"
                if tr == "rand_I":
                    assert int((y != gold).sum()) == p["n_differ_from_gold"]
                add(tr, s, p["draw_seed"] if p["draw_seed"] is not None else "—", y, gold, c_same)

    deg = ", ".join(f"seed {s}: task {infos[s].degenerate}" for s in SEEDS)
    write_table("T7_condition_stats", "T7 — RQ3 label conditions (no training results)",
                header, rows, [
                    "Built from results/conditions/*.json (the committed draws the runs check "
                    "against); reference rows from the frozen features.",
                    "noise vs gold = fraction of rows whose training label differs from gold; for "
                    "the full teacher track a gold label outside the 17 classes counts as a "
                    "difference.",
                    f"Rows per task follow each row's training label. † = degenerate task "
                    f"(holds {DEGENERATE_CLASS}): {deg}.",
                    "draw = numpy default_rng seed of the draw (seed + 100r for replicate r); "
                    "gold_I and teacher_I have no draw.",
                    "perm_I: its confusion matrix against gold equals teacher_I's exactly "
                    "(asserted per draw)."], csv_rows)

    # T7b: the full teacher track's noise, broken down (Gate 1 amendment).
    tn = np.array([DIALOGRE_RELATIONS[i] for i in feats["y_teacher"]])
    gn = np.array([DIALOGRE_RELATIONS[i] for i in feats["y_gold"]])
    kept = list(infos[SEEDS[0]].ci)
    t_in, g_in = np.isin(tn, kept), np.isin(gn, kept)
    unans = t_in & (gn == "unanswerable")
    sub = t_in & ~g_in & (gn != "unanswerable")
    lost = g_in & ~t_in
    top = lambda mask, arr, k: ", ".join(f"{_short(n) if ':' in n else n} {v}"
                                         for n, v in Counter(arr[mask].tolist()).most_common(k))
    lost_c = Counter(tn[lost].tolist())
    lost_other = int(lost.sum()) - lost_c["per:acquaintance"] - lost_c["unanswerable"]
    b_rows = [
        ["teacher label = gold", int((t_in & (tn == gn)).sum()), ""],
        ["in-space error (gold in the 17, teacher ≠ gold)", int((t_in & g_in & (tn != gn)).sum()),
         "the confusion matrix C (results/diagnostics/row_accounting.md)"],
        ["gold = unanswerable", int(unans.sum()), "teacher labels: " + top(unans, tn, 4)],
        ["gold = a sub-threshold class", int(sub.sum()), "gold: " + top(sub, gn, 10)],
        ["teacher track (sum)", int(t_in.sum()), ""],
        ["gold-track rows lost (teacher label outside the 17)", int(lost.sum()),
         f"teacher labels: acquaintance {lost_c['per:acquaintance']}, unanswerable "
         f"{lost_c['unanswerable']}, other {lost_other}"]]
    assert b_rows[0][1] + b_rows[1][1] + b_rows[2][1] + b_rows[3][1] == b_rows[4][1]
    write_table("T7b_teacher_noise_breakdown", "T7b — full teacher track: where its noise comes "
                "from", ["component", "rows", "detail"], b_rows, [
                    "From the frozen features; the same for every seed.",
                    "The first four rows sum to the 879 teacher-track rows. RQ3's conditions "
                    "live on I (the first two rows, 710), so they test in-space noise only.",
                    "The last row is not part of the 879: gold-track rows the teacher track "
                    "never sees because the teacher's label is outside the 17 classes."])


def _short(name: str) -> str:
    return name.split(":", 1)[1] if ":" in name else name


def table_T8(R: RQ3) -> None:
    header = ["condition", "regime"] + [x for s in SEEDS for x in (f"{s} macro-F1", f"{s} acc")]
    rows, csv_rows = [], []
    for c in RQ3_TRACKS:
        for g in [g for g in RQ3_JOINT if g in RQ3_REGIMES[c]]:
            row = [c, g]
            for s in SEEDS:
                F_, A_ = R.vals(R.F, c, g, s), R.vals(R.acc, c, g, s)
                row += [fv(F_), f(A_[0])]
                for r, fv_, av in zip(R.reps, F_, A_):
                    csv_rows.append({"condition": c, "regime": g, "seed": s, "replicate": r,
                                     "final_macro_f1": fv_, "final_accuracy": av})
            rows.append(row)
    for tr, n in (("gold", "1,327"), ("teacher", "879")):
        row = [f"{tr} (full {n} rows, reference)", "joint"]
        for s in SEEDS:
            row += [f(R.ref[s][tr]["test_macro_f1"]), f(R.ref[s][tr]["test_accuracy"])]
            csv_rows.append({"condition": f"{tr} (reference)", "regime": "joint", "seed": s,
                             "replicate": 0, "final_macro_f1": R.ref[s][tr]["test_macro_f1"],
                             "final_accuracy": R.ref[s][tr]["test_accuracy"]})
        rows.append(row)
    write_table("T8_rq3_joint", "T8 — RQ3 joint and joint_balanced, final 17-class test scores",
                header, rows, [
                    "Replicate 0, with the range over replicates 0 to 2 in brackets "
                    "(init seed seed + 100r; rand_I, perm_I and gold_sub also redraw)." if R.ran
                    else "Replicate 0 only (no replicates ran).",
                    "Reference rows: the Phase 4 joint baselines (joint_baseline_seed*.json), on "
                    "the full tracks; no replicates.",
                    "gold_sub runs joint only (RQ3_SPEC §2.4)."], csv_rows)


def table_T9(R: RQ3) -> None:
    header = ["condition", "regime"] + [x for s in SEEDS for x in
                                        (f"{s} ACC_f1", f"{s} ACC_acc", f"{s} forget", f"{s} final F1")]
    rows, csv_rows = [], []
    for c in RQ3_TRACKS:
        for g in [g for g in ("naive", "replay") if g in RQ3_REGIMES[c]]:
            row = [c, g]
            for s in SEEDS:
                info = R.infos[s]
                nondeg = [j for j in range(info.K - 1) if j != info.degenerate]
                m0 = R.runs[(c, g, s, 0)].m
                row += [fv(R.vals(lambda *k: R.runs[k].m["ACC_f1"], c, g, s)),
                        f(m0["ACC_acc"]),
                        ", ".join(f"t{j} {f(m0['forgetting_f1'][str(j)], True)}" for j in nondeg),
                        fv(R.vals(R.F, c, g, s))]
                for r in R.reps:
                    m = R.runs[(c, g, s, r)].m
                    csv_rows.append({"condition": c, "regime": g, "seed": s, "replicate": r,
                                     "ACC_f1": m["ACC_f1"], "BWT_f1": m["BWT_f1"],
                                     "ACC_acc": m["ACC_acc"], "degenerate_task": info.degenerate,
                                     **{f"forgetting_t{j}_f1": m["forgetting_f1"][str(j)]
                                        for j in range(info.K - 1)},
                                     "mean_forgetting_nondegenerate_f1":
                                         m["mean_forgetting_nondegenerate_f1"],
                                     "final_macro_f1": m["final_macro_f1_all"],
                                     "run_dir": R.runs[(c, g, s, r)].dir.name})
            rows.append(row)
    deg = ", ".join(f"seed {s}: task {R.infos[s].degenerate}" for s in SEEDS)
    write_table("T9_rq3_cl", "T9 — RQ3 continual runs (naive, replay)", header, rows, [
        "ACC_f1 and final F1: replicate 0, with the replicate range in brackets." if R.ran
        else "Replicate 0 only (no replicates ran).",
        f"forget = R[K−1][j] − R[j][j] on the non-degenerate old tasks only (PREREG 9). "
        f"Degenerate task: {deg}. Every task's value is in the CSV.",
        "final F1 = final model, all 17 classes, full test set (T3's measure)."], csv_rows)


def table_T10(R: RQ3) -> None:
    cols = [("Δ_replay", R.d_replay, False), ("Δ_bal", R.d_bal, False),
            ("Δ_share replay", R.d_share_replay, True), ("Δ_share bal", R.d_share_bal, True)]
    header = ["condition", "seed"] + [c[0] for c in cols]
    rows, csv_rows = [], []
    for c in RQ3_TRACKS:
        for s in SEEDS:
            row, rec = [c, s], {"condition": c, "seed": s}
            for name, fn, pct in cols:
                if "joint_balanced" not in RQ3_REGIMES[c] and "bal" in name:
                    row.append("—")
                    continue
                v = fn(c, s)
                sg = R.sign(v)
                row.append(fv(v, signed=True, pct=pct) + (" real" if sg else ""))
                rec.update({f"{name} r{r}": round(x, 6) for r, x in zip(R.reps, v)})
                rec[f"{name} real"] = sg
            rows.append(row)
            csv_rows.append(rec)
        summ = [c, "2 of 3"]
        for name, fn, pct in cols:
            summ.append("—" if ("joint_balanced" not in RQ3_REGIMES[c] and "bal" in name)
                        else R.dirn(fn, c))
        rows.append(summ)
    write_table("T10_rq3_key_comparison", "T10 — the key comparison: replay and balancing "
                "against joint", header, rows, [
                    "Δ_replay = replay − joint, Δ_bal = joint_balanced − joint, final 17-class "
                    "macro-F1. Δ_share = the same differences in broad-class share "
                    f"({BROAD}), in percentage points.",
                    "real = |Δ| > 0.03 (3 points for shares)" +
                    (" and the replicate range doesn't cross zero (PREREG 16)." if R.ran else "."),
                    "Each condition's last row applies the 2-of-3 rule across seeds.",
                    "Differences are paired within a replicate (same init seed and draw)."],
                csv_rows)


def table_T11(R: RQ3) -> None:
    header = ["condition", "regime"] + [str(s) for s in SEEDS] + ["gold test share"]
    rows, csv_rows = [], []
    for c in RQ3_TRACKS:
        for g in RQ3_REGIMES[c]:
            vs = {s: R.vals(R.S, c, g, s) for s in SEEDS}
            rows.append([c, g] + [fv(vs[s], pct=True) for s in SEEDS] +
                        [f"{100 * R.gold_share[SEEDS[0]]:.1f}"])
            csv_rows += [{"condition": c, "regime": g, "seed": s, "replicate": r, "share": round(v, 6),
                          "gold_test_share": round(R.gold_share[s], 6)}
                         for s in SEEDS for r, v in zip(R.reps, vs[s])]
    write_table("T11_rq3_broad_share", f"T11 — share of test predictions to the broad class "
                f"({BROAD}), %", header, rows, [
                    "Final model, full test set (1,144). Replicate range in brackets." if R.ran
                    else "Final model, full test set (1,144). Replicate 0 only.",
                    "gold test share = fraction of test examples whose gold label is the broad "
                    "class (the same on every seed)."], csv_rows)


def prereg_rq3(R: RQ3) -> Dict:
    from continual.prereg_rq3 import head_prereg, rq3_section, parse_scoring
    pred = parse_scoring(rq3_section(head_prereg()))
    out = []

    def add(line, prediction, outcome, verdict):
        out.append({"line": line, "prediction": prediction, "outcome": outcome,
                    "verdict": verdict})

    seeds_ok = lambda xs: sum(bool(x) for x in xs) >= 2
    per_seed = lambda d: ", ".join(f"{s}: {fv(d[s], signed=True)}" for s in SEEDS)

    # line 10
    d_a = {s: [R.F("gold_I", "joint", s, r) - R.ref[s]["gold"]["test_macro_f1"] for r in R.reps]
           for s in SEEDS}
    d_b = {s: [R.F("gold_sub", "joint", s, r) - R.ref[s]["gold"]["test_macro_f1"] for r in R.reps]
           for s in SEEDS}
    ratio = {s: R.F("teacher_I", "joint", s, 0) / R.F("gold_I", "joint", s, 0) for s in SEEDS}
    dir_a = direction([R.sign(d_a[s]) for s in SEEDS])
    a_ok = (dir_a != "not real") == pred["line10_gold_I_real"]
    pa_ok = seeds_ok([abs(d_a[s][0] - pred["line10_gold_I_minus_gold"]) <= POINT_TOL for s in SEEDS])
    pb_ok = seeds_ok([abs(d_b[s][0] - pred["line10_gold_sub_minus_gold"]) <= POINT_TOL for s in SEEDS])
    c_ok = seeds_ok([abs(ratio[s] - pred["line10_ratio"]) <= RATIO_TOL for s in SEEDS])
    parts = {"(a) real/not real": a_ok, "(b) gold_I value": pa_ok, "(b) gold_sub value": pb_ok,
             "(c) ratio": c_ok}
    add(10, f"gold_I − gold {pred['line10_gold_I_minus_gold']:+.3f} "
            f"({'real' if pred['line10_gold_I_real'] else 'not real'}); gold_sub − gold "
            f"{pred['line10_gold_sub_minus_gold']:+.3f}; teacher_I / gold_I {pred['line10_ratio']:.2f}",
        f"gold_I − gold {per_seed(d_a)} ({dir_a}); gold_sub − gold {per_seed(d_b)}; ratio "
        + ", ".join(f"{s}: {ratio[s]:.3f}" for s in SEEDS),
        "holds" if all(parts.values()) else
        "fails: " + ", ".join(k for k, v in parts.items() if not v))

    # lines 11, 12
    obs_word = {"real-negative": "lower", "real-positive": "higher", "not real": "same"}
    observed12 = None
    for line, other in ((11, "rand_I"), (12, "perm_I")):
        d = {s: [R.F("teacher_I", "joint", s, r) - R.F(other, "joint", s, r) for r in R.reps]
             for s in SEEDS}
        obs = obs_word[direction([R.sign(d[s]) for s in SEEDS])]
        if line == 12:
            observed12 = obs
        add(line, f"joint: teacher_I {pred[f'line{line}']} than {other}" if pred[f"line{line}"] != "same"
            else f"joint: teacher_I the same as {other}",
            f"teacher_I − {other}: {per_seed(d)} → {obs}",
            "holds" if obs == pred[f"line{line}"] else "fails")

    # line 13
    dirs = {c: R.dirn(R.d_replay, c) for c in CL_CONDS}
    ctrl = ((lambda d: d == "not real") if pred["line13_controls"] == "not_real"
            else (lambda d: d != "real-positive"))
    clause = {"teacher_I": dirs["teacher_I"] == "real-positive",
              "perm_I": dirs["perm_I"] == "real-positive",
              "rand_I": ctrl(dirs["rand_I"]), "gold_I": ctrl(dirs["gold_I"])}
    held = tuple(sorted(c for c in CL_CONDS if clause[c]))
    outcome = (("holds", ()) if len(held) == len(CL_CONDS) else
               ("fails", ()) if not held else ("partly", held))
    show = lambda o: o[0] + (f": {', '.join(o[1])}" if o[0] == "partly" else "")
    add(13, f"hypothesis {show(pred['line13'])} (controls read as {pred['line13_controls']})",
        "Δ_replay: " + "; ".join(f"{c} {dirs[c]}" for c in CL_CONDS) +
        f" → hypothesis {show(outcome)}",
        "holds" if pred["line13"] == outcome else "fails")

    # line 14
    pct = pred["line14_pct"] / 100.0
    n_pos, counted, detail = 0, 0, []
    for s in SEEDS:
        vr, vb = R.d_replay("teacher_I", s), R.d_bal("teacher_I", s)
        if R.sign(vr) != 1:
            detail.append(f"{s}: Δ_replay not real-positive")
            continue
        n_pos += 1
        rec = [round(b / a, 6) for a, b in zip(vr, vb)]
        ok = all(x >= round(pct, 6) for x in rec)
        counted += ok
        detail.append(f"{s}: recovery {fv(rec)}")
    v14 = (f"fails: Δ_replay real-positive on {n_pos} of 3 seeds, no replay gain to recover"
           if n_pos < 2 else ("holds" if counted >= 2 else "fails"))
    add(14, f"teacher_I: Δ_bal ≥ {pred['line14_pct']:g}% of Δ_replay on ≥ 2 of 3 seeds",
        "; ".join(detail), v14)

    # line 15
    gs = R.gold_share
    a_tp = {c: seeds_ok([all(R.S(c, "joint", s, r) > 2 * gs[s] for r in R.reps) for s in SEEDS])
            for c in ("teacher_I", "perm_I")}
    a_rand = seeds_ok([all(R.S("rand_I", "joint", s, r) <= 2 * gs[s] for r in R.reps) for s in SEEDS])
    pp = pred["line15_pp"] / 100.0
    b_tp = {c: seeds_ok([all(R.S(c, "joint", s, r) - R.S(c, "replay", s, r) > pp for r in R.reps)
                         for s in SEEDS]) for c in ("teacher_I", "perm_I")}
    shares = "; ".join(f"{c} joint " + ", ".join(f"{s}: {fv(R.vals(R.S, c, 'joint', s), pct=True)}"
                                                 for s in SEEDS)
                       for c in ("teacher_I", "perm_I", "rand_I"))
    red = "; ".join(f"{c} joint − replay " + ", ".join(
        f"{s}: {fv([-x for x in R.d_share_replay(c, s)], signed=True, pct=True)}" for s in SEEDS)
        for c in ("teacher_I", "perm_I"))
    p15 = {"(a) teacher_I > 2×": a_tp["teacher_I"], "(a) perm_I > 2×": a_tp["perm_I"],
           "(a) rand_I ≤ 2×": a_rand, "(b) teacher_I reduction": b_tp["teacher_I"],
           "(b) perm_I reduction": b_tp["perm_I"]}
    add(15, f"joint share > 2× gold ({200 * gs[SEEDS[0]]:.1f}%) for teacher_I and perm_I, not "
            f"rand_I; replay cuts it by > {pred['line15_pp']:g} points in teacher_I and perm_I",
        f"{shares}. {red} (points)",
        "holds" if all(p15.values()) else "fails: " + ", ".join(k for k, v in p15.items() if not v))

    add(16, "reporting: a cell is real only if the threshold holds and the replicate range "
            "doesn't cross zero; otherwise the 2-of-3 rule alone",
        f"replicates ran (r = {', '.join(map(str, R.reps))})" if R.ran else "no replicates ran",
        "applied")

    # framing map
    branches = []
    if outcome[0] == "holds":
        branches.append("13 holds: the thesis stands as drafted (structured noise; balanced replay "
                        "counters it).")
    if outcome[0] != "holds" and dirs["rand_I"] == "real-positive":
        branches.append("13 fails because rand_I also shows Δ_replay > 0.03: reframe as 'replay is "
                        "robust to LLM label noise and distillation is redundant'; drop "
                        "'structured' from the title.")
    if out[[o["line"] for o in out].index(14)]["verdict"] == "holds":
        branches.append("14 holds (Δ_bal ≈ Δ_replay): the mechanism is class balancing, not "
                        "continual learning.")
    branches.append("perm_I ≈ teacher_I (line 12 observed 'same'): the structure that matters is "
                    "class-conditional." if observed12 == "same" else
                    "perm_I ≠ teacher_I (line 12 observed "
                    f"'{observed12}'): instance dependence matters too.")
    sh = {c: R.dirn(R.d_share_replay, c) for c in ("teacher_I", "perm_I")}
    if all(sh[c] != "not real" for c in sh) and all(dirs[c] == "not real" for c in sh):
        branches.append("Δ_share real on teacher_I and perm_I but Δ_replay not real: the thesis "
                        "becomes 'class balancing removes the collapse signature without improving "
                        "macro-F1; distillation is inert'.")

    header = ["line", "prediction", "outcome", "verdict"]
    write_table("prereg_outcomes_rq3", "PREREG lines 10–16 against outcomes", header,
                [[o[k] for k in header] for o in out],
                ["Computed by make_tables.py with the RQ3 definitions in its source, fixed "
                 "before the sweep. Predictions read from PREREG.md at HEAD (rq3-scoring block).",
                 "Values: replicate 0, replicate range in brackets." if R.ran else
                 "Values: replicate 0 (no replicates)."] +
                ["Framing map, branches that apply: " + " | ".join(branches)], out)
    return {"outcomes": out, "branches": branches}


def figure_rq3_delta(R: RQ3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.axhspan(-THRESH, THRESH, color=GRID, alpha=0.7, linewidth=0, zorder=0)
    ax.axhline(0, color=BASE, linewidth=1, zorder=1)
    off = {1234: -0.2, 1: 0.0, 2: 0.2}
    for i, c in enumerate(FIG_ORDER):
        for s in SEEDS:
            v = R.d_replay(c, s)
            x = i + off[s]
            if R.ran:
                ax.vlines(x, min(v), max(v), color=INK2, linewidth=2, zorder=2)
            ax.plot(x, v[0], marker=SEED_MARKER[s], color=INK, markersize=8, linestyle="none",
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                    label=f"seed {s}" if i == 0 else None)
    ax.set_xticks(range(len(FIG_ORDER)))
    ax.set_xticklabels(FIG_ORDER, fontsize=9, color=INK2)
    ax.set_xlim(-0.6, len(FIG_ORDER) - 0.4)
    ax.set_ylabel("replay − joint, final macro-F1", fontsize=9, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=3)
    ax.set_title("Δ_replay by label condition", fontsize=11, color=INK, loc="left")
    note = "shaded: |Δ| ≤ 0.03, not a real difference"
    if R.ran:
        note += "   ·   vertical line: range over replicates r = 0, 1, 2"
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "rq3_delta_replay.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_rq3_share(R: RQ3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    gs = R.gold_share[SEEDS[0]]
    ax.axhline(gs, color=INK2, linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(2 * gs, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    goff = {"joint": -0.26, "joint_balanced": 0.0, "replay": 0.26}
    soff = {1234: -0.06, 1: 0.0, 2: 0.06}
    top = 2 * gs
    for i, c in enumerate(FIG_ORDER):
        for g in ("joint", "joint_balanced", "replay"):
            if g not in RQ3_REGIMES[c]:
                continue
            col, mk = RQ3_STYLE[g]
            for s in SEEDS:
                v = R.vals(R.S, c, g, s)
                x = i + goff[g] + soff[s]
                top = max(top, max(v))
                if R.ran:
                    ax.vlines(x, min(v), max(v), color=col, linewidth=1.5, alpha=0.8, zorder=2)
                ax.plot(x, v[0], marker=mk, color=col, markersize=8, linestyle="none",
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                        label=g if (i == FIG_ORDER.index("teacher_I") and s == SEEDS[0]) else None)
    ax.annotate(f"gold test share {100 * gs:.1f}%", (len(FIG_ORDER) - 0.45, gs),
                xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                fontsize=7.5, color=INK2)
    ax.annotate("2× gold (PREREG 15)", (len(FIG_ORDER) - 0.45, 2 * gs), xytext=(0, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(FIG_ORDER)))
    ax.set_xticklabels(FIG_ORDER, fontsize=9, color=INK2)
    ax.set_xlim(-0.6, len(FIG_ORDER) - 0.4)
    ax.set_ylim(0, top * 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel(f"test predictions as {_short(BROAD)}", fontsize=9, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=3)
    ax.set_title(f"Broad-class share ({BROAD})", fontsize=11, color=INK, loc="left")
    note = "three points per regime = seeds 1234, 1, 2"
    if R.ran:
        note += "   ·   vertical line: range over replicates"
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "rq3_broad_share.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def rq3(hash_: Optional[str] = None) -> None:
    infos = {s: SplitInfo(s) for s in SEEDS}
    table_T7(infos)
    print("rq3: T7, T7b (condition stats) written")
    runs, h, reps = select_rq3_runs(hash_)
    if runs is None:
        print("rq3: no RQ3 runs yet; T8-T11, figures and PREREG 10-16 skipped")
        return
    R = RQ3(runs, reps, infos)
    print(f"rq3: {len(runs)} runs at {h}, replicates {reps}")
    table_T8(R)
    table_T9(R)
    table_T10(R)
    table_T11(R)
    figure_rq3_delta(R)
    figure_rq3_share(R)
    prereg_rq3(R)
    print(f"rq3 tables -> {TABLE_DIR}   figures -> {FIG_DIR}")

def main():
    ap = argparse.ArgumentParser(description="Phase 7 and RQ3 tables and figures")
    ap.add_argument("--only", choices=["all", "phase7", "rq3"], default="all")
    ap.add_argument("--hash-primary", default=None)
    ap.add_argument("--hash-fixed", default=None)
    ap.add_argument("--hash-rq3", default=None)
    args = ap.parse_args()
    if args.only in ("all", "phase7"):
        phase7(args.hash_primary, args.hash_fixed)
    if args.only in ("all", "rq3"):
        rq3(args.hash_rq3)


if __name__ == "__main__":
    main()
