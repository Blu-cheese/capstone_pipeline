"""
Phase 7 sweep (PHASE6_7_SPEC §4.1 and §4.4, v4).

    venv/bin/python -m continual.run_all                    # primary: naive, replay, fkd
    venv/bin/python -m continual.run_all --schedule fixed   # §4.4: replay, fkd, kd with a
                                                            # fixed 10-epoch replay stage
    venv/bin/python -m continual.run_all --plan rq3         # RQ3_SPEC §2.4: 5 label
                                                            # conditions x their regimes x
                                                            # 3 seeds x 3 replicates

RQ3 (RQ3_SPEC, Gate 1 amendments). Tracks are label conditions
(continual/conditions.py). Two more regimes, joint and joint_balanced, write
ordinary run dirs with test predictions. Replicate r runs with init seed
seed + 100r and, for rand_I / perm_I / gold_sub, draw seed seed + 100r. The
rq3 plan refuses to start unless PREREG.md at HEAD has the RQ3 section and
its scoring block (continual/prereg_rq3.py), and every run refuses unless its
labels equal the committed results/conditions/ file.

Tracks x regimes x seeds through the unmodified harness. Tracks are a list,
so a third track slots in without code changes.

Provenance (v4). The harness takes its git hash AFTER writing the run's own
output directory, so it always records "-dirty". This script records the
hash and a dirty flag itself, BEFORE each run, with results/ excluded from
the dirtiness check. It refuses to start on a dirty tree, never commits
(commit all outputs once after the sweep), and skips a run only if a
metrics.json with the same track / regime / seed / schedule and the HEAD
hash already exists - found from that metadata, not from the path.

Each run directory gets, next to the harness's R_matrix.json, config.json
and train_log.jsonl:
  metrics.json          metadata, ACC/BWT (existing definitions), per-task
                        forgetting, observed-relation accuracy, final macro-F1
  predictions.npz       after each task: test and train predictions with
                        probability vectors (see Recorder)
  memory_indices.json   replay, fkd, kd
  replay_log.jsonl      replay, fkd, kd
  fkd_diagnostics.json  fkd, kd
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.conditions import (
    PHASE7_TRACKS, RQ3_TRACKS, RQ3_REGIMES, REPLICATES, check_saved,
    draw_seed_for, init_seed_for, train_rows, head_labels,
)
from continual.eval_sets import build_eval_sets
from continual.features import build_features
from continual.harness import NaiveRegime, TaskData, apply_mask, mask_vector, run_sequential
from continual.regimes import (
    ReplayRegime, FKDRegime, KDRegime, write_artifacts, prereg_committed,
    resolve_feature_rows,
)
from continual.prereg_rq3 import prereg_rq3_ready
from continual.student import (
    macro_f1, accuracy, train_joint, train_joint_balanced, _predict,
)
from continual.task_split import load_split, class_index
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS

REPO = Path(__file__).resolve().parent.parent
TRACKS = ["teacher", "gold"]
SCHEDULE_REGIMES = {"primary": ["naive", "replay", "fkd"],
                    "fixed": ["replay", "fkd", "kd"]}
DEGENERATE_CLASS = "per:alternate_names"
JOINT_REGIMES = ("joint", "joint_balanced")


# ------------------------------------------------------------ provenance

def git_state() -> Tuple[str, bool, str]:
    """(short HEAD hash, dirty?, porcelain status) with results/ excluded."""
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                       capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", ".",
                             ":(exclude)results"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    return h, bool(status), status


def completed_runs(results_root: Path) -> Dict[Tuple, Path]:
    """(track, regime, seed, schedule, git_hash, replicate) -> run dir, from metrics.json."""
    found = {}
    for p in sorted(results_root.glob("*/metrics.json")):
        meta = json.loads(p.read_text(encoding="utf-8")).get("meta", {})
        if meta.get("dirty") is False:
            found[(meta["track"], meta["regime"], meta["seed"],
                   meta["schedule"], meta["git_hash"], meta.get("replicate", 0))] = p.parent
    return found


# ------------------------------------------------------------- recording

class Recorder:
    """
    Wraps a regime and, after the regime's own after_task, records the
    model's predictions: the same model the harness evaluates for R row t
    (after_task runs after best weights are restored and after any replay
    stage, just before the harness's eval). Eval-mode forward passes only,
    so no RNG is consumed and training is untouched - checked by naive's
    R matrices staying bit-identical to the Phase 5 files.

    Test predictions: every frozen-eval-set test example (id = row in the
    eval set, identical across seeds), gold label, predicted label and the
    probability vector under the seen-through-t mask. Test examples have no
    training label - DialogRE's dev/test were never teacher-labelled.
    Train predictions: every training row of the track (id = row in the
    feature matrix), training label, gold label (-1 if gold is outside the
    17 kept classes), predicted label, probability vector.
    """

    def __init__(self, inner, track: str, seed: int, ev_test: Dict,
                 class_names: Dict[int, str],
                 feature_rows: Optional[np.ndarray] = None,
                 feats: Optional[Dict] = None, draw_seed: Optional[int] = None):
        self.inner, self.name = inner, inner.name
        self.track, self.seed, self.draw_seed = track, seed, draw_seed
        self.X_test = torch.from_numpy(ev_test["X"])
        self.y_test = np.asarray(ev_test["y"])
        self.class_names = class_names
        self.feature_rows, self.feats = feature_rows, feats
        self.test_pred: List[np.ndarray] = []
        self.test_prob: List[np.ndarray] = []
        self.train_pred: List[np.ndarray] = []
        self.train_prob: List[np.ndarray] = []
        self.train_label = self.train_gold = None

    # the four Regime hooks, delegated
    def train_class_set(self, t, task_classes):
        return self.inner.train_class_set(t, task_classes)

    def build_train_batchset(self, t, task_data):
        return self.inner.build_train_batchset(t, task_data)

    def loss(self, masked_logits, targets, features, t):
        return self.inner.loss(masked_logits, targets, features, t)

    def after_task(self, t: int, model, task_data: TaskData) -> None:
        self.inner.after_task(t, model, task_data)
        if self.train_label is None:
            self._init_train(task_data)
        seen = set().union(*task_data.task_classes[:t + 1])
        mask = mask_vector(seen, model.n_classes)
        model.eval()
        for X, preds, probs in ((self.X_test, self.test_pred, self.test_prob),
                                (torch.from_numpy(task_data.X), self.train_pred,
                                 self.train_prob)):
            with torch.no_grad():
                logits = apply_mask(model(X), mask)
            preds.append(logits.argmax(dim=1).numpy().astype(np.int64))
            probs.append(torch.softmax(logits, dim=1).numpy().astype(np.float32))

    def _init_train(self, task_data: TaskData) -> None:
        if self.feature_rows is None:
            self.feature_rows = resolve_feature_rows(self.track, self.class_names,
                                                     task_data.X, self.draw_seed)
        feats = self.feats or build_features(config.CORPUS)
        to_head = {DIALOGRE_RELATIONS.index(n): i for i, n in self.class_names.items()}
        self.train_label = np.asarray(task_data.y, dtype=np.int64)
        self.train_gold = np.array([to_head.get(int(v), -1)
                                    for v in feats["y_gold"][self.feature_rows]],
                                   dtype=np.int64)

    def save(self, out_dir: Path) -> None:
        names = np.array([self.class_names[i] for i in sorted(self.class_names)])
        np.savez_compressed(
            out_dir / "predictions.npz",
            class_names=names,
            test_id=np.arange(len(self.y_test)), test_gold=self.y_test,
            test_pred=np.stack(self.test_pred), test_prob=np.stack(self.test_prob),
            train_feature_row=np.asarray(self.feature_rows, dtype=np.int64),
            train_label=self.train_label, train_gold=self.train_gold,
            train_pred=np.stack(self.train_pred), train_prob=np.stack(self.train_prob),
        )


# --------------------------------------------------------------- metrics

def run_metrics(r: Dict, rec: Recorder, task_classes: List[List[int]],
                degenerate: int, n_classes: int) -> Dict:
    """
    Everything T1-T3 need, from the R matrix and the recorded predictions.
    Also re-derives both R matrices from the predictions and checks them
    against the harness's - the predictions must be the evaluated model's.
    """
    K = len(task_classes)
    F, A = r["R_f1"], r["R_acc"]
    y = rec.y_test
    slices = [np.flatnonzero(np.isin(y, cls)) for cls in task_classes]

    mismatches = []
    for t in range(K):
        for j in range(t + 1):
            f1, _ = macro_f1(y[slices[j]], rec.test_pred[t][slices[j]], n_classes)
            acc = accuracy(y[slices[j]], rec.test_pred[t][slices[j]])
            if round(f1, 6) != F[t][j] or round(acc, 6) != A[t][j]:
                mismatches.append((t, j))
    if mismatches:
        raise RuntimeError(f"recorded predictions disagree with R at {mismatches}")

    forgetting_f1 = {j: round(F[K - 1][j] - F[j][j], 6) for j in range(K - 1)}
    forgetting_acc = {j: round(A[K - 1][j] - A[j][j], 6) for j in range(K - 1)}
    nondeg = [j for j in range(K - 1) if j != degenerate]

    observed = []                                   # T2: Zhao Table 2 style
    for t in range(K):
        idx = np.concatenate(slices[:t + 1])
        observed.append(round(accuracy(y[idx], rec.test_pred[t][idx]), 6))
    final_f1, _ = macro_f1(y, rec.test_pred[K - 1], n_classes)   # T3

    return {
        "degenerate_task": degenerate,
        "ACC_f1": r["ACC_f1"], "BWT_f1": r["BWT_f1"],
        "ACC_acc": r["ACC_acc"], "BWT_acc": r["BWT_acc"],
        "forgetting_f1": forgetting_f1, "forgetting_acc": forgetting_acc,
        "mean_forgetting_nondegenerate_f1":
            round(float(np.mean([forgetting_f1[j] for j in nondeg])), 6) if nondeg else None,
        "mean_forgetting_nondegenerate_acc":
            round(float(np.mean([forgetting_acc[j] for j in nondeg])), 6) if nondeg else None,
        "observed_accuracy_after_task": observed,
        "average_observed_accuracy": round(float(np.mean(observed)), 6),
        "final_macro_f1_all": round(final_f1, 6),
        "R_rederived_from_predictions": "match",
    }


# ------------------------------------------------------------------ runs

def make_regime(name: str, track: str, seed: int, schedule: str,
                draw_seed: Optional[int] = None):
    if name == "naive":
        assert schedule == "primary", "naive has no replay stage to schedule"
        return NaiveRegime()
    cls = {"replay": ReplayRegime, "fkd": FKDRegime, "kd": KDRegime}[name]
    return cls(seed, track, schedule="fixed" if schedule == "fixed" else "early_stop",
               draw_seed=draw_seed)


def _meta(track: str, regime_name: str, run_name: str, seed: int, schedule: str,
          git_hash: str, replicate: int, started: str, lam=None) -> Dict:
    meta = {"track": track, "regime": regime_name, "seed": seed,
            "schedule": schedule, "run_name": run_name,
            "git_hash": git_hash, "dirty": False,
            "hash_recorded_before_run": True, "started": started,
            "lambda": lam, "spec": "PHASE6_7_SPEC v4"}
    if track in RQ3_TRACKS:
        init = init_seed_for(seed, replicate)
        meta.update({"spec": "RQ3_SPEC", "replicate": replicate,
                     "init_seed": seed if init is None else init,
                     "draw_seed": draw_seed_for(track, seed, replicate)})
    return meta


def _check_condition(track: str, seed: int, replicate: int, feats: Dict, ci: Dict,
                     require_committed: bool) -> None:
    if track in RQ3_TRACKS:
        rows, y = train_rows(track, feats, ci, draw_seed_for(track, seed, replicate))
        check_saved(track, seed, replicate, rows, y, require_committed=require_committed)


def run_joint(track: str, regime_name: str, seed: int, git_hash: str,
              results_root: Path = config.RESULTS_DIR, replicate: int = 0,
              require_committed: bool = False) -> Path:
    """
    joint / joint_balanced as a run dir (RQ3_SPEC §2.3, Gate 1). joint calls
    the unchanged student.train_joint, so for gold and teacher it reproduces
    joint_baseline_seed*.json exactly (tests/test_conditions.py, test 1).
    Deterministic algorithms are switched on as in the harness, so a joint
    run's result doesn't depend on what ran before it in the process.
    """
    assert regime_name in JOINT_REGIMES, regime_name
    if replicate:
        assert track in RQ3_TRACKS, "replicates are an RQ3 notion"
    torch.use_deterministic_algorithms(True)
    split = load_split(seed)
    ci = class_index(split)
    names = sorted(ci, key=ci.get)
    feats = build_features(config.CORPUS)
    ev = build_eval_sets(seed)
    _check_condition(track, seed, replicate, feats, ci, require_committed)
    init = init_seed_for(seed, replicate)
    draw = draw_seed_for(track, seed, replicate) if track in RQ3_TRACKS else None
    rows, y = train_rows(track, feats, ci, draw)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    fn = train_joint if regime_name == "joint" else train_joint_balanced
    model, info = fn(feats["X"][rows], y, ev["dev"]["X"], ev["dev"]["y"], len(ci),
                     seed=seed if init is None else init, verbose=False)

    Xte = torch.from_numpy(ev["test"]["X"])
    pred = _predict(model, Xte)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte), dim=1).numpy().astype(np.float32)
    y_test = ev["test"]["y"]
    f1, per_class = macro_f1(y_test, pred, len(ci))
    acc = accuracy(y_test, pred)

    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = "" if init is None else f"_i{init}"
    out_dir = results_root / f"{track}_{regime_name}_{seed}{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out_dir / "predictions.npz",
        class_names=np.array(names), test_id=np.arange(len(y_test)), test_gold=y_test,
        test_pred=pred[None].astype(np.int64), test_prob=prob[None],
        train_feature_row=rows.astype(np.int64), train_label=y,
        train_gold=head_labels(feats["y_gold"], ci)[rows])
    with open(out_dir / "train_log.jsonl", "w", encoding="utf-8") as fh:
        for e in info["history"]:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "track": track, "regime": regime_name, "seed": seed,
        "init_seed": seed if init is None else init, "draw_seed": draw,
        "train_examples": int(len(y)), "n_classes": len(ci),
        "sampler": info.get("sampler"), "wall_seconds": round(time.time() - t0, 1),
        **config.summary()}, indent=2, sort_keys=True))
    metrics = {
        "meta": _meta(track, regime_name, regime_name, seed, "primary", git_hash,
                      replicate, started),
        "final_macro_f1_all": round(f1, 6), "final_accuracy": round(acc, 6),
        "test_macro_f1_exact": f1, "test_accuracy_exact": acc,
        "best_dev_macro_f1": info["best_dev_macro_f1"], "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"], "train_examples": int(len(y)),
        "per_class_f1": {names[c]: round(v, 4) for c, v in sorted(per_class.items())},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return out_dir


def run_one(track: str, regime_name: str, seed: int, schedule: str,
            git_hash: str, results_root: Path = config.RESULTS_DIR,
            replicate: int = 0, require_committed: bool = False) -> Path:
    if regime_name in JOINT_REGIMES:
        assert schedule == "primary", "joint regimes have no replay stage to schedule"
        return run_joint(track, regime_name, seed, git_hash, results_root, replicate,
                         require_committed)
    if replicate:
        assert track in RQ3_TRACKS, "replicates are an RQ3 notion"
    split = load_split(seed)
    ci = class_index(split)
    class_names = {i: r for r, i in ci.items()}
    degenerate = next(t["task"] for t in split["tasks"]
                      if DEGENERATE_CLASS in t["relations"])
    init = init_seed_for(seed, replicate)
    draw = draw_seed_for(track, seed, replicate) if track in RQ3_TRACKS else None
    _check_condition(track, seed, replicate, build_features(config.CORPUS), ci,
                     require_committed)
    regime = make_regime(regime_name, track, seed, schedule, draw_seed=draw)
    rec = Recorder(regime, track, seed, build_eval_sets(seed)["test"], class_names,
                   draw_seed=draw)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = run_sequential(track, seed, rec, results_root=results_root, verbose=False,
                             init_seed=init, draw_seed=draw)
    r = json.loads((out_dir / "R_matrix.json").read_text(encoding="utf-8"))

    rec.save(out_dir)
    if isinstance(regime, ReplayRegime):
        write_artifacts(regime, out_dir)
    metrics = {
        "meta": _meta(track, regime_name, regime.name, seed, schedule, git_hash,
                      replicate, started, getattr(regime, "lam", None)),
        **run_metrics(r, rec, r["task_classes"], degenerate, len(ci)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return out_dir


def rq3_plan(seeds: List[int], replicates: List[int]) -> List[Tuple[str, str, int, int]]:
    """(track, regime, seed, replicate), RQ3_SPEC §2.4: 54 runs per replicate."""
    return [(tr, rg, sd, r) for r in replicates for tr in RQ3_TRACKS
            for rg in RQ3_REGIMES[tr] for sd in seeds]


def main():
    ap = argparse.ArgumentParser(description="Phase 7 sweep, or the RQ3 sweep (--plan rq3)")
    ap.add_argument("--plan", choices=["phase7", "rq3"], default="phase7")
    ap.add_argument("--schedule", choices=sorted(SCHEDULE_REGIMES), default="primary")
    ap.add_argument("--tracks", nargs="+", default=TRACKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--regimes", nargs="+", default=None,
                    help="default: all regimes for the schedule")
    ap.add_argument("--replicates", nargs="+", type=int, default=list(REPLICATES),
                    help="rq3 plan only")
    args = ap.parse_args()

    ok, why = prereg_committed()
    if not ok:
        sys.exit(f"refusing to run: {why}")
    if args.plan == "rq3":
        if args.schedule != "primary":
            sys.exit("the rq3 plan runs the primary replay schedule only")
        ok, why = prereg_rq3_ready()
        if not ok:
            sys.exit(f"refusing to run the RQ3 sweep: {why}")
    head, dirty, status = git_state()
    if dirty:
        sys.exit(f"refusing to run: tree is dirty outside results/:\n{status}")

    done = completed_runs(config.RESULTS_DIR)
    if args.plan == "rq3":
        plan = rq3_plan(args.seeds, args.replicates)
    else:
        regimes = args.regimes or SCHEDULE_REGIMES[args.schedule]
        plan = [(tr, rg, sd, 0) for tr in args.tracks for rg in regimes for sd in args.seeds]
    print(f"sweep: plan={args.plan} schedule={args.schedule} commit={head} runs={len(plan)}")
    for track, regime_name, seed, r in plan:
        h, d, status = git_state()                 # before EACH run (v4)
        if h != head or d:
            sys.exit(f"tree changed mid-sweep (hash {h}, dirty={d}):\n{status}")
        key = (track, regime_name, seed, args.schedule, h, r)
        if key in done:
            print(f"  skip  {track:9s} {regime_name:14s} {seed:5d} r{r}  ({done[key].name})")
            continue
        t0 = time.time()
        out = run_one(track, regime_name, seed, args.schedule, h, replicate=r,
                      require_committed=args.plan == "rq3")
        m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        score = (f"ACC_f1={m['ACC_f1']:.4f} BWT_f1={m['BWT_f1']:+.4f}" if "ACC_f1" in m
                 else f"macro_f1={m['final_macro_f1_all']:.4f}")
        print(f"  done  {track:9s} {regime_name:14s} {seed:5d} r{r}  {score}  "
              f"{time.time() - t0:4.1f}s  -> {out.name}")
    print("sweep complete. Commit all outputs once now (no commits between runs).")


if __name__ == "__main__":
    main()
