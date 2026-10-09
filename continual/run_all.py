"""
Phase 7 sweep (PHASE6_7_SPEC §4.1 and §4.4, v4).

    venv/bin/python -m continual.run_all                    # primary: naive, replay, fkd
    venv/bin/python -m continual.run_all --schedule fixed   # §4.4: replay, fkd, kd with a
                                                            # fixed 10-epoch replay stage

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
from continual.eval_sets import build_eval_sets
from continual.features import build_features
from continual.harness import NaiveRegime, TaskData, apply_mask, mask_vector, run_sequential
from continual.regimes import (
    ReplayRegime, FKDRegime, KDRegime, write_artifacts, prereg_committed,
    resolve_feature_rows,
)
from continual.student import macro_f1, accuracy
from continual.task_split import load_split, class_index
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS

REPO = Path(__file__).resolve().parent.parent
TRACKS = ["teacher", "gold"]
SCHEDULE_REGIMES = {"primary": ["naive", "replay", "fkd"],
                    "fixed": ["replay", "fkd", "kd"]}
DEGENERATE_CLASS = "per:alternate_names"


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
    """(track, regime, seed, schedule, git_hash) -> run dir, from metrics.json."""
    found = {}
    for p in sorted(results_root.glob("*/metrics.json")):
        meta = json.loads(p.read_text(encoding="utf-8")).get("meta", {})
        if meta.get("dirty") is False:
            found[(meta["track"], meta["regime"], meta["seed"],
                   meta["schedule"], meta["git_hash"])] = p.parent
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
                 feats: Optional[Dict] = None):
        self.inner, self.name = inner, inner.name
        self.track, self.seed = track, seed
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
                                                     task_data.X)
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

def make_regime(name: str, track: str, seed: int, schedule: str):
    if name == "naive":
        assert schedule == "primary", "naive has no replay stage to schedule"
        return NaiveRegime()
    cls = {"replay": ReplayRegime, "fkd": FKDRegime, "kd": KDRegime}[name]
    return cls(seed, track, schedule="fixed" if schedule == "fixed" else "early_stop")


def run_one(track: str, regime_name: str, seed: int, schedule: str,
            git_hash: str, results_root: Path = config.RESULTS_DIR) -> Path:
    split = load_split(seed)
    ci = class_index(split)
    class_names = {i: r for r, i in ci.items()}
    degenerate = next(t["task"] for t in split["tasks"]
                      if DEGENERATE_CLASS in t["relations"])
    regime = make_regime(regime_name, track, seed, schedule)
    rec = Recorder(regime, track, seed, build_eval_sets(seed)["test"], class_names)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = run_sequential(track, seed, rec, results_root=results_root, verbose=False)
    r = json.loads((out_dir / "R_matrix.json").read_text(encoding="utf-8"))

    rec.save(out_dir)
    if isinstance(regime, ReplayRegime):
        write_artifacts(regime, out_dir)
    metrics = {
        "meta": {"track": track, "regime": regime_name, "seed": seed,
                 "schedule": schedule, "run_name": regime.name,
                 "git_hash": git_hash, "dirty": False,
                 "hash_recorded_before_run": True, "started": started,
                 "lambda": getattr(regime, "lam", None),
                 "spec": "PHASE6_7_SPEC v4"},
        **run_metrics(r, rec, r["task_classes"], degenerate, len(ci)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 7 sweep")
    ap.add_argument("--schedule", choices=sorted(SCHEDULE_REGIMES), default="primary")
    ap.add_argument("--tracks", nargs="+", default=TRACKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--regimes", nargs="+", default=None,
                    help="default: all regimes for the schedule")
    args = ap.parse_args()
    regimes = args.regimes or SCHEDULE_REGIMES[args.schedule]

    ok, why = prereg_committed()
    if not ok:
        sys.exit(f"refusing to run: {why}")
    head, dirty, status = git_state()
    if dirty:
        sys.exit(f"refusing to run: tree is dirty outside results/:\n{status}")

    done = completed_runs(config.RESULTS_DIR)
    plan = [(tr, rg, sd) for tr in args.tracks for rg in regimes for sd in args.seeds]
    print(f"sweep: schedule={args.schedule} commit={head} runs={len(plan)}")
    for track, regime_name, seed in plan:
        h, d, status = git_state()                 # before EACH run (v4)
        if h != head or d:
            sys.exit(f"tree changed mid-sweep (hash {h}, dirty={d}):\n{status}")
        key = (track, regime_name, seed, args.schedule, h)
        if key in done:
            print(f"  skip  {track:7s} {regime_name:6s} {seed:5d}  ({done[key].name})")
            continue
        t0 = time.time()
        out = run_one(track, regime_name, seed, args.schedule, h)
        m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        print(f"  done  {track:7s} {regime_name:6s} {seed:5d}  ACC_f1={m['ACC_f1']:.4f} "
              f"BWT_f1={m['BWT_f1']:+.4f}  {time.time() - t0:4.1f}s  -> {out.name}")
    print("sweep complete. Commit all outputs once now (no commits between runs).")


if __name__ == "__main__":
    main()
