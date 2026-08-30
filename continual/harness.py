"""
Continual-learning harness (PHASE5_SPEC §2).

Sequential train/eval loop, regime-agnostic. The regime is a pluggable hook
object (§2.2); Phase 6's replay and FKD must be implementable WITHOUT editing
this file - if they can't, this phase failed its interface acceptance test.

The loop, for t = 0..K-1:
  1. Fresh optimizer (model WEIGHTS persist across tasks - that is the point;
     carried Adam state would confound the forgetting measurement).
  2. Train on regime.build_train_batchset(t) under the TRAIN mask
     = regime.train_class_set(t).
  3. Early-stop on the CURRENT task's gold dev slice only, same train mask,
     patience 5, cap 60. [CORRECTNESS] Peeking at other tasks' dev sets
     during training leaks the sequence.
  4. Restore best-dev weights before evaluation and before task t+1.
  5. Evaluate on gold test of every task j <= t under the EVAL mask
     = union of classes seen through t. Write row t of both R matrices.

[CORRECTNESS] Logit masking - the single place the meaningless-table bug
lives - is one function, `apply_mask`, used for train loss, dev early
stopping, and test evaluation alike:
  * TRAIN mask comes from the regime (naive: all classes seen through t -
    see NaiveRegime's docstring for why this diverges from PHASE5_SPEC 2.4;
    replay will return current + memory classes - the hook exists for this).
  * EVAL mask is classes seen through t. Masking eval to only task j's
    classes would hide forgetting entirely; evaluating unmasked over all 17
    would let the head emit never-trained classes and deflate early rows.
  * Train-mask convention vs the CRE literature: Zhao et al.'s Eq. 2 fixes
    the softmax denominator at all SEEN relations for every loss - Eq. 3 vs
    Eq. 9 differ only in which dataset is summed (new train data vs memory),
    not in prediction space. Naive's seen-through-t mask matches that
    convention. The replay reference (Efeoglu et al., PAPER_NOTES §2; PDF
    not in repo) fine-tunes with memory in the batch, which this interface
    expresses via build_train_batchset() rather than the mask.

Eval labels are GOLD for both tracks, from the frozen eval sets. `track`
selects TRAINING labels only. Task membership of a training example follows
its own track's label; task membership of an eval example follows its gold
label via the same frozen split.

CPU only. Deterministic: same invocation twice -> identical R matrices.

Usage:
    venv/bin/python -m continual.harness --track gold --seed 1234 --regime naive
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.eval_sets import build_eval_sets
from continual.features import build_features
from continual.metrics import (
    RMatrix, r_matrix_payload, validate_payload,
    warn_on_diagonal, load_joint_baseline_ref,
)
from continual.student import Student, macro_f1, accuracy
from continual.task_split import load_split, class_index, task_of_class
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS


# ------------------------------------------------------------------ masking

def apply_mask(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """
    [CORRECTNESS] The single logit-masking implementation.

    `active` is a boolean tensor over classes; inactive classes are filled
    with the dtype minimum so they can never be argmax nor receive softmax
    mass. Used identically in train loss, dev early stopping and test eval.
    """
    return logits.masked_fill(~active, torch.finfo(logits.dtype).min)


def mask_vector(classes: Set[int], n_classes: int) -> torch.Tensor:
    v = torch.zeros(n_classes, dtype=torch.bool)
    v[sorted(classes)] = True
    return v


def masked_predict(model: Student, X: torch.Tensor,
                   active: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return apply_mask(model(X), active).argmax(dim=1).numpy()


# ------------------------------------------------------------------- regime

class TaskData:
    """Track-specific training data, sliced by task."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 task_classes: List[List[int]]):
        self.X = X
        self.y = y
        self.task_classes = task_classes

    def current(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isin(self.y, self.task_classes[t])
        return self.X[mask], self.y[mask]


class Regime(Protocol):
    """Phase 6 implements replay/FKD purely against this interface."""
    name: str

    def train_class_set(self, t: int,
                        task_classes: List[List[int]]) -> Set[int]: ...

    def build_train_batchset(self, t: int,
                             task_data: TaskData) -> Tuple[np.ndarray, np.ndarray]: ...

    def loss(self, masked_logits: torch.Tensor, targets: torch.Tensor,
             features: torch.Tensor, t: int) -> torch.Tensor: ...

    def after_task(self, t: int, model: Student,
                   task_data: TaskData) -> None: ...


class NaiveRegime:
    """
    Sequential fine-tuning, no mitigation. The forgetting baseline.

    TRAIN MASK = ALL CLASSES SEEN THROUGH t, not current-task-only. Changed
    from PHASE5_SPEC 2.4's original convention after the Phase 5 diagnostic:
    under a current-only mask the model never contrasts new classes against
    old logits, and 84% of task-2 predictions leaked into earlier-task
    classes at eval - the diagonal depression measured calibration mismatch
    rather than interference. With seen-through-t, the train and eval
    prediction spaces coincide (class-incremental convention) and new-task
    examples act as implicit negatives for old classes, so measured
    forgetting is genuine interference.

    COMPARABILITY NOTE for the write-up (corrected - the first version of
    this note read the equations backwards): in Zhao et al., Eq. 2 fixes the
    softmax denominator at |R-tilde_k| - ALL seen relations - for every loss.
    Eq. 3's "over R_k" describes the label domain of the dataset being
    summed (D_train_k), not the prediction space; Eq. 9 is the same form
    summed over the memory buffer. Seen-through-t masking therefore MATCHES
    their convention. The earlier current-task-mask runs (kept on disk) were
    the deviation, not this.
    """
    name = "naive"

    def train_class_set(self, t, task_classes):
        return set().union(*task_classes[:t + 1])

    def build_train_batchset(self, t, task_data):
        return task_data.current(t)

    def loss(self, masked_logits, targets, features, t):
        return F.cross_entropy(masked_logits, targets)

    def after_task(self, t, model, task_data):
        pass


REGIMES = {"naive": NaiveRegime}


# ------------------------------------------------------------------ helpers

def _git_hash() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True,
                           cwd=Path(__file__).resolve().parent.parent
                           ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True,
                               cwd=Path(__file__).resolve().parent.parent
                               ).stdout.strip()
        return f"{h}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _track_train_data(track: str, feats: Dict, split: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Map the 37-way label space onto the head; keep rows surviving the split."""
    ci = class_index(split)
    full_to_head = {DIALOGRE_RELATIONS.index(r): h for r, h in ci.items()}
    y_full = feats["y_teacher"] if track == "teacher" else feats["y_gold"]
    keep = np.isin(y_full, list(full_to_head))
    X = feats["X"][keep]
    y = np.array([full_to_head[int(v)] for v in y_full[keep]], dtype=np.int64)
    return X, y


def _eval_slices(y_eval: np.ndarray,
                 task_classes: List[List[int]]) -> List[np.ndarray]:
    """
    Index arrays for each task's eval examples, derived from LABEL MEMBERSHIP
    in the frozen split rather than a stored task column - works identically
    for the real K=3 split and the degenerate K=1 acceptance run.
    """
    return [np.where(np.isin(y_eval, cls))[0] for cls in task_classes]


# --------------------------------------------------------------------- loop

def run_sequential(
    track: str,
    seed: int,
    regime,
    task_classes_override: Optional[List[List[int]]] = None,
    results_root: Path = None,
    verbose: bool = True,
) -> Path:
    """Run the sequential CL loop. Returns the results directory."""
    assert track in ("teacher", "gold"), track
    t_start = time.time()

    # Determinism: seed everything before ANY torch op, then flag it.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    split = load_split(seed)
    ci = class_index(split)
    n_classes = len(ci)

    if task_classes_override is not None:
        task_classes = [sorted(c) for c in task_classes_override]
    else:
        toc = task_of_class(split)
        task_classes = [sorted(c for c, tt in toc.items() if tt == k)
                        for k in range(split["k"])]
    K = len(task_classes)

    feats = build_features(config.CORPUS)
    ev = build_eval_sets(seed)
    X_train_all, y_train_all = _track_train_data(track, feats, split)
    task_data = TaskData(X_train_all, y_train_all, task_classes)

    dev_slices = _eval_slices(ev["dev"]["y"], task_classes)
    test_slices = _eval_slices(ev["test"]["y"], task_classes)

    Xdv = torch.from_numpy(ev["dev"]["X"])
    Xte = torch.from_numpy(ev["test"]["X"])

    # Model is created ONCE; weights persist across tasks.
    model = Student(n_classes)

    r_f1, r_acc = RMatrix(K), RMatrix(K)
    train_log: List[dict] = []
    seen: Set[int] = set()

    for t in range(K):
        train_classes = regime.train_class_set(t, task_classes)
        seen |= set(task_classes[t])
        train_mask = mask_vector(train_classes, n_classes)
        eval_mask = mask_vector(seen, n_classes)

        Xt_np, yt_np = regime.build_train_batchset(t, task_data)
        Xt, yt = torch.from_numpy(Xt_np), torch.from_numpy(yt_np)

        dev_idx = dev_slices[t]
        y_dev_t = ev["dev"]["y"][dev_idx]
        Xdv_t = Xdv[dev_idx]

        # 1. Fresh optimizer per task [CORRECTNESS].
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)

        if verbose:
            print(f"\n--- task {t}: {len(yt)} train examples, "
                  f"classes {sorted(train_classes)}, "
                  f"{len(dev_idx)} dev examples ---")

        best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
        stop_reason = "epoch_cap"

        for epoch in range(config.MAX_EPOCHS_PER_TASK):
            model.train()
            perm = torch.randperm(len(Xt))
            total_loss = 0.0
            for i in range(0, len(perm), config.BATCH_SIZE):
                idx = perm[i:i + config.BATCH_SIZE]
                optimizer.zero_grad()
                logits = model(Xt[idx])
                loss = regime.loss(apply_mask(logits, train_mask),
                                   yt[idx], Xt[idx], t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)

            # 3. Early stopping on CURRENT task's gold dev only [CORRECTNESS].
            dev_pred = masked_predict(model, Xdv_t, train_mask)
            dev_f1, _ = macro_f1(y_dev_t, dev_pred, n_classes)
            train_log.append({"task": t, "epoch": epoch,
                              "loss": round(total_loss / max(len(Xt), 1), 6),
                              "dev_macro_f1": round(dev_f1, 6)})

            if dev_f1 > best_f1:
                best_f1, best_epoch, patience = dev_f1, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= config.EARLY_STOPPING_PATIENCE:
                    stop_reason = f"early_stop(patience={config.EARLY_STOPPING_PATIENCE})"
                    break

        # 4. Restore best-dev weights before eval and before task t+1.
        if config.RESTORE_BEST_WEIGHTS and best_state is not None:
            model.load_state_dict(best_state)
        train_log.append({"task": t, "stop_reason": stop_reason,
                          "best_epoch": best_epoch,
                          "best_dev_macro_f1": round(best_f1, 6)})
        if verbose:
            print(f"    {stop_reason}; best dev {best_f1:.4f} @ epoch {best_epoch}")

        regime.after_task(t, model, task_data)

        # 5. Eval row t: every task j <= t, EVAL mask = seen-through-t.
        for j in range(t + 1):
            idx = test_slices[j]
            pred = masked_predict(model, Xte[idx], eval_mask)
            y_j = ev["test"]["y"][idx]
            f1_j, _ = macro_f1(y_j, pred, n_classes)
            r_f1.set(t, j, f1_j)
            r_acc.set(t, j, accuracy(y_j, pred))
            if verbose:
                print(f"    R[{t}][{j}]  f1={f1_j:.4f}  acc={accuracy(y_j, pred):.4f}")

    # ------------------------------------------------------------- outputs
    elapsed = time.time() - t_start
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{track}_{regime.name}_{seed}_{ts}"
    out_dir = (results_root or config.RESULTS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_ref = load_joint_baseline_ref(seed, track)
    warnings = warn_on_diagonal(r_f1, joint_ref["test_macro_f1"] if joint_ref else None)

    payload = r_matrix_payload(run_id, track, regime.name, seed, task_classes,
                               r_f1, r_acc, joint_ref, warnings)
    problems = validate_payload(payload)
    if problems:
        raise RuntimeError(f"R_matrix.json schema invalid: {problems}")

    (out_dir / "R_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True))
    (out_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "track": track, "regime": regime.name, "seed": seed,
        "git_hash": _git_hash(), "n_classes": n_classes, "k": K,
        "task_classes_override": task_classes_override is not None,
        "train_examples": int(len(y_train_all)),
        "eval": {"dev": int(len(ev["dev"]["y"])), "test": int(len(ev["test"]["y"])),
                 "label_source": "gold (both tracks)"},
        "wall_seconds": round(elapsed, 1),
        **config.summary(),
    }, indent=2, sort_keys=True))
    with open(out_dir / "train_log.jsonl", "w", encoding="utf-8") as fh:
        for entry in train_log:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    if verbose:
        print(f"\nACC_f1={payload['ACC_f1']}  BWT_f1={payload['BWT_f1']}  "
              f"ACC_acc={payload['ACC_acc']}  BWT_acc={payload['BWT_acc']}")
        print(f"wall time {elapsed:.1f}s -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Sequential CL harness")
    ap.add_argument("--track", choices=["teacher", "gold"], required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--regime", choices=sorted(REGIMES), default="naive")
    args = ap.parse_args()
    run_sequential(args.track, args.seed, REGIMES[args.regime]())


if __name__ == "__main__":
    main()
