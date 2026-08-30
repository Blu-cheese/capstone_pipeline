"""
The continual student (METHOD_A_SPEC_V2 Phase 4).

    Linear(1152 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> N_CLASSES)

N_CLASSES comes from the frozen task-split file, never hard-coded. The head is
fixed-width across all tasks and is NEVER resized between them - the harness
uses logit masking instead (Phase 5).

Plain PyTorch, CPU only. ~300K parameters, seconds per epoch.

This module also provides joint training - all classes at once, no continual
learning - which is the UPPER BOUND every CL regime should sit below. If a CL
regime later beats it, that is eval leakage, not a result.

Usage:
    venv/bin/python -m continual.student --seed 1234
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.eval_sets import build_eval_sets
from continual.features import build_features
from continual.task_split import load_split, class_index, kept_relations
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS


class Student(nn.Module):
    """Fixed-width MLP head over frozen sentence-encoder features."""

    def __init__(self, n_classes: int,
                 feature_dim: int = config.FEATURE_DIM,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ metrics

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
             n_classes: int) -> Tuple[float, Dict[int, float]]:
    """
    Macro-F1 and per-class F1.

    Averaged over classes PRESENT IN y_true only. Averaging over all
    n_classes would silently score absent classes as 0.0 and drag the mean
    down for reasons that have nothing to do with the model.
    """
    per_class: Dict[int, float] = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fn == 0:
            continue  # class absent from the reference set
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mean = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean, per_class


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


# ----------------------------------------------------------------- training

def _predict(model: Student, X: torch.Tensor,
             active: Optional[torch.Tensor] = None) -> np.ndarray:
    """Predict, optionally masking inactive classes to -inf (logit masking)."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        if active is not None:
            logits = logits.masked_fill(~active, float("-inf"))
        return logits.argmax(dim=1).numpy()


def train_joint(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    n_classes: int,
    seed: int = 1234,
    verbose: bool = True,
) -> Tuple[Student, Dict]:
    """
    Joint training: every class at once, no continual learning.

    Uses the same optimizer, batch size, epoch cap and early-stopping rule as
    the CL regimes (config.py), so the upper bound is measured under identical
    conditions rather than a more generous setup.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Student(n_classes)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xdv = torch.from_numpy(X_dev)

    best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
    history = []

    for epoch in range(config.MAX_EPOCHS_PER_TASK):
        model.train()
        perm = torch.randperm(len(Xtr))
        total_loss = 0.0
        for i in range(0, len(perm), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(idx)

        dev_f1, _ = macro_f1(y_dev, _predict(model, Xdv), n_classes)
        history.append({"epoch": epoch, "loss": total_loss / len(Xtr),
                        "dev_macro_f1": dev_f1})

        if dev_f1 > best_f1:
            best_f1, best_epoch, patience = dev_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                if verbose:
                    print(f"    early stop at epoch {epoch} "
                          f"(best {best_f1:.4f} @ epoch {best_epoch})")
                break

    if config.RESTORE_BEST_WEIGHTS and best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_dev_macro_f1": best_f1, "best_epoch": best_epoch,
                   "epochs_run": len(history), "history": history}


# ------------------------------------------------------------------- driver

def run_joint_baseline(seed: int = 1234, verbose: bool = True) -> Dict:
    """Joint-training upper bound for both label tracks."""
    split = load_split(seed)
    ci = class_index(split)
    n_classes = len(ci)
    relations = kept_relations(split)

    feats = build_features(config.CORPUS)
    ev = build_eval_sets(seed)

    # Map the frozen 37-way label space onto the student's N_CLASSES head,
    # keeping only rows whose label for THIS track survives the split.
    full_to_head = {DIALOGRE_RELATIONS.index(r): ci[r] for r in relations}

    def track_data(y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isin(y_full, list(full_to_head))
        X = feats["X"][mask]
        y = np.array([full_to_head[int(v)] for v in y_full[mask]], dtype=np.int64)
        return X, y

    results = {"seed": seed, "n_classes": n_classes, "relations": relations,
               "config": config.summary(), "tracks": {}}

    for track, y_full in (("teacher", feats["y_teacher"]), ("gold", feats["y_gold"])):
        X, y = track_data(y_full)
        if verbose:
            print(f"\n{'='*66}\nJOINT BASELINE - {track} track")
            print(f"{'='*66}")
            print(f"  train: {len(y)} examples over {len(set(y.tolist()))}/{n_classes} classes")
            print(f"  dev  : {len(ev['dev']['y'])}   test: {len(ev['test']['y'])}  (gold, both tracks)")

        model, info = train_joint(X, y, ev["dev"]["X"], ev["dev"]["y"],
                                  n_classes, seed=seed, verbose=verbose)

        pred = _predict(model, torch.from_numpy(ev["test"]["X"]))
        test_f1, per_class = macro_f1(ev["test"]["y"], pred, n_classes)
        test_acc = accuracy(ev["test"]["y"], pred)

        # Majority-class baseline on the same test set, for scale.
        counts = np.bincount(ev["test"]["y"], minlength=n_classes)
        majority_acc = float(counts.max() / counts.sum())

        results["tracks"][track] = {
            "train_examples": int(len(y)),
            "n_params": model.n_params(),
            "best_dev_macro_f1": info["best_dev_macro_f1"],
            "best_epoch": info["best_epoch"],
            "epochs_run": info["epochs_run"],
            "test_macro_f1": test_f1,
            "test_accuracy": test_acc,
            "majority_class_accuracy": majority_acc,
            "per_class_f1": {relations[c]: round(f, 4) for c, f in sorted(per_class.items())},
        }

        if verbose:
            print(f"  params: {model.n_params():,}   best dev macro-F1 "
                  f"{info['best_dev_macro_f1']:.4f} @ epoch {info['best_epoch']}")
            print(f"  TEST macro-F1 {test_f1:.4f}   accuracy {test_acc:.4f}   "
                  f"(majority-class {majority_acc:.4f})")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"joint_baseline_seed{seed}.json"
    out.write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"\nsaved to {out}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Student model + joint baselines")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    run_joint_baseline(args.seed)


if __name__ == "__main__":
    main()
