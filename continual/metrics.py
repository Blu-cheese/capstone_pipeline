"""
R-matrix container and CL metrics (PHASE5_SPEC §3).

Two K x K matrices per run:
    R_f1[t][j]  - macro-F1 over task j's test examples, eval-masked space
    R_acc[t][j] - accuracy over the same predictions

Both are logged because macro-F1 is this project's internal convention while
accuracy is what the CRE literature (Zhao et al., RP-CRE line) reports.

[CORRECTNESS] Matrices are LOWER-TRIANGULAR ONLY. Entries j > t are None in
Python and null in JSON, never 0.0 - a zero above the diagonal silently
poisons any aggregate computed later.

    ACC = mean_j R[K-1][j]
    BWT = mean_{j<K-1} ( R[K-1][j] - R[j][j] )     # negative = forgetting

[CORRECTNESS] Diagonal sanity: positive BWT can be an artifact of DEPRESSED
DIAGONALS (masking or output failures at time j), not genuine backward
transfer - a known failure mode in the 2025 CRE literature. warn_on_diagonal()
flags any diagonal cell more than 0.15 macro-F1 below the joint-baseline test
figure so BWT is never interpreted before the diagonal is trusted.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

DIAGONAL_WARN_MARGIN = 0.15


class RMatrix:
    """Lower-triangular result matrix for one metric."""

    def __init__(self, k: int):
        self.k = k
        self.rows: List[List[Optional[float]]] = [
            [None] * k for _ in range(k)
        ]

    def set(self, t: int, j: int, value: float) -> None:
        if j > t:
            raise ValueError(f"upper-triangular write R[{t}][{j}] refused")
        self.rows[t][j] = round(float(value), 6)

    def get(self, t: int, j: int) -> Optional[float]:
        return self.rows[t][j]

    def is_complete(self) -> bool:
        return all(self.rows[t][j] is not None
                   for t in range(self.k) for j in range(t + 1))

    def acc(self) -> Optional[float]:
        """Mean of the final row over all tasks."""
        last = self.rows[self.k - 1]
        vals = [last[j] for j in range(self.k)]
        if any(v is None for v in vals):
            return None
        return round(sum(vals) / self.k, 6)

    def bwt(self) -> Optional[float]:
        """Mean over j < K-1 of R[K-1][j] - R[j][j]. None when K == 1."""
        if self.k == 1:
            return None
        last = self.rows[self.k - 1]
        deltas = []
        for j in range(self.k - 1):
            if last[j] is None or self.rows[j][j] is None:
                return None
            deltas.append(last[j] - self.rows[j][j])
        return round(sum(deltas) / len(deltas), 6)


def warn_on_diagonal(r_f1: RMatrix, joint_test_f1: Optional[float]) -> List[str]:
    """
    Flag diagonal cells sitting far below the joint baseline.

    Returns the warnings (also printed) so they can be stored in the run's
    JSON rather than scrolling away in a terminal.
    """
    warnings = []
    if joint_test_f1 is None:
        return warnings
    for j in range(r_f1.k):
        d = r_f1.get(j, j)
        if d is not None and d < joint_test_f1 - DIAGONAL_WARN_MARGIN:
            w = (f"R_f1[{j}][{j}]={d:.3f} is more than {DIAGONAL_WARN_MARGIN} "
                 f"below the joint baseline ({joint_test_f1:.3f}) - a depressed "
                 f"diagonal makes BWT uninterpretable; check masking/output at "
                 f"task {j} before reading any BWT from this run")
            warnings.append(w)
            print(f"  [DIAGONAL WARNING] {w}")
    return warnings


def r_matrix_payload(
    run_id: str,
    track: str,
    regime: str,
    seed: int,
    task_classes: List[List[int]],
    r_f1: RMatrix,
    r_acc: RMatrix,
    joint_baseline_ref: Optional[Dict],
    diagonal_warnings: List[str],
) -> Dict:
    """Assemble the R_matrix.json payload (PHASE5_SPEC §3 schema)."""
    return {
        "run_id": run_id,
        "track": track,
        "regime": regime,
        "seed": seed,
        "task_classes": task_classes,
        "R_f1": r_f1.rows,
        "R_acc": r_acc.rows,
        "ACC_f1": r_f1.acc(),
        "BWT_f1": r_f1.bwt(),
        "ACC_acc": r_acc.acc(),
        "BWT_acc": r_acc.bwt(),
        "joint_baseline_ref": joint_baseline_ref,
        "diagonal_warnings": diagonal_warnings,
    }


def validate_payload(payload: Dict) -> List[str]:
    """AC4 schema check. Returns failure strings; empty list = valid."""
    problems = []
    for key in ("run_id", "track", "regime", "seed", "task_classes",
                "R_f1", "R_acc", "ACC_f1", "ACC_acc", "joint_baseline_ref"):
        if key not in payload:
            problems.append(f"missing key {key}")
    for name in ("R_f1", "R_acc"):
        rows = payload.get(name)
        if not isinstance(rows, list):
            problems.append(f"{name} is not a list")
            continue
        k = len(rows)
        for t, row in enumerate(rows):
            if len(row) != k:
                problems.append(f"{name}[{t}] has length {len(row)} != {k}")
                continue
            for j, v in enumerate(row):
                if j <= t and not isinstance(v, (int, float)):
                    problems.append(f"{name}[{t}][{j}] should be numeric, got {v!r}")
                if j > t and v is not None:
                    problems.append(f"{name}[{t}][{j}] must be null above diagonal, got {v!r}")
    if "BWT_f1" not in payload or "BWT_acc" not in payload:
        problems.append("missing BWT fields")
    return problems


def load_joint_baseline_ref(seed: int, track: str) -> Optional[Dict]:
    """Joint-baseline reference for diagonal checks; None if not yet run."""
    path = Path(f"results/joint_baseline_seed{seed}.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    tr = data.get("tracks", {}).get(track)
    if tr is None:
        return None
    return {"track": track, "test_macro_f1": tr["test_macro_f1"],
            "test_accuracy": tr["test_accuracy"],
            "epoch_cap": data.get("config", {}).get("max_epochs_per_task")}
