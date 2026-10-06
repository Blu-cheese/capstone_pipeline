"""
Replay and FKD regimes (PHASE6_7_SPEC §3).

Both are NaiveRegime with an after_task() that runs, for task index t:

  2. memory selection for the relations just learned          (Zhao §4.3)
  3. prototypes for those relations - FKD only                (Eq. 4)
  4. a replay stage on the memory of ALL seen relations,      (Zhao §4.6)
     t >= 1 only (k >= 2 in the paper's 1-based numbering; the official
     code skips memory training at the first task, CEAR main.py L88)

and FKD then snapshots the model as the next task's teacher.

after_task() runs after new-task training has restored its best weights and
before the harness records R row t (harness.py, loop step 5), so the replay
stage lands exactly where §3.1 puts it without editing harness.py.

New-task training (step 1) is inherited from NaiveRegime unchanged: Eq. 3,
softmax over all seen relations, no FKD term ever.

Replay and FKD differ in exactly one term of one stage - the replay-stage
loss is Eq. 9 for replay and Eq. 9 + lambda * L_fkd for FKD. Memory, schedule,
optimiser and early stopping are shared code, so the comparison is the
paper's own "w/o FKD" ablation (Zhao Table 3).

Index convention. The harness numbers tasks from 0. OLD = R-tilde_{k-1} is
the union of task_classes[:t], SEEN = R-tilde_k the union of
task_classes[:t+1]. Both are sets of GLOBAL head indices. Head indices follow
DIALOGRE_RELATIONS order, not task order, so old classes are never "the first
n_old columns": they are always selected with the OLD index set
(PHASE6_7_SPEC silent error 8; the official code's narrow(1, 0, n_old) only
works because its classes are numbered in arrival order).

Usage (refuses to run until PREREG.md lines 1-6 are committed):
    venv/bin/python -m continual.regimes --track gold --seed 1234 --regime fkd
"""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.eval_sets import build_eval_sets
from continual.features import EMBED_DIM, build_features
from continual.harness import (
    NaiveRegime, TaskData, apply_mask, mask_vector, masked_predict, run_sequential,
)
from continual.student import Student, macro_f1
from continual.task_split import load_split, class_index
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS

# features.py:114 builds X = concat([window, subject, object]), 384 each.
# Memory selection and prototypes use subject + object only (CLAUDE.md §5,
# RECAP_AUG26 §3): Eq. 1 builds h_x from head and tail entities alone.
ENTITY_COLS = slice(EMBED_DIM, 3 * EMBED_DIM)

# PHASE6_7_SPEC §4.2 T4. The last pair is within-task on every seed and is
# kept as the reference row.
ANALOGOUS_PAIRS = [
    ("per:positive_impression", "per:negative_impression"),
    ("per:friends", "per:girl/boyfriend"),
    ("per:friends", "per:roommate"),
    ("per:girl/boyfriend", "per:negative_impression"),
]


# ------------------------------------------------------- pure functions

def select_memory(Z: np.ndarray, y: np.ndarray, classes: Sequence[int],
                  m: int, seed: int) -> Dict[int, List[int]]:
    """
    Zhao §4.3: k-means over each relation's training samples, m clusters,
    keep one sample per cluster.

    Returns {class: [row indices into Z]}. "Labelled r" means the TRAINING
    label (teacher labels in the teacher track). Each cluster contributes the
    member of THAT cluster nearest its centre, so no sample is picked twice -
    the paper's per-cluster wording. The official code searches all of r's
    samples for each centroid instead (duplicates possible) and fixes
    random_state=0; both are declared deviations (PHASE6_7_SPEC §3.2).
    """
    memory: Dict[int, List[int]] = {}
    for r in classes:
        rows = np.flatnonzero(y == r)
        if len(rows) <= m:
            memory[int(r)] = [int(i) for i in rows]
            continue
        km = KMeans(n_clusters=m, random_state=seed, n_init=10).fit(Z[rows])
        picked = []
        for c in range(m):
            members = rows[km.labels_ == c]
            if len(members) == 0:
                raise RuntimeError(f"k-means left cluster {c} of class {r} empty")
            dist = np.linalg.norm(Z[members].astype(np.float64)
                                  - km.cluster_centers_[c].astype(np.float64), axis=1)
            picked.append(int(members[np.argmin(dist)]))
        assert len(set(picked)) == len(picked)
        memory[int(r)] = picked
    return memory


def prototype_eq4(Z: np.ndarray, y: np.ndarray, memory_rows: Sequence[int],
                  r: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eq. 4 for one relation, computed once, in the task where r first appears.
    Returns (p, p_static, p_dynamic), float64. No LayerNorm (declared
    deviation; CEAR applies one after mixing, Eq. 4 doesn't).
    """
    Z64 = Z.astype(np.float64)
    p_static = Z64[y == r].mean(axis=0)                  # Eq. 4: mean over ALL training samples of r
    p_dynamic = Z64[list(memory_rows)].mean(axis=0)      # Eq. 4: (1/|M_r|) sum over x_i in M_r of h_xi
    p = (1.0 - beta) * p_static + beta * p_dynamic       # Eq. 4: (1-beta) p_static + beta * dynamic
    return p, p_static, p_dynamic


def prototype_similarity(h: torch.Tensor, protos: torch.Tensor,
                         old_idx: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Eq. 10. h: [N, D] sample representations; protos: [C, D] indexed by
    GLOBAL class; old_idx: global indices of OLD = R-tilde_{k-1}.
    Returns s: [N, |OLD|], column j <-> old_idx[j], each row summing to 1.
    """
    p_old = protos[old_idx]                                      # Eq. 10: r_j and r_m range over OLD only
    cos = F.normalize(h, dim=1) @ F.normalize(p_old, dim=1).T    # Eq. 10: sim(h_xi, p_rj) = cosine
    return torch.softmax(cos / tau, dim=1)                       # Eq. 10: exp(sim/tau2) / sum over r_m in OLD


def fkd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
             s: torch.Tensor, y: torch.Tensor, old_idx: torch.Tensor,
             seen_mask: torch.Tensor, gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eqs. 11-13 for one batch of memory samples.

    student_logits, teacher_logits: [B, C] raw logits of theta_k and
    theta_{k-1} over all C head units. s: [B, |OLD|] from Eq. 10, columns in
    old_idx order. y: [B] training labels, global indices (may be a NEW
    class: every memory sample gets the FKD term, as in the paper and code).

    Returns (L_fkd, focal), focal being the per-sample Eq. 11 factor (for
    diagnostics). w, the teacher distribution and a are constants: computed
    under no_grad, as in CEAR main.py L271, L281.
    """
    with torch.no_grad():
        p_seen = torch.softmax(apply_mask(student_logits, seen_mask), dim=1)   # Eq. 11: P(.|x_i; theta_k), softmax over SEEN
        focal = (1.0 - p_seen.gather(1, y.unsqueeze(1)).squeeze(1)) ** gamma   # Eq. 11: (1 - P(y_i|x_i; theta_k))^gamma
        w = s * focal.unsqueeze(1)                                             # Eq. 11: w = s * focal - BOTH factors
        t = torch.softmax(teacher_logits[:, old_idx], dim=1)                   # Eq. 12: P(r_j|x_i; theta_{k-1}), softmax over OLD
        a = w * t                                                              # Eq. 12: a = w * P(r_j|x_i; theta_{k-1})
    log_q = torch.log_softmax(student_logits[:, old_idx], dim=1)               # Eq. 13: log P(r_j|x_i; theta_k), renormalised over OLD (CEAR L290-291)
    per_sample = -(a * log_q).sum(dim=1)                                       # Eq. 13: inner sum over r_j in OLD; weighted soft-target CE, not KL
    return per_sample.mean(), focal                                            # Eq. 13: 1/|M| sum over x_i (batch mean)


def param_hash(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _entropy_rows(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)


# -------------------------------------------------------------- regimes

class ReplayRegime(NaiveRegime):
    """
    Experience replay: naive new-task training, then a separate replay
    stage on the memory of all seen relations (Eq. 9).

    Replay-stage schedule (PHASE6_7_SPEC §3.6, declared deviation - CEAR runs
    a fixed 10 epochs): fresh optimiser with the new-task settings, batch 32
    with the last partial batch kept, patience 5, cap 120, early stopping on
    GOLD dev macro-F1 over all seen relations (dev under the seen mask). The
    pre-replay model is not a candidate, mirroring new-task training.
    """
    name = "replay"

    def __init__(self, seed: int, track: str, dev: Optional[Dict] = None,
                 feature_rows: Optional[np.ndarray] = None):
        assert track in ("teacher", "gold"), track
        self.seed, self.track = seed, track
        if dev is None:
            dev = build_eval_sets(seed)["dev"]
        self.dev_X = torch.from_numpy(dev["X"])
        self.dev_y = np.asarray(dev["y"])
        # task_data row -> row of the full feature matrix; resolved lazily
        # from the feature cache unless injected (tests).
        self.feature_rows = feature_rows
        ci = class_index(load_split(seed))
        self.class_names = {i: r for r, i in ci.items()}
        self.memory: Dict[int, List[int]] = {}
        self.replay_log: List[dict] = []
        self.pre_replay_hash: Dict[int, str] = {}   # lets tests prove the FKD
        self.stage = "new_task"                     # snapshot comes AFTER replay

    # ---- harness hook ------------------------------------------------

    def after_task(self, t: int, model: Student, task_data: TaskData) -> None:
        if self.feature_rows is None:
            self.feature_rows = self._resolve_feature_rows(task_data)
        Z = task_data.X[:, ENTITY_COLS]
        new = task_data.task_classes[t]
        assert not set(new) & set(self.memory), "memory is selected once per relation"

        self.memory.update(select_memory(Z, task_data.y, new,                 # step 2
                                         config.REPLAY_MEMORY_PER_RELATION, self.seed))
        self._after_memory(t, Z, task_data)                                   # step 3 (FKD)
        if t >= 1:
            self._replay_stage(t, model, task_data)                           # step 4
        self._end_of_task(t, model)                                           # FKD: teacher

    # ---- extension points (FKD overrides) ----------------------------

    def _after_memory(self, t: int, Z: np.ndarray, task_data: TaskData) -> None:
        pass

    def _stage_context(self, t: int, old: List[int], seen: List[int],
                       rows: np.ndarray, task_data: TaskData,
                       seen_mask: torch.Tensor) -> Dict:
        return {}

    def _replay_loss(self, l_cls: torch.Tensor, logits: torch.Tensor,
                     Xb: torch.Tensor, yb: torch.Tensor, idx: torch.Tensor,
                     ctx: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        return l_cls, {}

    def _finish_stage(self, t: int, ctx: Dict, epochs: List[dict]) -> None:
        pass

    def _end_of_task(self, t: int, model: Student) -> None:
        pass

    # ---- replay stage --------------------------------------------------

    def _replay_stage(self, t: int, model: Student, task_data: TaskData) -> None:
        self.pre_replay_hash[t] = param_hash(model)
        old = sorted(set().union(*task_data.task_classes[:t]))
        seen = sorted(set().union(*task_data.task_classes[:t + 1]))
        rows = np.array([i for r in seen for i in self.memory[r]])   # M-tilde_k: ALL seen relations, incl. R_k
        Xm = torch.from_numpy(task_data.X[rows])
        ym = torch.from_numpy(task_data.y[rows])
        n_classes = model.n_classes
        seen_mask = mask_vector(set(seen), n_classes)

        dev_idx = np.flatnonzero(np.isin(self.dev_y, seen))
        Xdv, ydv = self.dev_X[dev_idx], self.dev_y[dev_idx]

        ctx = self._stage_context(t, old, seen, rows, task_data, seen_mask)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)
        best_f1, best_state, best_epoch, patience = -1.0, None, -1, 0
        stop_reason = "epoch_cap"
        epochs: List[dict] = []

        self.stage = "replay"
        for epoch in range(config.MAX_EPOCHS_PER_TASK):
            model.train()
            perm = torch.randperm(len(Xm))
            sums: Counter = Counter()
            for i in range(0, len(perm), config.BATCH_SIZE):
                idx = perm[i:i + config.BATCH_SIZE]
                optimizer.zero_grad()
                logits = model(Xm[idx])
                l_cls = F.cross_entropy(apply_mask(logits, seen_mask), ym[idx])  # Eq. 9: CE on M-tilde_k, softmax over SEEN
                loss, parts = self._replay_loss(l_cls, logits, Xm[idx], ym[idx], idx, ctx)
                loss.backward()
                optimizer.step()
                sums["loss"] += loss.item() * len(idx)
                sums["L_cls"] += l_cls.item() * len(idx)
                for k, v in parts.items():
                    sums[k] += v
            dev_pred = masked_predict(model, Xdv, seen_mask)
            dev_f1, _ = macro_f1(ydv, dev_pred, n_classes)

            entry = {"task": t, "epoch": epoch,
                     "loss": round(sums["loss"] / len(Xm), 6),
                     "L_cls": round(sums["L_cls"] / len(Xm), 6),
                     "dev_macro_f1_seen": round(dev_f1, 6)}
            if "lam_L_fkd" in sums:
                entry["lam_L_fkd"] = round(sums["lam_L_fkd"] / len(Xm), 6)
                entry["focal_mean"] = round(sums["focal_sum"] / len(Xm), 6)
                entry["focal_gt_0.5"] = round(sums["focal_gt_half"] / len(Xm), 6)
            epochs.append(entry)
            self.replay_log.append(entry)

            if dev_f1 > best_f1:
                best_f1, best_epoch, patience = dev_f1, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= config.EARLY_STOPPING_PATIENCE:
                    stop_reason = f"early_stop(patience={config.EARLY_STOPPING_PATIENCE})"
                    break
        self.stage = "new_task"

        if config.RESTORE_BEST_WEIGHTS and best_state is not None:
            model.load_state_dict(best_state)
        self.replay_log.append({"task": t, "stop_reason": stop_reason,
                                "best_epoch": best_epoch,
                                "best_dev_macro_f1_seen": round(best_f1, 6),
                                "n_memory": int(len(rows)),
                                "n_dev_seen": int(len(dev_idx))})
        self._finish_stage(t, ctx, epochs)

    # ---- bookkeeping ---------------------------------------------------

    def _resolve_feature_rows(self, task_data: TaskData) -> np.ndarray:
        """Mirror harness._track_train_data's row filter, and check it."""
        feats = build_features(config.CORPUS)
        kept = [DIALOGRE_RELATIONS.index(self.class_names[i])
                for i in sorted(self.class_names)]
        y_full = feats["y_teacher"] if self.track == "teacher" else feats["y_gold"]
        rows = np.flatnonzero(np.isin(y_full, kept))
        assert np.array_equal(feats["X"][rows], task_data.X), "feature row map is wrong"
        return rows

    def memory_payload(self) -> Dict:
        return {
            "track": self.track, "seed": self.seed, "regime": self.name,
            "memory_size": config.REPLAY_MEMORY_PER_RELATION,
            "selection": "kmeans per relation, nearest member within each cluster, "
                         "random_state=seed, n_init=10, subject+object features",
            "classes": {
                self.class_names[r]: {
                    "head_index": r,
                    "train_rows": rows,
                    "feature_rows": [int(self.feature_rows[i]) for i in rows],
                }
                for r, rows in sorted(self.memory.items())
            },
        }


class FKDRegime(ReplayRegime):
    """
    Replay plus focal knowledge distillation in the replay stage:
    L_replay = Eq. 9 + lambda * L_fkd (Eq. 14, linear method only).

    The teacher theta_{k-1} is a frozen deep copy taken at the end of
    after_task(k-1), i.e. AFTER task k-1's replay stage (CEAR main.py L115).
    Features are frozen, so prototypes and every s(x, r) are constant within
    a task and s is precomputed once per replay stage.
    """
    name = "fkd"

    def __init__(self, seed: int, track: str, lam: float = config.FKD_LAMBDA,
                 **kw):
        super().__init__(seed, track, **kw)
        self.lam = lam
        self.gamma = config.FKD_GAMMA
        self.tau = config.FKD_TAU2
        self.beta = config.FKD_BETA
        self.protos: Optional[np.ndarray] = None   # [C, D], NaN until first learned
        self.p_static: Dict[int, np.ndarray] = {}
        self.p_dynamic: Dict[int, np.ndarray] = {}
        self.teacher: Optional[Student] = None
        self.snapshot_hash: Dict[int, str] = {}
        self.diagnostics: List[dict] = []

    def _after_memory(self, t, Z, task_data):
        if self.protos is None:
            n_classes = len(self.class_names)
            self.protos = np.full((n_classes, Z.shape[1]), np.nan)
        for r in task_data.task_classes[t]:
            assert r not in self.p_static, "p_static is stored once, never recomputed"
            p, ps, pd = prototype_eq4(Z, task_data.y, self.memory[r], r, self.beta)
            self.protos[r], self.p_static[r], self.p_dynamic[r] = p, ps, pd

    def _stage_context(self, t, old, seen, rows, task_data, seen_mask):
        assert self.teacher is not None, "teacher snapshot missing"
        old_idx = torch.tensor(old, dtype=torch.long)
        assert not np.isnan(self.protos[old]).any(), "prototype missing for an OLD relation"
        h = torch.from_numpy(task_data.X[rows][:, ENTITY_COLS].astype(np.float64))
        s = prototype_similarity(h, torch.from_numpy(self.protos), old_idx, self.tau)  # Eq. 10, once per task
        return {"old": old, "seen": seen, "rows": rows, "old_idx": old_idx,
                "seen_mask": seen_mask, "s": s.float(),
                "y": task_data.y[rows],
                "teacher_hash_before": param_hash(self.teacher)}

    def _replay_loss(self, l_cls, logits, Xb, yb, idx, ctx):
        with torch.no_grad():
            teacher_logits = self.teacher(Xb)                    # theta_{k-1}, eval mode, frozen
        l_fkd, focal = fkd_loss(logits, teacher_logits, ctx["s"][idx], yb,
                                ctx["old_idx"], ctx["seen_mask"], self.gamma)
        loss = l_cls + self.lam * l_fkd                          # Eq. 14, linear method: L_cls + lambda_2 * L_l_fkd
        n = len(idx)
        return loss, {"lam_L_fkd": (self.lam * l_fkd).item() * n,
                      "focal_sum": focal.sum().item(),
                      "focal_gt_half": int((focal > 0.5).sum())}

    def _finish_stage(self, t, ctx, epochs):
        hash_after = param_hash(self.teacher)
        if hash_after != ctx["teacher_hash_before"]:
            raise RuntimeError(f"teacher changed during the task-{t} replay stage")

        s = ctx["s"].double().numpy()
        n_old = len(ctx["old"])
        col = {r: j for j, r in enumerate(ctx["old"])}
        name_to_idx = {n: i for i, n in self.class_names.items()}
        pairs = []
        for a, b in ANALOGOUS_PAIRS:
            ia, ib = name_to_idx[a], name_to_idx[b]

            def directed(src, dst):
                # mean s(x_i, dst) over memory samples labelled src; defined
                # only when dst is OLD and src has memory
                mask = ctx["y"] == src
                if dst not in col or not mask.any():
                    return None
                return round(float(s[mask, col[dst]].mean()), 6)

            pairs.append({"a": a, "b": b, "a_to_b": directed(ia, ib),
                          "b_to_a": directed(ib, ia)})

        self.diagnostics.append({
            "task": t,
            "n_old": n_old,
            "n_memory": int(len(ctx["rows"])),
            "s": {"mean_row_entropy": round(float(_entropy_rows(s).mean()), 6),
                  "uniform_entropy": round(math.log(n_old), 6),
                  "mean_row_max": round(float(s.max(axis=1).mean()), 6),
                  "uniform_max": round(1.0 / n_old, 6)},
            "analogous_pairs_mean_s": pairs,
            "epochs": [{k: e[k] for k in ("epoch", "L_cls", "lam_L_fkd",
                                           "focal_mean", "focal_gt_0.5")}
                       for e in epochs],
            "teacher": {"snapshot_of_task": t - 1,
                        "hash_before": ctx["teacher_hash_before"],
                        "hash_after": hash_after,
                        "snapshot_hash_at_end_of_prev_task": self.snapshot_hash[t - 1],
                        "eval_mode": not self.teacher.training,
                        "any_requires_grad": any(p.requires_grad
                                                 for p in self.teacher.parameters())},
        })

    def _end_of_task(self, t, model):
        # Snapshot AFTER this task's replay stage: the teacher for task t+1.
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.snapshot_hash[t] = param_hash(model)


REGIMES = {"replay": ReplayRegime, "fkd": FKDRegime}


# -------------------------------------------------------------- outputs

def write_artifacts(regime: ReplayRegime, out_dir: Path) -> None:
    """
    Write memory_indices.json, replay_log.jsonl and (FKD) fkd_diagnostics.json
    into the run directory, then assert the memory matches any other run of
    the same track and seed in the same results root.
    """
    payload = regime.memory_payload()
    (out_dir / "memory_indices.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with open(out_dir / "replay_log.jsonl", "w", encoding="utf-8") as fh:
        for entry in regime.replay_log:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    if isinstance(regime, FKDRegime):
        (out_dir / "fkd_diagnostics.json").write_text(json.dumps({
            "lambda": regime.lam, "gamma": regime.gamma, "tau2": regime.tau,
            "beta": regime.beta, "tasks": regime.diagnostics,
        }, indent=2, sort_keys=True))

    for other in out_dir.parent.glob("*/memory_indices.json"):
        if other.parent == out_dir:
            continue
        theirs = json.loads(other.read_text(encoding="utf-8"))
        if (theirs.get("track"), theirs.get("seed")) != (regime.track, regime.seed):
            continue
        mine = {k: v["train_rows"] for k, v in payload["classes"].items()}
        if {k: v["train_rows"] for k, v in theirs["classes"].items()} != mine:
            raise RuntimeError(f"memory differs from {other.parent.name} "
                               f"for track={regime.track} seed={regime.seed}")


def prereg_committed() -> Tuple[bool, str]:
    """PREREG.md at HEAD must have no blanks left (lines 1-6 filled in)."""
    repo = Path(__file__).resolve().parent.parent
    try:
        text = subprocess.run(["git", "show", "HEAD:PREREG.md"], cwd=repo,
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return False, "PREREG.md is not committed"
    if "____" in text or "YYYY-MM-DD" in text:
        return False, "PREREG.md at HEAD still has blanks (lines 1-6 not committed)"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description="Replay / FKD continual-learning run")
    ap.add_argument("--track", choices=["teacher", "gold"], required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--regime", choices=sorted(REGIMES), required=True)
    args = ap.parse_args()

    ok, why = prereg_committed()
    if not ok:
        sys.exit(f"refusing to run: {why}")

    regime = REGIMES[args.regime](args.seed, args.track)
    out_dir = run_sequential(args.track, args.seed, regime)
    write_artifacts(regime, out_dir)
    print(f"artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
