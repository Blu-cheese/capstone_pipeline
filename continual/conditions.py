"""
Training-label conditions, i.e. tracks (RQ3_SPEC §2.2 with the Gate 1 amendments).

A track maps (frozen features, class index, draw seed) to the training rows and
their head labels. It is the ONLY place a run's training labels come from:
harness._track_train_data, regimes.resolve_feature_rows and run_all's joint
regimes all call train_rows().

    gold, teacher      Phase 4-7 tracks. Rows and order exactly as before
                       (every training row whose label is in the 17 classes).
    gold_I             gold labels on I
    teacher_I          teacher labels on I
    perm_I             teacher labels permuted within each gold class (drawn)
    rand_I             gold with exactly round(eps_I * |I|) rows flipped to a
                       uniform other class (drawn)
    gold_sub           gold on a uniform subset of the gold-track rows, size |I|
                       (drawn)

I = rows whose gold AND teacher labels are both in the 17 kept classes (710).

Replicates (Gate 1): replicate r runs with init seed = seed + 100r and, for the
drawn tracks only, draw seed = seed + 100r. Replicate 0 passes init_seed=None,
so it takes exactly the pre-RQ3 code path. KMeans memory selection stays on the
task-order seed (regimes.py), so it is not touched here.

Every draw is frozen to results/conditions/ before the sweep:
    {track}_{seed}.json          replicate 0, every RQ3 track
    {track}_{seed}_r{r}.json     replicates 1 and 2, drawn tracks only
and run_all refuses to start a run whose labels differ from the committed file.

    venv/bin/python -m continual.conditions           # write all files
    venv/bin/python -m continual.conditions --check   # verify them
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS

PHASE7_TRACKS = ("teacher", "gold")
RQ3_TRACKS = ("gold_I", "teacher_I", "perm_I", "rand_I", "gold_sub")
DRAWN_TRACKS = ("perm_I", "rand_I", "gold_sub")
ALL_TRACKS = PHASE7_TRACKS + RQ3_TRACKS
REPLICATES = (0, 1, 2)
REPLICATE_OFFSET = 100

# RQ3_SPEC §2.4: gold_sub runs joint and replay only.
RQ3_REGIMES = {t: ("joint", "joint_balanced", "naive", "replay") for t in RQ3_TRACKS}
RQ3_REGIMES["gold_sub"] = ("joint", "replay")


def conditions_dir() -> Path:
    return config.RESULTS_DIR / "conditions"


# ------------------------------------------------------------- seeds

def init_seed_for(seed: int, replicate: int) -> Optional[int]:
    """None for replicate 0 (the unchanged default path), else seed + 100r."""
    return None if replicate == 0 else seed + REPLICATE_OFFSET * replicate


def draw_seed_for(track: str, seed: int, replicate: int) -> Optional[int]:
    """seed + 100r for the drawn tracks; None for tracks with no draw."""
    return seed + REPLICATE_OFFSET * replicate if track in DRAWN_TRACKS else None


# ------------------------------------------------------------ labels

def _full_to_head(ci: Dict[str, int]) -> Dict[int, int]:
    return {DIALOGRE_RELATIONS.index(r): h for r, h in ci.items()}


def head_labels(y_full: np.ndarray, ci: Dict[str, int]) -> np.ndarray:
    """37-way labels -> head indices, -1 outside the kept classes."""
    f2h = _full_to_head(ci)
    return np.array([f2h.get(int(v), -1) for v in y_full], dtype=np.int64)


def intersection_rows(feats: Dict, ci: Dict[str, int]) -> np.ndarray:
    yt, yg = head_labels(feats["y_teacher"], ci), head_labels(feats["y_gold"], ci)
    return np.flatnonzero((yt >= 0) & (yg >= 0))


def train_rows(track: str, feats: Dict, ci: Dict[str, int],
               draw_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (feature-matrix row ids, head labels) for one track. Row ids ascending.

    For gold and teacher this is the pre-RQ3 computation verbatim (boolean
    filter on the track's own label, labels mapped to the head), so those
    tracks are bit-identical by construction - and by test.
    """
    if track not in ALL_TRACKS:
        raise ValueError(f"unknown track {track!r}")
    if (track in DRAWN_TRACKS) != (draw_seed is not None):
        raise ValueError(f"{track}: draw_seed must be "
                         f"{'given' if track in DRAWN_TRACKS else 'None'}")
    f2h = _full_to_head(ci)

    if track in PHASE7_TRACKS:
        y_full = feats["y_teacher"] if track == "teacher" else feats["y_gold"]
        keep = np.isin(y_full, list(f2h))
        rows = np.flatnonzero(keep)
        y = np.array([f2h[int(v)] for v in y_full[keep]], dtype=np.int64)
        return rows, y

    yt, yg = head_labels(feats["y_teacher"], ci), head_labels(feats["y_gold"], ci)
    I = np.flatnonzero((yt >= 0) & (yg >= 0))
    g, t = yg[I], yt[I]
    n_classes = len(ci)

    if track == "gold_I":
        return I, g.copy()
    if track == "teacher_I":
        return I, t.copy()

    rng = np.random.default_rng(draw_seed)
    if track == "perm_I":
        # Permute the teacher labels among the rows of each gold class: C[g][t]
        # and both marginals are kept exactly, only the row <-> wrong-label
        # pairing changes.
        y = t.copy()
        for c in range(n_classes):
            idx = np.flatnonzero(g == c)
            y[idx] = t[idx][rng.permutation(len(idx))]
        return I, y
    if track == "rand_I":
        # Exact flip count (not Bernoulli); a flipped row never keeps its gold.
        n_flip = int(round((1.0 - float((g == t).mean())) * len(I)))
        y = g.copy()
        for i in np.sort(rng.choice(len(I), size=n_flip, replace=False)):
            y[i] = rng.choice([c for c in range(n_classes) if c != g[i]])
        return I, y
    # gold_sub: uniform subset of the gold-track rows, size |I|, no teacher selection
    gold_rows = np.flatnonzero(yg >= 0)
    sub = np.sort(rng.choice(gold_rows, size=len(I), replace=False))
    return sub, yg[sub]


# ------------------------------------------------------- saved draws

def condition_path(track: str, seed: int, replicate: int) -> Path:
    if replicate and track not in DRAWN_TRACKS:
        replicate = 0       # no draw: every replicate uses the replicate-0 labels
    name = f"{track}_{seed}.json" if replicate == 0 else f"{track}_{seed}_r{replicate}.json"
    return conditions_dir() / name


def _sha(rows: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(rows, dtype=np.int64).tobytes())
    h.update(np.asarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def condition_payload(track: str, seed: int, replicate: int, feats: Dict,
                      ci: Dict[str, int]) -> Dict:
    draw = draw_seed_for(track, seed, replicate)
    rows, y = train_rows(track, feats, ci, draw)
    gold = head_labels(feats["y_gold"], ci)[rows]
    teacher = head_labels(feats["y_teacher"], ci)[rows]
    return {
        "track": track, "seed": seed, "replicate": replicate if track in DRAWN_TRACKS else 0,
        "draw_seed": draw, "n_rows": int(len(rows)),
        "n_differ_from_gold": int((y != gold).sum()),
        "noise_rate_vs_gold": round(float((y != gold).mean()), 6),
        "head_order": sorted(ci, key=ci.get),
        "sha256_rows_labels": _sha(rows, y),
        "row_ids": [int(r) for r in rows],
        "labels": [int(v) for v in y],
        "gold": [int(v) for v in gold],          # -1 = gold outside the 17 (never on I)
        "teacher": [int(v) for v in teacher],    # -1 = teacher outside the 17
    }


def all_condition_keys():
    """(track, seed, replicate) for every file that must exist."""
    for seed in config.SEEDS:
        for track in RQ3_TRACKS:
            for r in (REPLICATES if track in DRAWN_TRACKS else (0,)):
                yield track, seed, r


def write_conditions(feats: Dict, ci_of_seed) -> int:
    conditions_dir().mkdir(parents=True, exist_ok=True)
    n = 0
    for track, seed, r in all_condition_keys():
        p = condition_payload(track, seed, r, feats, ci_of_seed(seed))
        condition_path(track, seed, r).write_text(json.dumps(p) + "\n", encoding="utf-8")
        n += 1
    return n


def load_condition(track: str, seed: int, replicate: int) -> Dict:
    return json.loads(condition_path(track, seed, replicate).read_text(encoding="utf-8"))


def check_saved(track: str, seed: int, replicate: int, rows: np.ndarray,
                y: np.ndarray, require_committed: bool = False) -> None:
    """Raise unless the saved file exists, matches (rows, y) and, if asked, is committed."""
    path = condition_path(track, seed, replicate)
    if not path.exists():
        raise RuntimeError(f"{path} missing - run `python -m continual.conditions`")
    saved = load_condition(track, seed, replicate)
    if saved["row_ids"] != [int(r) for r in rows] or saved["labels"] != [int(v) for v in y]:
        raise RuntimeError(f"{path} does not match the labels built for this run")
    if require_committed:
        repo = Path(__file__).resolve().parent.parent
        rel = path.resolve().relative_to(repo)
        ok = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=repo,
                            capture_output=True).returncode == 0
        if not ok:
            raise RuntimeError(f"{path} is not committed")


def main():
    from continual.features import build_features
    from continual.task_split import load_split, class_index
    ap = argparse.ArgumentParser(description="Freeze RQ3 label conditions to disk")
    ap.add_argument("--check", action="store_true", help="verify instead of writing")
    args = ap.parse_args()
    feats = build_features(config.CORPUS)
    ci_of_seed = lambda s: class_index(load_split(s))
    if args.check:
        for track, seed, r in all_condition_keys():
            rows, y = train_rows(track, feats, ci_of_seed(seed), draw_seed_for(track, seed, r))
            check_saved(track, seed, r, rows, y)
        print("all condition files match")
        return
    n = write_conditions(feats, ci_of_seed)
    print(f"wrote {n} files to {conditions_dir()}")
    for track, seed, r in all_condition_keys():
        p = load_condition(track, seed, r)
        print(f"  {track:9s} seed {seed:4d} r{r}  rows {p['n_rows']:4d}  "
              f"noise {p['noise_rate_vs_gold']:.4f}  sha {p['sha256_rows_labels']}")


if __name__ == "__main__":
    main()
