"""
Feature extraction for the continual student (METHOD_A_SPEC Phase 2).

Each training example is one entity pair in one dialogue. Three strings are
embedded with a FROZEN sentence encoder and concatenated:

    [ dialogue/window text | subject span | object span ]  ->  384 * 3 = 1152

The encoder is never fine-tuned and never backpropagated through. That is
what makes the student cheap enough to train sequentially on CPU in seconds
per epoch, which is the whole reason Fusion A is feasible on this timeline.

CPU ONLY. No CUDA, no MPS, no device-selection logic - see CLAUDE.md 9.

Usage:
    venv/bin/python -m continual.features
    venv/bin/python -m continual.features --rebuild
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.dialogre_parser import DIALOGRE_RELATIONS, RELATION_TO_INDEX

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_SLUG = "minilm-l6-v2"
EMBED_DIM = 384
FEATURE_DIM = EMBED_DIM * 3

FEATURE_DIR = Path("data/features")
LABELED_DIR = Path("data/labeled")


def _load_encoder():
    """Load the frozen encoder, pinned to CPU."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER_NAME, device="cpu")
    model.eval()
    return model


def build_features(
    corpus: str = "dialogre_train",
    rebuild: bool = False,
    batch_size: int = 64,
) -> Dict:
    """
    Build (or load) the cached feature matrix for a labelled corpus.

    Returns a dict with:
        X            (N, 1152) float32 feature matrix
        y_teacher    (N,)      int64 teacher label indices - what to TRAIN on
        y_gold       (N,)      int64 gold label indices    - eval only
        relations    list[str] index -> relation name
        meta         dict
    """
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cache = FEATURE_DIR / f"{corpus}_{ENCODER_SLUG}.npz"

    if cache.exists() and not rebuild:
        t0 = time.time()
        data = np.load(cache, allow_pickle=False)
        meta = json.loads((FEATURE_DIR / f"{corpus}_{ENCODER_SLUG}.meta.json").read_text())
        print(f"[features] loaded cache {cache} in {time.time()-t0:.2f}s  "
              f"X={data['X'].shape}")
        return {
            "X": data["X"], "y_teacher": data["y_teacher"], "y_gold": data["y_gold"],
            "relations": DIALOGRE_RELATIONS, "meta": meta,
        }

    src = LABELED_DIR / f"{corpus}.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found - run `python -m labeling.label_corpus` first."
        )

    rows: List[dict] = json.loads(src.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"{src} is empty")

    # Drop rows whose teacher label is outside the frozen label space. Should
    # be impossible (the teacher is enum-constrained) but a silent bad index
    # would corrupt the student's head.
    usable = [r for r in rows
              if r.get("relation") in RELATION_TO_INDEX
              and r.get("gold_relation") in RELATION_TO_INDEX]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"[features] dropped {dropped} rows with out-of-space labels")

    print(f"[features] encoding {len(usable)} examples x 3 fields on CPU...")
    model = _load_encoder()

    t0 = time.time()
    windows = model.encode([r["window_text"] for r in usable],
                           batch_size=batch_size, show_progress_bar=False,
                           convert_to_numpy=True)
    subjects = model.encode([r["subject"] for r in usable],
                            batch_size=batch_size, show_progress_bar=False,
                            convert_to_numpy=True)
    objects = model.encode([r["object"] for r in usable],
                           batch_size=batch_size, show_progress_bar=False,
                           convert_to_numpy=True)
    elapsed = time.time() - t0

    X = np.concatenate([windows, subjects, objects], axis=1).astype(np.float32)
    y_teacher = np.array([RELATION_TO_INDEX[r["relation"]] for r in usable], dtype=np.int64)
    y_gold = np.array([RELATION_TO_INDEX[r["gold_relation"]] for r in usable], dtype=np.int64)

    assert X.shape == (len(usable), FEATURE_DIM), f"bad feature shape {X.shape}"
    assert y_teacher.shape == (len(usable),)

    np.savez_compressed(cache, X=X, y_teacher=y_teacher, y_gold=y_gold)

    agreement = float((y_teacher == y_gold).mean())
    meta = {
        "corpus": corpus,
        "encoder": ENCODER_NAME,
        "n_examples": int(len(usable)),
        "feature_dim": FEATURE_DIM,
        "n_classes": len(DIALOGRE_RELATIONS),
        "dropped_rows": dropped,
        "encode_seconds": round(elapsed, 1),
        "teacher_gold_agreement": round(agreement, 4),
    }
    (FEATURE_DIR / f"{corpus}_{ENCODER_SLUG}.meta.json").write_text(
        json.dumps(meta, indent=2))

    print(f"[features] encoded in {elapsed:.1f}s -> X={X.shape}  cached at {cache}")
    print(f"[features] teacher/gold agreement: {100*agreement:.1f}%")
    _print_distribution(y_teacher, y_gold)

    return {"X": X, "y_teacher": y_teacher, "y_gold": y_gold,
            "relations": DIALOGRE_RELATIONS, "meta": meta}


def _print_distribution(y_teacher: np.ndarray, y_gold: np.ndarray) -> None:
    from collections import Counter
    tc, gc = Counter(y_teacher.tolist()), Counter(y_gold.tolist())
    present = sorted(set(tc) | set(gc), key=lambda i: -tc.get(i, 0))
    print(f"\n{'relation':32s} {'teacher':>8s} {'gold':>8s}")
    print("-" * 50)
    for i in present:
        print(f"{DIALOGRE_RELATIONS[i]:32s} {tc.get(i,0):8d} {gc.get(i,0):8d}")
    thin = [DIALOGRE_RELATIONS[i] for i in tc if tc[i] < 20]
    if thin:
        print(f"\n{len(thin)} teacher classes under 20 examples "
              f"(Phase 3 merges or drops these): {thin}")


def main():
    ap = argparse.ArgumentParser(description="Build cached features for the student")
    ap.add_argument("--corpus", default="dialogre_train")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    build_features(args.corpus, args.rebuild, args.batch_size)


if __name__ == "__main__":
    main()
