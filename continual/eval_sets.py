"""
Frozen evaluation sets (METHOD_A_SPEC_V2 §Global config).

Built ONCE, written to disk, then read-only. The harness reads them; nothing
else touches them. Replay buffers and FKD prototypes draw exclusively from
training data - never from these.

Policy, per V2's corrected label-source table:

  * Eval sets use GOLD labels for BOTH tracks. Evaluating the teacher-track
    student against teacher labels would measure fidelity to Gemini, not
    accuracy. The tracks differ in exactly one variable - training-label
    source - and that is what makes the comparison mean anything.

  * Eval pairs are taken from gold dev/test AS-IS. They are NOT filtered by
    what the teacher would have labelled; doing so would reintroduce teacher
    bias into the measurement.

  * Multi-label pairs are DROPPED, not collapsed to first (75 dev, 49 test).

  * Only the split's kept relations are retained; dropped classes have no
    output unit on the student head.

This works because the teacher was prompted with DialogRE's native relation
set, so both label spaces already coincide.

Usage:
    venv/bin/python -m continual.eval_sets --seed 1234
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.features import ENCODER_NAME, _load_encoder
from continual.task_split import load_split, class_index, kept_relations
from preprocessing.dialogre_parser import load_dialogre


def build_eval_sets(seed: int = 1234, rebuild: bool = False,
                    batch_size: int = 64) -> Dict:
    """
    Build (or load) frozen dev/test eval sets for a task split.

    Returns {split_name: {"X", "y", "task"}} where `task` gives each example's
    task index, so the harness can slice D_test_j without re-deriving it.
    """
    split = load_split(seed)
    ci = class_index(split)
    r2t = split["relation_to_task"]
    n_classes = len(ci)

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.EVAL_DIR / f"eval_seed{seed}_{config.ENCODER_SLUG}.npz"
    meta_path = config.EVAL_DIR / f"eval_seed{seed}_{config.ENCODER_SLUG}.meta.json"

    if cache.exists() and not rebuild:
        d = np.load(cache, allow_pickle=False)
        meta = json.loads(meta_path.read_text())
        print(f"[eval_sets] loaded frozen sets from {cache}")
        return {
            "dev": {"X": d["dev_X"], "y": d["dev_y"], "task": d["dev_task"]},
            "test": {"X": d["test_X"], "y": d["test_y"], "task": d["test_task"]},
            "meta": meta, "class_index": ci, "n_classes": n_classes,
        }

    model = _load_encoder()
    out, meta_splits = {}, {}

    for split_name in ("dev", "test"):
        units = load_dialogre(f"data/raw/dialogre/{split_name}.json")

        windows, subjects, objects, labels, tasks = [], [], [], [], []
        dropped_multilabel = dropped_class = 0

        for unit in units:
            for pair in unit.pairs:
                if pair.is_multilabel:
                    dropped_multilabel += 1
                    continue
                rel = pair.gold
                if rel not in ci:
                    dropped_class += 1
                    continue
                windows.append(unit.text)
                subjects.append(pair.x)
                objects.append(pair.y)
                labels.append(ci[rel])
                tasks.append(r2t[rel])

        if not labels:
            raise ValueError(f"no usable eval examples for {split_name}")

        print(f"[eval_sets] encoding {len(labels)} {split_name} examples on CPU...")
        enc = lambda xs: model.encode(xs, batch_size=batch_size,
                                      show_progress_bar=False, convert_to_numpy=True)
        X = np.concatenate([enc(windows), enc(subjects), enc(objects)],
                           axis=1).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        t = np.array(tasks, dtype=np.int64)

        assert X.shape == (len(labels), config.FEATURE_DIM)
        out[split_name] = {"X": X, "y": y, "task": t}

        per_task = {int(k): int(v) for k, v in zip(*np.unique(t, return_counts=True))}
        meta_splits[split_name] = {
            "n_examples": int(len(labels)),
            "per_task": per_task,
            "dropped_multilabel": dropped_multilabel,
            "dropped_not_in_kept_classes": dropped_class,
        }
        print(f"[eval_sets]   {split_name}: {len(labels)} kept, "
              f"{dropped_multilabel} multi-label dropped, "
              f"{dropped_class} outside the {n_classes} kept classes")

    np.savez_compressed(
        cache,
        dev_X=out["dev"]["X"], dev_y=out["dev"]["y"], dev_task=out["dev"]["task"],
        test_X=out["test"]["X"], test_y=out["test"]["y"], test_task=out["test"]["task"],
    )
    meta = {
        "seed": seed,
        "encoder": ENCODER_NAME,
        "n_classes": n_classes,
        "label_source": "gold (both tracks)",
        "multi_label_policy": "dropped, not collapsed to first",
        "relations": kept_relations(split),
        "splits": meta_splits,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[eval_sets] frozen to {cache}")

    return {**out, "meta": meta, "class_index": ci, "n_classes": n_classes}


def main():
    ap = argparse.ArgumentParser(description="Freeze gold dev/test eval sets")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    ev = build_eval_sets(args.seed, args.rebuild)
    print(f"\n{'='*62}\nFROZEN EVAL SETS  seed={args.seed}  "
          f"N_CLASSES={ev['n_classes']}\n{'='*62}")
    for name in ("dev", "test"):
        m = ev["meta"]["splits"][name]
        print(f"  {name:5s}: {m['n_examples']:5d} examples   per-task "
              f"{m['per_task']}   [{m['dropped_multilabel']} multi-label dropped]")


if __name__ == "__main__":
    main()
