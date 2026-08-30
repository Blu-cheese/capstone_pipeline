"""
Harness acceptance tests (PHASE5_SPEC §4).

    AC2 - mask unit test: masked-out classes have logit <= finfo.min and are
          never argmax-predicted, for train-style and eval-style masks.
    AC4 - R_matrix.json schema validation (on a synthetic payload here; the
          harness also validates every real payload before writing it).

Run:  venv/bin/python tests/test_harness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from continual.harness import apply_mask, mask_vector, NaiveRegime
from continual.metrics import RMatrix, r_matrix_payload, validate_payload

_passed, _failed = 0, 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}")
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f"  -- {extra}" if extra else ""))


def test_ac2_masking():
    torch.manual_seed(0)
    n = 17
    logits = torch.randn(512, n) * 10  # large spread: try to beat the mask

    # Train-style mask: a single task's classes (task 1 of seed 1234: 4 classes)
    train_mask = mask_vector({3, 7, 11, 15}, n)
    m = apply_mask(logits, train_mask)
    fmin = torch.finfo(logits.dtype).min
    check("AC2 train mask: masked logits at finfo.min",
          bool((m[:, ~train_mask] <= fmin).all()))
    pred = m.argmax(dim=1)
    check("AC2 train mask: argmax never masked-out",
          bool(train_mask[pred].all()))

    # Eval-style mask: union of tasks 0+1 (11 of 17 classes)
    eval_mask = mask_vector(set(range(11)), n)
    m2 = apply_mask(logits, eval_mask)
    check("AC2 eval mask: masked logits at finfo.min",
          bool((m2[:, 11:] <= fmin).all()))
    check("AC2 eval mask: argmax never masked-out",
          bool(eval_mask[m2.argmax(dim=1)].all()))

    # The mask must not mutate the input tensor (masked_fill is out-of-place).
    check("AC2 apply_mask is non-destructive",
          bool((logits[:, ~train_mask] > fmin).any()))

    # Gradient path: CE through a masked target class stays finite.
    lg = torch.randn(8, n, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(
        apply_mask(lg, train_mask), torch.tensor([3, 7, 11, 15, 3, 7, 11, 15]))
    loss.backward()
    check("AC2 masked CE loss finite with active targets",
          bool(torch.isfinite(loss)))


def test_ac4_schema():
    r_f1, r_acc = RMatrix(3), RMatrix(3)
    for t in range(3):
        for j in range(t + 1):
            r_f1.set(t, j, 0.5)
            r_acc.set(t, j, 0.6)
    payload = r_matrix_payload("x_naive_1234_t", "gold", "naive", 1234,
                               [[0], [1], [2]], r_f1, r_acc,
                               {"track": "gold", "test_macro_f1": 0.45},
                               [])
    check("AC4 valid payload passes", validate_payload(payload) == [])

    # Upper-triangular zero must be rejected.
    bad = json._default_decoder.decode(json.dumps(payload)) if False else dict(payload)
    bad["R_f1"] = [[0.5, 0.0, None], [0.5, 0.5, None], [0.5, 0.5, 0.5]]
    problems = validate_payload(bad)
    check("AC4 upper-triangular 0.0 rejected", any("null above diagonal" in p for p in problems), str(problems))

    # RMatrix refuses upper-triangular writes outright.
    try:
        RMatrix(3).set(0, 2, 0.5)
        check("AC4 RMatrix refuses upper-tri write", False, "no error raised")
    except ValueError:
        check("AC4 RMatrix refuses upper-tri write", True)

    # BWT undefined at K=1 -> None, not 0.
    r1 = RMatrix(1)
    r1.set(0, 0, 0.9)
    check("AC4 K=1 BWT is None", r1.bwt() is None)
    check("AC4 K=1 ACC works", r1.acc() == 0.9)


def test_regime_interface():
    """NaiveRegime satisfies the four-hook protocol trivially."""
    r = NaiveRegime()
    tc = [[0, 1], [2, 3], [4, 5]]
    check("regime: train_class_set t=0 = task 0", r.train_class_set(0, tc) == {0, 1})
    check("regime: train_class_set = seen through t", r.train_class_set(1, tc) == {0, 1, 2, 3})
    check("regime: train_class_set t=2 = all", r.train_class_set(2, tc) == {0, 1, 2, 3, 4, 5})
    lg = torch.randn(4, 6)
    m = mask_vector({2, 3}, 6)
    loss = r.loss(apply_mask(lg, m), torch.tensor([2, 3, 2, 3]), lg, 1)
    check("regime: default loss is masked CE, finite", bool(torch.isfinite(loss)))
    check("regime: after_task is a no-op", r.after_task(0, None, None) is None)


import json  # noqa: E402  (used in test_ac4_schema)

if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{'='*50}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
