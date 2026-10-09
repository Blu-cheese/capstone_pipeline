"""
Tests for the replay and FKD regimes (PHASE6_7_SPEC §3.5). Run from
capstone_pipeline:

    venv/bin/python tests/test_fkd.py

No pytest - plain asserts and a tally, like the other test files.

The seven spec tests, plus checks that pin silent errors 4-6 directly:

  1. hand-calculated L_fkd, non-contiguous OLD indices (silent errors 3, 7, 8)
  2. s index sets: shape, row sums, new prototype closest (silent error 3)
  3. FKD with lambda = 0 reproduces replay bit for bit
  4. replay's and FKD's R row 0 equal naive's bit for bit
  5. teacher frozen, and snapshotted AFTER the previous replay stage (error 5)
  6. no FKD call during new-task training (error 1)
  7. naive untouched: 2 tracks x 3 seeds bit-identical to the Phase 5 files

Tests 3 and 4 run end to end on SYNTHETIC data (random class-clustered
features through the real harness and the real seed-1234 split). Their
real-data versions are replay/FKD runs, so they run only once PREREG.md
lines 1-6 are committed - the same guard the regimes CLI uses. Until then
they print SKIP. Test 7 is naive-only and always runs on real data.
"""

import json
import math
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import continual.harness as H
import continual.regimes as RG
from continual.harness import NaiveRegime, run_sequential, apply_mask
from continual.regimes import (
    ReplayRegime, FKDRegime, fkd_loss, prototype_similarity, prototype_eq4,
    select_memory, write_artifacts, prereg_committed,
)
from continual.task_split import load_split, class_index
from preprocessing.dialogre_parser import DIALOGRE_RELATIONS

torch.use_deterministic_algorithms(True)

_passed, _failed, _skipped = 0, 0, 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}" + (f"  ({extra})" if extra else ""))
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f"  ({extra})" if extra else ""))


def skip(name, why):
    global _skipped
    _skipped += 1
    print(f"SKIP  {name}  ({why})")


# Phase 5 naive reference runs: the current-convention runs (seen-through-t
# mask), with the cap-120 re-run for teacher/1234. Re-verified bit-identical
# on HEAD before Phase 6 code was written.
NAIVE_REFERENCE = {
    ("gold", 1234): "gold_naive_1234_20260829-151821",
    ("gold", 1): "gold_naive_1_20260829-152634",
    ("gold", 2): "gold_naive_2_20260829-152708",
    ("teacher", 1234): "teacher_naive_1234_20260829-235307",
    ("teacher", 1): "teacher_naive_1_20260829-152635",
    ("teacher", 2): "teacher_naive_2_20260829-152709",
}
R_KEYS = ("R_f1", "R_acc", "ACC_f1", "BWT_f1", "ACC_acc", "BWT_acc")
REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- fixtures

def _synthetic_data(seed=0, n_train=25, n_dev=8, n_test=8):
    """Class-clustered random features over the real 17-class label space."""
    ci = class_index(load_split(1234))
    full_index = [DIALOGRE_RELATIONS.index(r) for r in ci]     # head -> 37-way
    rng = np.random.RandomState(seed)
    means = rng.normal(size=(len(ci), 1152))

    def draw(n_per):
        X = np.concatenate([0.6 * means[h] + rng.normal(size=(n_per, 1152))
                            for h in range(len(ci))]).astype(np.float32)
        y = np.repeat(np.arange(len(ci)), n_per).astype(np.int64)
        return X, y

    Xtr, ytr = draw(n_train)
    y_full = np.array([full_index[h] for h in ytr], dtype=np.int64)
    feats = {"X": Xtr, "y_teacher": y_full, "y_gold": y_full,
             "relations": DIALOGRE_RELATIONS, "meta": {}}
    (Xdv, ydv), (Xte, yte) = draw(n_dev), draw(n_test)
    ev = {"dev": {"X": Xdv, "y": ydv}, "test": {"X": Xte, "y": yte},
          "meta": {}, "class_index": ci, "n_classes": len(ci)}
    return feats, ev


@contextmanager
def synthetic_harness(feats, ev):
    orig = H.build_features, H.build_eval_sets
    H.build_features = lambda corpus=None, **kw: feats
    H.build_eval_sets = lambda seed=None, **kw: ev
    try:
        yield
    finally:
        H.build_features, H.build_eval_sets = orig


@contextmanager
def count_fkd_calls():
    """
    Wrap regimes.fkd_loss and classify each call by whether the replay stage
    is on the call stack - independent of the regime's own bookkeeping.
    """
    calls = Counter()
    orig = RG.fkd_loss

    def counting(*a, **k):
        f, names = sys._getframe(), set()
        while f is not None:
            names.add(f.f_code.co_name)
            f = f.f_back
        calls["replay_stage" if "_replay_stage" in names else "outside_replay"] += 1
        return orig(*a, **k)

    RG.fkd_loss = counting
    try:
        yield calls
    finally:
        RG.fkd_loss = orig


_SYN = {}


def synthetic_runs():
    """naive, replay, FKD(lambda=0) and FKD(lambda=0.7) on the same synthetic data."""
    if _SYN:
        return _SYN
    feats, ev = _synthetic_data()
    rows = np.arange(len(feats["X"]))
    root = Path(tempfile.mkdtemp(prefix="test_fkd_syn_"))
    with synthetic_harness(feats, ev), count_fkd_calls() as calls:
        out = run_sequential("gold", 1234, NaiveRegime(), results_root=root, verbose=False)
        _SYN["naive"] = (None, json.loads((out / "R_matrix.json").read_text(encoding="utf-8")))
        for key, make in (("replay", lambda: ReplayRegime(1234, "gold", dev=ev["dev"], feature_rows=rows)),
                          ("fkd0", lambda: FKDRegime(1234, "gold", lam=0.0, dev=ev["dev"], feature_rows=rows)),
                          ("fkd", lambda: FKDRegime(1234, "gold", dev=ev["dev"], feature_rows=rows))):
            reg = make()
            out = run_sequential("gold", 1234, reg, results_root=root, verbose=False)
            write_artifacts(reg, out)     # also asserts memory matches the other runs
            _SYN[key] = (reg, json.loads((out / "R_matrix.json").read_text(encoding="utf-8")))
    _SYN["fkd_calls"] = calls
    return _SYN


# ------------------------------------------------------------------ tests

def test_1_hand_calculation():
    """
    Toy case: global classes 0..3, OLD = {0, 2, 3} (non-contiguous), new = {1}.
    gamma = 2, tau2 = 0.5.

      h1 = (1, 0), h2 = (0, 1);  p0 = (1, 0), p2 = (0, 1), p3 = (-1, 0),
      p1 = (1, 1)/sqrt2 (new class - must not enter Eq. 10)
      cos(h1, p0/p2/p3) = 1, 0, -1   ->  s1 = [e^2, 1, e^-2] / (e^2 + 1 + e^-2)
      cos(h2, p0/p2/p3) = 0, 1, 0    ->  s2 = [1, e^2, 1] / (e^2 + 2)

      student z1 = [2, -2, 1, 0], y1 = 0 (old class)
      student z2 = [0, 1, -1, 2], y2 = 1 (NEW class)
      teacher t1 = [1, 5, 0, -1], t2 = [-1, 5, 0, 1]  (column 1 is garbage the
      teacher never trained: it must not enter Eq. 12)

      focal_i = (1 - softmax(z_i over all 4)[y_i])^2
      teacher over OLD: softmax(t_i[[0, 2, 3]]); student over OLD: log_softmax(z_i[[0, 2, 3]])
      L = -(1/2) sum_i sum_{j in OLD} s_ij * focal_i * teacher_ij * log q_ij

    Evaluated scalar by scalar with Python's math module (not with the code
    under test): 0.2037091481625989. Slicing old classes by position
    ([:, :3]) gives 0.1225479..., 40% away, so error 8 cannot pass.
    """
    expected = 0.2037091481625989
    d = torch.float64
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=d)
    protos = torch.tensor([[1.0, 0.0], [1 / math.sqrt(2), 1 / math.sqrt(2)],
                           [0.0, 1.0], [-1.0, 0.0]], dtype=d)
    old_idx = torch.tensor([0, 2, 3])
    seen_mask = torch.tensor([True, True, True, True])
    z = torch.tensor([[2.0, -2.0, 1.0, 0.0], [0.0, 1.0, -1.0, 2.0]], dtype=d)
    t = torch.tensor([[1.0, 5.0, 0.0, -1.0], [-1.0, 5.0, 0.0, 1.0]], dtype=d)
    y = torch.tensor([0, 1])

    s = prototype_similarity(h, protos, old_idx, tau=0.5)
    loss, focal = fkd_loss(z, t, s, y, old_idx, seen_mask, gamma=2.0)
    check("1: L_fkd matches the hand-calculated literal",
          abs(loss.item() - expected) < 1e-12, f"got {loss.item()!r}, want {expected!r}")
    check("1: focal factors match hand values",
          abs(focal[0].item() - 0.11748920063698731) < 1e-12
          and abs(focal[1].item() - 0.5823478333263973) < 1e-12)

    zz = z.clone().requires_grad_(True)
    loss2, _ = fkd_loss(zz, t, s, y, old_idx, seen_mask, gamma=2.0)
    loss2.backward()
    check("1: no gradient reaches the new-class logit (Eq. 13 sums over OLD only)",
          torch.all(zz.grad[:, 1] == 0).item())


def test_2_index_sets():
    """s is [|M|, |OLD|], rows sum to 1, and the new prototype never enters."""
    d = torch.float64
    protos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=d)
    old_idx = torch.tensor([0, 2, 3])            # new class 1 has prototype (0, 0, 1)
    h = torch.tensor([[0.0, 0.0, 1.0],           # identical to the NEW prototype
                      [0.1, 0.2, 0.9],           # closest to the NEW prototype
                      [1.0, 0.5, 0.0]], dtype=d)
    s = prototype_similarity(h, protos, old_idx, tau=0.5)
    check("2: s has shape [|M|, |OLD|]", tuple(s.shape) == (3, 3), str(tuple(s.shape)))
    check("2: every row of s sums to 1", torch.allclose(s.sum(dim=1), torch.ones(3, dtype=d)))

    # Row 0: cos with p0, p2, p3 is 0, 0, 0 -> uniform over OLD. If SEEN leaked
    # into the denominator, the new prototype (cos 1) would take most of the mass.
    check("2: sample identical to the new prototype gets uniform s over OLD",
          torch.allclose(s[0], torch.full((3,), 1 / 3, dtype=d)), str(s[0].tolist()))

    # Row 1, by hand: cosines with p0, p2, p3 are 0.1, 0.2, -0.1 over |h| = sqrt(0.86).
    n = math.sqrt(0.86)
    e = [math.exp(c / n / 0.5) for c in (0.1, 0.2, -0.1)]
    want = [v / sum(e) for v in e]
    check("2: s row matches a hand softmax over OLD cosines only",
          all(abs(a - b) < 1e-12 for a, b in zip(s[1].tolist(), want)))


def test_memory_selection():
    rng = np.random.RandomState(0)
    Z = rng.normal(size=(60, 8)).astype(np.float32)
    y = np.array([0] * 40 + [1] * 12 + [2] * 8)
    mem = select_memory(Z, y, [0, 1, 2], m=10, seed=1234)
    check("memory: m picks per relation, all of a relation with n_r <= m",
          [len(mem[r]) for r in (0, 1, 2)] == [10, 10, 8])
    check("memory: no sample picked twice", all(len(set(v)) == len(v) for v in mem.values()))
    check("memory: every pick carries the relation's training label",
          all((y[v] == r).all() for r, v in mem.items()))
    check("memory: deterministic for a fixed seed",
          select_memory(Z, y, [0, 1, 2], m=10, seed=1234) == mem)


def test_prototype_static_term():
    """Silent error 4: the prototype must include the static (all-samples) mean."""
    Z = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [10.0, 10.0]])
    y = np.array([0, 0, 0, 1])
    p, ps, pd = prototype_eq4(Z, y, memory_rows=[2], r=0, beta=0.2)
    check("prototype: static = mean over ALL training samples of r", np.allclose(ps, [2.0, 0.0]))
    check("prototype: dynamic = mean over memory only", np.allclose(pd, [4.0, 0.0]))
    check("prototype: p = 0.8 static + 0.2 dynamic", np.allclose(p, [2.4, 0.0]), str(p))


def test_3_lambda_zero_equals_replay():
    syn = synthetic_runs()
    (rep, r_rep), (fk0, r_fk0) = syn["replay"], syn["fkd0"]
    check("3 (synthetic): FKD lambda=0 R matrix bit-identical to replay",
          all(r_rep[k] == r_fk0[k] for k in R_KEYS))
    check("3 (synthetic): FKD lambda=0 memory identical to replay", rep.memory == fk0.memory)
    fk = syn["fkd"][0]
    check("3 (synthetic): FKD lambda=0.7 actually adds a nonzero term",
          any(e.get("lam_L_fkd", 0) > 0 for e in fk.replay_log))

    ok, why = prereg_committed()
    if not ok:
        skip("3 (real data): FKD lambda=0 vs replay, seed 1234, both tracks", why)
        return
    root = Path(tempfile.mkdtemp(prefix="test_fkd_real3_"))
    for track in ("gold", "teacher"):
        a, b = ReplayRegime(1234, track), FKDRegime(1234, track, lam=0.0)
        ra = json.loads((run_sequential(track, 1234, a, results_root=root, verbose=False)
                         / "R_matrix.json").read_text(encoding="utf-8"))
        rb = json.loads((run_sequential(track, 1234, b, results_root=root, verbose=False)
                         / "R_matrix.json").read_text(encoding="utf-8"))
        check(f"3 (real, {track}): FKD lambda=0 R matrix bit-identical to replay",
              all(ra[k] == rb[k] for k in R_KEYS))
        check(f"3 (real, {track}): FKD lambda=0 memory identical to replay", a.memory == b.memory)


def test_4_row_zero_equals_naive():
    syn = synthetic_runs()
    naive_row0 = (syn["naive"][1]["R_f1"][0], syn["naive"][1]["R_acc"][0])
    for key in ("replay", "fkd"):
        r = syn[key][1]
        check(f"4 (synthetic): {key} R row 0 bit-identical to naive",
              (r["R_f1"][0], r["R_acc"][0]) == naive_row0)

    ok, why = prereg_committed()
    if not ok:
        skip("4 (real data): replay/FKD row 0 vs naive, seed 1234, both tracks", why)
        return
    root = Path(tempfile.mkdtemp(prefix="test_fkd_real4_"))
    for track in ("gold", "teacher"):
        ref = json.loads((REPO / "results" / NAIVE_REFERENCE[(track, 1234)]
                          / "R_matrix.json").read_text(encoding="utf-8"))
        for cls in (ReplayRegime, FKDRegime):
            r = json.loads((run_sequential(track, 1234, cls(1234, track), results_root=root,
                                           verbose=False) / "R_matrix.json").read_text(encoding="utf-8"))
            check(f"4 (real, {track}): {cls.name} R row 0 bit-identical to naive",
                  (r["R_f1"][0], r["R_acc"][0]) == (ref["R_f1"][0], ref["R_acc"][0]))


def test_5_teacher_frozen():
    syn = synthetic_runs()
    fk = syn["fkd"][0]
    check("5: one diagnostics entry per replay stage (tasks 1 and 2)",
          [d["task"] for d in fk.diagnostics] == [1, 2])
    for d in fk.diagnostics:
        tch = d["teacher"]
        check(f"5: task {d['task']} teacher hash unchanged across the replay stage",
              tch["hash_before"] == tch["hash_after"])
        check(f"5: task {d['task']} teacher is the snapshot from the end of task {d['task'] - 1}",
              tch["hash_before"] == tch["snapshot_hash_at_end_of_prev_task"])
        check(f"5: task {d['task']} teacher in eval mode with no grad",
              tch["eval_mode"] and not tch["any_requires_grad"])
    # Silent error 5: the task-2 teacher must be the model AFTER task 1's replay
    # stage, not before it.
    t2 = fk.diagnostics[1]["teacher"]
    check("5: task-2 teacher was taken after task 1's replay stage",
          t2["hash_before"] != fk.pre_replay_hash[1])


def test_6_no_fkd_in_new_task_training():
    calls = synthetic_runs()["fkd_calls"]
    check("6: L_fkd never called outside the replay stage",
          calls["outside_replay"] == 0, dict(calls))
    check("6: L_fkd called during the replay stage", calls["replay_stage"] > 0, dict(calls))


def test_replay_runs_on_full_seen_memory():
    """Silent error 6: replay at task t uses M-tilde over ALL seen relations."""
    rep = synthetic_runs()["replay"][0]
    stops = [e for e in rep.replay_log if "stop_reason" in e]
    tc = H.task_of_class(load_split(1234))
    seen_counts = [sum(1 for v in tc.values() if v <= t) for t in (1, 2)]
    check("replay: stages at tasks 1 and 2 only (none at task 0)",
          [e["task"] for e in stops] == [1, 2])
    check("replay: memory size = m x |SEEN| at each stage",
          [e["n_memory"] for e in stops] == [10 * n for n in seen_counts],
          f"{[e['n_memory'] for e in stops]} vs {[10 * n for n in seen_counts]}")


def test_kd_hand_calculation():
    """
    Plain KD (§4.4) = FKD with w = 1, through KDRegime's real _replay_loss.
    Same toy as test 1, every weight 1:
      L = -(1/2) sum_i sum_{j in OLD} softmax(t_i[OLD])_j * log softmax(z_i[OLD])_j
    Evaluated with Python's math module: 0.9582440804506892.
    """
    expected = 0.9582440804506892
    d = torch.float64
    z = torch.tensor([[2.0, -2.0, 1.0, 0.0], [0.0, 1.0, -1.0, 2.0]], dtype=d)
    t = torch.tensor([[1.0, 5.0, 0.0, -1.0], [-1.0, 5.0, 0.0, 1.0]], dtype=d)
    kd = object.__new__(RG.KDRegime)          # no data loading; just the loss path
    kd.lam, kd.teacher = 1.0, (lambda X: t)
    ctx = {"s": torch.rand(2, 3, dtype=d),    # must be ignored by KD
           "old_idx": torch.tensor([0, 2, 3]),
           "seen_mask": torch.ones(4, dtype=torch.bool)}
    loss, _ = kd._replay_loss(torch.tensor(0.0, dtype=d), z, None,
                              torch.tensor([0, 1]), torch.tensor([0, 1]), ctx)
    check("kd: L_kd through KDRegime matches the hand literal (w = 1)",
          abs(loss.item() - expected) < 1e-12, f"got {loss.item()!r}")


_FIXED = {}


def fixed_runs():
    """Fixed-schedule replay, FKD(lambda=0) and KD on the synthetic data."""
    if _FIXED:
        return _FIXED
    feats, ev = _synthetic_data()
    rows = np.arange(len(feats["X"]))
    root = Path(tempfile.mkdtemp(prefix="test_fkd_fixed_"))
    with synthetic_harness(feats, ev):
        for key, make in (
                ("replay", lambda: ReplayRegime(1234, "gold", dev=ev["dev"], feature_rows=rows, schedule="fixed")),
                ("fkd0", lambda: FKDRegime(1234, "gold", lam=0.0, dev=ev["dev"], feature_rows=rows, schedule="fixed")),
                ("kd", lambda: RG.KDRegime(1234, "gold", dev=ev["dev"], feature_rows=rows, schedule="fixed"))):
            reg = make()
            out = run_sequential("gold", 1234, reg, results_root=root, verbose=False)
            _FIXED[key] = (reg, json.loads((out / "R_matrix.json").read_text(encoding="utf-8")))
    return _FIXED


def test_fixed_schedule():
    fx = fixed_runs()
    rep = fx["replay"][0]
    check("fixed: run name carries the schedule", rep.name == "replay-fixed10", rep.name)
    for t in (1, 2):
        n = sum(1 for e in rep.replay_log if e.get("task") == t and "epoch" in e)
        check(f"fixed: task {t} replay stage ran exactly 10 epochs", n == 10, str(n))
        check(f"fixed: task {t} keeps the final weights (no dev-based restore)",
              rep.post_replay_hash[t] == rep.last_epoch_hash[t])
    stops = [e for e in rep.replay_log if "stop_reason" in e]
    check("fixed: stop reason says fixed_schedule(10)",
          all(e["stop_reason"] == "fixed_schedule(10)" for e in stops))
    check("fixed: FKD lambda=0 R matrix bit-identical to fixed replay",
          all(fx["replay"][1][k] == fx["fkd0"][1][k] for k in R_KEYS))
    check("fixed: KD diagnostics say w = 1",
          all(d["weighting"] == "none (w = 1)" for d in fx["kd"][0].diagnostics))
    # The primary schedule must still restore the best epoch somewhere.
    prim = synthetic_runs()["replay"][0]
    check("primary: early-stop path still restores best weights",
          any(prim.post_replay_hash[t] != prim.last_epoch_hash[t] for t in (1, 2)))


def test_recorder():
    """run_all's Recorder must not perturb training, and must reproduce R."""
    from continual.run_all import Recorder, run_metrics
    feats, ev = _synthetic_data()
    rows = np.arange(len(feats["X"]))
    names = {i: r for r, i in class_index(load_split(1234)).items()}
    root = Path(tempfile.mkdtemp(prefix="test_fkd_rec_"))
    with synthetic_harness(feats, ev):
        out_plain = run_sequential("gold", 1234, NaiveRegime(), results_root=root, verbose=False)
        rec = Recorder(NaiveRegime(), "gold", 1234, ev["test"], names,
                       feature_rows=rows, feats=feats)
        out_rec = run_sequential("gold", 1234, rec, results_root=root, verbose=False)
    a = json.loads((out_plain / "R_matrix.json").read_text(encoding="utf-8"))
    b = json.loads((out_rec / "R_matrix.json").read_text(encoding="utf-8"))
    check("recorder: wrapped naive R matrix bit-identical to unwrapped",
          all(a[k] == b[k] for k in R_KEYS))
    try:
        m = run_metrics(b, rec, b["task_classes"], degenerate=1, n_classes=len(names))
        ok = m["R_rederived_from_predictions"] == "match"
    except RuntimeError as e:
        ok, m = False, str(e)
    check("recorder: R re-derived from recorded predictions matches the harness", ok)
    check("recorder: one prediction set per task",
          len(rec.test_pred) == 3 and len(rec.train_pred) == 3)
    rec.save(root)
    z = np.load(root / "predictions.npz")
    check("recorder: probability rows sum to 1 and unseen classes get 0 after task 0",
          np.allclose(z["test_prob"].sum(axis=2), 1, atol=1e-5)
          and float(z["test_prob"][0][:, [c for c in range(17)
                    if c not in H.task_of_class(load_split(1234)) or
                    H.task_of_class(load_split(1234))[c] != 0]].max()) == 0.0)


def test_7_naive_untouched():
    root = Path(tempfile.mkdtemp(prefix="test_fkd_naive_"))
    for (track, seed), ref_dir in NAIVE_REFERENCE.items():
        ref = json.loads((REPO / "results" / ref_dir / "R_matrix.json").read_text(encoding="utf-8"))
        out = run_sequential(track, seed, NaiveRegime(), results_root=root, verbose=False)
        new = json.loads((out / "R_matrix.json").read_text(encoding="utf-8"))
        check(f"7: naive {track}/{seed} bit-identical to {ref_dir}",
              all(ref[k] == new[k] for k in R_KEYS))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{'='*50}\n{_passed} passed, {_failed} failed, {_skipped} skipped")
    sys.exit(1 if _failed else 0)
