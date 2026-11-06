"""
Tests for the RQ3 label conditions, regimes and tables (RQ3_SPEC §3, with the
Gate 1 amendments). Run from capstone_pipeline:

    venv/bin/python tests/test_conditions.py

Plain asserts and a tally, like the other suites. RULE: nothing here prints a
metric from an RQ3 condition. The RQ3 runs made below go to temp dirs and
are checked for structure and identity only, so the PREREG can still be
written blind.

  1. existing tracks untouched: naive 2 tracks x 3 seeds bit-identical to the
     Phase 5 files; joint (via run_all) identical to joint_baseline_seed*.json;
     replay seed 1234 (via run_all) identical to the Phase 7 primary runs
  2. teacher_I labels = teacher-track labels on the 710 shared rows
  3. perm_I keeps C exactly and changes at least one label (every draw)
  4. rand_I: exactly round(eps_I * |I|) labels differ from gold, none equals gold
  5. determinism: same seed twice identical, different seeds / replicates differ
  6. gold_sub: size |I|, subset of the gold rows, no duplicates
  7. balanced sampler within 2% of uniform on a toy set; joint never builds one
  8. init seed: explicit init = seed equals the default; an offset changes weights
  9. Phase 7 tables byte-identical with an RQ3 run at another hash present
 plus: saved draws match, the RQ3 run guard, memory consistency across
 replicates, joint run-dir contents, and the PREREG 10-17 scoring rules on
 synthetic values.
"""

import json
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import continual.make_tables as MT
import continual.student as ST
from continual import config
from continual.conditions import (
    DRAWN_TRACKS, REPLICATES, RQ3_TRACKS, check_saved, draw_seed_for, head_labels,
    intersection_rows, load_condition, train_rows,
)
from continual.features import build_features
from continual.harness import NaiveRegime, run_sequential
from continual.prereg_rq3 import head_prereg, parse_scoring, prereg_rq3_ready, rq3_section
from continual.regimes import param_hash
from continual.run_all import run_joint, run_one
from continual.task_split import load_split, class_index

torch.use_deterministic_algorithms(True)
REPO = Path(__file__).resolve().parent.parent
SEEDS = config.SEEDS
R_KEYS = ("R_f1", "R_acc", "ACC_f1", "BWT_f1", "ACC_acc", "BWT_acc")
NAIVE_REFERENCE = {          # the six Phase 5 naive runs (CLAUDE.md, test_fkd.py)
    ("gold", 1234): "gold_naive_1234_20260829-151821",
    ("gold", 1): "gold_naive_1_20260829-152634",
    ("gold", 2): "gold_naive_2_20260829-152708",
    ("teacher", 1234): "teacher_naive_1234_20260829-235307",
    ("teacher", 1): "teacher_naive_1_20260829-152635",
    ("teacher", 2): "teacher_naive_2_20260829-152709",
}

_passed, _failed = 0, 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}")
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f"  -- {extra}" if extra else ""))


FEATS = build_features(config.CORPUS)
CI = {s: class_index(load_split(s)) for s in SEEDS}


def tmp(prefix):
    return Path(tempfile.mkdtemp(prefix=prefix))


# ------------------------------------------------------------------ 1

def test_1_existing_tracks_untouched():
    root = tmp("test_cond_naive_")
    for (track, seed), ref_dir in NAIVE_REFERENCE.items():
        ref = json.loads((REPO / "results" / ref_dir / "R_matrix.json").read_text())
        out = run_sequential(track, seed, NaiveRegime(), results_root=root, verbose=False)
        new = json.loads((out / "R_matrix.json").read_text())
        check(f"1: naive {track}/{seed} R matrices bit-identical to {ref_dir}",
              all(ref[k] == new[k] for k in R_KEYS))

    root = tmp("test_cond_joint_")
    for seed in SEEDS:
        ref = json.loads((REPO / "results" / f"joint_baseline_seed{seed}.json").read_text())["tracks"]
        for track in ("gold", "teacher"):
            out = run_joint(track, "joint", seed, "test", results_root=root)
            m = json.loads((out / "metrics.json").read_text())
            r = ref[track]
            same = (m["test_macro_f1_exact"] == r["test_macro_f1"]
                    and m["test_accuracy_exact"] == r["test_accuracy"]
                    and m["best_epoch"] == r["best_epoch"] and m["epochs_run"] == r["epochs_run"]
                    and m["best_dev_macro_f1"] == r["best_dev_macro_f1"]
                    and m["per_class_f1"] == r["per_class_f1"]
                    and m["train_examples"] == r["train_examples"])
            check(f"1: joint {track}/{seed} via run_all identical to joint_baseline_seed{seed}.json",
                  same)

    P, h = MT.select_runs("primary", MT.PRIMARY, None)
    root = tmp("test_cond_replay_")
    for track in ("gold", "teacher"):
        ref = P[(track, "replay", 1234)]
        out = run_one(track, "replay", 1234, "primary", "test", results_root=root)
        R_new = json.loads((out / "R_matrix.json").read_text())
        mem_new = json.loads((out / "memory_indices.json").read_text())["classes"]
        mem_ref = ref.json("memory_indices.json")["classes"]
        z_new, z_ref = np.load(out / "predictions.npz"), ref.P
        check(f"1: replay {track}/1234 via run_all identical to Phase 7 ({ref.dir.name}): "
              "R, memory, predictions",
              all(R_new[k] == ref.R[k] for k in R_KEYS)
              and {k: v["train_rows"] for k, v in mem_new.items()} ==
                  {k: v["train_rows"] for k, v in mem_ref.items()}
              and all(np.array_equal(z_new[k], z_ref[k]) for k in
                      ("test_pred", "test_prob", "train_pred", "train_prob", "train_label",
                       "train_gold", "train_feature_row")))
        check(f"1: replay {track}/1234 memory file has no draw_seed key (Phase 7 format)",
              "draw_seed" not in json.loads((out / "memory_indices.json").read_text()))


# ------------------------------------------------------------------ 2-6

def test_2_teacher_I_equals_teacher_on_shared_rows():
    for s in SEEDS:
        rows_I, y_I = train_rows("teacher_I", FEATS, CI[s])
        rows_T, y_T = train_rows("teacher", FEATS, CI[s])
        pos = {int(r): i for i, r in enumerate(rows_T)}
        check(f"2: seed {s}: |I| = 710, I inside the teacher track, labels identical on it",
              len(rows_I) == 710 and all(int(r) in pos for r in rows_I)
              and all(y_T[pos[int(r)]] == v for r, v in zip(rows_I, y_I)))
    check("2: I = rows with gold and teacher both in the 17 classes",
          np.array_equal(intersection_rows(FEATS, CI[1234]),
                         train_rows("gold_I", FEATS, CI[1234])[0]))


def _confusion(g, t, n=17):
    C = np.zeros((n, n), dtype=int)
    np.add.at(C, (g, t), 1)
    return C


def test_3_perm_I_keeps_C():
    for s in SEEDS:
        rows, g = train_rows("gold_I", FEATS, CI[s])
        _, t = train_rows("teacher_I", FEATS, CI[s])
        for r in REPLICATES:
            rows_p, p = train_rows("perm_I", FEATS, CI[s], draw_seed_for("perm_I", s, r))
            check(f"3: perm_I seed {s} r{r}: C identical to teacher_I's, ≥ 1 label changed",
                  np.array_equal(rows_p, rows)
                  and np.array_equal(_confusion(g, p), _confusion(g, t)) and (p != t).any())


def test_4_rand_I_rate():
    _, g = train_rows("gold_I", FEATS, CI[1234])
    _, t = train_rows("teacher_I", FEATS, CI[1234])
    n_flip = int(round((1 - (g == t).mean()) * len(g)))
    check("4: round(eps_I * |I|) = 282", n_flip == 282)
    for s in SEEDS:
        for r in REPLICATES:
            _, y = train_rows("rand_I", FEATS, CI[s], draw_seed_for("rand_I", s, r))
            differ = y != g
            check(f"4: rand_I seed {s} r{r}: exactly 282 differ from gold, labels in range",
                  int(differ.sum()) == 282 and ((y >= 0) & (y < 17)).all())


def test_5_determinism():
    for tr in DRAWN_TRACKS:
        a = train_rows(tr, FEATS, CI[1234], 1234)
        b = train_rows(tr, FEATS, CI[1234], 1234)
        c = train_rows(tr, FEATS, CI[1], 1)
        d = train_rows(tr, FEATS, CI[1234], 1334)
        same = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        diff = lambda u, v: not (np.array_equal(u[0], v[0]) and np.array_equal(u[1], v[1]))
        check(f"5: {tr}: same seed twice identical; other seed and other replicate differ",
              same and diff(a, c) and diff(a, d))
    for tr in ("gold_I", "teacher_I"):
        a, b = train_rows(tr, FEATS, CI[1234]), train_rows(tr, FEATS, CI[2])
        check(f"5: {tr} has no draw (identical across seeds)",
              np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    try:
        train_rows("rand_I", FEATS, CI[1234], None)
        ok = False
    except ValueError:
        ok = True
    check("5: a drawn track without a draw seed is refused", ok)


def test_6_gold_sub():
    for s in SEEDS:
        gold_rows, y_gold = train_rows("gold", FEATS, CI[s])
        lab = dict(zip(gold_rows.tolist(), y_gold.tolist()))
        for r in REPLICATES:
            rows, y = train_rows("gold_sub", FEATS, CI[s], draw_seed_for("gold_sub", s, r))
            check(f"6: gold_sub seed {s} r{r}: 710 rows, subset of the 1,327, no duplicates, gold labels",
                  len(rows) == 710 and len(set(rows.tolist())) == 710
                  and set(rows.tolist()) <= set(gold_rows.tolist())
                  and all(lab[int(i)] == v for i, v in zip(rows, y)))


# ------------------------------------------------------------------ 7

@contextmanager
def count_samplers():
    calls = Counter()
    orig = ST.WeightedRandomSampler

    class Counting(orig):
        def __init__(self, *a, **k):
            calls["n"] += 1
            super().__init__(*a, **k)

    ST.WeightedRandomSampler = Counting
    try:
        yield calls
    finally:
        ST.WeightedRandomSampler = orig


def test_7_balanced_sampler():
    y = np.array([0] * 500 + [1] * 100 + [2] * 20 + [3] * 5)
    torch.manual_seed(0)
    draws = np.array(list(ST.balanced_sampler(y, 10_000)))
    freq = np.bincount(y[draws], minlength=4) / len(draws)
    check("7: 10,000 balanced draws on a 500/100/20/5 set are within 2% of uniform per class",
          len(draws) == 10_000 and np.all(np.abs(freq - 0.25) <= 0.02), f"{freq}")

    rng = np.random.RandomState(0)
    Xtr = rng.normal(size=(60, 1152)).astype(np.float32)
    ytr = np.array([0] * 40 + [1] * 15 + [2] * 5, dtype=np.int64)
    Xdv = rng.normal(size=(20, 1152)).astype(np.float32)
    ydv = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int64)
    with count_samplers() as calls:
        ST.train_joint(Xtr, ytr, Xdv, ydv, 3, seed=0, verbose=False)
        n_joint = calls["n"]
        ST.train_joint_balanced(Xtr, ytr, Xdv, ydv, 3, seed=0, verbose=False)
        n_bal = calls["n"] - n_joint
    check("7: train_joint never instantiates the sampler; train_joint_balanced does, once",
          n_joint == 0 and n_bal == 1)
    with count_samplers() as calls:
        run_joint("gold", "joint", 1234, "test", results_root=tmp("test_cond_js_"))
    check("7: the run_all joint regime never instantiates the sampler", calls["n"] == 0)
    torch.manual_seed(3)
    m_a = ST.Student(3)
    torch.manual_seed(3)
    m_b = ST.Student(3)
    check("7: same seed gives joint and joint_balanced the same initial weights",
          param_hash(m_a) == param_hash(m_b))


# ------------------------------------------------------------------ 8

class Capture(NaiveRegime):
    def __init__(self):
        self.hashes = {}

    def after_task(self, t, model, task_data):
        self.hashes[t] = param_hash(model)


def test_8_init_seed():
    root = tmp("test_cond_init_")
    a, b, c = Capture(), Capture(), Capture()
    run_sequential("gold", 1234, a, results_root=root, verbose=False)
    run_sequential("gold", 1234, b, results_root=root, verbose=False, init_seed=1234)
    run_sequential("gold", 1234, c, results_root=root, verbose=False, init_seed=1334)
    check("8: harness: init_seed = seed reproduces the default exactly", a.hashes == b.hashes)
    check("8: harness: init_seed offset changes the weights",
          all(a.hashes[t] != c.hashes[t] for t in a.hashes))
    rows, y = train_rows("gold", FEATS, CI[1234])
    from continual.eval_sets import build_eval_sets
    ev = build_eval_sets(1234)
    m1, _ = ST.train_joint(FEATS["X"][rows], y, ev["dev"]["X"], ev["dev"]["y"], 17, seed=1234,
                           verbose=False)
    m2, _ = ST.train_joint(FEATS["X"][rows], y, ev["dev"]["X"], ev["dev"]["y"], 17, seed=1334,
                           verbose=False)
    check("8: joint: an init offset changes the weights", param_hash(m1) != param_hash(m2))
    names = sorted(p.name for p in root.iterdir())
    check("8: replicate run dirs carry the init seed, so they can't collide",
          sum("_i1334_" in n for n in names) == 1 and sum("_i1234_" in n for n in names) == 1)


# ------------------------------------------------------------------ 9

def test_9_phase7_tables_unchanged():
    real = config.RESULTS_DIR.resolve()
    root = tmp("test_cond_tables_")
    for p in real.iterdir():
        if p.name not in ("tables", "figures"):
            (root / p.name).symlink_to(p)
    fake = root / "teacher_I_naive_1234_fake"
    fake.mkdir()
    (fake / "metrics.json").write_text(json.dumps({"meta": {
        "track": "teacher_I", "regime": "naive", "seed": 1234, "schedule": "primary",
        "git_hash": "0000000", "dirty": False, "replicate": 0, "spec": "RQ3_SPEC"}}))
    saved = config.RESULTS_DIR, MT.TABLE_DIR, MT.FIG_DIR
    config.RESULTS_DIR, MT.TABLE_DIR, MT.FIG_DIR = root, root / "tables", root / "figures"
    try:
        try:
            MT.select_runs("primary", MT.PRIMARY, None, tracks=MT.TRACKS + ["teacher_I"])
            sensitive = False
        except SystemExit:
            sensitive = True
        check("9: without the track filter the RQ3 run would break Phase 7 selection", sensitive)
        MT.phase7()
    finally:
        config.RESULTS_DIR, MT.TABLE_DIR, MT.FIG_DIR = saved
    new_tables = sorted((root / "tables").iterdir())
    phase7_names = {p.name for p in (real / "tables").iterdir()
                    if not p.name.startswith(("T7", "T8", "T9", "T10", "T11", "prereg_outcomes_rq3"))}
    diffs = [p.name for p in new_tables if p.read_bytes() != (real / "tables" / p.name).read_bytes()]
    check(f"9: all {len(new_tables)} Phase 7 table files rebuild byte-identical (the full set)",
          not diffs and {p.name for p in new_tables} == phase7_names and len(new_tables) == 22,
          f"differ: {diffs}")
    figs = sorted((root / "figures").iterdir())
    fdiffs = [p.name for p in figs if p.read_bytes() != (real / "figures" / p.name).read_bytes()]
    check(f"9: all {len(figs)} Phase 7 figures rebuild byte-identical", not fdiffs,
          f"differ: {fdiffs}")


# ------------------------------------------------------------- extras

def test_saved_conditions():
    n = 0
    for s in SEEDS:
        for tr in RQ3_TRACKS:
            for r in (REPLICATES if tr in DRAWN_TRACKS else (0,)):
                rows, y = train_rows(tr, FEATS, CI[s], draw_seed_for(tr, s, r))
                check_saved(tr, s, r, rows, y)
                p = load_condition(tr, s, r)
                assert p["gold"] == head_labels(FEATS["y_gold"], CI[s])[rows].tolist()
                n += 1
    check(f"saved: all {n} results/conditions files match the labels the runs build", n == 33)
    diag = json.loads((REPO / "results/diagnostics/task_rows_by_condition.json").read_text())["seeds"]
    ok = True
    for s in SEEDS:
        toc = {CI[s][r]: t for r, t in load_split(s)["relation_to_task"].items()}
        for tr in RQ3_TRACKS:
            y = load_condition(tr, s, 0)["labels"]
            ok &= [sum(toc[v] == k for v in y) for k in range(3)] == diag[str(s)][tr]["per_task"]
    check("saved: replicate-0 rows per task equal the Gate 1 table", ok)


EXAMPLE_PREREG = """# Pre-registration

## RQ3 predictions (written 2026-11-06, after Gate 1, before any §2.4 run)

10. Size: ...

rq3-scoring:
  line10_gold_I_minus_gold   = -0.05
  line10_gold_I_real         = yes
  line10_gold_sub_minus_gold = −0.04
  line10_ratio               = 0.75
  line11                     = lower
  line12                     = same
  line13                     = partly: perm_I, teacher_I
  line13_controls            = not_real_positive
  line14_pct                 = 50
  line14_expect              = fails
  line15_pp                  = 5
  line15_expect              = holds
  line17_pct                 = 50
  line17_expect              = holds

## Something else
"""


COMMITTED_SCORING = {   # PREREG.md at 38ec218; a change here means the PREREG moved
    "line10_gold_I_minus_gold": -0.06, "line10_gold_sub_minus_gold": -0.09,
    "line10_ratio": 0.78, "line14_pct": 50.0, "line15_pp": 3.0, "line17_pct": 60.0,
    "line10_gold_I_real": True, "line11": "lower", "line12": "same",
    "line13": ("partly", ("gold_I", "rand_I")), "line13_controls": "not_real_positive",
    "line14_expect": "fails", "line15_expect": "holds", "line17_expect": "holds"}


def test_guard_and_parser():
    ok, why = prereg_rq3_ready()
    check("guard: PREREG.md at HEAD has the committed RQ3 section, so RQ3 runs may start",
          ok, why)
    head = head_prereg()
    check("guard: the committed scoring block parses to exactly the PREREG values",
          parse_scoring(rq3_section(head)) == COMMITTED_SCORING)
    i = head.find("## RQ3 predictions")
    ok, why = prereg_rq3_ready(head[:i])
    check("guard: HEAD's PREREG without the RQ3 section is refused", not ok and "RQ3" in why, why)
    p = parse_scoring(rq3_section(EXAMPLE_PREREG))
    check("guard: the example scoring block parses (unicode minus, sorted partly set)",
          p["line10_gold_sub_minus_gold"] == -0.04 and p["line13"] == ("partly", ("perm_I", "teacher_I"))
          and p["line10_gold_I_real"] is True and p["line13_controls"] == "not_real_positive")
    check("guard: the example passes prereg_rq3_ready", prereg_rq3_ready(EXAMPLE_PREREG)[0])
    bad = [EXAMPLE_PREREG.replace("= lower", "= smaller"),
           EXAMPLE_PREREG.replace("  line15_pp                  = 5\n", ""),
           EXAMPLE_PREREG.replace("  line17_pct                 = 50\n", ""),
           EXAMPLE_PREREG.replace("line17_expect              = holds", "line17_expect = maybe"),
           EXAMPLE_PREREG.replace("partly: perm_I, teacher_I", "partly: gold_sub"),
           EXAMPLE_PREREG.replace("2026-11-06", "YYYY-MM-DD"),
           EXAMPLE_PREREG.replace("-0.05", "____")]
    check("guard: malformed or blank blocks are refused",
          all(not prereg_rq3_ready(t)[0] for t in bad))
    # The CLI refusal (`run_all --plan rq3` exiting with "refusing to run") was
    # checked here until the PREREG was committed at 38ec218; with the guard now
    # passing, that call would start real runs, so it is no longer made.


def test_memory_across_replicates():
    root = tmp("test_cond_mem_")
    errors = []
    for tr, r in (("gold_I", 0), ("gold_I", 1), ("rand_I", 0), ("rand_I", 1)):
        try:
            run_one(tr, "replay", 1234, "primary", "test", results_root=root, replicate=r)
        except RuntimeError as e:
            errors.append(str(e))
    check("memory: gold_I replicates share memory; rand_I draws aren't cross-checked", not errors,
          "; ".join(errors))
    mems = {p.parent.name: json.loads(p.read_text()) for p in root.glob("*/memory_indices.json")}
    g = [m for n, m in mems.items() if n.startswith("gold_I")]
    rd = [m for n, m in mems.items() if n.startswith("rand_I")]
    check("memory: KMeans stays on the task-order seed (gold_I r0 and r1 select the same rows)",
          len(g) == 2 and {k: v["train_rows"] for k, v in g[0]["classes"].items()} ==
          {k: v["train_rows"] for k, v in g[1]["classes"].items()})
    check("memory: rand_I files record their draw seed", sorted(m["draw_seed"] for m in rd) == [1234, 1334])
    metas = [json.loads(p.read_text())["meta"] for p in root.glob("*/metrics.json")]
    check("memory: RQ3 run metadata carries spec, replicate, init and draw seeds",
          all(m["spec"] == "RQ3_SPEC" and "replicate" in m and "init_seed" in m and "draw_seed" in m
              for m in metas) and {(m["replicate"], m["init_seed"]) for m in metas} ==
          {(0, 1234), (1, 1334)})


def test_joint_run_dir():
    root = tmp("test_cond_jdir_")
    out = run_joint("gold", "joint_balanced", 1234, "test", results_root=root)
    z = np.load(out / "predictions.npz")
    m = json.loads((out / "metrics.json").read_text())
    check("joint dir: predictions saved (1 x 1,144), probabilities sum to 1, argmax = pred",
          z["test_pred"].shape == (1, 1144) and z["test_prob"].shape == (1, 1144, 17)
          and np.allclose(z["test_prob"].sum(axis=2), 1, atol=1e-5)
          and np.array_equal(z["test_prob"][0].argmax(axis=1), z["test_pred"][0]))
    check("joint dir: metrics.json has the run_all metadata and final scores",
          m["meta"]["regime"] == "joint_balanced" and m["meta"]["dirty"] is False
          and "final_macro_f1_all" in m and "final_accuracy" in m
          and {p.name for p in out.iterdir()} ==
          {"metrics.json", "predictions.npz", "train_log.jsonl", "config.json"})


class FakeRQ3(MT.RQ3):
    """RQ3 accessors over hand-set F and S values, for the scoring rules."""

    def __init__(self, F, S, reps=(0, 1, 2), ref_gold=0.48, gold_share=0.024):
        self.reps, self.ran = list(reps), len(reps) > 1
        self._F, self._S = F, S
        self.ref = {s: {"gold": {"test_macro_f1": ref_gold}} for s in SEEDS}
        self.gold_share = {s: gold_share for s in SEEDS}

    def F(self, c, g, s, r):
        return self._F[(c, g)][SEEDS.index(s)][self.reps.index(r)]

    def S(self, c, g, s, r):
        return self._S[(c, g)][SEEDS.index(s)][self.reps.index(r)]


def _flat(v):
    return [[v] * 3 for _ in SEEDS]


def _score(F, S, block_overrides=None, reps=(0, 1, 2)):
    text = EXAMPLE_PREREG
    for k, v in (block_overrides or {}).items():
        text = text.replace(k, v)
    import continual.prereg_rq3 as PR
    orig, saved = PR.head_prereg, MT.TABLE_DIR
    PR.head_prereg = lambda: text
    MT.TABLE_DIR = tmp("test_cond_score_")
    try:
        return MT.prereg_rq3(FakeRQ3(F, S, reps))
    finally:
        PR.head_prereg, MT.TABLE_DIR = orig, saved


def _base():
    F, S = {}, {}
    for c in RQ3_TRACKS:
        for g in ("joint", "joint_balanced", "naive", "replay"):
            F[(c, g)] = _flat(0.40)
            S[(c, g)] = _flat(0.05)
    return F, S


def test_scoring_rules():
    check("score: real_sign needs |d| > 0.03 and, with replicates, one sign",
          MT.real_sign([0.05, 0.04, 0.06], True) == 1 and MT.real_sign([0.05, -0.01, 0.04], True) == 0
          and MT.real_sign([0.05, -0.01], False) == 1 and MT.real_sign([0.03, 0.05, 0.05], True) == 0
          and MT.real_sign([-0.04, -0.05, -0.01], True) == -1)
    check("score: direction is the 2-of-3 rule",
          MT.direction([1, 1, 0]) == "real-positive" and MT.direction([1, -1, 0]) == "not real"
          and MT.direction([-1, -1, 1]) == "real-negative")

    F, S = _base()
    for c in ("teacher_I", "perm_I"):
        F[(c, "replay")] = _flat(0.45)                         # Δ_replay +0.05 everywhere
    res = _score(F, S, {"partly: perm_I, teacher_I": "holds"})
    v = {o["line"]: o["verdict"] for o in res["outcomes"]}
    check("score: line 13 holds when teacher_I/perm_I gain and the controls don't",
          v[13] == "holds" and any(b.startswith("13 holds") for b in res["branches"]), v[13])

    F2 = dict(F)
    F2[("teacher_I", "replay")] = [[0.45, 0.39, 0.44]] * 3     # replicate range crosses zero
    res = _score(F2, S)
    v = {o["line"]: o["verdict"] for o in res["outcomes"]}
    o13 = next(o for o in res["outcomes"] if o["line"] == 13)["outcome"]
    check("score: a replicate range crossing zero makes the cell not real (partly: rand_I, gold_I, perm_I)",
          "partly: gold_I, perm_I, rand_I" in o13 and v[13] == "fails", o13)

    F3 = dict(F)
    F3[("gold_I", "replay")] = _flat(0.34)                      # gold_I Δ_replay real-negative
    o_nr = next(o for o in _score(F3, S, {"partly: perm_I, teacher_I": "holds",
                                           "not_real_positive": "not_real"})["outcomes"]
                if o["line"] == 13)
    o_np = next(o for o in _score(F3, S, {"partly: perm_I, teacher_I": "holds"})["outcomes"]
                if o["line"] == 13)
    check("score: line13_controls decides whether a real-negative control breaks the clause",
          o_nr["verdict"] == "fails" and o_np["verdict"] == "holds")

    F4, S4 = _base()                                             # no replay gain; share falls
    for c in ("teacher_I", "perm_I"):
        S4[(c, "joint")] = _flat(0.15)
        S4[(c, "replay")] = _flat(0.06)
    res = _score(F4, S4)
    o14 = next(o for o in res["outcomes"] if o["line"] == 14)
    check("score: line 14 fails with 'no replay gain to recover' when Δ_replay isn't real-positive",
          o14["verdict"].startswith("fails: Δ_replay real-positive on 0 of 3"), o14["verdict"])
    check("score: the Δ_share branch fires when share falls but macro-F1 doesn't move",
          any(b.startswith("Δ_share real on teacher_I and perm_I") for b in res["branches"]))
    o15 = next(o for o in res["outcomes"] if o["line"] == 15)
    check("score: line 15 (a) and (b) hold for teacher_I/perm_I here; rand_I 5% > 2×2.4% fails (a)",
          o15["verdict"] == "fails: (a) rand_I ≤ 2×", o15["verdict"])

    F5, S5 = _base()
    for c in ("teacher_I", "perm_I"):
        F5[(c, "replay")] = _flat(0.46)
        F5[(c, "joint_balanced")] = _flat(0.43)                  # recovers 50%
    res = _score(F5, S5, {"partly: perm_I, teacher_I": "holds"})
    v = {o["line"]: o["verdict"] for o in res["outcomes"]}
    check("score: line 14 holds at exactly 50% recovery with line14_pct = 50",
          v[14] == "holds" and any(b.startswith("14 holds") for b in res["branches"]), v[14])

    F6, S6 = _base()
    F6[("rand_I", "joint")] = _flat(0.45)                        # teacher_I 0.05 below rand_I
    F6[("gold_I", "joint")] = _flat(0.43)                        # gold_I − ref = −0.05
    F6[("gold_sub", "joint")] = _flat(0.44)                      # gold_sub − ref = −0.04
    F6[("teacher_I", "joint")] = _flat(0.3225)                   # ratio 0.75
    F6[("perm_I", "joint")] = _flat(0.3225)
    res = _score(F6, S6)
    v = {o["line"]: o["verdict"] for o in res["outcomes"]}
    check("score: lines 10, 11, 12 hold when the values match the example predictions",
          v[10] == "holds" and v[11] == "holds" and v[12] == "holds", str(v))
    res = _score(F6, S6, reps=(0,))
    check("score: line 16 reports when no replicates ran",
          next(o for o in res["outcomes"] if o["line"] == 16)["outcome"] == "no replicates ran")

    F7 = dict(F6)                                                 # r0 matches; r1, r2 don't
    F7[("gold_sub", "joint")] = [[0.44, 0.30, 0.30]] * 3
    F7[("gold_I", "joint")] = [[0.43, 0.41, 0.42]] * 3          # still real-negative in every replicate
    F7[("teacher_I", "joint")] = [[0.3225, 0.20, 0.20]] * 3
    F7[("rand_I", "joint")] = [[0.45, 0.33, 0.33]] * 3            # keep line 11's pairing at −0.05..−0.13
    F7[("perm_I", "joint")] = [[0.3225, 0.20, 0.20]] * 3
    o10 = next(o for o in _score(F7, S6)["outcomes"] if o["line"] == 10)
    check("score: line 10 point values and ratio are scored on replicate 0",
          o10["verdict"] == "holds", o10["verdict"])


def test_line13_expected_world():
    """Adithya's expected world: no condition's Δ_replay is real-positive."""
    F, S = _base()
    F[("gold_I", "replay")] = _flat(0.34)        # gold_I real-negative (as the full gold track)
    F[("teacher_I", "replay")] = _flat(0.35)     # teacher_I real-negative
    # perm_I, rand_I: Δ_replay 0, not real
    res = _score(F, S, {"partly: perm_I, teacher_I": "partly: rand_I, gold_I"})
    o13 = next(o for o in res["outcomes"] if o["line"] == 13)
    check("line 13: expected world scores as 'hypothesis partly: gold_I, rand_I'",
          o13["outcome"].endswith("→ hypothesis partly: gold_I, rand_I"), o13["outcome"])
    check("line 13: 'line13 = partly: rand_I, gold_I' (any order) names it, verdict holds",
          o13["verdict"] == "holds" and "hypothesis partly: gold_I, rand_I" in o13["prediction"],
          o13["prediction"])
    res = _score(F, S, {"partly: perm_I, teacher_I": "fails"})
    check("line 13: predicting 'fails' for that world is scored wrong",
          next(o for o in res["outcomes"] if o["line"] == 13)["verdict"] == "fails")


def test_line17_and_expectations():
    F, S = _base()
    S[("teacher_I", "joint")] = _flat(0.15)
    S[("teacher_I", "replay")] = _flat(0.06)             # D_rep 9 points
    S[("teacher_I", "joint_balanced")] = _flat(0.105)    # D_bal 4.5 points = 50%
    res = _score(F, S)
    o17 = next(o for o in res["outcomes"] if o["line"] == 17)
    check("line 17: holds at exactly 50% with line17_pct = 50; expectation 'holds' correct",
          o17["verdict"] == "holds" and o17["expected"] == "holds"
          and o17["expectation_correct"] == "yes", str(o17))
    res = _score(F, S, {"line17_pct                 = 50": "line17_pct = 60"})
    check("line 17: fails at line17_pct = 60; expectation now wrong",
          next(o for o in res["outcomes"] if o["line"] == 17)["expectation_correct"] == "no")
    S2 = dict(S)
    S2[("teacher_I", "replay")] = [[0.06, 0.13, 0.06]] * 2 + [[0.06, 0.06, 0.06]]   # r1 under 3 pts on 2 seeds
    o17 = next(o for o in _score(F, S2)["outcomes"] if o["line"] == 17)
    check("line 17: D_rep must clear 3 points in EVERY replicate; on 1 of 3 seeds it fails by rule",
          o17["verdict"].startswith("fails: D_rep clears 3 points on 1 of 3 seeds"), o17["verdict"])
    S3 = dict(S)
    S3[("teacher_I", "joint_balanced")] = [[0.105, 0.105, 0.14]] * 3                # r2 recovers 11%
    check("line 17: the recovery bar applies in every replicate",
          next(o for o in _score(F, S3)["outcomes"] if o["line"] == 17)["verdict"] == "fails")
    o14 = next(o for o in res["outcomes"] if o["line"] == 14)
    o15 = next(o for o in res["outcomes"] if o["line"] == 15)
    o13 = next(o for o in res["outcomes"] if o["line"] == 13)
    check("expectations: lines 14 and 15 carry them, lines 10-13 don't",
          o14["expected"] == "fails" and o14["expectation_correct"] == "yes"
          and o15["expected"] == "holds" and o13["expected"] == "" and o13["expectation_correct"] == "")


def test_rq3_tables_build_on_synthetic_grid():
    """T7-T11, F1, F2 and the scorecard from a full fake 162-run grid (random numbers only)."""
    from continual.conditions import RQ3_REGIMES
    import continual.prereg_rq3 as PR
    real = config.RESULTS_DIR.resolve()
    root = tmp("test_cond_rq3tables_")
    for name in ["conditions"] + [f"joint_baseline_seed{s}.json" for s in SEEDS]:
        (root / name).symlink_to(real / name)
    rng = np.random.default_rng(0)
    y_test = np.load(real / "gold_naive_1234_20261009-130538" / "predictions.npz")["test_gold"]
    n = 0
    for r in REPLICATES:
        for c in RQ3_TRACKS:
            for g in RQ3_REGIMES[c]:
                for s in SEEDS:
                    d = root / f"{c}_{g}_{s}_r{r}_fake"
                    d.mkdir()
                    K = 1 if g in ("joint", "joint_balanced") else 3
                    np.savez_compressed(d / "predictions.npz", test_gold=y_test,
                                        test_pred=rng.integers(0, 17, size=(K, len(y_test))))
                    m = {"meta": {"track": c, "regime": g, "seed": s, "schedule": "primary",
                                  "git_hash": "fffffff", "dirty": False, "replicate": r,
                                  "spec": "RQ3_SPEC"},
                         "final_macro_f1_all": float(rng.uniform(0.2, 0.5))}
                    if K == 1:
                        m["final_accuracy"] = float(rng.uniform(0.3, 0.6))
                    else:
                        m.update({"ACC_f1": 0.3, "ACC_acc": 0.4, "BWT_f1": -0.1,
                                  "forgetting_f1": {"0": -0.2, "1": -0.1},
                                  "mean_forgetting_nondegenerate_f1": -0.15,
                                  "observed_accuracy_after_task": [0.5, 0.45, 0.4]})
                    (d / "metrics.json").write_text(json.dumps(m))
                    n += 1
    saved = config.RESULTS_DIR, MT.TABLE_DIR, MT.FIG_DIR, PR.head_prereg
    config.RESULTS_DIR, MT.TABLE_DIR, MT.FIG_DIR = root, root / "tables", root / "figures"
    PR.head_prereg = lambda: EXAMPLE_PREREG
    try:
        MT.rq3()
    finally:
        config.RESULTS_DIR, MT.TABLE_DIR, MT.FIG_DIR, PR.head_prereg = saved
    want = {f"{t}.{e}" for t in ("T7_condition_stats", "T7b_teacher_noise_breakdown", "T8_rq3_joint",
                                 "T9_rq3_cl", "T10_rq3_key_comparison", "T11_rq3_broad_share",
                                 "prereg_outcomes_rq3") for e in ("md", "csv")}
    got = {p.name for p in (root / "tables").iterdir()}
    figs = {p.name for p in (root / "figures").iterdir()}
    check(f"tables: the full RQ3 set builds from a synthetic {n}-run grid",
          n == 162 and got == want and figs == {"rq3_delta_replay.png", "rq3_broad_share.png"},
          f"{sorted(want - got)} {sorted(figs)}")
    sc = (root / "tables" / "prereg_outcomes_rq3.md").read_text(encoding="utf-8")
    check("tables: the scorecard has a verdict for every line 10-17 and a framing line",
          all(f"| {k} |" in sc for k in range(10, 18)) and "Framing map, branches that apply" in sc
          and "expectation_correct" in sc)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{'=' * 50}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
