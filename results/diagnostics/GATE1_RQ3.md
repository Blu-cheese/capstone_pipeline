# RQ3 Gate 1 report (RQ3_SPEC §1.1 to §1.6)

November 5, 2026. Every number comes from `continual/rq3_diagnostics.py`
(`venv/bin/python -m continual.rq3_diagnostics`), whose outputs sit next to this
file. If this file and a generated `.md`/`.json` disagree, the generated file wins.
Tasks are numbered from 0, so t = 1, 2 are the paper's k = 2, 3.

No training run was made, with one declared exception (§1.4: joint predictions).
Nothing in `PHASE6_7_SPEC.md` §0 or §0.1 conflicts with anything below.

---

## 1.1 Housekeeping

Pushed `ae75ccb..2dd8886` to `origin/master` (`c530981`, `ee96337`, `a940992`,
`2dd8886`). The tree was clean. Docs repo remote left alone.

## 1.2 KD targets on the memory (`kd_memory_agreement.md`)

From saved files, no rerun. `run_all`'s Recorder saves `train_prob[t−1]` right
after `after_task(t−1)` returns, from the same weights the KD snapshot copies, in
eval mode, under the OLD mask. So it is the exact Eq. 12 target. Memory rows
come from `memory_indices.json` and the count matches `replay_log.jsonl`.

| track | seed | t | old samples | argmax = label | mean P(label) | new samples | new: entropy / max |
|---|---|---|---|---|---|---|---|
| gold | 1234 | 1 | 70 | 0.986 | 0.819 | 40 | 0.434 |
| gold | 1234 | 2 | 110 | 0.736 | 0.541 | 60 | 0.734 |
| gold | 1 | 1 | 70 | 0.871 | 0.712 | 40 | 0.521 |
| gold | 1 | 2 | 110 | 0.700 | 0.482 | 60 | 0.759 |
| gold | 2 | 1 | 70 | 0.943 | 0.893 | 60 | 0.479 |
| gold | 2 | 2 | 130 | 0.869 | 0.676 | 40 | 0.632 |
| teacher | 1234 | 1 | 70 | 0.986 | 0.919 | 40 | 0.366 |
| teacher | 1234 | 2 | 110 | 0.773 | 0.597 | 60 | 0.711 |
| teacher | 1 | 1 | 70 | 0.843 | 0.622 | 40 | 0.621 |
| teacher | 1 | 2 | 110 | 0.845 | 0.512 | 60 | 0.758 |
| teacher | 2 | 1 | 70 | 1.000 | 0.872 | 60 | 0.521 |
| teacher | 2 | 2 | 130 | 0.946 | 0.704 | 40 | 0.572 |

**The claim half holds.** The snapshot's argmax repeats the memory label on
70 to 100% of old-class samples (84% or more on 9 of 12 cells; the three lower
cells are all t = 2). The soft targets don't just repeat the label, though: mean
P(label) is 0.48 to 0.70 at t = 2, so a third to a half of the target mass is
off the label. On new-class samples the snapshot's OLD distribution sits at 37 to
76% of max entropy, so it pulls them toward particular old classes. Defensible
wording for the paper: "KD's targets on the memory agree with the memory's own
labels on 70 to 100% of old-class samples, and the remaining soft mass doesn't
reduce forgetting either (PREREG 8)". Not "KD is redundant because its targets
are the labels".

## 1.3 Row accounting (`row_accounting.md`, `rowset_I.json`)

| step | dialogues | pairs / rows |
|---|---|---|
| DialogRE train.json | 1,073 | 5,963 |
| teacher-labelled (first 400 in file order) | 400 | 2,144 |
| minus multi-label pairs | | −103 |
| minus pairs the teacher returned no label for | | −23 |
| single-label rows (feature matrix) | 397 | 2,018 |

The other 673 dialogues (3,819 pairs) are used by **neither** track. The gold
track never drew gold labels from outside the 400.

Two corrections:
- **Multi-label is 103, not 124.** 124 is dev 75 + test 49.
- **23 pairs were never labelled.** Each is the last pair of its dialogue: 23
  responses returned indices 0..n−2, the other 377 returned 0..n−1. No response
  is off by one, so no label is shifted. It's a teacher truncation, and those
  pairs are missing from both tracks.

| label space (2,018 rows, 17 classes) | rows |
|---|---|
| gold in 17 = **gold track** | 1,327 |
| teacher in 17 = **teacher track** | 879 |
| both = **I** | **710** |
| gold only | 617 (teacher said acquaintance 319, unanswerable 274, other 24) |
| teacher only | 169 (gold says unanswerable 125, siblings 23, other thin classes 21) |
| neither | 522 |

- Exact agreement on both: 428 / 710 = **0.6028**. On all 2,018 rows: 0.3736
  (the ~37% headline).
- **ε_I = 0.3972; rand_I flips round(ε_I · |I|) = 282 rows.**
- **I is not the 879 teacher rows.** It is a strict subset of both tracks. The
  169 teacher rows outside I all have a gold label outside the 17: unanswerable
  (the dropped null class) 125, per:siblings 23 (teacher has only 10, so it
  fell below 15), and 21 rows across 9 other sub-threshold classes. Their teacher
  labels lean on other_family (47) and alternate_names (35).
- So the teacher track's total noise is 451 / 879 = 51.3%: 282 in-space errors
  (captured by C) plus 169 rows that shouldn't be in a 17-class training set at
  all. **RQ3 on I measures only the first kind.** Comparing the full teacher
  track with teacher_I in T8 measures the second.
- Classes under 15 on I (frozen threshold not re-applied, reported only):
  gold_I employees_or_members 14, pet 13, visited_place 9, works 9; teacher_I
  visited_place 10.
- Per-class counts and the full 17 × 17 C are in `row_accounting.md`. The biggest
  off-diagonal cells: alternate_names → other_family 23, parents → other_family
  20, positive_impression → girl/boyfriend 20, children → other_family 16,
  children → parents 14, alternate_names → title 14.

## 1.4 The other_family check (`broad_class_check.md`)

Most over-predicted class (T5 definition): **per:other_family**, 144 teacher vs
26 gold (5.54×), the same on all seeds (the kept set is identical across splits).

| seed | teacher joint | teacher naive | teacher replay | teacher FKD | gold joint | gold test | T3: teacher replay vs joint |
|---|---|---|---|---|---|---|---|
| 1234 | 0.185 | 0.000 | 0.114 | 0.087 | 0.006 | 0.024 | 0.298 vs 0.312 (−0.014) |
| 1 | 0.205 | 0.000 | 0.094 | 0.128 | 0.007 | 0.024 | 0.350 vs 0.328 (+0.022) |
| 2 | 0.135 | 0.002 | 0.095 | 0.101 | 0.007 | 0.024 | 0.262 vs 0.344 (−0.083) |

- **Seed 1 figures confirmed:** 20.5% → 9.4%, and 0.350 vs 0.328.
- **Correction:** teacher replay beats teacher joint on **seed 1 only**. On seeds
  1234 and 2 it is below joint. The share reduction from joint to replay holds on
  every seed (18.5 → 11.4, 20.5 → 9.4, 13.5 → 9.5).
- **Relevant to PREREG line 13:** on the existing teacher track, Δ_replay
  (replay − joint, final macro-F1) is −0.014 / +0.022 / −0.083. That's not "real
  and positive" by the 2-of-3 rule; it leans negative. Line 13's prediction should
  be written knowing this.
- **Declared deviation from "saved files only":** joint test predictions were
  never saved. They were recomputed in memory with the unchanged `train_joint()`
  and used only because all six (2 tracks × 3 seeds) reproduce
  `joint_baseline_seed*.json` exactly: macro-F1, accuracy, best epoch, epochs run,
  best dev F1 and per-class F1. Nothing was written to the joint files. If that's
  not acceptable, the joint share column goes and everything else stands.

## 1.5 Read and report

**1. Can a new condition be a new entry in `run_all.py`'s track list? No.**
The track name is hard-wired in three places: `harness.run_sequential` asserts
`track in ("teacher", "gold")` (`harness.py:227`) and `_track_train_data` picks
`y_teacher` if the track is "teacher", else `y_gold` (`harness.py:199`); `ReplayRegime`
asserts the same (`regimes.py:218`) and `resolve_feature_rows` repeats the
selection (`regimes.py:188`). The joint baseline isn't a `run_all` regime at all:
`student.run_joint_baseline` loops over a fixed (teacher, gold) tuple
(`student.py:195`) and writes both into one `joint_baseline_seed{seed}.json`.
(`CAPSTONE_CONTEXT.md` §7.2 says conditions slot in without code changes. They
don't.)

Plan that keeps gold and teacher bit-identical: a new `continual/conditions.py`
with one function, `(track, seed) → (feature rows, head labels)`. For "gold" and
"teacher" it returns exactly what `_track_train_data` returns today (same rows,
same order); the new tracks build from I. `_track_train_data` and
`resolve_feature_rows` call it, and the two asserts accept the new names. `joint`
and `joint_balanced` become `run_all` regimes writing ordinary run dirs
(`metrics.json` + `predictions.npz`), calling the unchanged `train_joint()`. Test 1
checks bit identity; the six joint baselines already reproduce exactly (§1.4).
Also needed: `make_tables.select_runs` requires exactly one git hash among all
clean runs of a schedule (`make_tables.py:107-110`), so once RQ3 runs exist at a
new hash the Phase 7 tables would refuse to build. It has to filter on the Phase 7
tracks first.

**2. Is there a class-balanced sampler? No,** none anywhere (no `Sampler` in the
code). It goes in a new `train_joint_balanced()` next to `train_joint()` in
`student.py`, a copy that differs only in where the batch indices come from:
`WeightedRandomSampler(1 / count[y], num_samples=len(y), replacement=True)` on its
own `torch.Generator` seeded from the init seed, cut into batches of 32 with the
last partial batch kept. `train_joint()` isn't touched, and test 7 checks that it
never builds a sampler.

**3. Separate model-init seed? Yes, and the change is small.** The harness
uses `seed` for four things: RNG seeding (`harness.py:231-233`, which covers init
and batch order), the split (`:236`), the eval sets (`:249`, the same X on every
seed) and, through the regimes, KMeans `random_state` (`regimes.py:102`).
`train_joint` uses its `seed` only for RNG seeding (`student.py:120-121`). Change:
`run_sequential(..., init_seed=None)`, with `init_seed if not None else seed`
going to the three seeding calls only; `init_seed` added to run metadata and to
the skip key in `completed_runs` (`run_all.py:80`, `:284`) and the table keys.
The default path makes identical calls, so existing runs stay bit-identical
(test 1); test 8 checks that an offset changes the weights. **KMeans stays on the
task-order seed**, so replicates share the memory as well as the task order and
noise draw. That also keeps `write_artifacts`' check that runs of the same track
and seed selected identical memory (`regimes.py:546`) valid.

**4. Degenerate task position: unchanged under every condition, by construction.**
The split maps classes to tasks and is frozen; only the rows in each task move.
Seed 1234 (degenerate = task 1, marked †). rand_I and gold_sub are one provisional
`default_rng(seed)` draw to the §2.2 wording. T7 will give the final counts.

| condition | rows | task 0 | task 1 † | task 2 | classes < 15 rows |
|---|---|---|---|---|---|
| gold (existing) | 1,327 | 405 (31%) | 494 (37%) | 428 (32%) | none |
| teacher (existing) | 879 | 323 (37%) | 261 (30%) | 295 (34%) | none |
| gold_I | 710 | 209 (29%) | 259 (36%) | 242 (34%) | employees_or_members 14, pet 13, visited_place 9, works 9 |
| teacher_I | 710 | 244 (34%) | 218 (31%) | 248 (35%) | visited_place 10 |
| perm_I | 710 | 244 (34%) | 218 (31%) | 248 (35%) | visited_place 10 |
| rand_I | 710 | 253 (36%) | 223 (31%) | 234 (33%) | none |
| gold_sub | 710 | 211 (30%) | 270 (38%) | 229 (32%) | 7 classes, down to 7 rows |

perm_I's per-task counts always equal teacher_I's (the permutation stays inside
each gold class; asserted). Every task in every condition and seed stays at or
above 27.5% of rows, well clear of the split's 15% floor. Seeds 1 and 2 are in
`task_rows_by_condition.md`. gold_sub is thinnest per class (pet drops to 5 on
seed 2), because a random 53.5% subset halves the classes that only just cleared
15, while I keeps those classes nearly whole: the teacher gets them right.

## 1.6 Disagreement audit (`audit_50.csv`, `audit_50.meta.json`)

- Pool: the 282 teacher ≠ gold rows in I.
- Top 3 over-predicted classes, by T5's ratio computed on I: other_family
  (6.06×), works (2.56×), negative_impression (1.59×). On all 2,018 rows the third
  would be employee_or_member_of instead. I used I because the sample comes from I.
- Strata: 84 / 17 / 25 / 156 other. Largest-remainder allocation gives 15 / 3 / 4 /
  28. pandas `sample(random_state=1234)` within each stratum.
- Columns are the spec's, plus `pair_index`, entity types, `stratum` and
  `reverse_pair_gold` (the gold label of the (object, subject) pair when DialogRE
  annotates it, which helps spot direction errors; in 4 of the 50 the teacher's label
  equals it).
- `gold_relations_original` never differs from the single label: I excludes
  multi-label pairs by construction.
- Full dialogue text in every row. The longest train dialogue is 2,258 characters,
  so no excerpting was needed.

---

## Decisions for Adithya at this gate

1. **Row set: I = 710, as the spec defines it** (recommended). teacher_I is then
   not the teacher track: test 2 becomes "labels on the shared rows are
   identical", and T8's full-teacher row against teacher_I shows what the 169
   out-of-space rows cost. ε_I = 0.397 is the noise rate RQ3 tests, not 0.513.
2. **Init seed with KMeans pinned to the task-order seed** (recommended), so the
   replicates go ahead: +108 runs, 162 in total, all CPU-seconds.
3. **joint / joint_balanced as `run_all` regimes writing run dirs**, rather than new
   keys in `joint_baseline_seed*.json` (recommended).
4. **Top-3 for the audit defined on I** (as done), or on all rows.
5. **The §1.4 joint recompute:** keep it (verified exact) or drop that column.
