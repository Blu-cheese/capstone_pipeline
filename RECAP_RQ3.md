# RECAP: RQ3, noise structure vs noise rate

November 6, 2026. Spec: `RQ3_SPEC.md` (docs repo, with its Gate 1 and Gate 2
amendments). Predictions: `PREREG.md` lines 10 to 17, committed at `38ec218`
before any sweep run. Runs: 162, all from code commit `504fb3e`, outputs in
`8dddaa1`. Tables in `results/tables/`: T7 to T11 and
`prereg_outcomes_rq3` come from `continual/make_tables.py`, unchanged since
before the sweep; T12, E1 and E2 come from `continual/rq3_extra.py`, added
after it. If this file and a table disagree, the table wins.

Notation: seeds 1234 / 1 / 2. Replicate r = 0, 1, 2, where init seed = seed +
100r, and rand_I, perm_I and gold_sub also redraw their labels or rows. Values
are replicate 0 unless a range is given in brackets. A cell is **real** if
|Δ at r0| > 0.03 and the paired replicate values have one sign; a direction
(real-positive / real-negative) needs 2 of 3 seeds (PREREG 16). "Final F1" is
final 17-class test macro-F1 on the gold test set. Tasks are numbered from 0.

---

## 1. Bottom line

1. **At a matched noise rate, the teacher's structured noise costs no more
   macro-F1 under joint training than uniform flips.** teacher_I − rand_I is
   −0.022 / +0.008 / +0.079: "same" by the rule, and the one real seed goes
   the teacher's way (PREREG 11 FAILED). Permuting the teacher's labels within
   each gold class, which keeps its confusion matrix exactly, gives the same
   result as the real labels (PREREG 12 HELD).
2. **Data size doesn't explain the teacher gap.** On the same 710 rows the
   teacher_I / gold_I ratio is 0.665 / 0.696 / 0.812, against 0.69 / 0.68 /
   0.71 on the full tracks (PREREG 10 FAILED on its predicted 0.78). The gap is
   the teacher's wrong labels on rows both tracks share.
3. **Replay never beats joint, in any condition** (PREREG 13 HELD, as
   predicted: "partly: gold_I, rand_I"). teacher_I's replicate-0 Δ_replay
   (+0.043, +0.057) doesn't survive the replicates. This retracts the Phase 7
   reading that teacher replay can beat the teacher joint baseline.
4. **The broad-class collapse is a class-conditional signature, and class
   balancing removes it without buying macro-F1.** Under joint, 14 to 19% of
   teacher_I and perm_I test predictions go to `per:other_family` (gold share
   2.4%), against 1 to 5% for rand_I. Replay cuts that by 3.4 to 10.8 points and
   joint_balanced by as much or more (PREREG 15 and 17 HELD). Δ_bal on macro-F1
   is not real for teacher_I or perm_I; it is real-positive only on clean gold_I.
5. **Exploratory, not preregistered: structure does matter under replay.**
   teacher_I replay ends at final F1 0.336 / 0.354 / 0.340, rand_I replay at
   0.166 / 0.225 / 0.198, and their replicate ranges don't overlap on any seed.
   E1 points to why: uniform flips fill the memory of small, clean classes with
   wrong labels (`age`, `pet`, `visited_place`: 6 to 9 of 10 memory rows wrong
   in rand_I, 0 or 1 in teacher_I), and the replay stage trains on that memory.

---

## 2. What ran

| condition | training rows | labels | regimes |
|---|---|---|---|
| gold_I | the 710 rows of I | gold | joint, joint_balanced, naive, replay |
| teacher_I | I | teacher (ε_I = 0.3972, 282 errors) | same four |
| perm_I | I | teacher labels permuted within each gold class (C kept exactly); redrawn per replicate | same four |
| rand_I | I | gold, with exactly 282 rows flipped uniformly to another class; redrawn per replicate | same four |
| gold_sub | 710 random gold-track rows (not restricted to I); redrawn per replicate | gold | joint, replay |

- I = rows whose gold and teacher labels are both single-label and both inside
  the 17 classes. It is a strict subset of both tracks (§3, 1.3).
- 3 seeds × 3 replicates × 18 condition-regime pairs = 162 runs. Replay is the
  Phase 7 primary replay (early-stopped replay stage, m = 10, k-means memory on
  the task-order seed). joint_balanced draws batches with a
  `WeightedRandomSampler` (weight 1 / class count, with replacement); it is our
  addition, not from Zhao et al. No FKD or KD under RQ3.
- **The first sweep stopped at 159 runs.** At code `458b11b`, gold_sub seed 1234
  r2 has 11 `per:age` rows but only 8 distinct points in the subject+object
  features (rows sharing a subject and object), so k-means memory with k = 10
  had to leave clusters empty and the per-cluster guard raised. The fix in
  `504fb3e` sets k = min(m, distinct points), so that class keeps 8 memory rows.
  A scan of every track, seed and replicate found no other class with more than
  10 rows and fewer than 10 distinct points. All 162 runs were redone at
  `504fb3e`. The 159 that exist at both commits are bit-identical (R matrices,
  memory, metrics, predictions, train and replay logs; only run ids differ), and
  the `458b11b` copies were deleted.

---

## 3. §1 diagnostics (Gate 1, `results/diagnostics/`)

- **1.1 Housekeeping.** Phase 7 commits pushed (`ae75ccb..2dd8886`) before any
  RQ3 work.
- **1.2 KD targets on the memory** (`kd_memory_agreement.md`, from saved
  files). The snapshot's argmax repeats the memory label on 70 to 100% of
  old-class memory samples, but mean P(label) is only 0.48 to 0.70 at t = 2, and
  on new-class samples its OLD distribution sits at 37 to 76% of max entropy.
  So the claim half holds: KD's targets mostly agree with the labels, and the
  extra soft mass doesn't reduce forgetting either (PREREG 8).
- **1.3 Row accounting** (`row_accounting.md`). Both tracks come from the first
  400 DialogRE train dialogues: 2,144 pairs, minus 103 multi-label (not 124:
  124 is dev + test), minus 23 last-of-dialogue pairs the teacher never
  labelled, leaves 2,018 rows. Gold in 17 = 1,327, teacher in 17 = 879, both =
  **I = 710**, with 428 agreeing (ε_I = 0.3972). The teacher track's 51.3% noise
  is 282 in-space errors plus 169 rows whose gold label is outside the 17
  (`unanswerable` 125). RQ3 on I measures only the first kind.
- **1.4 The other_family check** (`broad_class_check.md`). `per:other_family` is
  the teacher's most over-predicted class on every seed (144 teacher vs 26 gold
  train labels). Seed 1's 20.5% → 9.4% share and 0.350 vs 0.328 (teacher replay
  vs joint) are confirmed. Teacher replay beats teacher joint on seed 1 only
  (Δ_replay −0.014 / +0.022 / −0.083). The joint shares come from a recompute
  with the unchanged `train_joint()` that reproduced all six
  `joint_baseline_seed*.json` exactly (macro-F1, accuracy, best epoch, epochs
  run, best dev F1, per-class F1).
- **1.5 Read and report.** Track names were hard-wired in the harness and
  replay regime, so conditions went into a new `continual/conditions.py`, with
  gold and teacher kept bit-identical (tested). There was no balanced sampler;
  `train_joint_balanced()` sits next to the untouched `train_joint()`. A
  separate init seed was a small change (tested). The degenerate task keeps its
  position under every condition, since the split maps classes to tasks.
- **1.6 Disagreement audit** (`audit_50.csv`). 50 of the 282 disagreeing rows
  in I, stratified on the teacher's top 3 over-predicted classes on I
  (other_family, works, negative_impression), with full dialogue text. The
  `teacher_defensible` and `note` columns are for Adithya and are still empty.
- **Gate 2 label stat.** For 14 of 17 gold classes, the teacher's most frequent
  label on I is the correct one. The three exceptions: `children` (5 correct vs
  16 to other_family), `parents` (9 vs 20 to other_family, and 9 to children)
  and `positive_impression` (11 vs 20 to girl/boyfriend).

---

## 4. Results

### T7: condition stats (`T7_condition_stats.md`, `T7b_teacher_noise_breakdown.md`)

All I conditions have 710 rows. teacher_I and perm_I carry 97 other_family rows
(13 of them correct), rand_I 21 to 36, gold_I 16. perm_I's C equals teacher_I's
on every draw (asserted). Classes under 15 rows: gold_I 4 (min 9), teacher_I
and perm_I 1 (`visited_place` 10), rand_I none, gold_sub 6 or 7 (min 4). T7b
breaks the full teacher track down: 428 correct + 282 in-space errors + 125 gold
`unanswerable` + 44 gold sub-threshold = 879.

### T8: joint and joint_balanced (`T8_rq3_joint.md`)

| condition | joint | joint_balanced |
|---|---|---|
| gold_I | 0.440 / 0.427 / 0.441 | 0.473 / 0.476 / 0.460 |
| teacher_I | 0.293 / 0.297 / 0.358 | 0.380 / 0.370 / 0.368 |
| perm_I | 0.358 / 0.287 / 0.342 | 0.363 / 0.298 / 0.348 |
| rand_I | 0.315 / 0.289 / 0.279 | 0.263 / 0.280 / 0.276 |
| gold_sub | 0.442 / 0.412 / 0.410 | — |
| gold, full 1,327 rows (reference) | 0.451 / 0.485 / 0.484 | — |
| teacher, full 879 rows (reference) | 0.312 / 0.328 / 0.344 | — |

- gold_I − gold_sub is −0.002 / +0.015 / +0.031, so I's rows are at most
  slightly easier than a random gold subset of the same size.
- teacher_I's joint range is 0.293–0.338, 0.297–0.357 and 0.357–0.368. The full
  teacher track sits inside it on seeds 1234 and 1 and just below on seed 2, so
  dropping the 169 out-of-space rows (and with them 19% of the data) changes
  little.

### T9: continual runs (`T9_rq3_cl.md`)

Final F1 under replay: gold_I 0.400 / 0.371 / 0.325, teacher_I 0.336 / 0.354 /
0.340, perm_I 0.311 / 0.282 / 0.275, rand_I 0.166 / 0.225 / 0.198, gold_sub
0.422 / 0.342 / 0.321. Replay ACC_f1 is 0.47 / 0.43 / 0.37 (gold_I), 0.40 /
0.43 / 0.42 (teacher_I), 0.38 / 0.35 / 0.34 (perm_I) and 0.25 / 0.33 / 0.28
(rand_I), against 0.08 to 0.21 under naive. Naive forgets as in Phase 5 (seed 2
naive is the majority predictor in every condition, since the degenerate task is
last). Under replay, teacher_I is within 0.02 of gold_I on seeds 1 and 2,
because replay costs gold_I 0.04 to 0.12 against joint and costs teacher_I
nothing measurable.

### T10: the key comparison (`T10_rq3_key_comparison.md`)

| condition | Δ_replay | Δ_bal | Δ_share replay (points) | Δ_share bal (points) |
|---|---|---|---|---|
| gold_I | −0.040 / −0.056 / −0.116: **real-negative** | +0.034 / +0.049 / +0.019: **real-positive** | not real | not real |
| teacher_I | +0.043 / +0.057 / −0.018: not real (ranges cross zero) | +0.088 / +0.073 / +0.010: not real (real on 1234 only) | −3.4 / −6.9 / −10.8: **real-negative** | −10.2 / −7.9 / −8.5: **real-negative** |
| perm_I | −0.047 / −0.005 / −0.067: **real-negative** | +0.005 / +0.012 / +0.006: not real | −6.0 / −5.7 / −3.4: **real-negative** | −14.2 / −14.1 / −14.4: **real-negative** |
| rand_I | −0.149 / −0.064 / −0.081: **real-negative** | −0.052 / −0.008 / −0.003: not real | not real | not real |
| gold_sub | −0.019 / −0.069 / −0.089: **real-negative** | — | not real | — |

### T11: broad-class share (`T11_rq3_broad_share.md`)

Share of the 1,144 test predictions going to `per:other_family`, %. Gold share
2.36% (27 of 1,144).

| condition | joint | joint_balanced | replay |
|---|---|---|---|
| gold_I | 0.6 / 0.0 / 0.7 | 0.8 / 0.8 / 1.4 | 2.3 / 1.5 / 0.9 |
| teacher_I | 14.1 / 14.2 / 19.4 | 3.8 / 6.4 / 10.9 | 10.7 / 7.3 / 8.7 |
| perm_I | 15.4 / 14.1 / 14.9 | 1.2 / 0.0 / 0.4 | 9.4 / 8.4 / 11.5 |
| rand_I | 1.1 / 0.9 / 4.9 | 2.7 / 3.5 / 4.3 | 4.5 / 2.7 / 3.8 |

joint_balanced pushes perm_I's share below the gold share, while teacher_I keeps
3.8 to 10.9%.

### T12: resolution (`T12_rq3_resolution.md`)

Final F1, min – max over replicates (spread):

| condition | regime | 1234 | 1 | 2 |
|---|---|---|---|---|
| gold_I | joint | 0.440 – 0.444 (0.004) | 0.427 – 0.446 (0.019) | 0.440 – 0.452 (0.013) |
| gold_I | joint_balanced | 0.458 – 0.486 (0.028) | 0.456 – 0.478 (0.022) | 0.460 – 0.479 (0.019) |
| gold_I | naive | 0.120 – 0.126 (0.006) | 0.116 – 0.117 (0.001) | 0.031 – 0.031 (0.000) |
| gold_I | replay | 0.383 – 0.400 (0.017) | 0.371 – 0.398 (0.028) | 0.321 – 0.337 (0.017) |
| teacher_I | joint | 0.293 – 0.338 (0.045) | 0.297 – 0.357 (0.060) | 0.357 – 0.368 (0.011) |
| teacher_I | joint_balanced | 0.380 – 0.388 (0.008) | 0.340 – 0.386 (0.046) | 0.368 – 0.397 (0.028) |
| teacher_I | naive | 0.116 – 0.125 (0.009) | 0.108 – 0.114 (0.006) | 0.031 – 0.042 (0.011) |
| teacher_I | replay | 0.313 – 0.336 (0.023) | 0.318 – 0.366 (0.048) | 0.279 – 0.340 (0.060) |
| perm_I | joint | 0.331 – 0.358 (0.027) | 0.287 – 0.336 (0.049) | 0.342 – 0.355 (0.013) |
| perm_I | joint_balanced | 0.352 – 0.384 (0.032) | 0.298 – 0.409 (0.111) | 0.348 – 0.366 (0.019) |
| perm_I | naive | 0.108 – 0.127 (0.019) | 0.046 – 0.121 (0.075) | 0.031 – 0.074 (0.043) |
| perm_I | replay | 0.311 – 0.328 (0.017) | 0.282 – 0.306 (0.024) | 0.275 – 0.306 (0.031) |
| rand_I | joint | 0.294 – 0.343 (0.049) | 0.289 – 0.329 (0.040) | 0.279 – 0.324 (0.045) |
| rand_I | joint_balanced | 0.263 – 0.302 (0.039) | 0.268 – 0.294 (0.026) | 0.276 – 0.290 (0.014) |
| rand_I | naive | 0.078 – 0.095 (0.017) | 0.071 – 0.100 (0.029) | 0.031 – 0.031 (0.000) |
| rand_I | replay | 0.152 – 0.217 (0.065) | 0.224 – 0.268 (0.044) | 0.179 – 0.213 (0.034) |
| gold_sub | joint | 0.428 – 0.442 (0.014) | 0.412 – 0.450 (0.039) | 0.400 – 0.440 (0.040) |
| gold_sub | replay | 0.363 – 0.422 (0.060) | 0.342 – 0.414 (0.072) | 0.275 – 0.333 (0.058) |

23 of 54 cells spread more than 0.03 over replicates, up to 0.111. So a single
run can't resolve a difference of 0.03. The PREREG rules compare paired
differences within a replicate, which is why they hold up, and why Phase 7's
single-init comparisons should be read with this spread in mind. gold_I spreads
least (at most 0.028). teacher_I, whose replicates also differ only in init,
spreads as wide as the conditions that redraw their labels (up to 0.060).

### Figures (`results/figures/`)

`rq3_delta_replay.png` (Δ_replay per condition and seed, replicate range as a
line, the ±0.03 band shaded) and `rq3_broad_share.png` (broad-class share per
condition and regime, with the gold share and 2× gold lines).

---

## 5. PREREG lines 10 to 17 against outcomes (`prereg_outcomes_rq3.md`)

| line | prediction | what happened | verdict | expected |
|---|---|---|---|---|
| 10 | gold_I ≈ gold − 0.06 (real); gold_sub ≈ gold − 0.09; teacher_I / gold_I ≈ 0.78 | gold_I − gold −0.011 / −0.058 / −0.043, real-negative (held). gold_sub − gold −0.009 / −0.074 / −0.074 (both values held on 2 of 3 seeds; seed 1234 off). Ratio 0.665 / 0.696 / 0.812, within 0.05 of 0.78 on 1 seed | **FAILED** (ratio) | — |
| 11 | joint: teacher_I lower than rand_I | −0.022 [−0.022, +0.034] / +0.008 / +0.079 [+0.044, +0.079]: real on seed 2 only, and positive → "same" | **FAILED** | — |
| 12 | joint: teacher_I the same as perm_I | −0.065 [−0.065, +0.007] / +0.010 / +0.016 → "same" | **HELD** | — |
| 13 | partly: gold_I, rand_I (no condition's Δ_replay real-positive) | teacher_I not real; perm_I, rand_I, gold_I real-negative → "partly: gold_I, rand_I" | **HELD** | — |
| 14 | Δ_bal ≥ 50% of Δ_replay | Δ_replay real-positive on 0 of 3 seeds, so it fails by rule | **FAILED** | fails: correct |
| 15 | joint share > 2× gold for teacher_I and perm_I, not rand_I; replay cuts > 3 points in both | all parts held (rand_I at or below 2× on seeds 1234 and 1; perm_I's cut under 3 points in one seed-1234 replicate) | **HELD** | holds: correct |
| 16 | replicates ran; reporting rule | replicates ran (r = 0, 1, 2), rule applied | applied | — |
| 17 | joint_balanced removes ≥ 60% of replay's share cut, teacher_I | 1234: D_bal / D_rep 0.84 to 3.00 (D_rep ≥ 3.4 points); 1: −0.06 to 1.14, fails; 2: 0.79 to 1.78 → 2 of 3 | **HELD** | holds: correct |

Line 13's "partly" means the two controls behaved as predicted and the two
structured conditions show no replay gain: no support for the original claim
that replay specifically helps under structured noise.

---

## 6. Framing map

Only one branch fired: **"perm_I ≈ teacher_I (line 12 observed 'same'): the
structure that matters is class-conditional."**

The Δ_share branch added at Gate 1 ("Δ_share real on teacher_I and perm_I but
Δ_replay not real") **did not fire by its rule**. Δ_share under replay is
real-negative for both conditions, but perm_I's Δ_replay is real-negative, not
"not real". The data is consistent with that branch's description: class
balancing removes the collapse signature without improving macro-F1. By the rule
written before the sweep, though, it is not the branch that applies.

---

## 7. Exploratory analyses (not preregistered, added after the sweep)

These come from saved files only (`continual/rq3_extra.py`). They suggest
explanations and claim nothing.

### E1: replay buffer purity (`E1_rq3_buffer_purity.md`)

Memory rows (selected by training label) whose gold label differs, replay, r0:

| | teacher_I | perm_I | rand_I |
|---|---|---|---|
| all classes | 69 / 69 / 70 of 170 (0.41) | 77 / 73 / 78 (0.43–0.46) | 87 / 85 / 80 (0.47–0.51) |
| expected if random within class | 0.41 | 0.41 | 0.49–0.50 |

- k-means memory is about as pure as a random draw within each class would be,
  in every condition. It doesn't select clean or dirty rows.
- Where the impurity sits is what differs. teacher_I and perm_I: `other_family`
  8 or 9 of 10 wrong, and 5 to 8 for `children`, `parents`,
  `positive_impression`, `negative_impression` and `works`; but `age` 0,
  `pet` 1 to 2, `visited_place` 1 (its whole 10-row class is the memory).
  rand_I: `age` 7 to 8, `pet` 6 to 9, `visited_place` 7 to 8, `works` 7 to 9.
  Uniform flips swamp small classes, since each class receives about the same
  number of wrong rows whatever its size.
- This fits rand_I's large replay penalty (Δ_replay −0.149 / −0.064 / −0.081),
  since the replay stage trains on this balanced, mostly wrong memory for the
  small classes. It is a suggested mechanism, not a tested one.

### E2: per-class F1 under joint (`E2_rq3_per_class_f1.md`, `E2b_rq3_largest_drops.md`)

The three classes each condition hurts most relative to gold_I (r0):

| seed | teacher_I | rand_I |
|---|---|---|
| 1234 | pet −0.600, roommate −0.340, positive_impression −0.266 | pet −0.600, works −0.480, employees_or_members −0.196 |
| 1 | pet −0.444, roommate −0.429, positive_impression −0.288 | pet −0.444, visited_place −0.361, positive_impression −0.263 |
| 2 | children −0.327, positive_impression −0.305, works −0.213 | pet −0.545, works −0.480, employees_or_members −0.386 |

- teacher_I hurts the classes in its confusion pattern: `children` (F1 0.000 /
  0.000 / 0.041), `parents` (0.04 to 0.10), `positive_impression` (0.03 to
  0.06), `roommate` (0.000 / 0.000 / 0.148), `spouse` (0.000 on all seeds). It
  does better than gold_I on `other_family` (0.16 to 0.18 vs 0.00 to 0.12) and
  `visited_place`.
- rand_I hurts the small classes: `pet` 0.000 on every seed, `works` 0.000 on
  two, `employees_or_members` and `age` on seed 2.
- So the macro-F1 cost is the same at a matched rate (PREREG 11), but different
  classes pay it.

---

## 8. Anomalies

1. **The first sweep's crash** (§2): one cell, fixed by capping k at the
   number of distinct points; no completed run changed.
2. **2× gold share.** The exact gold test share is 27 / 1,144 = 2.36%, so 2× is
   4.7%. PREREG 15 says 4.8% (from the rounded 2.4%). The scorer used the exact
   value. No outcome changes: the only seed near the line is rand_I seed 2, and
   its replicate-0 share (4.9%) is above both values.
3. **Line 17's seed 1234 cleared the 3-point bar by 0.4 points** (smallest
   D_rep 3.4).
4. **teacher_I joint's replicate 0 is the bottom of its range on seeds 1234 and
   1** (0.293 in 0.293–0.338, 0.297 in 0.297–0.357). That is what made
   replicate-0 Δ_replay look positive there.
5. **teacher_I `pet` F1 is 0.000 under joint on seeds 1234 and 1**, although 13
   of its 15 `pet` training rows are correct. gold_I gets 0.600 / 0.444 from 13
   rows. Not investigated.
6. **Smoke-test disclosure (Gate 2).** Before the PREREG, final F1 for six
   seed-1234 RQ3 test runs was printed in the session (deleted, not part of the
   sweep). The PREREG records this and states the values were not seen.

---

## 9. What this changes for the paper

- **Retract** "teacher replay can beat the teacher joint baseline". It held on
  one Phase 7 seed, and under replicates on I it is not real.
- **The ~69% headline survives the size control.** On matched rows the ratio is
  0.665 / 0.696 / 0.812.
- **Structure vs rate:** at a matched rate, structure doesn't change joint
  macro-F1 (PREREG 11 failed). Under joint, what the structure changes is which
  classes pay (E2, exploratory) and the broad-class collapse (T11, PREREG 15).
- **The collapse signature is class-conditional** (perm_I ≈ teacher_I, the
  framing branch that fired), and class balancing removes it without a real
  macro-F1 gain on noisy labels.
- **Replay vs noise structure** (T9, E1) is exploratory. To claim it, it needs
  its own PREREG entry and runs it hasn't been fitted to.
- **Single teacher** is a limitation: the second teacher (llama3.1:8b) was
  dropped in RQ3_SPEC.

## 10. Open

- `audit_50.csv` annotation (Adithya).
- Docs repo remote (Adithya).
