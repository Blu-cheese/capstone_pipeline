# RECAP — Phases 6 and 7: replay, FKD and the sweep

October 9, 2026. Spec: `PHASE6_7_SPEC.md` v4 (docs repo). Predictions:
`PREREG.md`. Every number below comes from the generated tables in
`results/tables/` (made by `continual/make_tables.py` from saved run files); if
this file and a table disagree, the table wins.

Tasks are numbered from 0. ACC_f1 / BWT_f1 are primary, accuracy versions
secondary (PREREG metric clarification). Forgetting on task j is
R[K−1][j] − R[j][j] (negative = forgetting), and forgetting conclusions use
only the non-degenerate tasks (PREREG 9).

---

## 1. Bottom line

1. **Replay prevents most catastrophic forgetting, in both tracks.** Final
   ACC_f1 goes from 0.08–0.21 under naive to 0.47–0.50 (gold) and 0.32–0.44
   (teacher). Forgetting on the non-degenerate tasks shrinks from about −0.37
   to −0.61 to between −0.18 and +0.03.
2. **FKD adds nothing measurable over replay.** With the fixed 10-epoch replay
   stage, FKD − replay is within ±0.008 ACC_f1 on all six track × seed cells.
   In the primary (early-stopped) runs it is within ±0.002 on four cells; the
   other two (+0.037, −0.035, opposite signs) trace to where early stopping
   landed, not to FKD. Neither schedule shows a real difference by the
   pre-registered rule.
3. **The reason is measurable.** FKD's prototype weighting is almost flat
   (s row entropy 98.7–99.5% of uniform in every run) and the FKD term is
   1.5–5.0% of the replay loss. Plain KD at full strength (no weighting) doesn't
   reduce forgetting either (PREREG 8 fails), so in this frozen-feature setup the
   memory itself is the lever, not distillation from the previous head.
4. **In the teacher track, balanced memory partly corrects the teacher's label
   bias.** The broad class `per:other_family` (144 teacher vs 26 gold training
   labels) takes 47–51% of seen-test predictions right after it is learned; replay
   brings that to 9–11% (gold share 2.4%). This is also why teacher-track replay
   can beat the teacher joint baseline (seed 1: 0.350 vs 0.328 final macro-F1)
   without any leakage.

## 2. What ran

| | |
|---|---|
| Code commit for all 36 runs | `c530981` |
| Primary sweep | naive, replay, fkd × gold, teacher × seeds 1234, 1, 2 = 18 runs |
| Fixed-schedule ablation (§4.4) | replay, fkd, kd with a 10-epoch replay stage, same grid = 18 runs |
| Outputs commit | `a940992` (all outputs, once, after the sweeps) |
| Provenance | `run_all.py` recorded hash + dirty flag before each run (all clean) |
| Reproduction checks | naive bit-identical to the 6 Phase 5 files; seed-1234 replay/FKD bit-identical to the 4 Phase 6 files; every run's R matrices re-derive exactly from its saved predictions |
| Tests | `tests/test_fkd.py` 59/59, `test_pipeline_fixes.py` 87/87, `test_parsers.py` 38/38 |
| Wall time | 26 s (primary) + 28 s (fixed), CPU |

Degenerate task (holds `per:alternate_names`): seed 1234 task 1, seed 1 task 1,
seed 2 task 2. So seeds 1234 and 1 have one non-degenerate old task (task 0),
seed 2 has two.

## 3. Results

### T1 — main (primary). `results/tables/T1_main.md`

| track | regime | ACC_f1 1234 / 1 / 2 | mean non-degenerate forgetting 1234 / 1 / 2 |
|---|---|---|---|
| gold | naive | 0.212 / 0.202 / 0.079 | −0.573 / −0.541 / −0.607 |
| gold | replay | 0.483 / 0.496 / 0.469 | −0.176 / −0.116 / −0.019 |
| gold | fkd | 0.520 / 0.494 / 0.470 | −0.182 / −0.112 / −0.019 |
| teacher | naive | 0.205 / 0.178 / 0.098 | −0.417 / −0.368 / −0.443 |
| teacher | replay | 0.369 / 0.442 / 0.316 | −0.104 / +0.009 / −0.057 |
| teacher | fkd | 0.369 / 0.407 / 0.316 | −0.101 / +0.027 / −0.053 |

BWT_f1 and the degenerate-task column are in the table. On seeds 1234 and 1
the degenerate task's "forgetting" is positive under replay and FKD (up to
+0.35), because its diagonal is depressed — BWT_f1 there is not read as
forgetting (PREREG 9). No cross-seed mean ± std, by design.

### T2 — accuracy on all observed relations (Zhao style). `T2_observed_accuracy.md`

Average accuracy over the three stages, gold: naive 0.42 / 0.41 / 0.45,
replay 0.45 / 0.46 / 0.52, fkd 0.50 / 0.46 / 0.52. Accuracy is dominated by
`per:alternate_names` (408 of 1,144 test examples), so naive looks better here
than on macro-F1 whenever the degenerate task was just learned.

### T3 — final macro-F1 over all 17 classes. `T3_final_macro_f1.md`

| track | naive | replay | fkd | joint |
|---|---|---|---|---|
| gold (1234 / 1 / 2) | 0.142 / 0.126 / 0.031 | 0.407 / 0.414 / 0.411 | 0.430 / 0.414 / 0.410 | 0.451 / 0.485 / 0.484 |
| teacher (1234 / 1 / 2) | 0.125 / 0.113 / 0.051 | 0.298 / 0.350 / 0.262 | 0.296 / 0.308 / 0.261 | 0.312 / 0.328 / 0.344 |

Gold replay reaches 85–90% of the joint upper bound. Teacher replay exceeds
the teacher joint baseline on seed 1 — see anomaly A3.

### T4 — analogous pairs. `T4_analogous_classes.md`, `T4_analogous_pairs.md`

Mean drop of the earlier-learned member over each seed's three cross-task
pairs (positive = F1 lost):

| track | naive | replay | fkd |
|---|---|---|---|
| gold (1234 / 1 / 2) | 0.54 / 0.57 / 0.48 | 0.36 / 0.35 / 0.18 | 0.37 / 0.35 / 0.19 |
| teacher (1234 / 1 / 2) | 0.40 / 0.41 / 0.43 | 0.22 / 0.17 / 0.27 | 0.21 / 0.18 / 0.27 |

Replay cuts analogous-pair loss by roughly a third to two thirds; FKD changes it
by at most 0.011. The within-task reference pair
(`girl/boyfriend ↔ negative_impression`) is in the table.

### T5 — teacher track post-hoc. `T5a_buffer_purity.md`, `T5b_broad_class_share.md`

- **Buffer purity:** 40–60% of the memory selected at each task carries the
  correct (gold) label; cumulative purity at the end is 0.49 on every seed.
  About half the teacher-track memory is mislabelled.
- **Broad class share** (`per:other_family`): naive 0.51 / 0.47 / 0.48 right
  after learning it, then ~0 (forgotten). Replay ends at 0.114 / 0.094 / 0.095,
  FKD 0.087 / 0.128 / 0.101; gold share 0.024.

### D1 — FKD diagnostics (§3.7). `D1_fkd_diagnostics.md`

s row entropy 98.7–99.5% of uniform; s row max 1.22–1.32× uniform; focal
factor 0.62–0.74 at the start of a replay stage, 0.10–0.28 at the end (the
student fits its 110–170 memory samples); λ·L_fkd is 1.5–5.0% of L_cls.
Analogous-pair s (seed 1234, task 2) is only 2–18% above uniform (teacher
`negative_impression → positive_impression`: 0.0925 vs uniform 0.0909).

### T6 — fixed 10-epoch replay stage (declared ablation). `T6_fixed_schedule.md`

| track | seed | replay | fkd | kd | fkd − replay | kd − replay (mean non-degen. forgetting) |
|---|---|---|---|---|---|---|
| gold | 1234 | 0.468 | 0.467 | 0.446 | −0.001 | +0.047 |
| gold | 1 | 0.462 | 0.464 | 0.429 | +0.001 | +0.006 |
| gold | 2 | 0.411 | 0.406 | 0.415 | −0.005 | +0.006 |
| teacher | 1234 | 0.356 | 0.363 | 0.347 | +0.007 | −0.006 |
| teacher | 1 | 0.436 | 0.435 | 0.414 | −0.000 | +0.008 |
| teacher | 2 | 0.328 | 0.336 | 0.363 | +0.008 | −0.001 |

(Columns 3–5 are final ACC_f1.) Plain KD's term is 23–27% of L_cls in the first
replay epoch and 77–112% in the last — the soft-target cross entropy has a floor
at the teacher's entropy, so the ratio rises as L_cls falls. Despite that
strength, KD lowers ACC_f1 in 4 of 6 cells and reduces forgetting by more than
0.03 only on gold seed 1234. The early-stopped replay stage gives higher gold
ACC_f1 than the fixed one (by 0.015–0.058); in the teacher track it is mixed.

### Figures. `results/figures/`

`acc_after_each_task.png` (ACC_f1 after each task, one panel per track × seed,
degenerate task shaded) and `confusion_seed1234_{gold,teacher}.png`.

## 4. PREREG against outcomes

Mechanically computed in `results/tables/prereg_outcomes.md`, using definitions
fixed in `make_tables.py` before the sweep ran (commit `c530981`). Verdicts are
as the predictions were worded; explanations do not change them.

| line | prediction (short) | outcome | verdict |
|---|---|---|---|
| 0 | joint teacher/gold ratio within 0.05 of 0.69 on seeds 1, 2 | 0.675 (Δ 0.016), 0.711 (Δ 0.020) | **holds** |
| 1 | replay beats naive on final ACC_f1, all 6 | 6 of 6 | **holds** |
| 2 | gold: FKD − replay within ±0.03 on every seed | +0.037 / −0.002 / +0.001 | **fails** (seed 1234 outside; no real difference by the threshold rule) |
| 3 | teacher: FKD − replay within ±0.03 on every seed | +0.000 / −0.035 / −0.000 | **fails** (seed 1 outside; no real difference by the threshold rule) |
| 4 | FKD's cross-task pair drop smaller than replay's by > 0.10 on at most 1 of 3 seeds | gold 0 of 3, teacher 0 of 3 (all |Δ| ≤ 0.011) | **holds** |
| 5 | gap the same in both tracks (both within threshold) | no real gap in either track | **holds** |
| 6 | teacher, replay: broad-class share falls after the task where it is learned | 0.510 → 0.114, 0.467 → 0.094, 0.108 → 0.095 | **holds** (2 of 3 seeds; on seed 2 the class is learned at task 1, whose own replay stage already caps it, so the later fall is only 0.013) |
| 7 | fixed: FKD − replay within ±0.03 on ≥ 2 of 3 seeds, both tracks | all six within ±0.008 | **holds** |
| 8 | fixed: KD forgets less than replay (> 0.03, ≥ 2 of 3 seeds), both tracks | gold 1 of 3, teacher 0 of 3 | **fails** |
| 9 | reporting rule (forgetting from non-degenerate tasks) | applied in T1 and T6 | applied |

Lines 2 and 3 fail on their literal "every seed" wording; in both, the outlying
seed is explained by anomaly A1, and by the threshold rule there is no real
difference — but the predictions as written failed, and should be reported so.
Line 8 failing is the more informative result: full-strength distillation does
not reduce forgetting here.

## 5. Anomalies and what they turned out to be

**A1. Primary FKD − replay gaps come from replay-stage early stopping.**
Gold seed 1234 (+0.037): the task-1 replay-stage dev curves match epoch by
epoch until replay's patience runs out at epoch 11 (best at 6), while FKD set a
new best by 0.0008 at epoch 7, survived, and trained to a best at epoch 27.
Teacher seed 1 (−0.035): at the task-2 replay stage replay kept epoch 31 and
FKD epoch 10, leaving FKD's task-2 F1 at 0.32 vs 0.45. Opposite signs, same
cause. With the fixed schedule every gap is within ±0.008. Model-selection
noise in the replay stage is about ±0.035 ACC_f1 — comparable to the 0.03
threshold.

**A2. Positive BWT_f1 / positive forgetting on the degenerate task** (seeds
1234 and 1). Naive's degenerate-task diagonal is the constant-majority
predictor; replay's balanced memory knocks the student off it, so the task's
diagonal is depressed and later rows recover (gold seed 1: +0.35). Not
backward transfer. Handled by PREREG 9.

**A3. Teacher replay beats the teacher joint baseline** (seed 1: 0.350 vs
0.328). Not leakage: memory and prototypes use training rows only (verified),
eval sets are frozen, and both use gold dev only for model selection. The joint
teacher model predicts `other_family` for 20.5% of test examples (gold share
2.4%); replay's final model, fine-tuned on 10 samples per class, 9.4%. Gains are
in the classes the teacher's prior crowds out (parents +0.20, spouse +0.16,
works +0.14, age +0.12 F1). Joint is an upper bound for training on the same
label distribution, not for a regime that ends with class-balanced fine-tuning.

**A4. Positive forgetting on a non-degenerate task, teacher seed 1** (task 0:
+0.009 replay, +0.027 FKD primary; +0.024 / +0.019 / +0.032 fixed). Same
mechanism as A3: task 0's diagonal starts with `other_family` taking 47% of its
predictions (gold 7.4%); balanced memory brings it to 9.6% and task-0 F1 rises.
All but fixed-KD's +0.032 are inside the 0.03 threshold.

**A5. Harness provenance bug.** `harness._git_hash()` runs after the run's
output directory is written, so every harness run, Phase 5 included, records
`-dirty` from its own output. `run_all.py` records the hash itself before each
run (spec v4 §4.1). The harness is unchanged; its `config.json` hash field
should be ignored in favour of `metrics.json`'s `meta.git_hash`.

## 6. For the report

- The CL contribution is a clean comparison with a clear negative result:
  **replay works, FKD adds nothing measurable, and the diagnostics say why**
  (flat s on frozen entity features, small FKD term, only two task boundaries).
  "FKD didn't help, and here's why" is a finding (CLAUDE.md §8).
- Plain KD failing too (PREREG 8) narrows the explanation: it isn't that FKD's
  weighting shrinks a useful signal; distillation from the previous head doesn't
  add retention in this frozen-feature setup.
- Teacher-track CL works on 37%-accurate labels, and the class-balanced replay
  stage partly undoes the teacher's broad-class bias — a link to the headline
  broad-class-collapse finding worth one paragraph.
- State the replay-stage selection noise (A1) and the early-stopping advantage
  replay/FKD get over naive (gold dev over all seen relations).
- Deviations from Zhao et al.: `docs/PAPER_NOTES.md`, "Full deviation list".

## 7. Open

- Optional ablations not run: plain KD on the primary schedule (6 runs), task
  masked (18 runs). Both need PREREG entries first.
- Docs repo has no remote yet (gh not installed; waiting for a private repo URL).
