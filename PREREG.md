# Pre-registration

Predictions are committed before the runs they predict. `git log -- PREREG.md`
is the record of when each line was written. A design change made after seeing a
number goes here as a declared ablation first; the primary result stays as is.

## Phase 6/7 predictions (line 0 written 2026-10-06, before the seed 1 and 2 joint baselines; lines 1 to 6 written 2026-10-08, before any replay/FKD run)

0. Joint baselines, seeds 1 and 2: the teacher/gold macro-F1 ratio stays within
   0.05 of seed 1234's 0.69 on both seeds.

Threshold: a difference counts as real only if it exceeds 0.03 (Phase 5 noise floor
on the final diagonal was ~0.03; joint baselines moved 0.034 across seeds from
initialisation alone) on at least 2 of 3 seeds.

Metric (clarified 2026-10-08, before any replay/FKD run): "final ACC" in predictions 1 to 3 means ACC_f1, and BWT means BWT_f1 (both from the macro-F1 R matrix); ACC_acc and BWT_acc are reported alongside as secondary.

1. Replay beats naive on final ACC in all 6 track x seed runs. (True by construction,
   since naive old-task rows are at zero. Recorded for completeness, not a finding.)
2. Gold track, FKD minus replay, final ACC: within ±0.03 on every seed (no real difference).
3. Teacher track, FKD minus replay, final ACC: within ±0.03 on every seed (no real difference).
4. Analogous pairs: FKD's drop on the cross-task pairs is smaller than replay's by more
   than 0.10 per-class F1 on at most 1 of 3 seeds, in the gold and teacher tracks.
5. The FKD minus replay gap is the same in the teacher track as the gold track (both
   within the threshold), because FKD acts at only two task boundaries. If a gap does
   appear, it is smaller in the teacher track, because the focal weight is computed
   from the noisy training label, so FKD concentrates on mislabeled memory samples.
6. Teacher track: the share of predictions going to the broad class falls in the tasks
   after the one where it is learned, under replay, because memory caps it at
   m = 10 samples like every other class.

### Lines 7 to 9 (written 2026-10-09, after Gate 2 and the seed-1234 primary runs, before any fixed-schedule run)

7. Fixed 10-epoch replay stage: FKD minus replay, final ACC_f1, is within ±0.03 on
   at least 2 of 3 seeds in both tracks, because s is near uniform (row entropy 98
   to 99% of max on seed 1234) and the FKD term is under 5% of the replay loss.
8. Fixed 10-epoch replay stage: plain KD forgets less than replay on the
   non-degenerate tasks (mean per-task forgetting, threshold 0.03, at least 2 of 3
   seeds) in both tracks, because it distils over OLD at full strength, where FKD's
   weighting shrinks the same term to a few percent of the loss.
9. Reporting: forgetting conclusions come from per-task forgetting on the
   non-degenerate tasks. BWT_f1 is reported but not read as forgetting when the
   degenerate task isn't last.

## RQ3 predictions (written 2026-11-05, after Gate 2, before any §2.4 run)

Disclosure: during Gate 2 smoke testing, Claude Code printed final macro-F1 for six
seed-1234 RQ3 runs (deleted, not part of the sweep). I have not seen those values.
These predictions were written with knowledge of the Gate 1 diagnostics (row
accounting, the confusion matrix C and its diagonal check, full-track broad-class
shares and Δ_replay) and the T7 label statistics. No RQ3 training result was used.

Threshold unchanged: real = |Δ| > 0.03 with the same sign across replicates,
on at least 2 of 3 seeds. Comparisons on final 17-class test macro-F1 unless stated.
Line 10 is scored on replicate 0.

10. Size. gold_I joint is about 0.06 below full gold joint (real): fewer rows and
    4 classes under 15, partly offset by I holding the rows the teacher handled,
    which are likely easier. gold_sub is about 0.09 below full gold: same size, no
    selection toward easy rows, and 6 to 7 thin classes. The matched-row ratio
    teacher_I / gold_I is about 0.78, above 0.68 to 0.71, because part of the
    unmatched gap was data size and the 169 out-of-space rows.

11. Rate vs structure, joint: teacher_I is lower than rand_I. Uniform flips leave
    the correct label as the local majority in every class region. The teacher has
    three classes where the wrong label is the majority (children and parents to
    other_family, positive_impression to girl/boyfriend), plus errors clustered on
    similar examples. Both change which class wins.

12. Instance dependence, joint: teacher_I is the same as perm_I. In the 14 diagonal-
    dominant classes, the teacher's clustered errors should hurt more than perm_I's
    scattered ones. In the 3 non-dominant classes the reverse is likely: perm_I
    spreads the wrong majority across the whole class region, while the teacher's
    correct rows stay clustered. Over 17 classes I expect these to roughly cancel.

13. Δ_replay is not real-positive for teacher_I, perm_I, rand_I or gold_I. So the
    controls' clauses hold and the structured conditions' clauses don't: "partly:
    gold_I, rand_I", which means no support for the core claim. Basis: full teacher
    track Δ_replay was -0.014 / +0.022 / -0.083, and joint is the usual upper bound
    under clean labels.

14. Balancing (macro-F1): Δ_bal recovers at least 50% of Δ_replay. Expected to
    fail by rule, since 13 predicts no real replay gain.

15. Broad-class share under joint: teacher_I and perm_I exceed 2x the gold test
    share (4.8%); rand_I does not. Replay reduces the share in teacher_I and
    perm_I by more than 3 points.

16. Reporting: replicates ran (r = 0, 1, 2). A cell is real only if the threshold
    holds and the sign agrees across replicates.

17. Balancing (share): in teacher_I, joint_balanced recovers at least 60% of
    replay's broad-class share reduction, on at least 2 of 3 seeds. Both bring
    other_family to about 1/17 of the effective training mass, so most of
    replay's effect on the share should be class balancing.

rq3-scoring:
  line10_gold_I_minus_gold   = -0.06
  line10_gold_I_real         = yes
  line10_gold_sub_minus_gold = -0.09
  line10_ratio               = 0.78
  line11                     = lower
  line12                     = same
  line13                     = partly: gold_I, rand_I
  line13_controls            = not_real_positive
  line14_pct                 = 50
  line14_expect              = fails
  line15_pp                  = 3
  line15_expect              = holds
  line17_pct                 = 60
  line17_expect              = holds
