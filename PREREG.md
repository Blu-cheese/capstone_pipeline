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
