# Pre-registration

Predictions are committed before the runs they predict. `git log -- PREREG.md`
is the record of when each line was written. A design change made after seeing a
number goes here as a declared ablation first; the primary result stays as is.

## Phase 6/7 predictions (line 0 written 2026-10-06, before the seed 1 and 2 joint baselines; lines 1 to 6 written YYYY-MM-DD, before any replay/FKD run)

0. Joint baselines, seeds 1 and 2: the teacher/gold macro-F1 ratio stays within
   0.05 of seed 1234's 0.69 on both seeds.

Threshold: a difference counts as real only if it exceeds ____ (Phase 5 noise floor
on the final diagonal was ~0.03) on at least ____ of 3 seeds.

1. Replay beats naive on final ACC in all 6 track x seed runs. (True by construction,
   since naive old-task rows are at zero. Recorded for completeness, not a finding.)
2. Gold track, FKD minus replay, final ACC: ____
3. Teacher track, FKD minus replay, final ACC: ____
4. Analogous pairs: FKD's drop on the cross-task pairs is smaller than replay's on
   ____ of 3 seeds, in the ____ track(s).
5. The FKD minus replay gap is larger / smaller / the same in the teacher track
   than the gold track, because ____.
6. Teacher track: the share of predictions going to the broad class rises / falls /
   stays flat across tasks under replay, because ____.
