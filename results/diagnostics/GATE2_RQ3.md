# RQ3 Gate 2 report (RQ3_SPEC §5)

November 5, 2026. No §2.4 sweep run has been made. The run guard refuses until
the PREREG RQ3 section is committed (tested through the real CLI).

| what | commit |
|---|---|
| Gate 1 diagnostics + outputs (pipeline) | `5521cac` |
| RQ3_SPEC.md as received (docs) | `5f2fd95` |
| RQ3_SPEC.md Gate 1 decisions and amendments, before any §2 code (docs) | `bcdaaf4` |
| §2 code, tests, the 33 frozen condition draws, T7/T7b (pipeline) | `a7c780a` |
| PREREG lines 10 to 16 | **not yet: Adithya** |

## Disclosure first

While smoke-testing the new run paths I printed final 17-class macro-F1 for six
seed-1234 RQ3 runs (temp dir, since deleted, not the sweep): gold_I naive,
teacher_I replay, rand_I replay r1, perm_I joint, teacher_I joint_balanced r2,
gold_sub replay. The values are in this session's collapsed tool output; I
haven't repeated them here and nothing uses them. The scoring rules below were
settled before that run. If you read that output before writing the PREREG, say
so under the RQ3 heading. Every check since prints pass/fail only.

## Tests (all suites)

| suite | result |
|---|---|
| tests/test_conditions.py (new) | 91 passed, 0 failed |
| tests/test_fkd.py | 59 passed, 0 failed, 0 skipped |
| tests/test_harness.py | 16 passed, 0 failed |
| tests/test_parsers.py | 38 passed, 0 failed |
| tests/test_pipeline_fixes.py | 87 passed, 0 failed |

test_conditions covers spec tests 1 to 9 and:
1. Naive, both tracks, 3 seeds: R bit-identical to Phase 5. Joint via `run_all`,
   both tracks, **all 3 seeds**: identical to `joint_baseline_seed*.json`.
   Replay seed 1234 via `run_all`, both tracks: R, memory and every prediction
   array identical to the Phase 7 primary runs.
2. teacher_I labels equal the teacher track's on the 710 shared rows.
3. perm_I keeps C exactly on all 9 draws, each changing labels.
4. rand_I: exactly 282 differ from gold, all 9 draws.
5. Determinism.
6. gold_sub subset and size.
7. Sampler within 2% of uniform; train_joint and the run_all joint path never
   build one.
8. init seed = seed reproduces the default; an offset changes weights.
9. With a fake RQ3 run at another hash present, all 22 Phase 7 table files
   **and** all 3 figures rebuild byte-identical, and without the track filter
   that run would have broken selection.

Extras: saved draws match what runs build; the guard and its parser; memory
consistency across replicates; joint run-dir contents; the 10 to 16 scoring
rules on hand-set values; the full RQ3 table set building from a synthetic
162-run grid of random numbers.

The real `make_tables` rebuild also left every Phase 7 file unchanged in place.

One fix to the Gate 1 commit: a bare `open()` in `rq3_diagnostics.py` failed
`test_pipeline_fixes`' encoding check. Fixed in `a7c780a`; the regenerated
diagnostics are byte-identical.

## T7 (`results/tables/T7_condition_stats.md`, `T7b_teacher_noise_breakdown.md`)

Seed 1234 shown; seeds 1 and 2 are in the file. † = degenerate task.

| condition | draw | rows | noise vs gold | other_family rows | task 0 | task 1 † | task 2 | classes < 15 |
|---|---|---|---|---|---|---|---|---|
| gold (full, ref) | — | 1327 | 0.0000 | 26 | 405 | 494 | 428 | — |
| teacher (full, ref) | — | 879 | 0.5131 | 144 | 323 | 261 | 295 | — |
| gold_I | — | 710 | 0.0000 | 16 | 209 | 259 | 242 | 4 (min 9) |
| teacher_I | — | 710 | 0.3972 | 97 | 244 | 218 | 248 | visited_place 10 |
| perm_I | 1234 / 1334 / 1434 | 710 | 0.3972 | 97 | 244 | 218 | 248 | visited_place 10 |
| rand_I | 1234 | 710 | 0.3972 | 29 | 253 | 223 | 234 | — |
| rand_I | 1334 / 1434 | 710 | 0.3972 | 21 / 25 | | | | — |
| gold_sub | 1234 / 1334 / 1434 | 710 | 0.0000 | 13 / 10 / 15 | | | | 6 to 7 classes, min 7 |

perm_I's C equals teacher_I's, asserted for every draw. The thinnest class in any
draw is gold_sub seed 2 r2 (`per:age`, 4 rows). rand_I has no broad
class: other_family gets 21 to 36 rows against teacher_I's 97.

T7b, the full teacher track: 428 correct + 282 in-space errors + 125 gold
`unanswerable` + 44 gold sub-threshold = 879. Separately, 617 gold-track rows
were lost (teacher said acquaintance 319, unanswerable 274, other 24).

## make_tables diff: the scoring code (`a7c780a`, `continual/make_tables.py:732` onwards)

The definitions are the comment block at the top of the RQ3 section. In short:

- **Real cell.** |Δ at r0| > 0.03 and, since replicates run, the same sign at
  r = 0, 1, 2 (PREREG 16). **Direction**: real-positive / real-negative on ≥ 2
  of 3 seeds, else "not real". Every Δ is paired within a replicate.
- **10.**
  - (a) Real / not real of gold_I − full gold joint.
  - (b) Both point values hold if within **0.03** of the prediction on ≥ 2 of
    3 seeds.
  - (c) The teacher_I / gold_I ratio holds if within **0.05** (line 0's
    tolerance).
  - The line holds only if all parts hold.
- **11, 12.** teacher_I − rand_I (or perm_I), joint. Lower / higher / same =
  real-negative / real-positive / not real.
- **13.** Δ_replay direction per condition. teacher_I and perm_I must be
  real-positive; rand_I and gold_I as set by `line13_controls`. The outcome is
  holds / fails / partly (with the set of conditions whose clause held), and
  the prediction must name it exactly.
- **14.** A seed counts if teacher_I's Δ_replay cell is real-positive and
  Δ_bal / Δ_replay ≥ X% in every replicate. Holds on ≥ 2 seeds. **If Δ_replay
  is real-positive on fewer than 2 seeds, the line fails: no replay gain to
  recover.**
- **15.**
  - (a) Joint share above 2 × gold share (4.8%) in every replicate for
    teacher_I and perm_I, and at or below it for rand_I, each on ≥ 2 seeds.
  - (b) Joint − replay share > X points in every replicate, for teacher_I and
    perm_I.
- **16.** "Applied", saying whether replicates ran.
- **Framing.** Every map branch whose condition is met is listed, including the
  new Δ_share branch.

`prereg_rq3()` is at `make_tables.py:1140`; T7 to T11, F1 and F2 sit between
732 and 1380.

## PREREG: what's needed before the sweep

Not committed yet. The guard needs, in PREREG.md at HEAD: a
`## RQ3 predictions (written <date>, ...)` section, no `____` / `YYYY-MM-DD`
anywhere, and **this scoring block** inside the section (your values):

```
rq3-scoring:
  line10_gold_I_minus_gold   = <number>
  line10_gold_I_real         = yes | no
  line10_gold_sub_minus_gold = <number>
  line10_ratio               = <number>
  line11                     = lower | higher | same
  line12                     = lower | higher | same
  line13                     = holds | fails | partly: <conditions>
  line13_controls            = not_real | not_real_positive
  line14_pct                 = <number>
  line15_pp                  = <number>
```

The block is my addition. The template's blanks are free text, and the scorer
needs your numbers and choices in a fixed form. It's spelled out in
`continual/prereg_rq3.py`. If you'd rather score by hand, say so and I'll drop
the guard's block requirement.

## Things to settle while writing lines 10 to 16

1. **Line 13 controls.** In the full gold track, replay − joint is already −0.044
   / −0.071 / −0.073 (T3): real and negative, because joint is the usual upper
   bound. Read literally, "not real for rand_I and gold_I" would likely fail on
   gold_I for a reason unrelated to the thesis. `line13_controls =
   not_real_positive` reads it as "no real gain". Your call; pick before the
   sweep.
2. **Line 14's premise.** On the existing teacher track, Δ_replay was −0.014 /
   +0.022 / −0.083 (Gate 1 §1.4). If teacher_I behaves the same, line 14 fails by
   rule.
3. **Tolerances I chose.** 0.03 for point predictions, 0.05 for the ratio.
   Change them now if you want different ones; after the sweep they stay.
4. **Line 15(b)** is scored as written (> X points). Whether that drop is "real"
   (3 points) is reported separately in T10.
5. **Sampler RNG.** Changed from the Gate 1 plan: `joint_balanced` draws from
   the global torch RNG, not its own generator, so joint and joint_balanced start
   from identical weights for a given seed (tested).
6. `harness.py` is no longer "unedited apart from docstrings": its training rows
   now come from `conditions.train_rows`, it has `init_seed` / `draw_seed`
   parameters, and run ids carry the init seed. Bit identity for existing runs is
   tested (point 1 above).

## After the PREREG commit

On your go-ahead: `venv/bin/python -m continual.run_all --plan rq3`. That's 162
runs from one commit, a few minutes on CPU. Then `make_tables` and one commit of
all outputs.

---

## Addendum: Gate 2 review changes (Nov 5)

Spec and PAPER_NOTES amended first (docs `8510bb5`), then code.

- **line13_controls = not_real_positive**, tolerances 0.03 / 0.05, the sampler RNG
  change and the harness edits approved. The harness edits are in
  `docs/PAPER_NOTES.md`, "RQ3 code changes".
- **Line 13, exact outcome.** Four clauses: teacher_I real-positive, perm_I
  real-positive, rand_I not real-positive, gold_I not real-positive. "holds" = all
  four, "fails" = none, otherwise "partly: <conditions whose clause holds,
  alphabetical>". In the expected world (no condition's Δ_replay real-positive)
  the controls hold and the two structured conditions don't, so the scorer writes
  `→ hypothesis partly: gold_I, rand_I`. The PREREG block line that names it is
  `line13 = partly: gold_I, rand_I` (any order). Tested: holds for that
  prediction, fails for `line13 = fails`.
- **Expectations.** `line14_expect`, `line15_expect` and `line17_expect` are
  required in the block. The scorecard gains `expected` and
  `expectation_correct` columns.
- **Line 17** is implemented as specified, reading the premise as "D_rep clears 3
  points in every replicate on fewer than 2 seeds, so the line fails". Both bars
  apply in every replicate. Tested at the 50% boundary, at 60%, with one replicate
  under 3 points, and with one replicate under the recovery bar.
- **Line 10's** point values and ratio are scored on replicate 0, stated in the
  comment block. Tested: r0 matching with r1/r2 off still holds.
- **Label stat** (`row_accounting.md`, "Is the diagonal the row maximum?"): 14 of
  17 gold classes have the correct label as the teacher's most frequent. The
  three that don't: children (diagonal 5, other_family 16), parents (9,
  other_family 20; children also 9) and positive_impression (11,
  girl/boyfriend 20).
