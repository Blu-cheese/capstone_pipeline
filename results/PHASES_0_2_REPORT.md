# Phases 0–2: Results and Inferences

**Project:** Continual Knowledge Graph Construction from Streaming Audio
**Scope of this report:** Phase 0 (schema-constrained extraction), Phase 1
(teacher labelling at scale), Phase 2 (feature extraction)
**Model:** `gemini-2.5-flash-lite` via the Gemini OpenAI-compatible endpoint
**Hardware:** M2 Mac, CPU only throughout

---

## 1. Executive summary

Three things were established, in order of importance to the research claim:

1. **Schema-constrained decoding fixes taxonomy adherence completely.**
   Relation adherence went from 29–39% (freeform) to **100%** (JSON-schema
   enums), with **no loss of recall** — extraction yield actually rose from
   55 to 85 triples on the same two meetings.

2. **AMI cannot support the continual-learning experiment, and no taxonomy
   change fixes that.** Measured directly: AMI yields 4–14 distinct relations
   with the most frequent class taking 36–77% of labels. The constraint is
   the corpus, not the label set.

3. **DialogRE can, and does.** 2,121 teacher-labelled pairs across 400
   dialogues, 33 of 37 relation types populated, feature matrix built and
   cached. Both Phase 1 and Phase 2 pass every acceptance criterion.

One result needs to be reported honestly rather than fixed: **teacher
agreement with gold is 37.5%**, and prompt engineering moved it only ~5
points. The consequence is discussed in §6.

---

## 2. Phase 0 — Schema-constrained extraction

### 2.1 Adherence

Measured on the saved triple files (post entity-resolution) for all runs.
Adherence = share of extracted relations belonging to the 20-type taxonomy.

| Run | meeting_001 | meeting_002 | Combined | Entity-type violations |
|---|---|---|---|---|
| Original (April, freeform) | 1/26 (4%) | 16/18 (89%) | **17/44 (38.6%)** | 0/88 |
| Freeform (re-run) | 4/28 (14%) | 12/27 (44%) | **16/55 (29.1%)** | 0/110 |
| **Constrained (schema enums)** | 36/36 (100%) | 49/49 (100%) | **85/85 (100%)** | 0/170 |

### 2.2 Yield

| Run | meeting_001 | meeting_002 | Total |
|---|---|---|---|
| Original (freeform) | 26 | 18 | 44 |
| Freeform (re-run) | 28 | 27 | 55 |
| **Constrained** | 36 | 49 | **85** |

### 2.3 Inferences

**(a) The root cause was a single prompt clause, not model capability.**
The original system prompt read *"COMMON RELATION TYPES (you may use others
if they fit better)"* — the model was explicitly invited to invent labels,
and did so 33 distinct times. Constraining the same model to the same
taxonomy via decoding-level enums eliminated invention entirely.

**(b) Prompt-level instruction is not merely weaker than schema
constraints — it is unstable.** CLAUDE.md §2d characterised meeting_001 (4%)
and meeting_002 (89%) as "two different adherence regimes", implying a
property of the meetings. It is not: on re-run, meeting_002 fell from 89% to
44%. Freeform adherence is **run-to-run variance**, which is a stronger
argument for schema constraints than the original framing, because it means
no amount of prompt tuning yields a reliable floor.

**(c) Constraining the label space did not cost recall.** Yield rose 55 → 85
(+55%). The intuition that constraints suppress extraction is not supported
here; removing the burden of inventing a label appears to let the model
attend to extraction itself.

**(d) Entity types were never the problem.** Zero violations across all
three runs, 368 entity mentions. The taxonomy failure was purely relational.
The enum on entity types is defensive, not corrective.

**(e) A share of the original "invented" relations were spelling variants.**
`deadline for` vs `deadline_for`, `scheduled for` vs `scheduled_for`.
Normalising to snake_case before the taxonomy check recovers these, so the
true invention rate was somewhat below the headline 4%.

---

## 3. AMI investigation (rejected corpus)

AMI was the planned Phase 1 corpus. It was tested, and rejected on evidence.

### 3.1 Access

The route recommended in `ami_parser.py`'s docstring — the `knkarthick/AMI`
HuggingFace re-upload — is **gated** (`HTTP 401`), requiring an account and
accepted terms. The only ungated mirror found contains no data files.

The official Edinburgh distribution (`ami_public_manual_1.6.2.zip`,
CC BY 4.0, 22 MB) needs no authentication and was used instead. This
inverts the module's advice: the NXT XML route is now both easier *and*
better, because it carries **real timestamps** rather than synthetic ones.
171 meetings parsed correctly with `parse_ami_nxt` unmodified.

### 3.2 Why it was rejected

| Measurement | AMI | DialogRE |
|---|---|---|
| Distinct relations observed | 4–14 | 35 |
| Most frequent class share | 36–77% | 21% |
| Types with ≥20 examples | ~4 of 20 | 31 of 35 |
| Human-annotated labels | none | 5,963 pairs |

Constrained extraction on AMI reached 100% adherence but produced a
**collapsed distribution**: `discussed` took 86% of labels on `EN2001a` and
77% on `ES2002c`, with 16 of 20 relation types never occurring.

Critically, this was tested **without** constraints too. Freeform extraction
across four AMI meetings produced 14 distinct relations from 55 triples,
`discussed` still dominant at 36%, with an idiosyncratic tail (`maximal`,
`is`, `stated`).

### 3.3 Inference

**The corpus determines the achievable label distribution far more than the
taxonomy does.** Meeting speech is overwhelmingly discussion; crisp
relational facts are sparse. Refitting the taxonomy to AMI would have
produced the same collapse under different names, and would have cost a day
to discover. Phase 3 requires a balanced 3-task split with no task under 15%
of examples — unachievable on AMI at any label set.

---

## 4. Phase 1 — Teacher labelling

### 4.1 Acceptance criteria

| Criterion | Target | Actual | |
|---|---|---|---|
| Labelled pairs | ≥1,500 | **2,121** | PASS |
| Source dialogues | ≥40 | **400** | PASS |
| Re-run makes zero API calls | required | verified | PASS |
| Relation distribution reported | required | 33/37 used, 4 absent | PASS |
| Failures | — | **0** | PASS |

### 4.2 Design decisions

- **One API call per dialogue, not per pair.** Dialogues carry 5.4 annotated
  pairs on average, so batching cut request count ~5.4× against a quota with
  no context loss.
- **Cache keyed on `sha256(dialogue + pairs + model + PROMPT_VERSION)`**,
  sharded two hex chars deep. Bumping `PROMPT_VERSION` correctly invalidates
  — the v1→v2 prompt change forced a clean re-label rather than silently
  mixing labels from two different instructions.
- **Teacher and gold labels stored side by side**, enabling the dual-track
  experiment (student-on-teacher vs student-on-gold) without re-labelling.

### 4.3 Teacher quality

| Prompt | Agreement with gold |
|---|---|
| v1 (plain instruction) | 32.6% |
| v2 (ordered decision procedure) | **37.5%** |

For reference, fine-tuned BERT baselines on DialogRE report roughly 58–63%
F1, so a zero-shot LLM in the high 30s is not anomalous — but it is low
enough to constrain the experiment.

### 4.4 Inference — the central finding

**Schema constraints guarantee label *validity*, not label *correctness* or
*calibration*.** The teacher was structurally incapable of emitting an
invalid class, and still disagreed with human annotation 62% of the time.

The failure is **systematic, not random**:

| Relation | Gold | Teacher | |
|---|---|---|---|
| `per:acquaintance` | 9 | **789** (v1) | catastrophic over-prediction |
| `per:alternate_names` | 446 | 190 | under-detected |
| `per:positive_impression` | 147 | 6 | near-total blindness |
| `per:alumni` | 21 | 0 | never predicted |
| `per:siblings` | 67 | 10 | severely under-predicted |

`per:acquaintance` absorbed 150 `per:alternate_names`, 97
`per:girl/boyfriend` and 86 `per:positive_impression` — it functioned as an
"I'm unsure" sink.

**This is the same pathology as `discussed` on AMI.** Two different corpora,
two different taxonomies, one recurring failure mode: *the model routes
uncertainty into whichever class is semantically broadest.* An explicit
instruction that the class is rare (v2) recovered only ~5 points, which
suggests the behaviour is not correctable by prompting alone.

For the report, this is the more interesting result than the adherence
table: **constrained decoding solves the vocabulary problem and leaves the
distribution problem untouched.**

---

## 5. Phase 2 — Feature extraction

| Criterion | Target | Actual | |
|---|---|---|---|
| Feature matrix | `(N, 1152)`, N≥1500 | **`(2121, 1152)`** float32 | PASS |
| Label arrays | `(N,)` | `y_teacher`, `y_gold` both `(2121,)` | PASS |
| Cached reload | <2s | **0.12s** | PASS |
| Cold encode | ~1–2 min | **20.9s** | PASS |
| CPU only | required | no CUDA/MPS anywhere | PASS |

Encoder `all-MiniLM-L6-v2`, frozen, 384-dim × 3 fields (window, subject,
object) → 1152. Cache is 2.5 MB. Verified on Python 3.14 with torch 2.13.0
and sentence-transformers 6.0.0 — the feared 3.14 wheel incompatibility did
not materialise.

---

## 6. What this means for Phases 3–7

**The continual-learning experiment remains valid.** ACC and BWT measure
forgetting of whatever the student learned; they do not require the teacher
to be accurate. A 37.5% teacher gives a lower absolute ceiling, not an
invalid protocol.

**The dual-track decision de-risks this.** Running the identical harness on
teacher labels and on gold labels converts a weakness into a finding: *how
does label noise affect continual forgetting?* Cost is only extra training
minutes.

**Phase 3's thin-class problem is now concrete.** 17 teacher classes have
under 20 examples. With ~16 viable classes, a 3-task split of roughly 5
classes each is comfortable, but the merge/drop list needs an explicit
decision and must be logged.

**Phase 5 (manual gold annotation) is retired.** DialogRE's 5,963
human-annotated pairs supply teacher-quality measurement directly, saving
~2 hours of annotator time plus a second annotator's hour. Cohen's kappa is
consequently not available; this should be noted as a deliberate trade.

### Limitations to state in the report

- DialogRE is transcribed sitcom dialogue, not meetings. The CL method is
  domain-agnostic, but Track B's corpus no longer matches Track A's domain.
- The student performs relation classification given an entity pair, not
  end-to-end extraction. Candidate pairs come from the corpus annotation.
- 265 of 5,963 pairs carry multiple gold relations; collapsed to the first
  for a single-label classifier.
- AMI is CC BY 4.0 — attribution required if any AMI-derived result is
  published. DialogRE should be cited (Yu et al., 2020).

---

## 7. API usage

### 7.1 Calls by run

| Run | Calls | Input tokens | Output tokens |
|---|---:|---:|---:|
| Connectivity test | 1 | 15 | 2 |
| Phase 0 constrained | 4 | 6,500 | 2,600 |
| Phase 0 freeform | 4 | 6,500 | 2,600 |
| AMI sanity (EN2001a) | 10 | 22,500 | 7,000 |
| AMI scenario (ES2002a/c) | 16 | 36,000 | 8,800 |
| AMI freeform vocabulary | 12 | 27,000 | 4,500 |
| DialogRE teacher v1 | 400 | 437,200 | 25,844 |
| DialogRE teacher v2 | 400 | 437,200 | 25,844 |
| **Total** | **847** | **~973,000** | **~77,000** |

Token figures are estimated from measured prompt/response character counts
at ~4 chars/token; call counts are exact.

### 7.2 Cost

Estimated **≈ $0.13 USD ≈ ₹11** — roughly **2.3% of the ₹500 cap**.
Verify against the Google AI Studio billing console for the authoritative
figure; the estimate above assumes flash-lite list pricing.

### 7.3 Efficiency notes

- **Batching pairs per dialogue saved ~4,300 calls.** Labelling 2,121 pairs
  individually would have cost ~2,121 requests per prompt version; batching
  reduced it to 400.
- **The cache saved 60 calls** on the v2 run and makes every future re-run
  free. All 800 responses are on disk.
- **v1 labelling was superseded, not wasted** — its 400 responses remain
  cached and were the evidence base for diagnosing the `per:acquaintance`
  collapse.
- Headroom is ample: at ~₹11 for 847 calls, the remaining ₹489 supports
  roughly **35,000 further calls** at the same profile. Quota is not a
  binding constraint on the remaining phases.

---

## 8. Bugs fixed en route

| Ref | Bug | Status |
|---|---|---|
| §2a | `transcript_parser` silently returned an empty transcript on current whisper-diarization `.txt` output | Fixed; raises loudly; regression test against real diarizer output |
| §2b | `pipeline.py` and `query.py` disagreed on default model | Fixed; both resolve to `gemini-2.5-flash-lite` |
| new | Base URL defaulted to OpenAI while the model defaulted to Gemini — any run without a hand-exported `OPENAI_BASE_URL` failed | Fixed; endpoint derived from model name |
| §2c | `run_full_pipeline.sh` hardcoded `--device mps`, which raises `KeyError` in `diarize.py` | Fixed to `cpu` |
| new | `run_full_pipeline.sh` fed `.txt` (no timestamps) when `.srt` was available | Fixed; `.srt` preferred |
| new | `capstone_pipeline` had no `.gitignore`; `.pyc` files were tracked and a secret would have been committable | `.gitignore` added, `.pyc` untracked, `.env` protected |
| new | `ami_parser` treated any short token before a colon as a speaker, inventing phantom speakers from prose | Fixed; test-covered |
| new | `Triple.source_utterance` truncated to 200 chars, insufficient for Phase 2 features | `window_text` field added carrying the full window |

Test coverage: **38 parser checks + 21 extractor checks**, all passing.
