# Pipeline hardening — August 30, 2026

First end-to-end run of **audio → diarization → knowledge graph** on all five
meetings. It surfaced four defects, all of which had been in the code since
April and none of which had ever fired. Every one of them failed *silently*:
the run reported success, and taxonomy adherence printed 100%, while content
was missing.

This document records what broke, why it was invisible until now, and what
changed. It is written for the report's methods and limitations sections.

---

## 1. Why nothing had failed before

The April and August-23 pipeline runs consumed `test_data/meeting_001.txt` —
a **hand-written** mock transcript, not diarized audio:

```
SPEAKER_00 (0:00:01 - 0:00:05): Good morning everyone, let's begin the department meeting.
SPEAKER_01 (0:00:06 - 0:00:14): Thank you Dr. Jayashree. Before we start...
SPEAKER_00 (0:00:15 - 0:00:28): ...Dr. Ramesh, could you present the syllabus proposal?
```

Real diarized output looks nothing like it:

```
Speaker 0: Good morning, everyone.
Speaker 0: Three things today.
```

| | Hand-written (April) | Diarized (Aug 30) |
|---|---|---|
| Utterances | 22 long turns | 89 short fragments |
| Names in dialogue | nearly every turn, with titles | sparse |
| `SPEAKER_n` extracted as entities | **0** | **170 of 385 (43%)** |

The hand-written transcript names people constantly, so the extractor always
had a real antecedent. Whisper splits speech into fragments, so a 15-utterance
window can contain no name at all — and with no antecedent, the model falls
back to the only identifier in the prompt: the speaker label.

**The defects were latent, not new. The first honest test exposed all four at
once.**

---

## 2. The four defects

### 2.1 Silent TTS line drops

`test_scripts/generate_audio.py` caught per-line Edge-TTS failures and
`continue`d, omitting that line from the audio. Detected by comparing speech
rate against script word count:

| Meeting | s/word before | after | verdict |
|---|---|---|---|
| 001 | 0.557 | 0.564 | baseline |
| 002 | 0.543 | 0.550 | baseline |
| **003** | **0.422** | **0.582** | ~22% of script was missing |
| 004 | 0.525 | 0.532 | baseline |
| **005** | **0.345** | **0.525** | ~36% of script was missing |

`meeting_005.mp3` was 316s of audio for a 916-word script; it opened
mid-conversation and ended before the chair's closing remarks.

**Fix:** retry with exponential backoff, treat a zero-byte write as a throttle
rather than a success, pace requests at 0.4s, and **abort the whole file**
rather than emit a partial meeting. The generator now prints s/word so
truncation is visible immediately.

### 2.2 Voice collision → merged speakers

`VOICE_MAP` assigned `en-IN-NeerjaNeural` to JAYASHREE and
`en-IN-NeerjaExpressiveNeural` to MEERA — two variants of the same voice. NeMo
diarization merged them, yielding **3 speakers for 4-speaker scripts in 4 of 5
meetings**.

This is worse than it sounds. A merged label gets one name applied to two
people's utterances, i.e. **fabricated attribution** in the graph. That is
strictly worse than an anonymous placeholder.

**Fix:** every speaker in a meeting now has a distinct base voice and locale.
All eight voices verified against the live Edge-TTS list before use.

**Result: 4/4 speakers detected in all five meetings.**

### 2.3 `SPEAKER_n` extracted as PERSON entities

`ConversationWindow.text` renders `[SPEAKER_0] (0.2s): ...` straight into the
prompt, and the extractor faithfully treated the placeholder as a person. It
was the highest-degree node in the graph.

**Fix:** new `preprocessing/speaker_naming.py`. One LLM call per meeting maps
labels to names spoken in the dialogue, applied **before chunking**, so the
extractor never sees a placeholder. Three safeguards:

1. **Roster constraint.** Attendee lists (`data/rosters.json`) restrict names
   to a closed set — the same move that took taxonomy adherence 29% → 100%,
   applied one layer up. A real deployment has this from the calendar invite.
2. **Elimination.** When labels and roster are both size *n* and exactly one
   of each is unassigned, the remaining assignment is forced. This is
   deduction, not guessing.
3. **Placeholder drop.** Any speaker still unresolved has its triples removed
   rather than guessed at.

The guardrail demonstrably worked. On both runs the model proposed a wrong
answer and was rejected:

```
meeting_003: [WARN] 'null' is not on the roster; leaving SPEAKER_1 unresolved
             Assigned SPEAKER_1 -> Kumar by elimination
meeting_005: [WARN] 'Ramesh' is not on the roster; leaving SPEAKER_1 unresolved
             Assigned SPEAKER_1 -> Keshavan by elimination
```

`Ramesh` does not attend meeting_005 at all. Without the roster the graph
would assert that a person from a different meeting series chaired it.

### 2.4 Adherence cannot see extraction failures

`adherence_report()` divides parsed triples by parsed triples. A window that
fails contributes to **neither numerator nor denominator**, so the metric
prints 100% while content is missing. Two windows of `meeting_003` were lost
to HTTP 503 and the run reported complete success.

**Fix:** `coverage_report()` reports windows extracted / failed / empty, and
`pipeline.py` prints it and writes `output/coverage.json`.

> **For the report:** taxonomy adherence measures *label validity*. It says
> nothing about *extraction completeness*. These are different claims and need
> different metrics.

---

## 3. Query interface — six further fixes

Found by testing, not by inspection.

| # | Defect | Symptom |
|---|---|---|
| 1 | No Cypher retry | One malformed generation killed the question outright |
| 2 | No API retry in `query.py` | A transient 503 surfaced as a raw traceback |
| 3 | Case-sensitive `CONTAINS` | `'Data Mining'` vs stored `data mining` → silently zero rows |
| 4 | `ORDER BY r.timestamp` | That is *pipeline write time*, not meeting time |
| 5 | Directed-only matching | "What happened to X" found nothing when X is the object |
| 6 | **Vocabulary drift** | The query prompt listed 18 relation types; the extractor emits 20 |

**#6 is the most instructive.** `replaced_by` — the relation carrying the
entire "Data Mining course was replaced" story, 8 occurrences — was absent from
the query layer's schema. A fact the graph held was unreachable from the query
side because two hardcoded lists had drifted. The schema is now derived from
`RELATION_TYPES`, so they cannot diverge again.

**#4 is the most dangerous.** It produced correct-looking answers purely
because the five meetings happened to be inserted in order. Re-run one meeting
alone and the chronology silently inverts — a wrong answer with no error. The
graph stores three time-ish properties and only two mean what a temporal query
needs:

- `source_meeting` — real chronology across meetings
- `utterance_time` — chronology within a meeting
- `timestamp` — **when the pipeline wrote the row**

Effect of fixes 3–6, same graph and same question:

> **Before:** "Ramesh proposed the Applied Deep Learning course in meeting_001.
> The outcome of this proposal is not yet known."
>
> **After:** "Ramesh proposed the Applied Deep Learning course in meeting_001.
> In meeting_003, it was decided that the data mining elective would be
> replaced by Applied Deep Learning. In meeting_004, the syllabus was
> discussed... A revised assessment was approved by the Board in meeting_004."

---

## 4. Final verification

| Check | Result |
|---|---|
| `SPEAKER_n` nodes in graph | **0** |
| Speakers resolved | **4/4 in all five meetings** |
| Failed extraction windows | **0** (was 2) |
| Triples dropped to unresolved speakers | **0** (was 44) |
| Meetings present | 5 — 64 / 91 / 121 / 122 / 92 relations |
| Taxonomy adherence | **100%** (536/536) |
| Parser tests | **38/38 pass** |

**Graph: 398 entities, 490 relations.** Up from 356 / 439 before the fixes.

**31 entities appear in more than one meeting** (2 span all five, 3 span four,
4 span three, 22 span two). This cross-meeting linkage is what makes the graph
a connected structure rather than five islands — the direct contrast with the
April screenshot's ~50 disconnected components.

ASR name variants are merged by Soundex against the roster, scoped to PERSON
entities only: `Jaishree` → `Jayashree`, `Mira` → `Meera`. Levenshtein at 0.85
cannot catch these (0.78 and 0.60 respectively) because ASR errors are
phonetic, not orthographic.

---

## 5. Known limitations

Deliberately not fixed; they belong in the report.

- **`Dr. Tetna`** — "Chetana" misheard. She is referenced in dialogue but
  attends no meeting, so no roster entry exists to anchor the correction.
- **Collective nouns typed PERSON** — `Students`, `faculty members`,
  `engineers on site`, `university relations head`.
- **`Meera and Kumar` as a single entity** — the extractor did not split a
  conjunction.
- **`budget_for` is noisy** — e.g. `Mini project → 30%` is not a budget.
- **Entity fragmentation.** `name` is the node primary key, so `four GPUs`,
  `four more GPUs` and `GPUs` are three nodes. Levenshtein at 0.85 cannot
  merge referentially-identical noun phrases.
- **Uniqueness is on `name` alone, not `(name, type)`.** Since `MERGE` matches
  on name and sets `type` only `ON CREATE`, the first mention wins the type
  permanently. A PERSON and a TOPIC sharing a name silently become one node.
- **Broad-class collapse persists.** `discussed` is 142 of 490 relations
  despite an explicit anti-`discussed` prompt rule. This is the same failure
  documented for AMI and DialogRE, now in a third corpus under a third
  taxonomy — further evidence it is not prompt-correctable.
- **One misattribution observed**: the graph holds both "Keshavan was assigned
  to provide a justification" and the correct "Kumar was assigned...". An
  extraction error in one window.

---

## 6. Scalability

Asked in the context of hours-long meetings.

- **Linear** in meeting length: ASR, chunking, extraction, graph insertion.
- **Quadratic** in distinct entities: `EntityResolver.resolve()` fuzzy-matches
  every mention against every known entity. Invisible at 404 entities;
  a 2-hour meeting might produce 3,000–5,000. It also matches **across types**,
  so a COURSE can merge into a PERSON — filtering candidates by `entity_type`
  fixes the bug and cuts the search space ~5×.
- **Wall-clock constraint is ASR**: ~3× realtime on CPU, so a 2-hour meeting is
  ~6 hours. Trivially parallelisable across meetings; a GPU removes it.
- **Insertion is one round trip per triple** (490 for this run). `UNWIND`
  batching would make it 2.
- **Not a constraint:** Neo4j at this scale, API rate limits, memory.

One-line summary: *the pipeline is linear in meeting length at every stage
except entity resolution, which is quadratic in distinct entities; ASR is the
wall-clock constraint and is trivially parallelisable.*

---

## 7. Deferred, with reasons

Considered and consciously not done the night before the review.

- **Template-based Cypher generation.** Constrained slot-filling against
  pre-validated query templates would make syntax errors and hallucinated
  relations *impossible*, rather than recoverable. This is the right design and
  the same principle as constrained extraction. Deferred because it rewrites
  the component being demoed; the repair loop covers the failure for now.
- **`tenacity` for retry policy.** Three hand-rolled policies have drifted —
  extraction waits up to 80s on a rate limit, the query layer 4s, and network
  errors use flat rather than exponential backoff. Worth unifying, but it adds
  a dependency to a deliberately stdlib-only HTTP path. Note it would *not*
  replace the Cypher repair loop, which mutates its input each attempt rather
  than retrying the same call.
- **Two-pass graph insertion** (nodes, then relationships). Would give `UNWIND`
  batching and, more importantly, allow deliberate resolution of entity-type
  conflicts instead of first-writer-wins.
- **Entity-type filter in fuzzy matching.** A real correctness bug, but it
  changes graph-building behaviour, and the current graph is verified and
  frozen for the demo.

---

## 8. Reproducing

```bash
# 1. Generate audio (base conda env — only it has edge_tts + pydub)
cd capstone_pipeline/test_scripts
/opt/homebrew/Caskroom/miniconda/base/bin/python generate_audio.py \
    script_01_department_meeting.txt ../../whisper-diarization/meeting_001.mp3

# 2. Transcribe (whisper-diarization env; --device cpu is mandatory,
#    mtypes has no "mps" key)
conda activate whisper-diarization
python diarize.py -a meeting_001.mp3 --device cpu \
    --whisper-model medium.en --batch-size 8 --no-stem

# 3. Build the graph (pipeline venv)
source venv/bin/activate
python pipeline.py --extractor llm --llm-model gemini-2.5-flash-lite \
    --neo4j-password capstone123 --clear-graph --request-delay 0.2 \
    --input test_data/processed/meeting_001.srt ...

# 4. Query
python query.py --interactive --verbose --neo4j-password capstone123
```

`--request-delay` defaults to 5.0s for free-tier keys; use 0.1–0.5 on a paid
tier. It is the single largest component of pipeline wall-clock time.

For Neo4j Browser colouring, nodes need a second label matching their type
(Browser styles by label, not by property). This is cosmetic and lives only in
the database — re-apply after any `--clear-graph` run:

```cypher
MATCH (e:Entity {type:'PERSON'}) SET e:PERSON;   // repeat per type
```
