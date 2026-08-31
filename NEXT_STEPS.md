# Next steps

Written Aug 31, 2026, after the pipeline-hardening session. Ordered by when
it matters, not by size. Each item says *why*, because in three weeks the
reason is the part that will have evaporated.

---

## 0. Before the demo

- [ ] **Merge the branch.** 7 commits on `pipeline-hardening-aug30` are not on
      `master`. The demo depends on several of them — notably the relevance
      check that stops the answer step rewriting unrelated facts as the
      asked-about subject.
      ```
      git checkout master && git merge pipeline-hardening-aug30 && git push
      ```
- [ ] **Screenshots.** See `benchmarks/README.md` for the commands. Note the
      QA benchmark now reads **13/15 (87%)**, not the 67% in older notes.
- [ ] **CLAUDE.md is stale.** Still says *"Phase 4 ⬅️ next"* when 4, 5 and
      naive-6 are done; §10 lists `config.py` and the eval-set builder as
      unchecked when both exist; none of this session is recorded. Its own
      header calls a stale CLAUDE.md "the single most likely thing to mislead
      a future session."

---

## 1. Evaluation integrity — the pinned question

**The problem:** the 15 QA questions were written by the same process that
then fixed the failures. Prompt rules and few-shot examples were added after
seeing which questions failed, four rounds of it, 5/15 -> 13/15. That is a
feedback loop, not an evaluation. It is unknown how much of 87% reflects the
system and how much reflects the tuning.

Roughly: the architectural fixes generalise (Cypher repair, 5xx retry,
case-insensitive CONTAINS, directed-not-undirected matching, zero-row
broadening, LIMIT + truncation disclosure, relevance check). The four worked
examples in the Cypher prompt are fitted — the money one literally contains
`gpu`, `lab`, `lakh`, the tokens the test questions need. The largest single
jump (10 -> 13) came mostly from that fitted example.

- [ ] **Write 8-10 held-out questions from the scripts, without assistance.**
      This has to be done independently or it isn't held out.
- [ ] Run them. **~80%+** means the fixes generalise and 87% was roughly
      honest. **~50%** means we fitted, and the held-out number is the real
      one.
- [ ] Freeze the current 15 as a **dev** set; keep the new ones as **test**.
      Same discipline as the frozen Phase 3 split, and for the same reason.
- [ ] **In the paper, report the held-out number.** A reviewer will ask this
      exact question.

---

## 2. Paper

- [ ] **`PAPER_OUTLINE.md`** — thesis, section arc, every claim mapped to the
      result file backing it. Proposed thesis:

      > Schema-constrained decoding buys validity, not correctness — and the
      > gap has a measurable downstream cost that standard metrics cannot see.

      The spine (§1-§7, ending at gold 0.451 vs teacher 0.312) does not
      depend on Phase 6/7. Only the CL section deepens when regimes land.

- [ ] **Joint baselines for seeds 1 and 2.** Only
      `joint_baseline_seed1234.json` exists, so the headline 0.139 gap rests
      on **one seed**. Three seeds with a spread is far more defensible.
      Highest-value CL item, roughly an hour.

- [ ] **Model ablation — paper-critical, not optimisation.** Re-label ~100
      DialogRE dialogues with a stronger model and measure agreement against
      gold.
      - agreement stays ~37% -> the central claim *hardens*: it survives a
        capability jump
      - agreement jumps to ~60% -> the claim becomes "small models struggle",
        a different and weaker paper, and better known before writing than
        after
      - The cache key already includes the model name, so both label sets
        coexist cleanly. No split rebuild — it is a measurement, not a
        migration.

---

## 3. The CL deliverable (§9.2) — still 1 of 3 regimes

- [ ] **`continual/regimes.py` does not exist.** Replay and FKD unbuilt.
- [ ] **Replay first.** It also un-degenerates BWT: with off-diagonals
      pinned at exactly 0, `BWT = -mean(diagonal)` algebraically (verified to
      4 decimals in `benchmarks/report_cl.py`), so BWT currently carries zero
      forgetting information. Replay is what makes the metric mean something
      again.
- [ ] **FKD must be verified by a human against Eqs. 10-13** in
      `docs/PAPER_NOTES.md`. CLAUDE.md §11 requires this personally; that
      formulation was wrong once already.
- [ ] Consider whether the majority-class task is worth fixing: the
      example-count balance constraint forces `per:alternate_names` (408 of
      1,144) in with the three thinnest classes, and that task collapses to
      the constant-majority predictor at every seed. Balancing by *class
      count* fixes it but invalidates the frozen split. Decide deliberately,
      write down the reasoning **before** running, and report both.

---

## 4. Extraction quality — the real ceiling

Two QA questions fail because the fact was never extracted, and several
others were hard to retrieve because context was lost at triple-formation
time. No amount of query tuning reaches these.

- [ ] **Amounts attach to the approver, not the funded thing.**
      `Jayashree --approved--> "2.2 lakhs for phase 1 procurement"` contains
      no token linking it to GPUs or the lab. This is why money questions
      needed a special retrieval rule. Fix at the prompt: keep the funded
      subject inside the triple.
- [ ] **Facts never extracted:** who chairs the Board of Studies (implicit —
      Keshavan runs the meeting but nobody says he chairs it), the enrolment
      count (47), the 4.5 lakh figure from meeting_002.
- [ ] **Collective nouns typed PERSON:** `Students`, `faculty members`,
      `engineers on site`, `university relations head`.
- [ ] **A contradiction sits in the graph:** 2.2 lakhs is recorded as both
      phase one (m2, correct) and phase two (m4, wrong).
- [ ] **`budget_for` is semantically polluted** — also used for scoring
      weights ("15 points for critical analysis"), which is why budget
      queries return noise.
- [ ] **Entity fragmentation:** `four GPUs` / `four more GPUs` / `GPUs` are
      three nodes. `name` is the primary key, so any surface variation
      creates a node. Levenshtein at 0.85 cannot merge referentially
      identical noun phrases.
- [ ] **Broad-class collapse persists:** `discussed` is ~30% of relations
      despite an explicit suppression rule. Third corpus, third taxonomy,
      same failure — this is a *finding*, not a bug to fix.

---

## 5. Deferred engineering, with reasons

Considered and consciously not done. None are blocking.

- [ ] **Template-based Cypher generation.** Constrained slot-filling against
      pre-validated query templates makes syntax errors and hallucinated
      relations *impossible* rather than recoverable. The right architecture,
      and the same principle as constrained extraction one layer up. Keep a
      free-form fallback with the repair loop for unmatched intents.
- [ ] **Entity-type filter in fuzzy matching.** A real correctness bug:
      `resolve()` matches across all types, so a COURSE can merge into a
      PERSON. Also cuts the search space ~5x.
- [ ] **Two-pass graph insertion** (nodes, then relationships). Gives
      `UNWIND` batching — 490 round trips becomes 2 — and, more importantly,
      lets entity-type conflicts be resolved deliberately instead of
      first-writer-wins. Uniqueness is on `name` alone, not `(name, type)`.
- [ ] **`tenacity` for retry policy.** Three hand-rolled policies have
      drifted: extraction waits up to 80s on a rate limit, the query layer 4s,
      and network errors use flat rather than exponential backoff. Unify as a
      *policy decision*, not a library swap — and note it would not replace
      the Cypher repair loop, which mutates its input rather than retrying.
- [ ] **Dynamic few-shot grounding.** Sample real entity names and relation
      frequencies from the graph into the prompt at query time, so the model
      learns the data shape empirically instead of from hand-written rules.
      Strictly better than static examples and adapts to any corpus — but
      measure it on the held-out set, not the dev set.
- [ ] **Windows portability:** `run_full_pipeline.sh` is bash-only. The
      Python path is clean (all `open()` calls carry `encoding="utf-8"`), and
      the demo path needs only `pip install neo4j` since the transcripts are
      committed. ASR needs WSL2 — `nemo_toolkit` is Linux-first.
- [ ] **No `requirements.txt`.** A fresh clone has to read imports to guess.
      Split by tier: demo (`neo4j`), audio (`edge-tts`, `pydub`), CL
      (`torch`, `numpy`, `sentence-transformers`, `datasets`).
- [ ] **No `.env.example`.**

---

## 6. If the model is swapped later

- [ ] **Check strict JSON-schema support FIRST.** The 100% adherence comes
      from `response_format: {"type": "json_schema", "strict": true}` with
      enums — not from the prompt. Many OpenAI-compatible providers accept
      `json_object` but silently ignore `json_schema`. If that happens,
      adherence falls back toward the freeform 29% and the headline result's
      setup evaporates. Test on ~10 windows and confirm `adherence_pct` is
      still 100.0 before anything else.
- [ ] Must be a hosted endpoint — CPU-only hardware, no CUDA or MPS.
- [ ] `default_base_url()` only knows Gemini vs OpenAI; set
      `OPENAI_BASE_URL` explicitly. No code change needed.
- [ ] **Swapping the teacher model cascades:** teacher labels change ->
      teacher features, the frozen Phase 3 split, and every naive run must be
      rebuilt. The gold track is unaffected. Do the sample ablation (§2)
      first and only rebuild if the result justifies it.
- [ ] **Meeting extraction is safe to swap anytime** — nothing downstream is
      frozen.

---

## Reference

- `results/PIPELINE_FIXES_AUG30.md` — full write-up of the defects fixed,
  written for the report's methods and limitations sections.
- `benchmarks/README.md` — how to run everything.
- Current state: 87 + 38 tests, graph benchmark 8/8, QA 13/15, graph at
  508 relations / 100% adherence / 59/59 windows.
