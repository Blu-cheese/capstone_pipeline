"""
Regression tests for the Aug 30 pipeline hardening. Run from capstone_pipeline:

    venv/bin/python tests/test_pipeline_fixes.py

No pytest, no network, no Neo4j - plain asserts and a tally, so this runs
anywhere including a fresh Windows clone.

Each test below corresponds to a defect that shipped silently: the run
reported success and taxonomy adherence printed 100% while content was
missing. The API-dependent paths are exercised by substituting the transport
function, so the real validation logic runs without a network call.

See results/PIPELINE_FIXES_AUG30.md for the full write-up.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models import Utterance, MeetingTranscript, Triple
from preprocessing import speaker_naming
from preprocessing.speaker_naming import (
    PLACEHOLDER, infer_speaker_names, apply_speaker_names,
    drop_placeholder_triples, load_rosters,
)
from preprocessing.entity_resolution import EntityResolver, soundex
from extractors import llm_extractor
from extractors.llm_extractor import LLMExtractor, RELATION_TYPES, ENTITY_TYPES
import query as query_mod

_passed, _failed = 0, 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}" + (f"  ({extra})" if extra else ""))
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f"  ({extra})" if extra else ""))


def fake_transcript(speaker_ids, meeting_id="m_test"):
    """Minimal transcript: one utterance per label, in the order given."""
    utts = [Utterance(speaker_id=s, text=f"line {i}", start_time=float(i),
                      end_time=float(i) + 1)
            for i, s in enumerate(speaker_ids)]
    return MeetingTranscript(meeting_id=meeting_id, utterances=utts,
                             duration=float(len(utts)))


def triple(subj, obj, rel="discussed"):
    return Triple(subject=subj, subject_type="PERSON", relation=rel,
                  object=obj, object_type="TOPIC", confidence=0.8,
                  source_meeting="m_test", timestamp=0.0)


# --------------------------------------------------------------------------
# Defect 3: SPEAKER_n extracted as PERSON entities (170 of 385 triples, 43%)
# --------------------------------------------------------------------------

def test_placeholder_detection():
    for s in ["SPEAKER_0", "SPEAKER_12", "speaker_3", "SPEAKER 0"]:
        check(f"placeholder: '{s}' recognised", bool(PLACEHOLDER.match(s)))
    for s in ["Jayashree", "Speaker Committee", "SPEAKER_0 and Ramesh", ""]:
        check(f"placeholder: '{s}' NOT a placeholder", not PLACEHOLDER.match(s))


def test_placeholder_triples_dropped():
    ts = [triple("SPEAKER_0", "budget"),
          triple("Jayashree", "budget", "approved"),
          triple("Ramesh", "SPEAKER_2")]
    kept = drop_placeholder_triples(ts)
    check("drop: only the clean triple survives", len(kept) == 1,
          f"{len(kept)} kept")
    check("drop: keeps the right one", kept and kept[0].subject == "Jayashree")


def test_roster_rejects_off_roster_name():
    """
    The live failure: for meeting_005 the model proposed 'Ramesh', who does
    not attend that meeting. Without the roster the graph would assert that a
    person from another meeting series was present.
    """
    speaker_naming._call_api = lambda *a, **k: json.dumps({
        "SPEAKER_0": "Priya", "SPEAKER_1": "Ramesh",
        "SPEAKER_2": "Jayashree", "SPEAKER_3": "Kumar",
    })
    t = fake_transcript(["SPEAKER_0", "SPEAKER_1", "SPEAKER_2", "SPEAKER_3"])
    m = infer_speaker_names(t, roster=["Jayashree", "Keshavan", "Kumar", "Priya"])
    check("roster: off-roster 'Ramesh' never assigned",
          "Ramesh" not in m.values(), f"got {m}")
    check("roster: on-roster names kept", m.get("SPEAKER_0") == "Priya")


def test_elimination_completes_forced_assignment():
    """
    Same case, one step later: three labels are assigned, one roster member
    is unclaimed, so the fourth assignment is forced. Deduction, not guessing.
    """
    speaker_naming._call_api = lambda *a, **k: json.dumps({
        "SPEAKER_0": "Priya", "SPEAKER_1": "Ramesh",
        "SPEAKER_2": "Jayashree", "SPEAKER_3": "Kumar",
    })
    t = fake_transcript(["SPEAKER_0", "SPEAKER_1", "SPEAKER_2", "SPEAKER_3"])
    m = infer_speaker_names(t, roster=["Jayashree", "Keshavan", "Kumar", "Priya"])
    check("elimination: SPEAKER_1 recovered as Keshavan",
          m.get("SPEAKER_1") == "Keshavan", f"got {m.get('SPEAKER_1')}")
    check("elimination: all four resolved", len(m) == 4, f"{len(m)} resolved")


def test_elimination_refuses_when_ambiguous():
    """Two labels unassigned is not forced - it must NOT guess."""
    speaker_naming._call_api = lambda *a, **k: json.dumps({
        "SPEAKER_0": "Priya", "SPEAKER_1": None,
        "SPEAKER_2": None, "SPEAKER_3": "Kumar",
    })
    t = fake_transcript(["SPEAKER_0", "SPEAKER_1", "SPEAKER_2", "SPEAKER_3"])
    m = infer_speaker_names(t, roster=["Jayashree", "Keshavan", "Kumar", "Priya"])
    check("elimination: refuses with 2 unknowns", len(m) == 2, f"{len(m)} resolved")


def test_no_duplicate_person_assignment():
    """One person must not be assigned to two speaker labels."""
    speaker_naming._call_api = lambda *a, **k: json.dumps({
        "SPEAKER_0": "Jayashree", "SPEAKER_1": "Jayashree",
    })
    t = fake_transcript(["SPEAKER_0", "SPEAKER_0", "SPEAKER_0", "SPEAKER_1"])
    m = infer_speaker_names(t, roster=["Jayashree", "Ramesh"])
    check("dedupe: Jayashree assigned at most once",
          list(m.values()).count("Jayashree") <= 1, f"got {m}")
    check("dedupe: kept the label with more utterances",
          m.get("SPEAKER_0") == "Jayashree", f"got {m}")


def test_api_failure_falls_back_not_crashes():
    def boom(*a, **k):
        raise RuntimeError("simulated 503")
    speaker_naming._call_api = boom
    t = fake_transcript(["SPEAKER_0", "SPEAKER_1"])
    m = infer_speaker_names(t, roster=["Jayashree", "Ramesh"])
    check("resilience: API failure yields {} not an exception", m == {})


def test_apply_speaker_names_rewrites_utterances():
    t = fake_transcript(["SPEAKER_0", "SPEAKER_1", "SPEAKER_0"])
    n = apply_speaker_names(t, {"SPEAKER_0": "Jayashree"})
    ids = [u.speaker_id for u in t.utterances]
    check("apply: renames every matching utterance", n == 2, f"{n} renamed")
    check("apply: unmapped label untouched", ids == ["Jayashree", "SPEAKER_1", "Jayashree"])


def test_rosters_file_loads():
    r = load_rosters("data/rosters.json")
    check("rosters: file loads", len(r) == 5, f"{len(r)} meetings")
    check("rosters: meeting_005 cast is correct",
          set(r.get("meeting_005", [])) == {"Jayashree", "Keshavan", "Kumar", "Priya"})
    check("rosters: missing file is not fatal", load_rosters("nope.json") == {})


# --------------------------------------------------------------------------
# ASR name variants: phonetic, not orthographic
# --------------------------------------------------------------------------

def test_soundex_matches_asr_variants():
    pairs = [("Jayashree", "Jaishree"), ("Meera", "Mira"), ("Priya", "Preya")]
    for a, b in pairs:
        check(f"soundex: {a} == {b}", soundex(a) == soundex(b),
              f"{soundex(a)} vs {soundex(b)}")
    check("soundex: Kumar != Keshavan", soundex("Kumar") != soundex("Keshavan"))


def test_resolver_collapses_variants_onto_roster():
    roster = ["Jayashree", "Meera", "Pavan", "Ramesh", "Keshavan", "Kumar", "Priya"]
    r = EntityResolver(roster_names=roster)
    check("resolver: Jaishree -> Jayashree", r.resolve("Jaishree", "PERSON") == "Jayashree")
    check("resolver: Mira -> Meera", r.resolve("Mira", "PERSON") == "Meera")
    check("resolver: roster name unchanged", r.resolve("Ramesh", "PERSON") == "Ramesh")


def test_resolver_scope_is_narrow():
    """The phonetic rule must not fire outside PERSON, or on phrases."""
    roster = ["Jayashree", "Meera", "Pavan", "Ramesh"]
    check("scope: non-PERSON untouched",
          EntityResolver(roster_names=roster).resolve("Mira", "TOPIC") == "Mira")
    check("scope: multi-token untouched",
          EntityResolver(roster_names=roster).resolve("Ramesh Kumar Committee", "PERSON")
          == "Ramesh Kumar Committee")
    check("scope: non-roster person untouched",
          EntityResolver(roster_names=roster).resolve("Vendor", "PERSON") == "Vendor")


# --------------------------------------------------------------------------
# Defect 4: adherence cannot see extraction failures
# --------------------------------------------------------------------------

class _Win:
    """Minimal stand-in for ConversationWindow."""
    def __init__(self, wid):
        self.window_id = wid
        self.meeting_id = "m_test"
        self.utterances = [Utterance("Jayashree", "we approved the budget", 0.0, 1.0)]
        self.text = "[Jayashree] (0.0s): we approved the budget"
    @property
    def speaker_ids(self):
        return {"Jayashree"}


def _extractor(responses):
    """LLMExtractor whose transport replays a canned list of responses."""
    e = LLMExtractor(model="gemini-2.5-flash-lite", api_key="x", request_delay=0)
    seq = list(responses)
    def transport(*a, **k):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    e._call_api = transport
    return e


def test_coverage_counts_failed_windows():
    """
    The live failure: 2 of 14 windows lost to HTTP 503. Adherence still
    printed 100% because a failed window contributes to neither numerator
    nor denominator.
    """
    good = json.dumps({"triples": [{"subject": "Jayashree", "subject_type": "PERSON",
                                    "relation": "approved", "object": "budget",
                                    "object_type": "TOPIC", "confidence": 0.9}]})
    e = _extractor([good, RuntimeError("HTTP 503"), good])
    e.extract_meeting([_Win(0), _Win(1), _Win(2)])
    c = e.coverage_report()
    check("coverage: counts all windows", c["windows_total"] == 3, str(c))
    check("coverage: counts the failure", c["windows_failed"] == 1, str(c))
    check("coverage: percentage reflects loss", c["coverage_pct"] == 66.7, str(c))
    check("coverage: adherence still reads 100% (why it was invisible)",
          e.adherence_report()["adherence_pct"] == 100.0)


def test_coverage_distinguishes_empty_from_failed():
    """A window of pure small talk yields nothing. That is not a failure."""
    empty = json.dumps({"triples": []})
    e = _extractor([empty, RuntimeError("HTTP 503")])
    e.extract_meeting([_Win(0), _Win(1)])
    c = e.coverage_report()
    check("coverage: empty != failed",
          c["windows_empty"] == 1 and c["windows_failed"] == 1, str(c))


# --------------------------------------------------------------------------
# Query layer: six defects found by testing
# --------------------------------------------------------------------------

def test_query_schema_knows_every_relation():
    """
    Guards the vocabulary-drift bug: the query prompt listed 18 relation
    types while the extractor emitted 20. `replaced_by` - which carries the
    whole "Data Mining was replaced" story - was unreachable from the query
    side. Two hardcoded lists had diverged.
    """
    missing = [r for r in RELATION_TYPES if r not in query_mod.GRAPH_SCHEMA]
    check("schema: every relation type is documented", not missing, f"missing {missing}")
    missing_e = [e for e in ENTITY_TYPES if e not in query_mod.GRAPH_SCHEMA]
    check("schema: every entity type is documented", not missing_e, f"missing {missing_e}")
    check("schema: replaced_by specifically present",
          "replaced_by" in query_mod.GRAPH_SCHEMA)


def test_query_prompt_forbids_the_known_failures():
    p = query_mod.CYPHER_SYSTEM_PROMPT
    check("prompt: forbids SQL constructs", "NOT SQL" in p and "STRING_AGG" in p)
    check("prompt: requires case-insensitive matching", "toLower" in p)
    check("prompt: forbids ordering by write time",
          "NEVER order by r.timestamp" in p or "Never order by this" in query_mod.GRAPH_SCHEMA)
    check("prompt: row cap is the raised one",
          "LIMIT 200" in p and "LIMIT 25" not in p)


def test_truncation_is_disclosed_to_the_answer_step():
    """
    A capped result is a prefix. Since rows are ordered oldest-first, the
    facts lost are the most recent - so the answer must not imply completeness.
    """
    captured = {}
    def fake_llm(system, user, **kw):
        captured["user"] = user
        return "ok"
    query_mod.call_llm = fake_llm

    rows = [{"a": i} for i in range(query_mod.MAX_ROWS)]
    query_mod.generate_answer("q", rows, truncated=True)
    check("truncation: answer step is told the result is partial",
          "partial" in captured["user"].lower() or "more" in captured["user"].lower())
    query_mod.generate_answer("q", [{"a": 1}], truncated=False)
    check("truncation: not claimed when complete",
          "partial picture" not in captured["user"].lower())


def test_row_cap_exceeds_real_query_sizes():
    """
    The old cap of 25 truncated real demo questions: 'infosys' matches 40
    rows and 'capstone' 32, of 490 total.
    """
    check("cap: 200 clears the widest observed question (40 rows)",
          query_mod.MAX_ROWS >= 40 * 2, f"MAX_ROWS={query_mod.MAX_ROWS}")


# --------------------------------------------------------------------------
# Cross-cutting: file encoding, for a Windows clone
# --------------------------------------------------------------------------

def test_no_bare_open_calls():
    """
    On Windows, open() defaults to the ANSI codepage. Rupee amounts and
    Indian names in the extracted triples would raise UnicodeEncodeError or
    corrupt silently. macOS defaults to UTF-8, which is why this never showed.
    """
    import re
    repo = Path(__file__).resolve().parent.parent
    bare = []
    for p in repo.rglob("*.py"):
        if "venv" in p.parts or p.name == Path(__file__).name:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]      # prose about open() is not a call
            if re.search(r'(?<![\w.])open\(', code) and "encoding=" not in code:
                bare.append(f"{p.relative_to(repo)}:{i}")
    check("encoding: no bare open() calls remain", not bare, f"{len(bare)} found: {bare[:3]}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{'='*50}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
