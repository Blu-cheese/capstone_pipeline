"""
Parser tests. Run from the capstone_pipeline root:

    venv/bin/python tests/test_parsers.py

No pytest dependency - plain asserts and a tally, so this runs anywhere.

The transcript_parser cases exist because of a silent failure: when
whisper-diarization changed its .txt format (commit 8d87f2e), the parser
matched zero lines and returned an EMPTY transcript with no error. The
pipeline then reported "no triples found" and exited 0. The regression guard
below parses the real diarizer output in ../whisper-diarization/, not the
hand-written legacy-format files in test_data/ - those still parse fine under
the old regex, which is exactly why the bug stayed hidden.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.transcript_parser import (
    parse_transcript, parse_diarized_text, parse_srt,
)
from preprocessing.ami_parser import parse_ami_dialogue, load_ami_jsonl, parse_ami_nxt
from preprocessing.chunking import chunk_transcript

REPO = Path(__file__).resolve().parent.parent
WHISPER_DIR = REPO.parent / "whisper-diarization"

_passed, _failed = 0, 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}")
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f"  -- {extra}" if extra else ""))


def raises(name, fn, exc=ValueError):
    try:
        fn()
        check(name, False, "did not raise")
    except exc:
        check(name, True)
    except Exception as e:
        check(name, False, f"raised {type(e).__name__} not {exc.__name__}")


def write_tmp(content, suffix, encoding="utf-8"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(path).write_text(content, encoding=encoding)
    return path


# ---------------------------------------------------------------- transcript

def test_legacy_format():
    p = write_tmp(
        "SPEAKER_00 (0:00:01 - 0:00:05): Good morning everyone.\n"
        "SPEAKER_01 (0:00:06 - 0:00:14): Thanks, let's begin.\n",
        ".txt",
    )
    t = parse_diarized_text(p, "legacy")
    check("legacy: 2 utterances", len(t.utterances) == 2)
    check("legacy: real timestamps", t.utterances[1].start_time == 6.0,
          str(t.utterances[1].start_time))
    check("legacy: speaker ids", [u.speaker_id for u in t.utterances]
          == ["SPEAKER_00", "SPEAKER_01"])
    os.unlink(p)


def test_current_plain_format():
    p = write_tmp(
        "Speaker 0: Good morning, everyone.\n\n"
        "Speaker 2: Thank you, ma'am.\n\n"
        "Speaker 0: Let's get started.\n",
        ".txt",
    )
    t = parse_diarized_text(p, "plain")
    check("plain: 3 utterances", len(t.utterances) == 3, str(len(t.utterances)))
    check("plain: speaker ids zero-padded",
          [u.speaker_id for u in t.utterances]
          == ["SPEAKER_00", "SPEAKER_02", "SPEAKER_00"],
          str([u.speaker_id for u in t.utterances]))
    check("plain: synthetic clock monotonic",
          [u.start_time for u in t.utterances] == [0.0, 4.0, 8.0])
    os.unlink(p)


def test_bom_handling():
    """diarize.py writes utf-8-sig; a BOM must not break utterance #1."""
    p = write_tmp("Speaker 0: First line matters.\n", ".txt", encoding="utf-8-sig")
    t = parse_diarized_text(p, "bom")
    check("BOM: first utterance parsed", len(t.utterances) == 1, str(len(t.utterances)))
    check("BOM: no \\ufeff leaked into speaker id",
          t.utterances[0].speaker_id == "SPEAKER_00", repr(t.utterances[0].speaker_id))
    os.unlink(p)


def test_prose_colon_not_a_speaker():
    p = write_tmp(
        "Speaker 0: Right.\nOkay: so the thing is.\nSpeaker 1: Mm.\n", ".txt")
    t = parse_diarized_text(p, "prose")
    check("prose colon: no phantom speaker", len(t.utterances) == 2,
          str([(u.speaker_id, u.text) for u in t.utterances]))
    check("prose colon: merged into previous turn",
          t.utterances[0].text == "Right. Okay: so the thing is.",
          repr(t.utterances[0].text))
    os.unlink(p)


def test_empty_raises_loudly():
    p = write_tmp("this file has no speaker lines whatsoever\n", ".txt")
    raises("unparseable .txt raises instead of returning empty",
           lambda: parse_diarized_text(p, "empty"))
    os.unlink(p)


def test_srt_preferred_over_txt():
    d = tempfile.mkdtemp()
    txt = Path(d) / "m.txt"
    srt = Path(d) / "m.srt"
    txt.write_text("Speaker 0: no timestamps here.\n", encoding="utf-8")
    srt.write_text(
        "1\n00:00:07,000 --> 00:00:09,500\n[SPEAKER_00]: no timestamps here.\n",
        encoding="utf-8")
    t = parse_transcript(str(txt))
    check("prefers sibling .srt", t.utterances[0].start_time == 7.0,
          f"start={t.utterances[0].start_time} (0.0 means it used the .txt)")
    t2 = parse_transcript(str(txt), prefer_srt=False)
    check("prefer_srt=False forces the .txt", t2.utterances[0].start_time == 0.0)


def test_real_diarizer_output_regression():
    """The actual bug: real whisper-diarization output must not parse empty."""
    real = WHISPER_DIR / "meeting_001.txt"
    if not real.exists():
        print(f"SKIP  real diarizer output not found at {real}")
        return
    t = parse_diarized_text(str(real), "real")
    check("REGRESSION: real diarizer .txt parses non-empty",
          len(t.utterances) > 0, "THIS IS BUG 2a")
    check("REGRESSION: real output yields a usable number of turns",
          len(t.utterances) >= 10, f"only {len(t.utterances)}")
    w = chunk_transcript(t, window_size=15, overlap=5)
    check("REGRESSION: real transcript chunks", len(w) > 0, f"{len(w)} windows")


def test_test_data_still_parses():
    """test_data/*.txt are legacy-format; they must keep working."""
    for name in ("meeting_001.txt", "meeting_002.txt"):
        p = REPO / "test_data" / name
        if not p.exists():
            continue
        t = parse_transcript(str(p))
        check(f"test_data/{name} parses", len(t.utterances) > 0)
        check(f"test_data/{name} has real timestamps",
              any(u.start_time > 0 for u in t.utterances))


# ---------------------------------------------------------------------- AMI

def test_ami_basic():
    t = parse_ami_dialogue(
        "Speaker A: Cool.\nSpeaker B: Um so shall we get started.\n"
        "Speaker A: Yeah okay.", "m1")
    check("ami: 3 utterances", len(t.utterances) == 3)
    check("ami: speakers", [u.speaker_id for u in t.utterances]
          == ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"])
    check("ami: monotonic clock",
          [u.start_time for u in t.utterances] == [0.0, 4.0, 8.0])


def test_ami_role_codes():
    t = parse_ami_dialogue("PM: Right.\nA: Okay.\nID: Sure.\nUI: Mm.\nME: Yeah.", "m")
    check("ami: bare role codes accepted",
          [u.speaker_id for u in t.utterances]
          == ["SPEAKER_PM", "SPEAKER_A", "SPEAKER_ID", "SPEAKER_UI", "SPEAKER_ME"],
          str([u.speaker_id for u in t.utterances]))


def test_ami_prose_colon():
    t = parse_ami_dialogue("Speaker A: Right.\nOkay: so the thing is.\nSpeaker B: Mm.", "m")
    check("ami: no phantom speaker from prose colon", len(t.utterances) == 2,
          str([(u.speaker_id, u.text) for u in t.utterances]))
    for bad in ("Right: yeah.", "Um: okay then.", "Anyway: moving on."):
        r = parse_ami_dialogue(f"Speaker A: x.\n{bad}", "t")
        check(f"ami: '{bad}' is prose", len(r.utterances) == 1)


def test_ami_continuation_and_errors():
    t = parse_ami_dialogue("Speaker A: A turn\nthat wraps.\nSpeaker B: Right.", "m")
    check("ami: continuation merged",
          len(t.utterances) == 2 and t.utterances[0].text == "A turn that wraps.")
    raises("ami: unparseable dialogue raises",
           lambda: parse_ami_dialogue("garbage no colons", "m"))
    raises("ami: all-prose-colon dialogue raises",
           lambda: parse_ami_dialogue("Okay: so.\nRight: yeah.", "m"))


def test_ami_jsonl():
    recs = [{"id": "EN2001a", "dialogue": "Speaker A: Hello.\nSpeaker B: Hi."},
            {"id": "EN2001b", "dialogue": "Speaker C: Test."},
            {"id": "BAD", "dialogue": "garbage no colons"},
            {"id": "NOFIELD"}]
    p = write_tmp("\n".join(json.dumps(r) for r in recs) + "\n", ".jsonl")
    ts = load_ami_jsonl(p)
    check("ami jsonl: 2 good, 2 skipped", len(ts) == 2, str(len(ts)))
    check("ami jsonl: id prefix", ts[0].meeting_id == "ami_EN2001a", ts[0].meeting_id)
    check("ami jsonl: limit", len(load_ami_jsonl(p, limit=1)) == 1)
    os.unlink(p)


def test_ami_nxt():
    d = tempfile.mkdtemp()
    (Path(d) / "EN2001a.A.words.xml").write_text(
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
        '<w starttime="0.5" endtime="0.8">Hello</w>'
        '<w starttime="0.9" endtime="1.2">there</w>'
        '<vocalsound starttime="1.3" endtime="1.5"/>'
        '<w starttime="9.0" endtime="9.4">Later</w>'
        '</nite:root>', encoding="utf-8")
    t = parse_ami_nxt(d, "EN2001a")
    check("ami nxt: gap splits utterances", len(t.utterances) == 2, str(len(t.utterances)))
    check("ami nxt: words merged", t.utterances[0].text == "Hello there",
          repr(t.utterances[0].text))
    check("ami nxt: speaker from filename", t.utterances[0].speaker_id == "SPEAKER_A")


def test_ami_feeds_chunker():
    t = parse_ami_dialogue(
        "\n".join(f"Speaker {'AB'[i % 2]}: utterance {i}." for i in range(20)), "m")
    w = chunk_transcript(t, window_size=15, overlap=5)
    check("ami: chunker consumes AMI transcript", len(w) > 0, f"{len(w)} windows")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{'='*50}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
