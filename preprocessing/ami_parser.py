"""
Parse AMI Meeting Corpus transcripts into MeetingTranscript objects.

The AMI Meeting Corpus (https://groups.inf.ed.ac.uk/ami/corpus/) is released
under CC BY 4.0. Attribution to the AMI project is REQUIRED wherever this data
or anything derived from it is published.

Two input routes are supported:

1. HF-style plain text (recommended, e.g. the `knkarthick/AMI` re-upload).
   Each meeting is one string of speaker-prefixed lines:

       Speaker A: Cool.
       Speaker B: Um so shall we get started.

   No timestamps are present, so sequential synthetic ones are generated.

2. Official NXT XML (`ami_public_manual_1.6.2`), which stores one word per XML
   tag in per-speaker files (e.g. `words/EN2001a.A.words.xml`). Words must be
   reassembled into utterances. Real timings are available on this route.

Route 1 is far less work; use it unless real timestamps are needed.
"""

import re
import json
import glob
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree

from utils.models import Utterance, MeetingTranscript


# Synthetic timing for sources without real timestamps. AMI utterances are
# short conversational turns; this only needs to be monotonic and plausible,
# since nothing downstream depends on it being true to the audio.
SECONDS_PER_UTTERANCE = 4.0

# "Speaker A:", "Speaker 0:", "A:", "PM:" etc. at line start.
SPEAKER_LINE = re.compile(
    r'^\s*(speaker\s+)?([A-Za-z0-9_]{1,12})\s*:\s*(.+)$',
    re.IGNORECASE,
)


def _is_plausible_bare_label(token: str) -> bool:
    """
    Whether a colon-prefixed token with no explicit "Speaker" keyword is a
    speaker label rather than ordinary text.

    AMI transcripts are disfluent and full of mid-turn colons ("Okay: so the
    thing is"). Accepting any short token before a colon invents phantom
    speakers and splits one turn into two, which silently corrupts speaker
    attribution downstream. Real bare labels in AMI are role codes (PM, ID,
    UI, ME) or single letters (A, B, C) - short and uppercase - so require
    that shape and let anything else fall through to the continuation branch.
    """
    return len(token) <= 4 and (token.isupper() or token.isdigit() or len(token) == 1)


def parse_ami_dialogue(
    dialogue: str,
    meeting_id: str,
    seconds_per_utterance: float = SECONDS_PER_UTTERANCE,
) -> MeetingTranscript:
    """
    Parse one HF-style AMI dialogue string into a MeetingTranscript.

    Lines that don't match the `Speaker X: text` pattern are appended to the
    previous utterance (handles turns that wrap across newlines) rather than
    being silently dropped.

    Raises ValueError if no utterances could be parsed - never returns an
    empty transcript silently.
    """
    utterances: List[Utterance] = []
    clock = 0.0

    for raw_line in dialogue.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SPEAKER_LINE.match(line)
        # A bare "Token: text" line is only a new turn if the token looks like
        # a speaker label; otherwise treat it as continuation prose.
        if match and not match.group(1) and not _is_plausible_bare_label(match.group(2)):
            match = None

        if match:
            speaker = "SPEAKER_" + match.group(2).strip().upper()
            text = match.group(3).strip()
            if not text:
                continue
            utterances.append(Utterance(
                speaker_id=speaker,
                text=text,
                start_time=clock,
                end_time=clock + seconds_per_utterance,
            ))
            clock += seconds_per_utterance
        elif utterances:
            # Continuation of the previous turn.
            utterances[-1].text += " " + line

    if not utterances:
        raise ValueError(
            f"No utterances parsed for meeting '{meeting_id}'. "
            f"Expected lines like 'Speaker A: text'. "
            f"First 200 chars received: {dialogue[:200]!r}"
        )

    return MeetingTranscript(
        meeting_id=meeting_id,
        utterances=utterances,
        audio_file="",
        duration=utterances[-1].end_time,
    )


def load_ami_jsonl(
    path: str,
    dialogue_key: str = "dialogue",
    id_key: str = "id",
    id_prefix: str = "ami_",
    limit: Optional[int] = None,
) -> List[MeetingTranscript]:
    """
    Load AMI meetings from a JSONL file (one JSON object per line).

    Export the HF dataset to JSONL first so this module has no `datasets`
    dependency and the corpus is pinned on disk rather than re-downloaded:

        from datasets import load_dataset
        ds = load_dataset("knkarthick/AMI")
        for split in ds:
            ds[split].to_json(f"data/ami_{split}.jsonl")

    Malformed records are reported and skipped rather than aborting the load.
    """
    transcripts: List[MeetingTranscript] = []
    skipped = 0

    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(transcripts) >= limit:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                print(f"  ! line {line_no}: invalid JSON, skipped")
                continue

            dialogue = record.get(dialogue_key)
            if not dialogue:
                skipped += 1
                print(f"  ! line {line_no}: no '{dialogue_key}' field, skipped")
                continue

            meeting_id = f"{id_prefix}{record.get(id_key, line_no)}"
            try:
                transcripts.append(parse_ami_dialogue(dialogue, meeting_id))
            except ValueError as exc:
                skipped += 1
                print(f"  ! line {line_no}: {exc}")

    if not transcripts:
        raise ValueError(f"No usable AMI meetings parsed from {path}")

    print(f"Parsed {len(transcripts)} AMI meetings from {path}"
          + (f" ({skipped} skipped)" if skipped else ""))
    return transcripts


def parse_ami_nxt(
    words_dir: str,
    meeting_id: str,
    max_gap: float = 1.0,
) -> MeetingTranscript:
    """
    Parse official NXT-format AMI word files into a MeetingTranscript.

    Reads `{words_dir}/{meeting_id}.*.words.xml` (one file per speaker), where
    each <w> tag is a single word carrying starttime/endtime attributes. Words
    from the same speaker separated by less than `max_gap` seconds are merged
    into one utterance; utterances from all speakers are then interleaved by
    start time.

    Only needed when real timestamps matter. Prefer load_ami_jsonl otherwise.
    """
    pattern = str(Path(words_dir) / f"{meeting_id}.*.words.xml")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No AMI word files matched: {pattern}")

    utterances: List[Utterance] = []

    for filepath in files:
        # EN2001a.A.words.xml -> speaker "A"
        parts = Path(filepath).name.split(".")
        speaker = "SPEAKER_" + (parts[1].upper() if len(parts) > 2 else "UNK")

        root = ElementTree.parse(filepath).getroot()
        current: Optional[Utterance] = None

        for node in root:
            # Skip vocal-sound / non-word markup, keep only timed word tags.
            if not node.tag.endswith("w"):
                continue
            text = (node.text or "").strip()
            if not text:
                continue
            try:
                start = float(node.attrib["starttime"])
                end = float(node.attrib["endtime"])
            except (KeyError, ValueError):
                continue

            # AMI tags punctuation as its own <w punc="true"> token. Joining
            # those with a space yields "Okay ." - fine for a parser, but it
            # is what the teacher LLM reads and what the student embeds, so
            # attach punctuation directly to the preceding word.
            is_punc = node.attrib.get("punc") == "true"

            if current is not None and start - current.end_time <= max_gap:
                current.text += text if is_punc else " " + text
                current.end_time = end
            else:
                current = Utterance(
                    speaker_id=speaker, text=text,
                    start_time=start, end_time=end,
                )
                utterances.append(current)

    if not utterances:
        raise ValueError(f"No words parsed for AMI meeting '{meeting_id}'")

    utterances.sort(key=lambda u: u.start_time)

    return MeetingTranscript(
        meeting_id=f"ami_{meeting_id}",
        utterances=utterances,
        audio_file="",
        duration=max(u.end_time for u in utterances),
    )
