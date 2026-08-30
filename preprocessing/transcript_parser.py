"""
Parse whisper-diarization output into MeetingTranscript.

Two .txt layouts exist in the wild, because whisper-diarization changed its
output format:

  LEGACY (pre-Sortformer, and the hand-written files in test_data/):
      SPEAKER 00 (0:00:01 - 0:00:05): Hello everyone, let's begin.

  CURRENT (commit 8d87f2e, Feb 2026 Sortformer refactor):
      Speaker 0: Hello everyone, let's begin.

The current format carries NO timestamps, so synthetic sequential ones are
generated for it. Both layouts are handled; the legacy regex is tried first
and the plain layout is the fallback.

diarize.py writes BOTH a .txt and a .srt for every run, and only the .srt
retains real timings - so parse_transcript() prefers a sibling .srt when one
exists. Prefer feeding .srt explicitly where you can.

Note: diarize.py writes with encoding="utf-8-sig", so these files carry a BOM.
Reading them as plain utf-8 leaves \\ufeff glued to the first line and breaks
the match on utterance #1. All reads here use utf-8-sig.
"""

import re
import json
from pathlib import Path
from typing import List

from utils.models import Utterance, MeetingTranscript


# Synthetic turn length for the timestamp-less format. Only needs to be
# monotonic and plausible - nothing downstream depends on it matching audio,
# and real timings are available via .srt.
SECONDS_PER_UTTERANCE = 4.0


def parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS or H:MM:SS or MM:SS to seconds."""
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


LEGACY_LINE = re.compile(
    r'(SPEAKER[\s_]*\d+)\s*'
    r'\(([^)]+?)\s*-\s*([^)]+?)\)\s*:\s*'
    r'(.+)',
    re.IGNORECASE
)

# Current format: "Speaker 0: text". The literal "Speaker" keyword is
# required - matching any token before a colon would turn ordinary prose
# ("Okay: so the thing is") into a phantom speaker turn.
PLAIN_LINE = re.compile(r'^speaker[\s_]*(\d+)\s*:\s*(.+)$', re.IGNORECASE)


def _parse_legacy_lines(lines: List[str]) -> List[Utterance]:
    """Parse the timestamped `SPEAKER XX (start - end): text` layout."""
    utterances: List[Utterance] = []
    for line in lines:
        match = LEGACY_LINE.match(line)
        if not match:
            continue
        content = match.group(4).strip()
        if not content:
            continue
        utterances.append(Utterance(
            speaker_id=match.group(1).strip().upper().replace(" ", "_"),
            text=content,
            start_time=parse_timestamp(match.group(2)),
            end_time=parse_timestamp(match.group(3)),
        ))
    return utterances


def _parse_plain_lines(
    lines: List[str],
    seconds_per_utterance: float = SECONDS_PER_UTTERANCE,
) -> List[Utterance]:
    """
    Parse the timestamp-less `Speaker N: text` layout, synthesizing a
    monotonic clock. Lines that don't start a new turn are appended to the
    previous one rather than dropped.
    """
    utterances: List[Utterance] = []
    clock = 0.0
    for line in lines:
        match = PLAIN_LINE.match(line)
        if match:
            content = match.group(2).strip()
            if not content:
                continue
            utterances.append(Utterance(
                speaker_id=f"SPEAKER_{int(match.group(1)):02d}",
                text=content,
                start_time=clock,
                end_time=clock + seconds_per_utterance,
            ))
            clock += seconds_per_utterance
        elif utterances:
            utterances[-1].text += " " + line
    return utterances


def parse_diarized_text(filepath: str, meeting_id: str = "") -> MeetingTranscript:
    """
    Parse whisper-diarization .txt output in either the legacy timestamped
    layout or the current timestamp-less one.

    Raises ValueError if no utterances are parsed. Returning an empty
    transcript here is what let the format change go unnoticed: the pipeline
    reported "no triples found" and exited 0 on every real diarizer output.
    """
    # utf-8-sig strips the BOM diarize.py writes; harmless on files without one.
    text = Path(filepath).read_text(encoding="utf-8-sig")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    utterances = _parse_legacy_lines(lines)
    synthetic_times = False
    if not utterances:
        utterances = _parse_plain_lines(lines)
        synthetic_times = bool(utterances)

    if not utterances:
        raise ValueError(
            f"No utterances parsed from {filepath}. Expected either "
            f"'SPEAKER 00 (0:00:01 - 0:00:05): text' or 'Speaker 0: text'. "
            f"First 200 chars: {text[:200]!r}"
        )

    if synthetic_times:
        print(f"  [NOTE] {Path(filepath).name} has no timestamps "
              f"(current whisper-diarization .txt format); synthesized "
              f"{SECONDS_PER_UTTERANCE}s/turn. Use the .srt for real timings.")

    if not meeting_id:
        meeting_id = Path(filepath).stem

    duration = max((u.end_time for u in utterances), default=0.0)

    return MeetingTranscript(
        meeting_id=meeting_id,
        utterances=utterances,
        audio_file=filepath,
        duration=duration,
    )


def parse_srt(filepath: str, meeting_id: str = "") -> MeetingTranscript:
    """
    Parse SRT subtitle files (alternative whisper-diarization output).
    Speaker labels may be embedded in the text as [SPEAKER_XX] prefix.
    """
    text = Path(filepath).read_text(encoding="utf-8-sig")
    blocks = re.split(r'\n\s*\n', text.strip())

    utterances: List[Utterance] = []
    srt_time_pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})'
    )
    speaker_pattern = re.compile(r'^\[?(SPEAKER[_ ]?\d+)\]?:?\s*', re.IGNORECASE)

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue

        # Find the timestamp line
        time_match = None
        time_line_idx = -1
        for i, line in enumerate(lines):
            time_match = srt_time_pattern.search(line)
            if time_match:
                time_line_idx = i
                break

        if not time_match:
            continue

        start_str = time_match.group(1).replace(",", ".")
        end_str = time_match.group(2).replace(",", ".")
        start = parse_timestamp(start_str)
        end = parse_timestamp(end_str)

        # Text is everything after the timestamp line
        content_lines = lines[time_line_idx + 1:]
        content = " ".join(l.strip() for l in content_lines if l.strip())

        # Extract speaker if present
        speaker = "UNKNOWN"
        sp_match = speaker_pattern.match(content)
        if sp_match:
            speaker = sp_match.group(1).upper().replace(" ", "_")
            content = content[sp_match.end():].strip()

        if content:
            utterances.append(Utterance(
                speaker_id=speaker,
                text=content,
                start_time=start,
                end_time=end,
            ))

    if not utterances:
        raise ValueError(
            f"No subtitle blocks parsed from {filepath}. "
            f"First 200 chars: {text[:200]!r}"
        )

    if not meeting_id:
        meeting_id = Path(filepath).stem

    duration = max((u.end_time for u in utterances), default=0.0)

    return MeetingTranscript(
        meeting_id=meeting_id,
        utterances=utterances,
        audio_file=filepath,
        duration=duration,
    )


def parse_transcript(
    filepath: str,
    meeting_id: str = "",
    prefer_srt: bool = True,
) -> MeetingTranscript:
    """
    Auto-detect format and parse.

    diarize.py emits a .txt and a .srt side by side for every run, and only
    the .srt carries real timings. When handed the .txt, transparently prefer
    the sibling .srt so downstream triples get true timestamps instead of
    synthesized ones. Pass prefer_srt=False to force the given file.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".json":
        return MeetingTranscript.load(filepath)

    if ext == ".srt":
        return parse_srt(filepath, meeting_id)

    if prefer_srt:
        sibling = path.with_suffix(".srt")
        if sibling.exists():
            print(f"  [NOTE] Using {sibling.name} instead of {path.name} "
                  f"(same run, but retains real timestamps).")
            return parse_srt(str(sibling), meeting_id or path.stem)

    return parse_diarized_text(filepath, meeting_id)
