"""
Parse whisper-diarization output into MeetingTranscript.

whisper-diarization (MahmoudAshraf97) outputs a text file with lines like:
    SPEAKER 00 (0:00:01 - 0:00:05): Hello everyone, let's begin.
    SPEAKER 01 (0:00:06 - 0:00:12): Thanks, so the agenda today is...

It can also produce SRT files. This module handles both.
"""

import re
import json
from pathlib import Path
from typing import List

from utils.models import Utterance, MeetingTranscript


def parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS or H:MM:SS or MM:SS to seconds."""
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def parse_diarized_text(filepath: str, meeting_id: str = "") -> MeetingTranscript:
    """
    Parse the standard whisper-diarization text output.
    
    Expected format per line:
        SPEAKER XX (START - END): text content here
    
    Handles minor variations in spacing and timestamp format.
    """
    pattern = re.compile(
        r'(SPEAKER[\s_]*\d+)\s*'
        r'\(([^)]+?)\s*-\s*([^)]+?)\)\s*:\s*'
        r'(.+)',
        re.IGNORECASE
    )

    utterances: List[Utterance] = []
    text = Path(filepath).read_text(encoding="utf-8")

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            speaker = match.group(1).strip().upper().replace(" ", "_")
            start = parse_timestamp(match.group(2))
            end = parse_timestamp(match.group(3))
            content = match.group(4).strip()

            if content:
                utterances.append(Utterance(
                    speaker_id=speaker,
                    text=content,
                    start_time=start,
                    end_time=end,
                ))

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
    text = Path(filepath).read_text(encoding="utf-8")
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

    if not meeting_id:
        meeting_id = Path(filepath).stem

    duration = max((u.end_time for u in utterances), default=0.0)

    return MeetingTranscript(
        meeting_id=meeting_id,
        utterances=utterances,
        audio_file=filepath,
        duration=duration,
    )


def parse_transcript(filepath: str, meeting_id: str = "") -> MeetingTranscript:
    """Auto-detect format and parse."""
    ext = Path(filepath).suffix.lower()
    if ext == ".srt":
        return parse_srt(filepath, meeting_id)
    elif ext == ".json":
        return MeetingTranscript.load(filepath)
    else:
        # Default: try diarized text format
        return parse_diarized_text(filepath, meeting_id)
