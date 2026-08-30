"""
Shared data models for the pipeline.
Both extractors (LLM and DHGAT) produce List[Triple].
Everything downstream consumes this format.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Utterance:
    """Single speaker turn from whisper-diarization."""
    speaker_id: str
    text: str
    start_time: float
    end_time: float

    def to_dict(self):
        return asdict(self)


@dataclass
class Triple:
    """
    One extracted knowledge triple.
    This is the universal format — both extractors produce these.
    """
    subject: str
    subject_type: str       # PERSON, COURSE, DEPARTMENT, COMMITTEE, etc.
    relation: str           # assigned_to, approved, teaches, etc.
    object: str
    object_type: str
    confidence: float = 1.0
    source_meeting: str = ""
    timestamp: float = 0.0  # from utterance start_time
    source_utterance: str = ""  # truncated preview (200 chars) for inspection
    # Full text of the window this triple came from. Kept separate because
    # source_utterance is a truncated preview, and Method A's feature
    # extraction needs the complete window to embed.
    window_text: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ConversationWindow:
    """
    A chunk of consecutive utterances for extraction.
    The extractor processes one window at a time.
    """
    window_id: int
    utterances: list        # List[Utterance]
    meeting_id: str = ""

    @property
    def text(self) -> str:
        """Formatted transcript for the LLM extractor."""
        lines = []
        for u in self.utterances:
            lines.append(f"[{u.speaker_id}] ({u.start_time:.1f}s): {u.text}")
        return "\n".join(lines)

    @property
    def speaker_ids(self) -> set:
        return {u.speaker_id for u in self.utterances}

    def to_dict(self):
        return {
            "window_id": self.window_id,
            "meeting_id": self.meeting_id,
            "utterances": [u.to_dict() for u in self.utterances],
        }


@dataclass
class MeetingTranscript:
    """Full transcript from one meeting."""
    meeting_id: str
    utterances: list        # List[Utterance]
    audio_file: str = ""
    duration: float = 0.0

    def to_dict(self):
        return {
            "meeting_id": self.meeting_id,
            "audio_file": self.audio_file,
            "duration": self.duration,
            "utterances": [u.to_dict() for u in self.utterances],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MeetingTranscript":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        utterances = [Utterance(**u) for u in data["utterances"]]
        return cls(
            meeting_id=data["meeting_id"],
            utterances=utterances,
            audio_file=data.get("audio_file", ""),
            duration=data.get("duration", 0.0),
        )
