"""
Chunk a MeetingTranscript into ConversationWindows for extraction.

Uses a sliding window with overlap so relations spanning a boundary
aren't missed entirely.
"""

from typing import List
from utils.models import MeetingTranscript, ConversationWindow


def chunk_transcript(
    transcript: MeetingTranscript,
    window_size: int = 15,
    overlap: int = 5,
) -> List[ConversationWindow]:
    """
    Split transcript into overlapping windows.
    
    Args:
        transcript: Full meeting transcript.
        window_size: Number of utterances per window.
        overlap: Number of utterances shared between consecutive windows.
    
    Returns:
        List of ConversationWindows ready for extraction.
    """
    utterances = transcript.utterances
    if not utterances:
        return []

    step = max(1, window_size - overlap)
    windows = []
    window_id = 0

    for start in range(0, len(utterances), step):
        end = min(start + window_size, len(utterances))
        window_utts = utterances[start:end]

        windows.append(ConversationWindow(
            window_id=window_id,
            utterances=window_utts,
            meeting_id=transcript.meeting_id,
        ))
        window_id += 1

        # Stop if we've reached the end
        if end >= len(utterances):
            break

    return windows
