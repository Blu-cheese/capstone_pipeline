#!/usr/bin/env python3
"""
Generate multi-speaker audio from a script text file using Edge-TTS (free).

Usage:
    pip install edge-tts pydub
    # Mac: brew install ffmpeg (pydub needs it)

    python generate_audio.py script_01_department_meeting.txt meeting_001.mp3
    python generate_audio.py script_02_department_followup.txt meeting_002.mp3
    python generate_audio.py script_03_board_of_studies.txt meeting_003.mp3

The script expects text files in the format:
    SPEAKER_NAME: line of dialogue here
    ANOTHER_SPEAKER: their response

Speakers are mapped to Edge-TTS voices automatically. Four Indian English
voices are used by default to simulate a college meeting.
"""

import asyncio
import os
import re
import sys
from pathlib import Path


# Speaker name → Edge-TTS voice mapping.
#
# Every speaker who shares a meeting with another MUST get a distinct base
# voice, not a variant of the same one. The original map gave JAYASHREE
# "en-IN-NeerjaNeural" and MEERA "en-IN-NeerjaExpressiveNeural" — two
# flavours of the same voice — and NeMo diarization merged them into a
# single speaker in every meeting they shared (3 speakers detected for a
# 4-speaker script, in 4 of 5 meetings). A merged speaker then gets one
# name applied to two people's lines, i.e. false attribution in the graph.
#
# The two casts are:
#   Scripts 1/2/4: JAYASHREE, RAMESH, MEERA, PAVAN
#   Scripts 3/5:   KESHAVAN, JAYASHREE, KUMAR, PRIYA
# JAYASHREE spans both, so she anchors; everyone else is spread across
# locales and genders to maximise acoustic distance within each cast.
VOICE_MAP = {
    # Female voices — four distinct locales
    "JAYASHREE": "en-IN-NeerjaNeural",      # Indian    (both casts)
    "MEERA":     "en-AU-NatashaNeural",     # Australian (cast A)
    "PRIYA":     "en-US-AriaNeural",        # US        (cast B)
    "CHETANA":   "en-GB-SoniaNeural",       # British
    # Male voices — four distinct locales
    "RAMESH":    "en-IN-PrabhatNeural",     # Indian    (cast A)
    "PAVAN":     "en-US-GuyNeural",         # US        (cast A)
    "KUMAR":     "en-GB-RyanNeural",        # British   (cast B)
    "KESHAVAN":  "en-US-ChristopherNeural", # US, deep  (cast B)
    # Generic fallbacks
    "SPEAKER_00": "en-IN-PrabhatNeural",
    "SPEAKER_01": "en-IN-NeerjaNeural",
    "SPEAKER_02": "en-US-GuyNeural",
    "SPEAKER_03": "en-AU-NatashaNeural",
}

DEFAULT_VOICE = "en-US-AriaNeural"


def parse_script(path: str):
    """Parse SPEAKER: text lines from a script file."""
    lines = []
    current_speaker = None
    current_text = []

    speaker_pattern = re.compile(r'^([A-Z][A-Z0-9_]*)\s*:\s*(.*)$')

    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip()

            # Skip empty lines and separators
            if not line or line.startswith("---") or line.startswith("==="):
                if current_speaker and current_text:
                    lines.append((current_speaker, " ".join(current_text).strip()))
                    current_speaker = None
                    current_text = []
                continue

            match = speaker_pattern.match(line)
            if match:
                # Save previous speaker's accumulated text
                if current_speaker and current_text:
                    lines.append((current_speaker, " ".join(current_text).strip()))

                current_speaker = match.group(1)
                first_line = match.group(2).strip()
                current_text = [first_line] if first_line else []
            else:
                # Continuation of previous speaker's line
                if current_speaker:
                    current_text.append(line.strip())

    # Flush last speaker
    if current_speaker and current_text:
        lines.append((current_speaker, " ".join(current_text).strip()))

    # Filter empty
    lines = [(s, t) for s, t in lines if t]

    return lines


async def generate_line(speaker: str, text: str, idx: int, out_dir: str,
                        max_attempts: int = 5) -> str:
    """
    Generate audio for a single line using Edge-TTS.

    Retries with exponential backoff. Edge-TTS throttles under rapid
    sequential requests and returns either an exception or a zero-byte
    file; both previously caused the line to be dropped silently, which
    cost meeting_003 ~22% and meeting_005 ~36% of their scripts.
    """
    import edge_tts

    voice = VOICE_MAP.get(speaker, DEFAULT_VOICE)
    filename = os.path.join(out_dir, f"temp_{idx:03d}_{speaker}.mp3")

    last_error = None
    for attempt in range(max_attempts):
        try:
            communicate = edge_tts.Communicate(text, voice, rate="+0%")
            await communicate.save(filename)
            # A silently-empty file is a throttle, not a success.
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                return filename
            last_error = RuntimeError("wrote a zero-byte file")
        except Exception as e:
            last_error = e

        if attempt < max_attempts - 1:
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{max_attempts - 1} in {wait}s ({last_error})")
            await asyncio.sleep(wait)

    raise RuntimeError(
        f"line {idx} ({speaker}) failed after {max_attempts} attempts: {last_error}"
    )


def concatenate_audio(files: list, output_path: str, pause_ms: int = 500):
    """Combine audio files with small pauses between speakers."""
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=pause_ms)

    for f in files:
        audio = AudioSegment.from_mp3(f)
        combined += audio + pause

    # Export
    combined.export(output_path, format="mp3")

    # Cleanup
    for f in files:
        if os.path.exists(f):
            os.remove(f)


async def main(script_path: str, output_path: str):
    """Generate audio for an entire script."""
    # Check dependencies
    try:
        import edge_tts
        from pydub import AudioSegment
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install edge-tts pydub")
        print("Also need ffmpeg: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)")
        sys.exit(1)

    # Parse script
    lines = parse_script(script_path)
    print(f"Parsed {len(lines)} lines from {script_path}")

    if not lines:
        print("No dialogue found. Check script format.")
        sys.exit(1)

    # Show speaker distribution
    speakers = {}
    for speaker, _ in lines:
        speakers[speaker] = speakers.get(speaker, 0) + 1
    print(f"Speakers: {dict(sorted(speakers.items(), key=lambda x: -x[1]))}")

    # Setup output directory
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "_temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate each line
    print(f"\nGenerating {len(lines)} audio segments...")
    audio_files = []

    failures = []

    for i, (speaker, text) in enumerate(lines):
        voice = VOICE_MAP.get(speaker, DEFAULT_VOICE)
        preview = text[:60] + ("..." if len(text) > 60 else "")
        print(f"  [{i+1}/{len(lines)}] {speaker} ({voice}): {preview}")

        try:
            filename = await generate_line(speaker, text, i, temp_dir)
            audio_files.append(filename)
        except Exception as e:
            print(f"    FAILED: {e}")
            failures.append((i, speaker, text))

        # Pace requests. Edge-TTS throttles on rapid sequential calls.
        await asyncio.sleep(0.4)

    # A partial script produces a plausible-sounding meeting that is
    # quietly missing content, which is worse than a hard failure.
    if failures:
        print(f"\n{'!' * 60}")
        print(f"ABORTING: {len(failures)} of {len(lines)} lines failed to generate.")
        for i, speaker, text in failures:
            print(f"  line {i} ({speaker}): {text[:70]}")
        print("Nothing was written. Re-run to retry.")
        print(f"{'!' * 60}")
        for f in audio_files:
            if os.path.exists(f):
                os.remove(f)
        sys.exit(1)

    # Concatenate
    print(f"\nConcatenating {len(audio_files)} segments into {output_path}...")
    assert len(audio_files) == len(lines), (
        f"segment count {len(audio_files)} != line count {len(lines)}"
    )
    concatenate_audio(audio_files, output_path)

    # Cleanup temp dir
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    # Report
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    words = sum(len(t.split()) for _, t in lines)
    from pydub import AudioSegment
    duration = len(AudioSegment.from_mp3(output_path)) / 1000.0
    print(f"\nDone! Output: {output_path} ({size_mb:.1f} MB, {duration:.1f}s)")
    # ~0.54 s/word is the observed rate for these voices. A materially
    # lower ratio means lines went missing somewhere.
    print(f"  {len(lines)} lines, {words} words -> {duration / words:.3f} s/word "
          f"(expect ~0.54; well below that means dropped content)")
    print(f"\nNext steps:")
    print(f"  1. Run whisper-diarization on this audio:")
    print(f"     python diarize.py -a {output_path} --device cpu --whisper-model medium.en")
    print(f"  2. Feed the resulting .txt into the pipeline:")
    print(f"     python pipeline.py --input {Path(output_path).stem}.txt --extractor llm --no-neo4j")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    script_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        sys.exit(1)

    asyncio.run(main(script_path, output_path))
