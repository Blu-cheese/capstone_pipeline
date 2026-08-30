# Generating Audio from Test Scripts

Three scripts are provided for testing the pipeline:

1. **script_01_department_meeting.txt** — CSE weekly meeting (~5 min, 4 speakers)
2. **script_02_department_followup.txt** — Follow-up showing temporal evolution (~6 min, 4 speakers)
3. **script_03_board_of_studies.txt** — Messier BOS meeting with interruptions (~7 min, 4 speakers)

Scripts 1 and 2 are designed to be processed sequentially to demonstrate how the knowledge graph evolves across meetings (the GPU budget going from 3 lakhs to 4.5 lakhs, capstone timeline changes, etc.).

## Option 1: Record yourselves (most realistic)

The ideal option. Four team members each read one speaker's lines. This gives you:
- Real voices with natural variation
- Realistic pauses, interruptions, "ums"
- Genuine overlapping speech (the hardest case for diarization)
- Audio quality similar to what you'd get from an actual meeting

**Setup:**
- Use any recording app (Voice Memos on Mac, Audacity, Zoom)
- One phone/laptop microphone in the center of a table works
- OR record a Zoom/Meet call with audio
- Export as `.wav` or `.mp3`
- 10-15 minutes total

**Tips:**
- Don't rehearse too much — stumbles and "let me restart that" are realistic
- Interrupt each other naturally when the script has back-and-forth
- Don't worry about perfect pronunciation

## Option 2: Free TTS services (automated)

If you can't record, generate audio using text-to-speech. Several free options:

### ElevenLabs (best quality, free tier)

1. Go to https://elevenlabs.io, sign up free
2. Free tier: 10,000 characters/month — enough for all three scripts
3. Use their "Studio" feature to assign different voices per speaker
4. Export as MP3

For multi-speaker audio, use their "Voice Design" or pick 4 distinct preset voices. Map them:
- JAYASHREE / MEERA / PRIYA → different female voices
- RAMESH / KUMAR / KESHAVAN / PAVAN → different male voices

### Edge-TTS (Microsoft, completely free via CLI)

```bash
pip install edge-tts

# Generate per-speaker audio, then concatenate
edge-tts --voice "en-IN-NeerjaNeural" --text "Good morning everyone." --write-media jayashree_line1.mp3
edge-tts --voice "en-IN-PrabhatNeural" --text "Thank you ma'am." --write-media ramesh_line1.mp3
```

Indian English voices available:
- Female: en-IN-NeerjaNeural, en-IN-NeerjaExpressiveNeural
- Male: en-IN-PrabhatNeural

For multi-speaker generation, a simple script:

```python
import asyncio
import edge_tts
from pydub import AudioSegment
import os

# Speaker → voice mapping
VOICES = {
    "JAYASHREE": "en-IN-NeerjaNeural",
    "RAMESH": "en-IN-PrabhatNeural",
    "MEERA": "en-IN-NeerjaExpressiveNeural",
    "PAVAN": "en-US-GuyNeural",  # different voice for contrast
}

async def generate_line(speaker, text, idx):
    voice = VOICES.get(speaker, "en-US-AriaNeural")
    communicate = edge_tts.Communicate(text, voice)
    filename = f"temp_{idx:03d}_{speaker}.mp3"
    await communicate.save(filename)
    return filename

def parse_script(path):
    """Parse SPEAKER: text format."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("---"):
                parts = line.split(":", 1)
                speaker = parts[0].strip()
                text = parts[1].strip()
                if speaker.isupper() and text:
                    lines.append((speaker, text))
    return lines

async def main(script_path, output_path):
    lines = parse_script(script_path)
    print(f"Parsed {len(lines)} lines")

    # Generate each line as separate audio
    files = []
    for i, (speaker, text) in enumerate(lines):
        print(f"  [{i+1}/{len(lines)}] {speaker}: {text[:60]}...")
        filename = await generate_line(speaker, text, i)
        files.append(filename)

    # Concatenate with small pauses
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=400)  # 400ms between speakers

    for f in files:
        audio = AudioSegment.from_mp3(f)
        combined += audio + pause
        os.remove(f)  # cleanup temp file

    combined.export(output_path, format="mp3")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main(
        "script_01_department_meeting.txt",
        "meeting_001.mp3"
    ))
```

Install: `pip install edge-tts pydub` and `brew install ffmpeg` (for pydub).

### OpenAI TTS (paid but cheap)

About $0.15 for all three scripts combined. Higher quality than Edge-TTS.

```bash
pip install openai
```

## Option 3: Colab notebook with TTS

If you want everything in the cloud, here's a Colab-compatible approach:

```python
# In a Colab cell
!pip install edge-tts pydub

# Upload your script via Colab UI
from google.colab import files
uploaded = files.upload()

# Then run the generation script above
```

## Testing the Output

Once you have audio files, run them through the full pipeline:

```bash
# Activate whisper-diarization conda env
conda activate whisper-diarization
cd whisper-diarization

# Transcribe with diarization
python diarize.py -a meeting_001.mp3 --device mps

# This creates meeting_001.txt next to the audio

# Switch to pipeline env
cd ../capstone_pipeline
source venv/bin/activate

# Process through pipeline
cp ../whisper-diarization/meeting_001.txt test_data/
python pipeline.py \
    --input test_data/meeting_001.txt \
    --extractor llm \
    --llm-model "gemini-2.0-flash" \
    --no-neo4j

# Query
python query.py --interactive --no-neo4j
# > Who proposed the Applied Deep Learning course?
# > What did they decide about the GPU lab?
```

## Comparing Real vs TTS Audio

For your capstone evaluation, test both:
- **TTS audio** → shows the pipeline works on clean input
- **Real recorded audio** → shows robustness to noise, accents, interruptions

Real audio will reveal pipeline failures that TTS hides. If you have time, do both — it strengthens your evaluation narrative.

## Recording Tips for Real Audio

If you go the recording route:

1. **Setup:** One room, four chairs around a table. Place one phone flat in the middle.
2. **Mic:** A smartphone built-in mic is fine. You don't need a podcast setup.
3. **Test first:** Record 30 seconds, transcribe it, check if whisper-diarization picks up all four speakers.
4. **Script-lite:** Use the script as a guide but don't read it verbatim. Paraphrase naturally.
5. **Timing:** Aim for ~5-7 minutes. Short enough to iterate, long enough to have meaningful content.
6. **Background:** Record in a quiet room. Avoid coffee shops for the test recordings (save those for the "robustness test" later).
7. **Export:** WAV format is best for whisper-diarization, but MP3 works fine too.

## What These Scripts Test

**Script 1** — Standard meeting dynamics. Expected extractions include:
- PERSON proposing COURSE (Ramesh proposed Applied Deep Learning)
- RESOURCE budget_for DEADLINE (GPU procurement, 3 lakhs)
- DEADLINE deadline_for EVENT (December 20th for mock interviews)
- PERSON assigned_to RESOURCE (Pavan assigned to GPU procurement)

**Script 2** — Tests temporal evolution. Processed after Script 1, should show:
- Budget evolution (3 lakhs → 4.5 lakhs → 2.2 lakhs approved)
- Deadline changes (December 8th → December 22nd)
- Status updates (TCS drive happened, results came in)

**Script 3** — Tests harder cases:
- Deferred/unresolved items (Data Mining review pending)
- Conditional approvals (two of three items approved)
- Multiple speakers in rapid exchange (interruptions)
- Prioritization decisions (Infosys > NVIDIA > Bosch)

If the pipeline handles all three, it's in good shape for the capstone demo.
