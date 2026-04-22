# Continual Knowledge Graph Construction from Meeting Audio

## Pipeline Overview

```
Audio → whisper-diarization → Transcript → Chunking → Extractor → Entity Resolution → Neo4j
                                                        ↑
                                              LLM (MVP) | DHGAT (Research)
```

## Quick Start

### 1. Test the pipeline (no dependencies needed)

```bash
cd capstone_pipeline

# Run with mock extractor — tests the full flow without API keys or Neo4j
python pipeline.py \
    --input test_data/meeting_001.txt test_data/meeting_002.txt \
    --extractor mock \
    --no-neo4j
```

### 2. Run with LLM extractor (needs API key)

```bash
export OPENAI_API_KEY="sk-..."

# With flat file graph (no Neo4j needed)
python pipeline.py \
    --input test_data/meeting_001.txt test_data/meeting_002.txt \
    --extractor llm \
    --llm-model gpt-4o-mini \
    --no-neo4j

# With Neo4j
python pipeline.py \
    --input test_data/meeting_001.txt test_data/meeting_002.txt \
    --extractor llm \
    --neo4j-password yourpassword
```

### 3. Run with DHGAT extractor (needs trained model)

```bash
# First, train DHGAT on DialogRE
cd capstone_dialouge-re
pip install -r requirements-colab.txt
python main.py --mode train
cd ..

# Then run pipeline with DHGAT
python pipeline.py \
    --input test_data/meeting_001.txt \
    --extractor dhgat \
    --dhgat-repo ./capstone_dialouge-re \
    --dhgat-ckpt ./capstone_dialouge-re/runs/<timestamp>/best.pt \
    --no-neo4j
```

## Project Structure

```
capstone_pipeline/
├── pipeline.py                          # Main orchestrator
├── utils/
│   └── models.py                        # Shared data models (Triple, Utterance, etc.)
├── preprocessing/
│   ├── transcript_parser.py             # Parse whisper-diarization output
│   ├── chunking.py                      # Sliding window chunker
│   └── entity_resolution.py             # Normalize + deduplicate entities
├── extractors/
│   ├── llm_extractor.py                 # LLM-based triple extraction (MVP)
│   └── dhgat_extractor.py               # DHGAT adapter (research baseline)
├── graph/
│   └── neo4j_graph.py                   # Neo4j + flat file fallback
└── test_data/
    ├── meeting_001.txt                  # Sample staff meeting transcript
    └── meeting_002.txt                  # Follow-up meeting (temporal evolution)
```

## Dependencies

### Minimal (mock/LLM extractor + flat file graph)
- Python 3.10+
- No pip packages required for mock mode
- `openai` or any OpenAI-compatible API for LLM mode

### Full (Neo4j + DHGAT)
```bash
pip install neo4j spacy
python -m spacy download en_core_web_sm

# For DHGAT — see capstone_dialouge-re/requirements-colab.txt
```

### Audio processing (Stage 1)
```bash
# whisper-diarization (run separately, produces transcript files)
git clone https://github.com/MahmoudAshraf97/whisper-diarization
pip install -c constraints.txt -r requirements.txt
python diarize.py -a your_meeting.wav
# Then feed the output .txt into this pipeline
```

## Team Workflow

| Person | Component | Key Files |
|--------|-----------|-----------|
| Person 1 | Audio pipeline (whisper-diarization) | External repo → feeds `test_data/` |
| Person 2 | Neo4j + entity resolution | `graph/neo4j_graph.py`, `preprocessing/entity_resolution.py` |
| Person 3 | LLM extractor + annotation | `extractors/llm_extractor.py`, annotation spreadsheets |
| Person 4 | DHGAT + continual learning | `extractors/dhgat_extractor.py`, `capstone_dialouge-re/` |

## Connecting whisper-diarization

After running `diarize.py` on your audio, it produces a text file.
Feed that directly into this pipeline:

```bash
# Step 1: Transcribe + diarize
cd whisper-diarization
python diarize.py -a /path/to/meeting_recording.wav
# This creates a .txt file with speaker labels

# Step 2: Process through pipeline
cd ../capstone_pipeline
python pipeline.py --input /path/to/meeting_recording.txt --extractor llm --no-neo4j
```

## Output

Each meeting produces a `triples_<meeting_id>.json` file in `./output/`:

```json
[
  {
    "subject": "Dr. Ramesh",
    "subject_type": "PERSON",
    "relation": "proposed",
    "object": "Applied Deep Learning",
    "object_type": "COURSE",
    "confidence": 0.95,
    "source_meeting": "meeting_001",
    "timestamp": 29.0
  }
]
```

With Neo4j, you can query temporal evolution:
```cypher
-- All decisions across meetings
MATCH (s)-[r:RELATION]->(o) 
WHERE r.type IN ['approved', 'rejected', 'postponed', 'decided_on']
RETURN s.name, r.type, o.name, r.source_meeting 
ORDER BY r.timestamp

-- How an entity evolved across meetings
MATCH (e:Entity {name: 'GPU procurement'})-[r:RELATION]-(other)
RETURN r.source_meeting, r.type, other.name 
ORDER BY r.timestamp
```
