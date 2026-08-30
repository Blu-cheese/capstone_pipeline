#!/usr/bin/env bash
#
# End-to-end pipeline: audio file(s) → knowledge graph
#
# Runs whisper-diarization first (in its conda env), then runs the
# extraction pipeline (in its venv), all from a single command.
#
# Usage:
#   ./run_full_pipeline.sh meeting_001.mp3
#   ./run_full_pipeline.sh meeting_001.mp3 meeting_002.mp3 meeting_003.mp3
#   WITH_NEO4J=1 ./run_full_pipeline.sh *.mp3
#
# Requirements:
#   - Both environments must already be set up (see SETUP_GUIDE.md)
#   - whisper-diarization conda env: conda env named "whisper-diarization"
#   - Pipeline venv: at capstone_pipeline/venv
#
# Environment variables (optional):
#   WITH_NEO4J=1          Use Neo4j instead of flat file graph
#   NEO4J_PASS=...        Neo4j password (default: capstone123)
#   LLM_MODEL=...         LLM model (default: gemini-2.5-flash-lite)
#   WHISPER_MODEL=...     Whisper model (default: medium.en)

set -e  # Exit on error

# -------- Configuration (override with env vars) --------
WHISPER_DIARIZATION_DIR="${WHISPER_DIARIZATION_DIR:-../whisper-diarization}"
PIPELINE_DIR="${PIPELINE_DIR:-.}"
WHISPER_MODEL="${WHISPER_MODEL:-medium.en}"
LLM_MODEL="${LLM_MODEL:-gemini-2.5-flash-lite}"
NEO4J_PASS="${NEO4J_PASS:-capstone123}"
CONDA_ENV="${CONDA_ENV:-whisper-diarization}"

# -------- Preflight checks --------
if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio_file> [<audio_file> ...]"
    echo ""
    echo "Example:"
    echo "  $0 meeting_001.mp3 meeting_002.mp3"
    echo ""
    echo "Environment variables:"
    echo "  WITH_NEO4J=1          Use Neo4j (default: flat file graph)"
    echo "  NEO4J_PASS=<pass>     Neo4j password (default: capstone123)"
    echo "  LLM_MODEL=<name>      Override LLM model"
    echo "  WHISPER_MODEL=<name>  Override Whisper model (default: medium.en)"
    exit 1
fi

# Check whisper-diarization exists
if [ ! -f "$WHISPER_DIARIZATION_DIR/diarize.py" ]; then
    echo "ERROR: whisper-diarization not found at $WHISPER_DIARIZATION_DIR"
    echo "Set WHISPER_DIARIZATION_DIR env var to its actual location."
    exit 1
fi

# Check pipeline exists
if [ ! -f "$PIPELINE_DIR/pipeline.py" ]; then
    echo "ERROR: pipeline.py not found at $PIPELINE_DIR"
    exit 1
fi

# Check conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first."
    exit 1
fi

# Check venv exists
if [ ! -f "$PIPELINE_DIR/venv/bin/activate" ]; then
    echo "ERROR: pipeline venv not found at $PIPELINE_DIR/venv"
    echo "Run: python3 -m venv venv && source venv/bin/activate"
    exit 1
fi

# Check API key for LLM (unless using Ollama)
if [ -z "$OPENAI_API_KEY" ] && [[ "$OPENAI_BASE_URL" != *"localhost"* ]]; then
    echo "ERROR: OPENAI_API_KEY not set."
    echo "Export it or set OPENAI_BASE_URL to a local Ollama instance."
    echo ""
    echo "For Gemini (the default model), OPENAI_API_KEY is your Google AI"
    echo "Studio key; the base URL is selected automatically from the model"
    echo "name, so OPENAI_BASE_URL does not need to be set."
    exit 1
fi

# -------- Stage 1: Audio → Transcript --------
echo ""
echo "=========================================="
echo "STAGE 1: Audio transcription + diarization"
echo "=========================================="

# Enable conda in this shell
eval "$(conda shell.bash hook)"

# Activate whisper-diarization env
conda activate "$CONDA_ENV"

# Set cmake workaround (required for M2 Macs with modern cmake)
export CMAKE_POLICY_VERSION_MINIMUM=3.5

TRANSCRIPT_PATHS=()

for audio in "$@"; do
    if [ ! -f "$audio" ]; then
        echo "WARNING: $audio not found, skipping."
        continue
    fi

    # Determine absolute path
    audio_abs="$(cd "$(dirname "$audio")" && pwd)/$(basename "$audio")"
    audio_dir="$(dirname "$audio_abs")"
    audio_base="$(basename "${audio%.*}")"
    # diarize.py writes BOTH .txt and .srt. Only the .srt retains real
    # timestamps in the current output format, so that is what we feed
    # the pipeline.
    transcript_path="$audio_dir/${audio_base}.srt"

    echo ""
    echo "Processing: $audio"

    # Skip if transcript already exists (cache)
    if [ -f "$transcript_path" ]; then
        echo "  ✓ Transcript already exists at $transcript_path (skipping)"
    else
        (
            cd "$WHISPER_DIARIZATION_DIR"
            # MUST stay cpu. diarize.py has mtypes = {"cpu":..., "cuda":...}
            # with no "mps" key, so --device mps dies with a KeyError.
            python diarize.py \
                -a "$audio_abs" \
                --device cpu \
                --whisper-model "$WHISPER_MODEL" \
                --batch-size 8
        )

        if [ ! -f "$transcript_path" ]; then
            echo "  ✗ Transcription failed — no transcript produced"
            continue
        fi
        echo "  ✓ Transcribed to $transcript_path"
    fi

    TRANSCRIPT_PATHS+=("$transcript_path")
done

if [ ${#TRANSCRIPT_PATHS[@]} -eq 0 ]; then
    echo ""
    echo "ERROR: No transcripts were produced. Exiting."
    exit 1
fi

# -------- Stage 2: Transcripts → Knowledge Graph --------
echo ""
echo "=========================================="
echo "STAGE 2: Extraction + graph insertion"
echo "=========================================="

# Switch environments: deactivate conda, activate pipeline venv
conda deactivate

cd "$PIPELINE_DIR"
source venv/bin/activate

# Copy transcripts into test_data for the pipeline
mkdir -p test_data/processed
for tp in "${TRANSCRIPT_PATHS[@]}"; do
    cp "$tp" test_data/processed/
done

# Build pipeline arguments
PIPELINE_ARGS=(
    --extractor llm
    --llm-model "$LLM_MODEL"
)

# Neo4j vs flat file
if [ "$WITH_NEO4J" = "1" ]; then
    PIPELINE_ARGS+=(--neo4j-password "$NEO4J_PASS")
else
    PIPELINE_ARGS+=(--no-neo4j)
fi

# Input files (use the copies in test_data/processed)
PIPELINE_ARGS+=(--input)
for tp in "${TRANSCRIPT_PATHS[@]}"; do
    PIPELINE_ARGS+=("test_data/processed/$(basename "$tp")")
done

# Run pipeline
python pipeline.py "${PIPELINE_ARGS[@]}"

# -------- Summary --------
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "Transcripts: ${#TRANSCRIPT_PATHS[@]}"
echo "Output:      $PIPELINE_DIR/output/"
echo ""
echo "Next steps:"
echo "  Inspect triples:  cat output/triples_*.json | python -m json.tool"
if [ "$WITH_NEO4J" = "1" ]; then
    echo "  Query graph:      python query.py --interactive --neo4j-password $NEO4J_PASS"
    echo "  Neo4j Browser:    http://localhost:7474"
else
    echo "  Query graph:      python query.py --interactive --no-neo4j"
fi
