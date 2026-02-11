#!/bin/bash
# MVP-Echo Scribe entrypoint (Sherpa-ONNX ASR + Pyannote diarization)
#
# Downloads ASR and VAD models from GitHub releases (public, no auth).
# Pyannote downloads its own diarization models via HuggingFace on first run.
# Models are cached in /models/sherpa-onnx (persistent Docker volume).
# Skips downloads if files already exist from a previous run.

set -e

MODEL_DIR="${MODEL_ID:-/models/sherpa-onnx}"

echo "[echo-scribe] Starting MVP-Echo Scribe (Sherpa ASR + Pyannote diarization)..."

# ── GPU check ──
if command -v nvidia-smi &>/dev/null; then
    echo "[echo-scribe] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[echo-scribe] WARNING: nvidia-smi not found"
fi

# ── Download models from GitHub releases (no auth required) ──
echo "[echo-scribe] Model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# --- ASR: Parakeet TDT 0.6B v2 (int8) ---
# Source: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
# Files: encoder.int8.onnx (622MB), decoder.int8.onnx (6.9MB), joiner.int8.onnx (1.7MB), tokens.txt
ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"

if [ -f "${MODEL_DIR}/encoder.int8.onnx" ] && [ -f "${MODEL_DIR}/tokens.txt" ]; then
    echo "[echo-scribe] ASR model already present, skipping download"
else
    echo "[echo-scribe] Downloading ASR model (Parakeet TDT 0.6B v2 int8, ~460MB)..."
    curl -fSL "${ASR_URL}" | tar xjf - -C "${MODEL_DIR}" --strip-components=1
    echo "[echo-scribe] ASR model downloaded"
fi

# ── Verify all models ──
echo "[echo-scribe] Verifying models..."
MISSING=0
for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do
    if [ -f "${MODEL_DIR}/${f}" ]; then
        SIZE=$(stat -c%s "${MODEL_DIR}/${f}" 2>/dev/null || echo "?")
        echo "[echo-scribe]   OK: ${f} (${SIZE} bytes)"
    else
        echo "[echo-scribe]   MISSING: ${f}"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "[echo-scribe] ERROR: Some models are missing. Cannot start."
    exit 1
fi

echo "[echo-scribe] All models verified"
echo "[echo-scribe] Note: Pyannote diarization models will be downloaded from HuggingFace on first use"

# ── Start FastAPI app ──
echo "[echo-scribe] Starting uvicorn on port ${PORT:-8001}..."
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8001}" --workers 1
