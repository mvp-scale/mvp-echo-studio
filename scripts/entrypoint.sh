#!/bin/bash
# MVP-Echo Scribe entrypoint: GPU check + start uvicorn

echo "[echo-scribe] Starting MVP-Echo Scribe..."

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    echo "[echo-scribe] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[echo-scribe] WARNING: nvidia-smi not found, running on CPU"
fi

# Check CUDA from Python
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'[echo-scribe] CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'[echo-scribe] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('[echo-scribe] WARNING: CUDA not available to PyTorch')
"

# Check HF token
if [ -n "$HF_TOKEN" ]; then
    echo "[echo-scribe] HuggingFace token configured (diarization enabled)"
else
    echo "[echo-scribe] WARNING: No HF_TOKEN set, speaker diarization will be disabled"
fi

echo "[echo-scribe] Starting uvicorn on port ${PORT:-8001}..."
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8001}" --workers 1
