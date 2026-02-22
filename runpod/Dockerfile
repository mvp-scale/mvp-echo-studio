# MVP-Echo Scribe: RunPod Serverless GPU Worker
#
# Build from project root:
#   docker build -f runpod/Dockerfile -t mvp-scribe-runpod .
#
# Single-stage build (no frontend needed for serverless API)

FROM nvcr.io/nvidia/nemo:25.04

ENV PYTHONUNBUFFERED=1

# PyTorch CUDA memory allocator: expandable segments reduce fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model cache → RunPod network volume (persists across cold starts)
ENV HF_HOME=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch
ENV NEMO_CACHE_DIR=/runpod-volume/nemo

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Fix pip metadata for NVIDIA's local torch wheels.
# The NeMo container installed them from /opt/transfer/*.whl files that were
# deleted after install. Removing direct_url.json lets pip treat them as
# normal installed packages during dependency resolution.
RUN for d in /usr/local/lib/python3.12/dist-packages/torch*.dist-info; do \
        rm -f "$d/direct_url.json"; \
    done

# Pin torch ecosystem versions so pip won't replace them.
RUN for pkg in torch torchvision torchaudio numpy; do \
        v=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('$pkg').split('+')[0])" 2>/dev/null) \
        && echo "$pkg==$v" >> /tmp/torch-pins.txt \
        || true; \
    done && echo '--- Pinned torch packages ---' && cat /tmp/torch-pins.txt

# Install pyannote-audio WITHOUT dependency resolution to avoid torch conflicts.
# Pin to 3.3.2 — version 4.x has exact torch==2.8.0 pin incompatible with NeMo.
RUN pip install --no-cache-dir --no-deps pyannote-audio==3.3.2

# torchaudio shim (pure-Python, soundfile-based — avoids C++ ABI issues)
COPY backend/torchaudio_compat /tmp/torchaudio_shim
RUN if python3 -c "import torchaudio" 2>/dev/null; then \
        echo "torchaudio already in container"; \
    else \
        echo "Installing torchaudio shim (soundfile-based, no C++ ABI dependency)"; \
        SITE=$(python3 -c "import site; print(site.getsitepackages()[0])"); \
        cp -r /tmp/torchaudio_shim "$SITE/torchaudio"; \
        python3 -c "import torchaudio; print(f'torchaudio {torchaudio.__version__} OK')"; \
    fi \
    && rm -rf /tmp/torchaudio_shim

# Install remaining deps with torch pinned so pip can't replace them.
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        -c /tmp/torch-pins.txt \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt /tmp/torch-pins.txt

# RunPod SDK
RUN pip install --no-cache-dir runpod

# spaCy English model for entity detection and topic extraction (~12MB)
RUN python3 -m spacy download en_core_web_sm

# Packages that trigger pip resolution loops due to torch 2.7.0a0 pre-release pin.
# ALL must use --no-deps to prevent pip from replacing NVIDIA's custom torch.
RUN pip install --no-cache-dir --no-deps \
        'torch-audiomentations>=0.11.0' \
        'torchmetrics>=1.0' \
        julius \
        'torch-pitch-shift>=1.2.2' \
        primePy \
        semver \
        tensorboardX

# Patch pyannote's version checker to handle NVIDIA's non-SemVer torch version.
# "2.7.0a0+7c8ec84dab" fails semver.parse; we strip everything after X.Y.Z.
RUN VFILE=$(python3 -c "import pyannote.audio.utils.version as v; print(v.__file__)") && \
    sed -i '1s/^/import re\n/' "$VFILE" && \
    sed -i "s/theirs = \".\".join(theirs.split(\".\")\\[:3\\])/&\n    theirs = re.sub(r'[^0-9.].*', '', theirs)/" "$VFILE" && \
    sed -i "s/mine = \".\".join(mine.split(\".\")\\[:3\\])/&\n    mine = re.sub(r'[^0-9.].*', '', mine)/" "$VFILE" && \
    python3 -c "from pyannote.audio.utils.version import check_version; check_version('test', '3.3.2', '2.7.0a0+foo'); print('version.py patch OK')"

# Copy backend code + handler
WORKDIR /app
COPY backend/ /app/
COPY runpod/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
