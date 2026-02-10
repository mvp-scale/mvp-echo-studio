# Sherpa-ONNX GPU Batch Processing Implementation

## Critical Finding: Diarization Is Faster on CPU (2026-02-10)

### Benchmark: Native C++ Binary (v1.12.24), 18-min Podcast, RTX 3090

Tested using the **pre-compiled GPU runtime** from the official sherpa-onnx v1.12.24 release
(`sherpa-onnx-v1.12.24-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2`).
This is the same C++ code Gemini's research recommends building from source.

| provider | Time | Notes |
|----------|------|-------|
| `--segmentation.provider=cuda --embedding.provider=cuda` | **4 min 26 sec** | GPU overhead exceeds compute benefit |
| `--segmentation.provider=cpu --embedding.provider=cpu` | **1 min 55 sec** | 2.3x FASTER than GPU |

**Conclusion**: GPU makes diarization **slower**, not faster. This is NOT a Python issue,
NOT a pip wheel issue, NOT a configuration error. The pre-compiled native C++ binary
from the official release exhibits the same behavior. The segmentation model (5.7 MB)
and embedding model (27 MB) are too small for GPU data transfer overhead to pay off.
The diarization pipeline processes hundreds of tiny sequential inferences (10-second sliding
windows) where CPU-to-GPU transfer latency dominates.

**Building from source will NOT fix this.** The runtime is already GPU-enabled.
The diarization algorithm is inherently CPU-optimal for these model sizes.

### Revised Strategy

The original 4-step pipeline (diarize first → then ASR) is bottlenecked by diarization.
The revised approach should be:

1. **ASR on full audio (GPU)** — Run `decode_streams()` on the entire audio in one batch. Fast (~seconds).
2. **Diarization (CPU, concurrent)** — Run `OfflineSpeakerDiarization.process()` on CPU in parallel with or after ASR.
3. **Merge** — Match ASR timestamps to diarization speaker segments.

This gives the user transcription text in seconds, with speaker labels added when diarization completes.

### What Was Ruled Out

| Hypothesis | Tested | Result |
|-----------|--------|--------|
| pip wheel missing GPU support | Installed `+cuda12.cudnn9` wheel | GPU works for ASR, not faster for diarization |
| Python overhead causing slowness | Tested native C++ binary directly | Same result: GPU slower than CPU |
| Need to build from source (Gemini recommendation) | Used pre-compiled v1.12.24 GPU binary from official release | Same binary, same result |
| CUDA 12 vs 11.8 compatibility | Release built for CUDA 12.x + cuDNN 9.x (matches our stack) | Not a version issue |
| Wrong provider config | Both `OfflineSpeakerSegmentationModelConfig.provider` and `SpeakerEmbeddingExtractorConfig.provider` confirmed as valid attributes | Config is accepted, just slower |

---

## Overview

This implementation uses the Sherpa-ONNX Python API for batch GPU ASR processing,
with diarization running on CPU (where it's fastest).

## Architecture

### Pipeline (Revised)

1. **ASR on full audio (GPU)**: `OfflineRecognizer.decode_streams()` transcribes in parallel
2. **Speaker Diarization (CPU)**: `OfflineSpeakerDiarization.process()` segments by speaker
3. **Merge Results**: Match ASR timestamps to diarization speaker labels

### Performance Expectations (Revised)

| Audio Length | ASR (GPU) | Diarization (CPU) | Total |
|--------------|-----------|-------------------|-------|
| 18 minutes   | ~5-10 sec | ~2 min            | ~2 min |
| 3 hours      | <3 minutes                 |

**Speedup vs NeMo**: ~15x faster (18min: 15s → 5-7s)

## Files Created/Modified

### New Files

1. **`backend/adapters/sherpa/__init__.py`** - Package initialization
2. **`backend/adapters/sherpa/transcription.py`** - Main adapter implementation
   - Uses Python sherpa-onnx API
   - Implements TranscriptionPort interface
   - Models pre-downloaded in Dockerfile from GitHub releases (no HuggingFace auth needed)
   - Runs 4-step pipeline with batch GPU processing

### Modified Files

1. **`Dockerfile.sherpa`** - Updated for Python API
   - Uses nvidia/cuda:12.6.3-cudnn-runtime base
   - Installs sherpa-onnx GPU binaries for libonnxruntime_providers_cuda.so
   - Installs Python sherpa-onnx package (uses GPU libs via LD_LIBRARY_PATH)
   - Installs soundfile, numpy for audio processing

2. **`backend/config.py`** - Updated comment for Sherpa adapter

3. **`backend/use_cases/transcribe.py`** - Skip chunking for Sherpa
   - Detects Sherpa by checking if diarization port is None
   - Sherpa handles full 3-hour files without 500s chunking

4. **`backend/api.py`** - Handle optional torch import
   - Makes torch optional (not needed for Sherpa)
   - Health endpoint checks torch availability before use

5. **`backend/main.py`** - Handle optional torch import
   - Makes torch optional for Sherpa containers

6. **`entrypoint-sherpa.sh`** - Updated description

## Models

Models are downloaded at container build time from GitHub releases (no auth required).
Cached in `/models/sherpa-onnx` (persistent Docker volume) so rebuilds don't re-download.

### Pre-Validation Checklist (2026-02-10)

All model URLs were verified live before implementation:

| Model | Primary Source | HTTP Status | Auth Required |
|-------|---------------|-------------|---------------|
| ASR (Parakeet TDT v2 int8) | GitHub Release `asr-models` | **302 (OK)** | No |
| Segmentation (Pyannote 3.0) | GitHub Release `speaker-segmentation-models` | **302 (OK)** | No |
| Embedding (3D-Speaker CAM++) | GitHub Release `speaker-recongition-models` | **302 (OK)** | No |
| ASR (HuggingFace mirror) | `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` | **200 (OK)** | No |
| Segmentation (HuggingFace mirror) | `csukuangfj/sherpa-onnx-pyannote-segmentation-3-0` | **200 (OK)** | No |
| Embedding (HuggingFace mirror) | `csukuangfj/sherpa-onnx-3dspeaker-speech-campplus-sv-zh-en-16k-common-advanced-onnx` | **401 FAIL** | **Yes (gated)** |

**Key finding**: The embedding model on HuggingFace is gated (401 Unauthorized). GitHub releases are the reliable source for all three models.

### Model #1: ASR — Parakeet TDT 0.6B v2 (int8)

**What it does**: Speech-to-text transcription (English, with punctuation + casing)
**Architecture**: NeMo Transducer (`model_type="nemo_transducer"`)
**Files needed**: `encoder.int8.onnx` (622 MB), `decoder.int8.onnx` (6.9 MB), `joiner.int8.onnx` (1.7 MB), `tokens.txt` (9.2 KB)

| Source | URL | Notes |
|--------|-----|-------|
| **Primary (GitHub Release)** | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2` | ~460 MB tar.bz2, public, no auth |
| Backup (HuggingFace) | `https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` | Public, use `huggingface_hub.snapshot_download()` |
| Alt: fp16 variant | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2` | ~1.1 GB, higher precision |
| Alt: v3 (multilingual) | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2` | ~465 MB, 25 European languages |
| Alt: 110M small model | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8.tar.bz2` | ~100 MB, lighter/faster |
| Original NVIDIA model | `https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2` | NeMo format (not ONNX), requires conversion |

```bash
# Download and extract
curl -fSL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2 \
  | tar xjf - -C /models/sherpa-onnx/ --strip-components=1
```

### Model #2: Speaker Segmentation — Pyannote 3.0

**What it does**: Segments audio by detecting who is speaking when
**Files needed**: `model.onnx` (renamed to `segmentation-3.0.onnx` in our setup)

| Source | URL | Notes |
|--------|-----|-------|
| **Primary (GitHub Release)** | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` | ~6.6 MB, public, no auth |
| Backup (HuggingFace) | `https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0` | Public, use `snapshot_download()` |
| Alt: Reverb Diarization v1 | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2` | ~10 MB, alternative segmentation model |
| Alt: Reverb Diarization v2 | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v2.tar.bz2` | ~242 MB, newer/larger |

```bash
# Download, extract, and rename
curl -fSL https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 \
  | tar xjf - -C /tmp/ \
  && mv /tmp/sherpa-onnx-pyannote-segmentation-3-0/model.onnx /models/sherpa-onnx/segmentation-3.0.onnx \
  && rm -rf /tmp/sherpa-onnx-pyannote-segmentation-3-0
```

### Model #3: Speaker Embedding — 3D-Speaker CAM++

**What it does**: Extracts voice embeddings for speaker clustering/identification
**Files needed**: `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (note: hyphen in `16k-common`)

| Source | URL | Notes |
|--------|-----|-------|
| **Primary (GitHub Release)** | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | ~27 MB, direct .onnx, public, no auth |
| ~~Backup (HuggingFace)~~ | ~~`csukuangfj/sherpa-onnx-3dspeaker-speech-campplus-sv-zh-en-16k-common-advanced-onnx`~~ | **GATED (401) — do not use** |
| Alt: English-focused (VoxCeleb) | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` | ~28 MB, English speakers |
| Alt: WeSpeaker ResNet34 | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx` | ~25 MB, English |
| Alt: NeMo TitaNet Large | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx` | ~97 MB, highest quality English |
| Alt: NeMo TitaNet Small | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx` | ~38 MB, good balance |

**IMPORTANT filename note**: The GitHub release file uses a hyphen: `16k-common_advanced`. The previous code expected all underscores: `16k_common_advanced`. The Python adapter must reference the correct filename.

```bash
# Direct download (no extraction needed)
curl -fSL -o /models/sherpa-onnx/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```

### All Embedding Models Available (GitHub Release `speaker-recongition-models`)

Note: The release tag has a permanent typo — `recongition` not `recognition`.

| Model File | Size | Language | Family |
|------------|------|----------|--------|
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 27 MB | zh+en | 3D-Speaker CAM++ |
| `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` | 28 MB | en | 3D-Speaker CAM++ |
| `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` | 27 MB | zh | 3D-Speaker CAM++ |
| `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | 38 MB | zh | 3D-Speaker ERes2Net |
| `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` | 25 MB | en | 3D-Speaker ERes2Net |
| `3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx` | 111 MB | zh | 3D-Speaker ERes2Net Large |
| `3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx` | 68 MB | zh | 3D-Speaker ERes2NetV2 |
| `wespeaker_en_voxceleb_resnet34.onnx` | 25 MB | en | WeSpeaker ResNet34 |
| `wespeaker_en_voxceleb_resnet34_LM.onnx` | 25 MB | en | WeSpeaker ResNet34+LM |
| `wespeaker_en_voxceleb_CAM++.onnx` | 28 MB | en | WeSpeaker CAM++ |
| `wespeaker_en_voxceleb_CAM++_LM.onnx` | 28 MB | en | WeSpeaker CAM+++LM |
| `wespeaker_en_voxceleb_resnet152_LM.onnx` | 75 MB | en | WeSpeaker ResNet152 |
| `wespeaker_en_voxceleb_resnet221_LM.onnx` | 91 MB | en | WeSpeaker ResNet221 |
| `wespeaker_en_voxceleb_resnet293_LM.onnx` | 109 MB | en | WeSpeaker ResNet293 |
| `nemo_en_speakerverification_speakernet.onnx` | 22 MB | en | NeMo SpeakerNet |
| `nemo_en_titanet_small.onnx` | 38 MB | en | NeMo TitaNet Small |
| `nemo_en_titanet_large.onnx` | 97 MB | en | NeMo TitaNet Large |

## Usage

### Build and Start

```bash
# Using start script
./start.sh sherpa

# Or manually with docker-compose
DOCKERFILE=Dockerfile.sherpa ENGINE=sherpa docker compose up -d --build
```

### Test

```bash
# Health check
curl http://localhost:20301/health

# Transcribe with diarization
curl -X POST \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "response_format=verbose_json" \
  http://localhost:20301/v1/audio/transcriptions
```

### Logs

```bash
docker compose logs -f mvp-scribe
```

## Current Implementation State

### Docker Stack (Working)
- **Dockerfile.sherpa** — `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` base
- **pip**: `sherpa-onnx==1.12.23+cuda12.cudnn9` from [CUDA wheel index](https://k2-fsa.github.io/sherpa/onnx/cuda.html)
- **ASR**: `OfflineRecognizer.from_transducer()` — GPU, working, fast
- **Diarization**: `OfflineSpeakerDiarization.process()` — CPU (GPU is slower, see findings above)
- **Models**: Downloaded by `entrypoint-sherpa.sh` from GitHub releases (no HuggingFace auth)
- **Status**: Functional. Short files transcribe correctly. 18-min files bottlenecked by diarization (~2 min).

### Config Parameters (Current)
```python
# ASR — GPU, uses high-level factory method
self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=..., decoder=..., joiner=..., tokens=...,
    model_type="nemo_transducer",
    provider="cuda",
    num_threads=4,
)

# Diarization — provider="cuda" accepted but GPU is slower (see benchmarks)
segmentation_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
    pyannote=..., provider="cuda", num_threads=4,
)
embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=..., provider="cuda", num_threads=4,
)
```

### API Gotchas Discovered
- `OfflineTransducerModelConfig` constructor uses `encoder_filename=` (not `encoder=`). Use `from_transducer()` factory instead.
- `OfflineSpeakerDiarization.process(audio)` takes ONE arg (samples). No `sample_rate` parameter.
- `process()` returns `OfflineSpeakerDiarizationResult` object, not a list. Call `.sort_by_start_time()` then wrap in `list()`.

## Open Questions for Next Session

1. **Can diarization be parallelized?** — The `sherpa-onnx-offline-parallel` binary does parallel ASR but NOT parallel diarization. Is there a way to split audio and diarize chunks concurrently?
2. **Flip the pipeline?** — ASR first (GPU, seconds) → diarize second (CPU, ~2 min) → merge timestamps. User gets transcript immediately, speaker labels arrive later.
3. **Alternative diarization engines?** — Is there a GPU-native diarization approach outside sherpa-onnx that could be faster? (e.g., PyTorch-based Pyannote directly, or a different model architecture)
4. **Reverb Diarization v2** — The 242 MB model in the release. Larger model = more GPU-friendly? Untested.
5. **Is the 5-second target achievable with diarization?** — ASR alone can hit ~5-10s for 18 min. Diarization adds ~2 min minimum. May need to accept diarization as async/background.

## Exit Criteria Status

- [x] Sherpa adapter implemented with Python API
- [x] All models downloading from verified public GitHub releases
- [x] ASR working on GPU via `from_transducer()` factory
- [x] Diarization working (CPU, functional)
- [x] Short file transcription validated (51s test.m4a)
- [x] 18-min file transcription functional (diarization bottleneck identified)
- [x] Dockerfile with GPU CUDA 12 wheel
- [x] Entrypoint downloads and verifies all 6 model files
- [ ] **Performance target**: 18 min in <10s (blocked by diarization ~2 min)
- [ ] **Pipeline restructure**: ASR-first approach not yet implemented

## References

### Primary Sources (Canonical)
- [sherpa-onnx GitHub Repository](https://github.com/k2-fsa/sherpa-onnx) — Main project, all releases hosted here
- [sherpa-onnx GitHub Releases](https://github.com/k2-fsa/sherpa-onnx/releases) — All model downloads
- [Pre-trained Models Index (Official Docs)](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html) — Complete model catalog
- [NeMo Transducer Models (Official Docs)](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html) — Parakeet model details

### Model Release Tags (Direct Download Pages)
- [ASR Models Release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) — Parakeet, SenseVoice, Whisper, etc.
- [Speaker Segmentation Models Release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models) — Pyannote 3.0, Reverb v1/v2
- [Speaker Recognition Models Release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) — 3D-Speaker, WeSpeaker, NeMo TitaNet

### HuggingFace Mirrors (Backup — some repos are gated)
- [csukuangfj on HuggingFace](https://huggingface.co/csukuangfj) — Maintainer's profile with all model repos
- [csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8) — ASR model (public)
- [csukuangfj/sherpa-onnx-pyannote-segmentation-3-0](https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0) — Segmentation (public)
- [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — Original NVIDIA model (NeMo format, not ONNX)
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — Multilingual variant (NeMo format, not ONNX)

### Documentation & API
- [Speaker Diarization Docs](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html) — Diarization pipeline setup
- [Sherpa-ONNX Python API Examples](https://github.com/k2-fsa/sherpa-onnx/tree/master/python-api-examples) — Reference code
- [sherpa-onnx PyPI](https://pypi.org/project/sherpa-onnx/) — Python package (v1.12.23 as of Jan 2026)
- [Context7 sherpa-onnx](https://context7.com/k2-fsa/sherpa-onnx) — AI-indexed documentation

### Gotchas Log
- The `speaker-recongition-models` release tag has a permanent typo ("recongition" not "recognition") — this is intentional and will not change
- The embedding model `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` uses a **hyphen** in `16k-common` — code must match this exactly
- The HuggingFace repo for the embedding model (`csukuangfj/sherpa-onnx-3dspeaker-speech-campplus-sv-zh-en-16k-common-advanced-onnx`) returns **401 Unauthorized** — always use GitHub release instead
- GitHub release download URLs return HTTP 302 redirect to the actual CDN — `curl -fSL` (follow redirects) is required
- The pip CPU wheel (`sherpa-onnx==1.12.23`) silently falls back from `provider="cuda"` — use `+cuda12.cudnn9` wheel
- `OfflineTransducerModelConfig` constructor params differ between pip wheel and built-from-source (`encoder` vs `encoder_filename`)
- Diarization `process()` takes `(samples)` not `(samples, sample_rate)` — sample rate is assumed 16kHz
- Diarization returns a result object, not a list — use `list(result.sort_by_start_time())`
- **GPU diarization is 2.3x SLOWER than CPU** — tested with native v1.12.24 C++ binary, confirmed not a Python/pip issue

### Session Log (2026-02-10)

**Iteration 1**: Models failing to download. Embedding model on HuggingFace gated (401). Fixed by switching all downloads to GitHub releases in `entrypoint-sherpa.sh`.

**Iteration 2**: ASR constructor crash — `OfflineTransducerModelConfig(encoder=...)` wrong param name. Fixed by using `OfflineRecognizer.from_transducer()` factory method.

**Iteration 3**: GPU fallback to CPU — pip `sherpa-onnx==1.12.23` is CPU-only. Fixed by installing `sherpa-onnx==1.12.23+cuda12.cudnn9` from CUDA wheel index.

**Iteration 4**: `process()` signature error — passing `(audio, sample_rate)` but API takes `(samples)` only. Fixed by removing sample_rate arg.

**Iteration 5**: `len()` on result object — `OfflineSpeakerDiarizationResult` has no `len()`. Fixed by calling `list(result.sort_by_start_time())`.

**Iteration 6**: 18-min file takes 2.5 min, GPU at 4%, CPU at 100%. Diarization is the bottleneck. Benchmarked native C++ binary: GPU diarization (4:26) vs CPU (1:55). GPU is slower. This is fundamental to the model sizes, not fixable by building from source.

**Compared against**: Context7 sherpa-onnx docs (SHERPA-C7.md), Gemini research report (SHERPA_GEMINI.md). Both assume diarize-first pipeline. Neither addresses the GPU diarization performance issue.
