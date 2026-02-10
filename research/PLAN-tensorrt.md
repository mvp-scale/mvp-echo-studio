# Plan: TensorRT Execution Provider for Sherpa ASR

## Status: Ready to implement

## What We've Done (this session)

### Hybrid Sherpa ASR + Pyannote Diarization — COMPLETE
- Stripped Sherpa's built-in OfflineSpeakerDiarization (slow C++ sequential processing)
- Injected PyannoteDiarizationAdapter for both NeMo and Sherpa engines
- Sherpa adapter now does: sub-chunk audio at 80s → batch GPU decode → token timestamps → group into segments
- Pyannote handles diarization separately, timestamps merge by overlap
- Added PyTorch + Pyannote to Dockerfile.sherpa (torchaudio shim, torch.load patch for PyTorch 2.10)
- **Result: 35s for 17min 2-speaker podcast, 12GB VRAM, quality matches NeMo**

### Files modified (all committed-ready on disk):
- `backend/adapters/sherpa/transcription.py` — batch sub-chunk ASR, no VAD/diarization
- `backend/adapters/nemo/diarization.py` — torch.load monkey-patch for PyTorch 2.6+ weights_only
- `backend/config.py` — sherpa branch now creates PyannoteDiarizationAdapter
- `backend/use_cases/transcribe.py` — removed Sherpa-specific chunking bypass, unified diarization
- `backend/api.py` — minor log message update
- `Dockerfile.sherpa` — added PyTorch cu126, pyannote 3.3.2, torchaudio shim, deps
- `entrypoint-sherpa.sh` — removed segmentation/embedding downloads, simplified to ASR-only

### Current performance breakdown (17min audio):
| Step | Duration | Notes |
|------|----------|-------|
| FFmpeg convert + split | ~0s | 3x 500s chunks |
| Pyannote diarization | 12s | GPU embeddings + CPU clustering |
| ASR chunk 1 (500s → 7x80s streams) | 11s | batch GPU decode |
| ASR chunk 2 (500s → 7x80s streams) | 10s | batch GPU decode |
| ASR chunk 3 (33s → 1 stream) | 1s | batch GPU decode |
| **Total** | **~35s** | Sequential diarization + ASR |

### Key issues found:
- Parakeet TDT encoder max attention window: ~100s (1250 frames at 12.5fps). 500s single-stream fails.
- ONNX Runtime pre-allocates ~11.7GB VRAM arena regardless of actual model needs
- 400% CPU during inference = ORT's 4 CPU threads + Pyannote clustering

---

## Next Step: TensorRT Execution Provider

### Research findings (thoroughly validated):

**1. sherpa-onnx natively supports TensorRT**
- `provider="trt"` is a valid option (implemented June 2024, PRs #921 and #1130)
- Encoder runs on TRT EP, decoder/joiner automatically fall back to CUDA EP
- sherpa-onnx pre-built GPU wheels include `libonnxruntime_providers_tensorrt.so`
- Just needs TensorRT runtime libs on the system (`libnvinfer.so`, etc.)

**2. Docker changes needed**
- Add `pip install tensorrt-lean-cu12` (~500MB) to Dockerfile.sherpa
- Set `provider="trt"` in SherpaTranscriptionAdapter
- Add TRT engine cache directory (persistent volume, first run compiles 5-15 min)
- Add env vars: `ORT_TENSORRT_ENGINE_CACHE_ENABLE=1`, `ORT_TENSORRT_CACHE_PATH=/models/trt_cache`

**3. Expected performance**
- Sherpa's own benchmarks: 15-30% encoder speedup (TRT vs CUDA EP)
- Encoder is >95% of compute time, so this is meaningful
- ASR: 22s → ~15-18s estimated
- First inference: 5-15 minutes (TRT compiles optimized kernels for the encoder)
- Subsequent: cached engines load in seconds

**4. Parallel diarization + ASR (separate optimization)**
- Diarization and ASR are independent — can run concurrently in threads
- Would give: max(12s, 16s) ≈ 16s instead of 12s + 16s = 28s
- Threading in use_cases/transcribe.py with concurrent.futures

**5. Why NeMo is still faster (for context)**
- NeMo uses label-looping algorithm (67% decoder compute reduction)
- CUDA Graphs with conditional nodes (eliminates kernel launch overhead)
- bfloat16 autocasting (eliminates AMP overhead)
- These are PyTorch-specific, not available in ONNX Runtime
- NVIDIA's RTFx=3386 uses Riva NIM + batch 128 + FP8 on datacenter GPUs

### TRT configuration for sherpa-onnx:
```python
# In SherpaTranscriptionAdapter.load():
self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=os.path.join(self._model_dir, "encoder.int8.onnx"),
    decoder=os.path.join(self._model_dir, "decoder.int8.onnx"),
    joiner=os.path.join(self._model_dir, "joiner.int8.onnx"),
    tokens=os.path.join(self._model_dir, "tokens.txt"),
    model_type="nemo_transducer",
    provider="trt",  # Changed from "cuda"
    num_threads=4,
)
```

Note: sherpa-onnx also exposes TensorrtConfig for fine-tuning:
```python
# sherpa_onnx.TensorrtConfig fields:
# trt_max_workspace_size=2147483647 (~2GB)
# trt_fp16_enable=True
# trt_engine_cache_enable=True
# trt_engine_cache_path="/models/trt_cache"
# trt_timing_cache_enable=True
# trt_timing_cache_path="/models/trt_cache"
```

Need to check if from_transducer() exposes provider_config parameter, or if we need env vars.

### Dockerfile.sherpa additions:
```dockerfile
# After sherpa-onnx install, before web stack:
RUN pip3 install --no-cache-dir tensorrt-lean-cu12

# Add TRT cache env vars:
ENV ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
ENV ORT_TENSORRT_CACHE_PATH=/models/trt_cache
ENV ORT_TENSORRT_FP16_ENABLE=1
ENV ORT_TENSORRT_MAX_WORKSPACE_SIZE=4294967296
```

### Implementation steps:
1. Add `tensorrt-lean-cu12` to Dockerfile.sherpa
2. Change `provider="cuda"` to `provider="trt"` in transcription.py
3. Add TRT env vars to Dockerfile.sherpa and docker-compose.yml
4. Add `mkdir -p /models/trt_cache` to entrypoint-sherpa.sh
5. Test: first run will take 5-15 min (engine compilation), subsequent runs use cache
6. Benchmark: compare ASR time with TRT vs CUDA EP
7. (Optional) Implement parallel diarization + ASR in transcribe.py

### Hard-won dependency fixes (don't lose these):
- `pyannote-audio==3.3.2` must be installed with `--no-deps` to avoid torch conflicts
- `torch_audiomentations` must be installed with `--no-deps` (its chain pulls real torchaudio)
- `torch_pitch_shift` must also be `--no-deps` (depends on torchaudio)
- Leaf deps to install explicitly: `julius`, `primePy`, `sentencepiece`, `hyperpyyaml`
- `speechbrain` must be `--no-deps` (also depends on torchaudio)
- `matplotlib` is required by pyannote at runtime (not in its declared deps)
- `semver>=3.0.0` and `tensorboardX>=2.6` are required by pyannote
- torchaudio shim MUST be installed AFTER all pip installs (pip can overwrite it)
- `huggingface_hub<0.24` pin required (pyannote 3.3.2 uses deprecated kwarg)
- PyTorch 2.10 `torch.load` defaults `weights_only=True`; lightning passes `weights_only=None` which ORT treats as True. Fix: monkey-patch in diarization adapter to force `False` when `None`
- All of these are already implemented in the current Dockerfile.sherpa and diarization.py on disk

### Open questions to verify during implementation:
- Does sherpa-onnx's from_transducer() accept provider_config for TRT settings, or only env vars?
- Does tensorrt-lean-cu12 provide all needed .so files, or do we need the full tensorrt-cu12?
- Will the int8 ONNX model (encoder.int8.onnx) work with TRT EP, or does TRT need the fp32/fp16 version?
- What TensorRT version does tensorrt-lean-cu12 install, and is it compatible with ORT bundled in sherpa-onnx 1.12.23?

### Version compatibility (critical):
- sherpa-onnx 1.12.23 bundles ORT ~1.22-1.23
- ORT 1.22 needs TensorRT 10.9
- ORT 1.23 needs TensorRT 10.x
- tensorrt-lean-cu12 latest = TRT 10.14 (should be backward compatible)
- Our base: nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 (CUDA 12.6, cuDNN 9)
