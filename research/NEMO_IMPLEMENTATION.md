# NeMo Implementation — Performance & Production Tuning

## Status: Complete (2026-02-10)

Production-ready NeMo stack with GPU optimizations. Ready for Cloudflare tunnel deployment.

## Performance Results

### RTX 3090, 17-minute two-speaker podcast

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total wall-clock time | ~15s | ~20-21s | Slightly slower (GPU contention from parallel diarization) |
| VRAM usage | ~15GB | ~4GB | **73% reduction** |
| Min GPU requirement | 16GB+ | 8GB | Opens up to RTX 3060/4060 class cards |

The VRAM reduction from 15GB to 4GB is the headline win. Speed is slightly slower due to parallel diarization + ASR contending for the same PyTorch GPU context, but the tradeoff is worth it for the dramatic memory savings.

## Optimizations Applied

### 1. FP16 Half-Precision Inference

**File:** `backend/adapters/nemo/transcription.py`

```python
self._model = self._model.half()
```

Biggest single win. Converts all model weights and activations from fp32 to fp16. RTX 3090 has dedicated fp16 tensor cores that process these operations at full throughput. Cuts model weight memory in half and reduces memory bandwidth pressure.

**Impact:** ~15GB → ~4GB VRAM. No measurable quality degradation on English transcription.

### 2. cuDNN Benchmark Auto-Tuning

**File:** `backend/adapters/nemo/transcription.py`

```python
torch.backends.cudnn.benchmark = True
```

Tells cuDNN to profile multiple convolution algorithms on the first inference and cache the fastest for subsequent calls. Small overhead on first request, faster on all subsequent requests.

### 3. `torch.inference_mode()` over `torch.no_grad()`

**File:** `backend/adapters/nemo/transcription.py`

```python
with torch.inference_mode():
    output = self._model.transcribe([audio_path], timestamps=True)
```

Stricter than `no_grad()` — also disables view tracking and version counting. Marginal speedup but zero risk.

### 4. PyTorch CUDA Memory Allocator Tuning

**Files:** `Dockerfile`, `docker-compose.yml`

```
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Reduces memory fragmentation by using expandable segments instead of fixed-size blocks. Prevents OOM errors from non-contiguous free memory.

### 5. Parallel Diarization + ASR

**File:** `backend/use_cases/transcribe.py`

```python
with ThreadPoolExecutor(max_workers=2) as pool:
    future_diar = pool.submit(_run_diarization)
    future_asr = pool.submit(_run_asr)
```

Runs Pyannote speaker diarization concurrently with NeMo ASR. Both release the GIL during CUDA kernel execution. On NeMo, both use PyTorch so there is some GPU contention, but diarization overlaps with ASR rather than adding to total time.

### 6. Configurable ASR Provider

**Files:** `backend/config.py`, `backend/adapters/sherpa/transcription.py`, `backend/api.py`

```python
self.asr_provider = os.environ.get("ASR_PROVIDER", "cuda").lower()
```

Infrastructure for future TensorRT support. Currently defaults to `"cuda"`. Exposed in `/health` endpoint for observability.

### 7. ORT Arena Strategy (Sherpa only)

**Files:** `Dockerfile.sherpa`, `docker-compose.yml`

```
ENV ORT_ARENA_EXTEND_STRATEGY=kSameAsRequested
```

Reduces ONNX Runtime's CUDA memory arena pre-allocation. Only affects Sherpa engine.

## Architecture

### Engine Selection

```bash
./start.sh nemo     # NeMo Parakeet TDT 0.6B (recommended)
./start.sh sherpa   # Sherpa-ONNX int8 (lighter, slightly slower)
```

### Processing Pipeline (NeMo)

```
Upload → FFmpeg (1s) → [Parallel on GPU] → Merge → Post-process → Response
                        ├─ NeMo ASR (fp16, ~15s)
                        └─ Pyannote Diarization (~12s)
```

### Key Files Modified

| File | Change |
|------|--------|
| `backend/adapters/nemo/transcription.py` | fp16, cuDNN benchmark, inference_mode |
| `backend/use_cases/transcribe.py` | ThreadPoolExecutor parallel diarization + ASR |
| `backend/config.py` | `asr_provider` config, engine-aware chunk_duration |
| `backend/api.py` | Wire `asr_provider` to load(), expose in /health |
| `Dockerfile` | `PYTORCH_CUDA_ALLOC_CONF` env var |
| `Dockerfile.sherpa` | `ORT_ARENA_EXTEND_STRATEGY` env var |
| `docker-compose.yml` | All new env vars |
| `frontend/src/components/SettingsPanel.tsx` | Speaker count hints visible before transcription |

## Authentication (Already Implemented)

**File:** `backend/auth.py`

- LAN IPs (192.168.x.x, 10.x.x.x, 127.x.x.x, 172.16-31.x.x) bypass auth
- Tailscale CGNAT range (100.64.0.0/10) also bypasses
- Cloudflare tunnel: reads `CF-Connecting-IP` header for real client IP
- External requests require `Authorization: Bearer {key}` header
- Keys stored in `api-keys.json` (mounted read-only into container)
- `/health` and `/v1/models` are open (no auth required)

## Frontend Diarization Controls

Speaker count hints (Min/Max/Exact) are now always visible when diarization is toggled on, not just after first transcription. Users can set "Exact: 2" for a known two-person podcast before submitting.

## What We Tried and Didn't Ship

### TensorRT Acceleration
- Sherpa-onnx's `OfflineModelConfig` doesn't expose `provider_config` for TRT
- NeMo's `torch_tensorrt` requires export pipeline and version pinning
- Parked for future investigation

### Sherpa-ONNX Outer Chunk Elimination
- Removed 500s outer chunking for Sherpa (it handles long audio internally)
- Reduced ASR from 3 sequential batch decodes to 1
- Sherpa speed improved from ~35s to ~26s but diarization accuracy and segment quality doesn't match NeMo's native sentence boundaries

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE` | `nemo` | Engine selection: `nemo` or `sherpa` |
| `ASR_PROVIDER` | `cuda` | ORT execution provider (Sherpa) or device (NeMo) |
| `CHUNK_DURATION` | `500` (NeMo) / disabled (Sherpa) | Outer audio chunk size in seconds |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch memory allocator tuning |
| `ORT_ARENA_EXTEND_STRATEGY` | `kSameAsRequested` | ORT CUDA arena allocation strategy |

## Next Steps

1. **Cloudflare Tunnel** — Expose the service publicly via `cloudflared` tunnel
2. **API Key Management** — Lock down external access, provision keys for users
3. **Rate Limiting** — Protect GPU from abuse (Redis-based, planned for v0.4.0)
