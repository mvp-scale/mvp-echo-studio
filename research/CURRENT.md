# MVP-Echo Scribe - v0.3.0 Session Context

## Current State (2026-02-10)

**Status**: v0.3.0 — Hexagonal architecture refactor + Sherpa-ONNX backend
**Prior work**: See `ARCHIVE.md` for v0.2.0 feature completion log
**Benchmark**: 18-minute multi-speaker audio file, segment-level timestamps + speaker labels

---

## Goals

| Priority | Goal | Definition |
|----------|------|------------|
| P0 | **Accuracy** | Transcription quality parity — Sherpa-ONNX output matches NeMo output for same audio |
| P0 | **Speed** | 18-minute audio transcribed in <10s (target: 5s). Current NeMo baseline: ~15s on RTX 3090 |
| P1 | **Adaptability** | Engine-swappable architecture — switch NeMo/Sherpa via env var, no code changes |
| P1 | **Reliability** | Both stacks pass identical test suite against same benchmark audio |
| P2 | **Security** | Auth works across LAN, Tailscale, and Cloudflare tunnel deployments |
| P2 | **Resilience** | Graceful fallback if preferred engine fails to load (config error, missing model) |
| P3 | **Multi-user** | Concurrent request handling (deferred — requires Redis job queue, v0.4.0) |

---

## Phase 1: Hexagonal Foundation ✅

**Goal**: Establish all ports (ML + infrastructure), domain models, and directory structure. Default adapters preserve current behavior. No external dependencies added — Redis adapters come in Phase 5.

### ML Ports (transcription engine boundary)

| Feature | Dependencies | Goals | Conf. | Status | Notes | Test Strategy |
|---------|-------------|-------|-------|--------|-------|---------------|
| **Domain models** | None | Adaptability | 🟢 | ✅ | `TranscriptSegment`, `DiarizationSegment`, `DiarizationResult` dataclasses in `domain/models.py`. Framework-agnostic. | Import domain models, verify fields match current segment data. |
| **TranscriptionPort ABC** | `domain/models.py` | Adaptability | 🟢 | ✅ | Abstract interface: `load()`, `transcribe()`, `model_name()`, `is_loaded()`. All adapters implement this. | ABC cannot be instantiated. Method signatures accept/return domain types. |
| **DiarizationPort ABC** | `domain/models.py` | Adaptability | 🟢 | ✅ | Abstract interface: `load()`, `diarize()`, `merge_with_transcription()`, `is_loaded()`. Speaker hint params on `diarize()`. | ABC verification. Speaker hint params forwarded. |
| **AudioProcessingPort ABC** | None | Adaptability | 🟢 | ✅ | Abstract interface: `convert_to_wav()`, `split_into_chunks()`. Shared by both engine stacks. | ABC verification. |

### Infrastructure Ports (scalability boundary)

| Feature | Dependencies | Goals | Conf. | Status | Notes | Test Strategy |
|---------|-------------|-------|-------|--------|-------|---------------|
| **JobQueuePort ABC** | `domain/models.py` | Multi-user | 🟢 | ✅ | `submit(func, *args) -> job_id`, `status(job_id)`, `result(job_id)`. Default: `SyncJobAdapter` runs inline. | Submit job, verify sync adapter returns result immediately. |
| **RateLimiterPort ABC** | None | Security | 🟢 | ✅ | `check(api_key) -> bool`, `remaining(api_key) -> int`. Default: `NoOpRateLimiter` always returns True. | Verify no-op never blocks. |
| **ProgressPort ABC** | None | Reliability | 🟢 | ✅ | `report(job_id, stage, progress, detail)`. Stages: converting, transcribing, diarizing, post_processing. Default: `LogProgressAdapter`. | Verify log lines emitted. |
| **KeyStorePort ABC** | None | Security | 🟢 | ✅ | `validate(key) -> bool`, `get_name(key) -> str`. Default: `JsonFileKeyStore`. | Validate known key returns True. |
| **UsagePort ABC** | None | Reliability | 🟢 | ✅ | `log(api_key, metadata)`. Default: `NoOpUsageAdapter` discards. | Verify no-op doesn't error. |

### Shared Foundation

| Feature | Dependencies | Goals | Conf. | Status | Notes | Test Strategy |
|---------|-------------|-------|-------|--------|-------|---------------|
| **Domain-to-DTO mappers** | `domain/models.py` | Reliability | 🟢 | ✅ | `mappers.py`: convert `TranscriptSegment` <-> `WhisperSegment`, preserve all fields. API response schema unchanged. | Round-trip domain -> DTO -> domain produces identical data. |
| **Directory scaffolding** | None | Adaptability | 🟢 | ✅ | `domain/`, `ports/`, `adapters/{nemo,sherpa,ffmpeg,local}/`, `use_cases/`. | All packages importable. |

---

## Phase 2: Extract NeMo Adapters ✅

**Goal**: Move existing NeMo/Pyannote code into adapter implementations. Pure refactor — current behavior preserved exactly. `api.py` slimmed to thin HTTP layer.

**Exit criteria**: `ENGINE=nemo docker compose up` produces identical output to current monolithic stack.

| Feature | Dependencies | Goals | Conf. | Status | Notes | Test Strategy |
|---------|-------------|-------|-------|--------|-------|---------------|
| **NeMoTranscriptionAdapter** | Phase 1, `transcription.py` | Accuracy, Adaptability | 🟢 | ✅ | `adapters/nemo/transcription.py`. Implements `TranscriptionPort`. Returns `list[TranscriptSegment]`. | Transcribe benchmark. Output must match current NeMo. |
| **PyannoteDiarizationAdapter** | Phase 1, `diarization/` | Accuracy, Adaptability | 🟢 | ✅ | `adapters/nemo/diarization.py`. Implements `DiarizationPort`. Uses `torchaudio_compat/` internally. | Diarize benchmark. Speaker labels match. |
| **FFmpegAudioAdapter** | Phase 1, `audio.py` | Adaptability | 🟢 | ✅ | `adapters/ffmpeg/audio.py`. Shared by both stacks. | Convert MP3/FLAC/M4A to WAV. Split 18min file. |
| **TranscribeAudioUseCase** | All ML + infra ports | Adaptability, Reliability | 🟢 | ✅ | `use_cases/transcribe.py`. Full pipeline: convert -> chunk -> transcribe (with progress) -> diarize -> post-process. | Full pipeline produces identical verbose_json. |
| **Slim api.py** | Use case | Adaptability | 🟢 | ✅ | Thin HTTP layer: parse form params, call use case, return response. No model imports, no orchestration. | All existing curl commands produce identical responses. |
| **Adapter factory** | All adapters, `config.py` | Adaptability | 🟢 | ✅ | `ENGINE` env var selects ML adapters. `INFRA` env var selects infra adapters. Factory functions in `config.py`. | `ENGINE=nemo INFRA=local` loads defaults. `ENGINE=invalid` shows clear error. |

---

## Phase 3: Sherpa-ONNX Adapters 🔄

**Goal**: Implement second engine stack. Same API, same output format, different runtime. Target: 18min < 10s (ideally 5s), <4GB VRAM.

**Exit criteria**: Both engines transcribe benchmark with comparable accuracy. Sherpa: <10s, <4GB VRAM.

**Status (2026-02-10)**: Functional but diarization bottleneck identified. ASR works on GPU (fast). Diarization proven to be CPU-optimal (GPU is 2.3x slower). Pipeline restructure needed. See `SHERPA_IMPLEMENTATION.md` for full session log and benchmarks.

### What Works
- `./start.sh sherpa` builds and runs cleanly
- Models auto-download from GitHub releases (no HuggingFace auth needed)
- ASR via `OfflineRecognizer.from_transducer()` — GPU, fast, correct output
- Diarization via `OfflineSpeakerDiarization.process()` — functional, produces speaker segments
- Short files (51s) transcribe with diarization in ~7s
- 18-min files transcribe but bottlenecked by diarization (~2 min)

### What Was Discovered
- **GPU diarization is slower than CPU**: Benchmarked native C++ binary (v1.12.24) — GPU 4:26 vs CPU 1:55 for 18-min file. Segmentation (5.7MB) and embedding (27MB) models are too small for GPU to help.
- **Building from source won't fix it**: Pre-compiled GPU release binary shows same behavior.
- **Pipeline needs restructuring**: Current: diarize first → ASR on clips. Proposed: ASR on full audio (GPU, seconds) → diarize (CPU, ~2 min) → merge timestamps.
- **5-second target for 18 min requires ASR-only or async diarization**

### Current Implementation

| Feature | Status | Notes |
|---------|--------|-------|
| **SherpaTranscriptionAdapter** | ✅ Working | `adapters/sherpa/transcription.py`. Python API with `from_transducer()` factory + `decode_streams()` batch GPU. |
| **Diarization (built-in)** | ✅ Working (CPU) | `OfflineSpeakerDiarization` with Pyannote segmentation 3.0 + 3D-Speaker CAM++ embedding. GPU proven slower — runs on CPU. |
| **Dockerfile.sherpa** | ✅ Working | CUDA 12.6 runtime + `sherpa-onnx==1.12.23+cuda12.cudnn9` pip wheel. ~4-5GB image. No PyTorch. |
| **entrypoint-sherpa.sh** | ✅ Working | Downloads 3 models from GitHub releases, verifies all 6 files, starts uvicorn. |
| **docker-compose integration** | ✅ Working | `./start.sh sherpa` selects Dockerfile.sherpa + ENGINE=sherpa. |

### Next Steps
1. **Restructure pipeline**: ASR first (GPU) → diarization (CPU, concurrent or background) → merge
2. **Benchmark ASR-only speed**: How fast is `decode_streams()` on full 18-min audio without diarization?
3. **Explore Reverb Diarization v2**: 242MB model — larger model may benefit from GPU
4. **Consider async diarization**: Return transcript immediately, add speaker labels when ready

---

## Phase 4: Auth & Deployment Hardening ✅

**Goal**: Fix auth for real-world deployments. Ensure both stacks deploy cleanly to remote hosts.

| Feature | Dependencies | Goals | Conf. | Status | Notes | Test Strategy |
|---------|-------------|-------|-------|--------|-------|---------------|
| **Tailscale CGNAT bypass** | `auth.py` | Security | 🟢 | ✅ | Added `ip_network("100.64.0.0/10")` to `PRIVATE_NETWORKS`. | Access from 100.65.x.x. No 401. Public IPs still require key. |
| **Cloudflare tunnel auth** | `auth.py`, `api.ts` | Security | 🟢 | ✅ | Backend: checks `CF-Connecting-IP` header first, then `x-forwarded-for`. Frontend: stores API key in localStorage, sends as `Bearer` on all requests. | Access through tunnel. Auth succeeds. |
| **Engine info in /health** | `/health` endpoint | Reliability | 🟢 | ✅ | Response includes `engine`, `model_name`, `model_id`, VRAM usage. | `curl /health` shows engine. |
| **Frontend API key support** | `api.ts` | Security | 🟢 | ✅ | `getStoredApiKey()` / `setStoredApiKey()` in `api.ts`. All API calls include `Authorization: Bearer {key}` header when key is set. | Set key in localStorage, verify header sent. |
| **Startup health validation** | Adapter factory | Resilience | 🟢 | 📋 | Load adapters, smoke test with 1s silent audio. Missing model -> clean error. | Start with missing files. Verify clear error. |
| **Remote deployment test** | All phases | Reliability | 🟢 | 📋 | Deploy Sherpa to second machine. Full pipeline through Cloudflare tunnel. | End-to-end on remote host. |

---

## Phase 5: Redis Adapter Swap (v0.4.0)

**Goal**: Replace default no-op/sync infrastructure adapters with Redis-backed implementations. Ports already defined in Phase 1 — this phase is purely adapter implementation. Set `INFRA=redis` to activate.

**Exit criteria**: `INFRA=redis docker compose up` runs with Redis. `INFRA=local` still works without Redis. Same API, same frontend.

| Feature | Dependencies | Goals | Conf. | Status | Notes | Test Strategy |
|---------|-------------|-------|-------|--------|-------|---------------|
| **Redis container** | `docker-compose.yml` | Multi-user | 🟢 | 📋 | Add `redis:7-alpine` service. `redis-data` volume for persistence. `--appendonly yes`. Only started when `INFRA=redis`. | `docker compose up redis`. Verify connectivity from backend. |
| **RedisJobAdapter** | `JobQueuePort`, Redis, `rq`/`arq` | Multi-user | 🟢 | 📋 | Implements `JobQueuePort`. `submit()` enqueues to Redis via rq. `status()` polls job state. `result()` returns completed output. Upload returns job ID; client polls `/jobs/{id}`. | Upload 3 files concurrently. Verify parallel processing. Poll status endpoint. |
| **RedisRateLimiter** | `RateLimiterPort`, Redis | Security | 🟢 | 📋 | Implements `RateLimiterPort`. Redis INCR with TTL. Key: `ratelimit:{api_key}:{hour}`. 100/hr guest, unlimited named. 429 + `Retry-After`. | Spam guest key, verify 429 after 100. Named keys bypass. Counter resets hourly. |
| **RedisUsageAdapter** | `UsagePort`, Redis | Reliability | 🟢 | 📋 | Implements `UsagePort`. Sorted set `usage:{api_key}`. Stores: file_size, duration, features, processing_time. Optional daily JSON export. | Query usage for key. Verify all fields. Check Redis memory. |
| **WebSocketProgressAdapter** | `ProgressPort`, Redis pubsub, FastAPI WS | Reliability | 🟡 | 📋 | Implements `ProgressPort`. Worker publishes to Redis pubsub channel per job. Frontend subscribes via WebSocket. Shows: "Converting...", "Chunk 2/5", "Diarizing...". | Upload 20min file, verify real-time updates. Test disconnect/reconnect. |
| **`/jobs/{id}` endpoint** | `RedisJobAdapter` | Multi-user | 🟢 | 📋 | New API endpoint. Returns job status, progress, and result when complete. Only active when `INFRA=redis`. | Poll non-existent job -> 404. Poll in-progress -> status. Poll complete -> result. |
| **Model caching** | `/models` Docker volume | Speed | 🟢 | ✅ | Already working. Models cached in volume. ~10s GPU load on startup. No re-download. Not a port — Docker volume strategy. | Restart container. No re-download. Check load time. |

---

## Performance Targets

| Metric | NeMo Baseline (RTX 3090) | Sherpa-ONNX Actual (2026-02-10) | Notes |
|--------|--------------------------|--------------------------------|-------|
| 18min ASR only | ~15s | ~5-10s (GPU batch decode) | `decode_streams()` on full audio — meets target |
| 18min diarization | ~5s (Pyannote+PyTorch) | **~2 min (CPU)** | Sherpa diarization slower — no PyTorch, CPU-only optimal. GPU is 2.3x slower. |
| 18min end-to-end | ~20s | **~2 min** | Bottlenecked by diarization, not ASR |
| Peak VRAM | ~6GB | <4GB | Confirmed lower |
| Container image | ~20GB | ~4-5GB | No PyTorch. Pure ONNX Runtime. |
| Model download | ~8GB | ~500MB | 3 models from GitHub releases, no auth |
| Cold start | ~10s | ~2s model load | ONNX loads faster |
| WER delta | Baseline | TBD — short file output looks correct | Needs formal comparison |

---

## Architecture

### Directory Structure

```
backend/
├── domain/
│   └── models.py                  # TranscriptSegment, DiarizationSegment, DiarizationResult
│
├── ports/
│   ├── transcription.py           # TranscriptionPort ABC
│   ├── diarization.py             # DiarizationPort ABC
│   ├── audio.py                   # AudioProcessingPort ABC
│   ├── job_queue.py               # JobQueuePort ABC
│   ├── rate_limiter.py            # RateLimiterPort ABC
│   ├── progress.py                # ProgressPort ABC
│   ├── key_store.py               # KeyStorePort ABC
│   └── usage.py                   # UsagePort ABC
│
├── adapters/
│   ├── nemo/
│   │   ├── transcription.py       # NeMoTranscriptionAdapter
│   │   └── diarization.py         # PyannoteDiarizationAdapter (shared by both stacks)
│   │
│   ├── sherpa/
│   │   └── transcription.py       # SherpaTranscriptionAdapter (VAD+ASR pipeline)
│   │
│   ├── ffmpeg/
│   │   └── audio.py               # FFmpegAudioAdapter (shared by both stacks)
│   │
│   └── local/                     # Default infra adapters (no external deps)
│       ├── sync_job.py            # SyncJobAdapter (inline processing)
│       ├── noop_rate_limiter.py   # NoOpRateLimiter (unlimited)
│       ├── log_progress.py        # LogProgressAdapter (stdout)
│       ├── json_key_store.py      # JsonFileKeyStore (current auth.py logic)
│       └── noop_usage.py          # NoOpUsageAdapter (discards)
│
├── use_cases/
│   └── transcribe.py              # TranscribeAudioUseCase (orchestration, all ports injected)
│
├── api.py                         # FastAPI routes (thin, delegates to use cases)
├── models.py                      # Pydantic DTOs for API request/response
├── mappers.py                     # Domain <-> DTO conversions
├── config.py                      # Configuration + adapter factory (ENGINE + INFRA)
├── auth.py                        # Auth middleware (LAN + Tailscale + Cloudflare bypass)
├── main.py                        # Entry point (unchanged)
│
├── post_processing.py             # Unchanged (operates on WhisperSegment DTOs)
├── entity_detection.py            # Unchanged (CPU-only)
├── sentiment_analysis.py          # Unchanged (CPU-only)
│
└── torchaudio_compat/             # Unchanged (internal to Pyannote adapter)
```

### Adapter Selection

```bash
# NeMo stack (default)
docker compose up -d --build

# Sherpa-ONNX stack
DOCKERFILE=Dockerfile.sherpa ENGINE=sherpa docker compose up -d --build

# Verify engine
curl http://localhost:20301/health | jq .engine
```

---

## Known Issues

### Auth: Tailscale + Cloudflare Tunnel ✅ Fixed
Added `100.64.0.0/10` (Tailscale CGNAT) to `PRIVATE_NETWORKS`. Backend now checks `CF-Connecting-IP` header for Cloudflare tunnel traffic. Frontend sends Bearer token from localStorage.

### Sherpa-ONNX Diarization Quality ✅ Resolved
Decision: Hybrid approach. Both stacks use Pyannote 3.1 for diarization. Sherpa-ONNX built-in diarization was unproven, so we keep the known-good Pyannote pipeline. Trade-off: Sherpa container includes PyTorch (~10-12GB instead of ~2GB), but diarization quality is identical to NeMo stack.

### Sherpa-ONNX Timestamp Format ✅ Resolved
Sherpa adapter uses VAD+GPU WebSocket pipeline: Silero VAD (CPU, pip sherpa-onnx) provides speech segment boundaries, GPU C++ WebSocket server transcribes each segment. Timestamps derived from VAD boundaries. Pre-built GPU binaries from sherpa-onnx GitHub releases.

### Sherpa WebSocket Server 300s Limit ✅ Fixed
**Symptom**: `Max utterance length is configured to 300 seconds, received length is 500 seconds. Payload is too large!`
**Fix**: Added `--max-utterance-length=600` to `sherpa-onnx-offline-websocket-server` in `entrypoint-sherpa.sh` line 63.

### Pyannote use_auth_token on Sherpa Stack ✅ Fixed
**Symptom**: `hf_hub_download() got an unexpected keyword argument 'use_auth_token'`
**Root cause**: pyannote-audio 3.3.2 passes deprecated kwarg. huggingface_hub >= 0.24 removed it.
**Fix**: Pinned `huggingface_hub<0.24` in Dockerfile.sherpa before installing pyannote.

### Sherpa VAD API Version Compat ✅ Fixed
**Symptom**: VAD always reported "No VAD segments"
**Root cause**: `vad.empty` and `vad.front` are methods in v1.12.23, not properties. Code did `while not self._vad.empty:` which is `while not <method>:` → always False.
**Fix**: Callable check in adapter: `vad.empty() if callable(vad.empty) else vad.empty`

---

**Last Updated**: 2026-02-10
