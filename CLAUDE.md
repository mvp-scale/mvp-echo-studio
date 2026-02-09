# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MVP-Echo Studio is a GPU-accelerated batch audio transcription service with speaker diarization. It uses NVIDIA NeMo Parakeet TDT 0.6B for transcription and Pyannote speaker-diarization-3.1 for speaker identification. Performance: ~15s to transcribe 17min audio on RTX 3090.

**Tech Stack:**
- Backend: Python 3.12, FastAPI, NVIDIA NeMo, Pyannote
- Frontend: React 18, TypeScript, Vite, Tailwind CSS
- Container: NVIDIA NeMo 25.04 base image with CUDA and PyTorch
- GPU: Requires NVIDIA GPU with 8GB+ VRAM (RTX 3090 recommended)

## Development Commands

### Docker Operations
```bash
# Build and start the container
docker compose up -d --build

# View logs
docker compose logs -f mvp-scribe

# Restart after code changes
docker compose down && docker compose up -d --build

# Interactive shell for debugging
docker exec -it mvp-scribe bash

# Check GPU memory usage
docker exec mvp-scribe nvidia-smi
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Development server (with hot reload)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Testing the API
```bash
# Health check
curl http://localhost:20301/health

# Basic transcription
curl -X POST -F "file=@audio.mp3" http://localhost:20301/v1/audio/transcriptions

# With speaker diarization
curl -X POST \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "response_format=verbose_json" \
  http://localhost:20301/v1/audio/transcriptions

# Access from LAN (replace IP)
curl -X POST -F "file=@audio.mp3" -F "diarize=true" \
  http://192.168.1.10:20301/v1/audio/transcriptions
```

## Architecture

### High-Level Flow
```
[React Frontend] → [FastAPI Backend] → [GPU Processing Pipeline]
                                         ├─ Audio Conversion (ffmpeg via pydub)
                                         ├─ Transcription (NeMo Parakeet TDT 0.6B)
                                         └─ Diarization (Pyannote 3.1)
```

### Backend Architecture

**Key Modules:**
- `main.py` - Application entry point, starts Uvicorn server
- `api.py` - FastAPI app factory, defines `/v1/audio/transcriptions` endpoint
- `transcription.py` - NeMo model wrapper for transcription
- `audio.py` - Audio preprocessing (ffmpeg conversion, chunking at 500s)
- `diarization/__init__.py` - Pyannote pipeline wrapper for speaker identification
- `models.py` - Pydantic models for request/response schemas
- `config.py` - Environment configuration loader
- `auth.py` - Authentication middleware (LAN bypass + Bearer token validation)
- `torchaudio_compat/` - Pure-Python torchaudio shim (solves ABI compatibility issue)

**Processing Pipeline:**
1. Upload audio file via multipart form data
2. Convert to WAV format using ffmpeg (via `audio.py`)
3. Split into 500s chunks if longer (to handle memory constraints)
4. Transcribe each chunk with NeMo Parakeet TDT 0.6B
5. Apply timestamp offsets for multi-chunk audio
6. Run speaker diarization on full audio (if enabled)
7. Merge diarization speaker labels into transcription segments
8. Format response (text, JSON, SRT, VTT)

### Frontend Architecture

**Key Files:**
- `App.tsx` - Main application component, manages state and orchestration
- `api.ts` - API client for backend communication
- `types.ts` - TypeScript type definitions

**Components:**
- `UploadZone.tsx` - Drag-and-drop file upload
- `TranscriptViewer.tsx` - Displays transcription results
- `SpeakerSegment.tsx` - Individual speaker segment display
- `AudioPlayer.tsx` - Audio playback with timestamp sync
- `SearchBar.tsx` - Search within transcript
- `ExportBar.tsx` - Export to various formats
- `HealthIndicator.tsx` - Backend health status display
- `ProgressBar.tsx` - Progress indicator for processing
- `SpeakerBadge.tsx` - Speaker label display component
- `Layout.tsx` - Page layout wrapper

### Critical Architectural Detail: torchaudio ABI Compatibility

**Problem:** NVIDIA NeMo ships a custom PyTorch build (`2.7.0a0+hash`) with incompatible C++ ABI. All PyPI torchaudio wheels fail with undefined symbol errors.

**Solution:** `backend/torchaudio_compat/` is a pure-Python shim that implements the subset pyannote-audio needs:
- Audio I/O via soundfile
- Transforms (Resample, MFCC, MelSpectrogram) as torch.nn.Module
- Kaldi compliance functions (pure torch, copied from torchaudio v2.7.0)

**Important:** When modifying dependencies or PyTorch versions, verify the torchaudio shim still works by testing diarization functionality.

## Authentication System

**API Key System:**
- Keys stored in `backend/api-keys.json` (loaded at startup)
- LAN requests (192.168.x.x, 10.x.x.x, 127.0.0.1) bypass authentication
- External requests require `Authorization: Bearer {key}` header
- Middleware in `auth.py` validates requests

**Key Schema:**
```json
{
  "keys": [
    {"key": "sk-corey-2026", "name": "Corey (dev)", "active": true},
    {"key": "sk-guest1-{hash}", "name": "Guest 1", "active": true}
  ]
}
```

## API Endpoints

### POST /v1/audio/transcriptions
OpenAI-compatible transcription endpoint.

**Form Parameters:**
- `file` (required) - Audio file (MP3, WAV, FLAC, M4A, OGG)
- `model` - Model identifier (default: "whisper-1")
- `response_format` - Output format: "json", "text", "srt", "vtt", "verbose_json" (default: "json")
- `diarize` - Enable speaker diarization (default: true)
- `include_diarization_in_text` - Include speaker labels in text output
- `language` - Language code (optional, auto-detected)
- `timestamps` - Include segment timestamps
- `word_timestamps` - Include word-level timestamps (if supported by model)

**Response (verbose_json):**
```json
{
  "text": "Full transcript...",
  "language": "en",
  "duration": 91.2,
  "model": "parakeet-tdt-0.6b-v2",
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Good morning everyone.",
      "speaker": "speaker_0"
    }
  ]
}
```

### GET /health
Health check endpoint with GPU status.

### GET /v1/models
List available models (OpenAI-compatible).

## Configuration

**Environment Variables:**
- `HF_TOKEN` - HuggingFace token (required for pyannote gated model)
- `MODEL_ID` - NeMo model identifier (default: nvidia/parakeet-tdt-0.6b-v2)
- `PORT` - Server port (default: 8001)
- `CHUNK_DURATION` - Audio chunk duration in seconds (default: 500)
- `ENABLE_DIARIZATION` - Enable diarization by default (default: true)
- `INCLUDE_DIARIZATION_IN_TEXT` - Include speaker labels in text (default: true)
- `TEMP_DIR` - Temporary file directory (default: /tmp/parakeet)
- `HF_HOME` - HuggingFace cache directory
- `TORCH_HOME` - PyTorch cache directory
- `NEMO_CACHE_DIR` - NeMo cache directory

**Volume Mounts:**
- `scribe-models:/models` - Persistent model cache (prevents re-downloading)
- `./api-keys.json:/data/api-keys.json:ro` - API keys (read-only mount)

## Common Development Tasks

### Adding a New Feature

1. **Backend changes:**
   - Update `models.py` with new request/response fields
   - Add processing logic in appropriate module
   - Add API parameter to `api.py` endpoint
   - Test with curl

2. **Frontend changes:**
   - Add UI controls in relevant component
   - Update `api.ts` with new parameters
   - Update `types.ts` with new TypeScript types
   - Test in browser

3. **Always rebuild container after backend changes:**
   ```bash
   docker compose down && docker compose up -d --build
   ```

### Debugging GPU Issues

```bash
# Check GPU visibility from host
nvidia-smi

# Check GPU visibility inside container
docker exec mvp-scribe nvidia-smi

# Verify PyTorch sees GPU
docker exec mvp-scribe python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory allocation
curl http://localhost:20301/health | jq .gpu_memory
```

### Testing Diarization

```bash
# Verify pyannote imports work (tests torchaudio shim)
docker exec mvp-scribe python3 -c "from pyannote.audio import Pipeline; print('OK')"

# Test with multi-speaker audio
curl -X POST \
  -F "file=@meeting.mp3" \
  -F "diarize=true" \
  -F "response_format=verbose_json" \
  http://localhost:20301/v1/audio/transcriptions | jq '.segments[0].speaker'
```

## Known Issues and Limitations

- **Parakeet TDT is English-only** - For other languages, consider whisper-large-v3 (slower)
- **Long audio chunking** - 500s chunks may cut mid-sentence, affects readability
- **Pyannote speaker limits** - Degrades with >10 speakers, use `num_speakers` hint if known
- **Synchronous processing** - Server blocks during transcription, no concurrent requests yet
- **No persistent storage** - Transcripts only returned in API response, not stored

## Implementation Roadmap

**Current Version: v0.1.0** - Fully working baseline

**Planned for v0.2.0 (Phase 1):**
- Paragraph grouping (speaker-aware, silence-based)
- Audio player with click-to-seek timestamps
- Speaker timeline visualization
- Speaker statistics (talk time %, word count)
- Transcript search with highlighting
- Post-processing: filler removal, find/replace, profanity filter
- Custom speaker labels
- Diarization hints (min/max/exact speaker count)
- Export formats: SRT, VTT, TXT, JSON

**Planned for v0.3.0 (Infrastructure):**
- Redis infrastructure (job queue + rate limiting + pattern storage)
- Multi-file concurrent upload (background jobs)
- PII redaction (regex-based with user-uploadable patterns)
- WebSocket progress updates

See `CURRENT.md` for detailed implementation plan with test strategies and acceptance criteria.

## Reference Documentation

- `README.md` - Project overview, quick start, feature list
- `QUICKSTART.md` - 5-minute setup guide
- `CURRENT.md` - Comprehensive session context, detailed roadmap, implementation tables
- `mockup.html` - Interactive feature demo (open in browser to see all planned features)
- `SETUP_GIT.md` - Git repository setup guide

## Important Notes for AI Assistants

1. **Never modify torchaudio_compat/ without understanding the ABI issue** - This shim is critical for pyannote compatibility
2. **Always test diarization after PyTorch/NeMo version changes** - Dependencies are tightly coupled
3. **Long audio handling uses chunking** - Be aware of 500s chunk boundaries when debugging timestamps
4. **Authentication has LAN bypass** - Don't require API keys for local network requests
5. **Models are cached in volumes** - First startup downloads ~8GB, subsequent starts are fast
6. **GPU memory is precious** - Monitor VRAM usage, current peak is ~6GB for transcription + diarization
7. **The mockup.html file shows the target UI** - Reference this when implementing frontend features
8. **Export format consistency** - Ensure speaker labels appear correctly in SRT/VTT/TXT exports
