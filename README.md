# MVP-Echo Studio

GPU-accelerated batch audio transcription service with speaker diarization. Built for local deployment on NVIDIA GPUs.

## What It Does

Upload audio files (MP3, WAV, FLAC, etc.) and get back:
- **Accurate transcription** via NVIDIA NeMo Parakeet TDT 0.6B
- **Speaker labels** via Pyannote speaker-diarization-3.1
- **Word-level timestamps** for precise alignment
- **Multiple export formats**: SRT, VTT, TXT, JSON

**Performance**: 17 minutes of audio transcribed in ~15 seconds on RTX 3090.

## Current Status

**v0.1.0** - ✅ **Fully Working**
- Core transcription + diarization pipeline operational
- FastAPI backend with OpenAI-compatible API (`/v1/audio/transcriptions`)
- React frontend with upload/view/export
- Docker deployment with GPU passthrough
- Interactive feature mockup (`mockup.html`) showing all planned features

## Quick Start

```bash
# 1. Prerequisites
# - NVIDIA GPU (CUDA 11.8+)
# - Docker + nvidia-container-toolkit
# - HuggingFace account (for pyannote gated model)

# 2. Setup
cp .env.example .env
# Edit .env and add your HF_TOKEN=hf_xxx

# 3. Build and run
docker compose up -d --build

# 4. Test
curl -X POST -F "file=@test.mp3" -F "diarize=true" \
  http://localhost:8001/v1/audio/transcriptions

# 5. Open frontend
# http://localhost:8001
```

## Architecture

```
[React Frontend] → [FastAPI Backend] → [GPU Processing Pipeline]
                                         ├─ Audio Conversion (ffmpeg)
                                         ├─ Transcription (NeMo Parakeet TDT 0.6B)
                                         └─ Diarization (Pyannote 3.1)
```

**Tech Stack**:
- **Backend**: Python 3.12, FastAPI, NVIDIA NeMo, Pyannote
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Container**: NVIDIA NeMo 25.04 base image (includes CUDA, PyTorch)
- **GPU**: RTX 3090 (24GB VRAM), ~4-6GB used during inference

## Key Features

### Working (v0.1.0)
- ✅ Audio transcription with word-level timestamps
- ✅ Speaker diarization (who said what)
- ✅ OpenAI-compatible API endpoint
- ✅ Multi-format audio support (MP3, WAV, FLAC, M4A, OGG)
- ✅ Long audio handling (auto-chunked at 500s)
- ✅ Export: SRT, VTT, TXT, JSON
- ✅ React web UI (upload, view, export)

### Planned (v0.2.0 - Phase 1)
- 📋 API key authentication (simple JSON file)
- 📋 Paragraph grouping (speaker-aware)
- 📋 Audio player with click-to-seek timestamps
- 📋 Speaker timeline visualization
- 📋 Speaker statistics (talk time %, word count)
- 📋 Transcript search with highlighting
- 📋 Post-processing: filler removal, find/replace, profanity filter
- 📋 Custom speaker labels ("Speaker 0" → "Alice")
- 📋 Diarization hints (min/max/exact speaker count)

### Planned (v0.3.0 - Infrastructure)
- 📋 Redis infrastructure (job queue + rate limiting + pattern storage)
- 📋 Multi-file concurrent upload (background jobs)
- 📋 PII redaction (regex-based: SSN, CC, email, phone + user-uploadable patterns)
- 📋 WebSocket progress updates

## Documentation

- **[CURRENT.md](./CURRENT.md)** - Comprehensive session context, feature roadmap, implementation plan
- **[QUICKSTART.md](./QUICKSTART.md)** - 5-minute setup guide
- **[mockup.html](./mockup.html)** - Interactive demo of all planned features (open in browser)

## Project Structure

```
mvp-echo-studio/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── api.py               # /v1/audio/transcriptions endpoint
│   ├── transcription.py     # NeMo model wrapper
│   ├── audio.py             # ffmpeg conversion, chunking
│   ├── diarization/         # Pyannote pipeline wrapper
│   ├── torchaudio_compat/   # Pure-Python torchaudio shim (ABI fix)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React app
│   │   └── components/      # Upload, settings, transcript viewer
│   └── package.json
├── Dockerfile               # Multi-stage: Node + NeMo base image
├── docker-compose.yml       # Service definition, GPU, ports
├── entrypoint.sh            # Startup script (GPU check, model load)
├── mockup.html              # Interactive feature demo
└── CURRENT.md               # Detailed implementation plan
```

## API Example

```bash
# Basic transcription
curl -X POST -F "file=@audio.mp3" http://localhost:8001/v1/audio/transcriptions

# With speaker diarization
curl -X POST \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "response_format=verbose_json" \
  http://localhost:8001/v1/audio/transcriptions

# Response
{
  "text": "Good morning everyone...",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "Good morning everyone.", "speaker": "SPEAKER_0"},
    {"start": 3.4, "end": 6.8, "text": "Thank you for joining.", "speaker": "SPEAKER_0"}
  ],
  "language": "en",
  "duration": 91.2
}
```

## Technical Highlight: torchaudio ABI Fix

**Problem**: NVIDIA NeMo ships a custom PyTorch build (`2.7.0a0+hash`) with incompatible C++ ABI. All PyPI torchaudio wheels fail with undefined symbol errors.

**Solution**: Created a pure-Python torchaudio shim (`backend/torchaudio_compat/`) that implements the subset pyannote-audio needs:
- Audio I/O via soundfile
- Transforms (Resample, MFCC, MelSpectrogram) as torch.nn.Module
- Kaldi compliance functions (pure torch, copied from torchaudio v2.7.0)

**Result**: Zero C++ dependencies, zero ABI issues, pyannote works perfectly.

## Development

```bash
# View logs
docker compose logs -f mvp-scribe

# Rebuild after code changes
docker compose down && docker compose up -d --build

# Run tests (TODO)
docker exec mvp-scribe pytest backend/tests/

# Interactive shell
docker exec -it mvp-scribe bash
```

## Performance

| Audio Length | Transcription Time | RTF (Real-Time Factor) | VRAM Usage |
|--------------|-------------------|------------------------|------------|
| 3.4s | 0.25s | 0.07 | 4.2 GB |
| 17 min | 15s | 0.015 | 5.8 GB |
| 60 min (est) | ~50s | 0.014 | 6.5 GB |

RTF < 0.02 means it's **50x faster than real-time** on RTX 3090.

## Known Issues

- Parakeet TDT 0.6B is English-only (use whisper-large-v3 for other languages)
- Long audio (>500s) chunked which may cut mid-sentence
- Pyannote degrades with >10 speakers (use `num_speakers` hint)
- No streaming support (batch processing only)

## Next Steps

See [CURRENT.md](./CURRENT.md) for detailed roadmap. Top priorities:

1. **API key authentication** (simple, LAN bypass)
2. **Paragraph grouping** (speaker-aware, silence-based)
3. **Frontend features**: audio player, timeline, stats, search
4. **Redis infrastructure** for multi-file upload
5. **PII redaction** (regex-based, user-uploadable patterns)

## Contributing

This is a proof-of-concept / MVP project. See CURRENT.md for implementation tasks.

## License

TBD

## Credits

- **Models**: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- **Base Image**: NVIDIA NeMo Framework 25.04
- **torchaudio shim kaldi.py**: Adapted from [pytorch/audio v2.7.0](https://github.com/pytorch/audio/blob/v2.7.0/src/torchaudio/compliance/kaldi.py) (BSD-2-Clause)
