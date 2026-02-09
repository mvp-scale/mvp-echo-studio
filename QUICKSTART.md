# Quick Start - MVP-Echo Studio

**Goal**: Get transcription working in 5 minutes.

## 1. Prerequisites Check

```bash
# NVIDIA GPU check
nvidia-smi
# Should show your GPU (RTX 3090, etc.)

# Docker check
docker --version
# Should be 20.10+ or newer

# NVIDIA Container Toolkit check
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
# Should show GPU inside container
```

## 2. Get HuggingFace Token

1. Sign up at https://huggingface.co (free)
2. Go to https://huggingface.co/settings/tokens
3. Create new token (read access)
4. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1

## 3. Configure

```bash
cd mvp-echo-studio

# Create .env file
echo "HF_TOKEN=hf_your_actual_token_here" > .env
```

## 4. Build and Start

```bash
docker compose up -d --build
```

**Wait 10-15 minutes** for first build (downloads ~8GB base image + models).

Watch logs:
```bash
docker compose logs -f mvp-scribe
```

Look for:
```
✓ GPU detected: NVIDIA GeForce RTX 3090
✓ Model loaded on GPU
✓ Diarization pipeline ready
✓ Uvicorn running on http://0.0.0.0:8001
```

## 5. Test

```bash
# Health check
curl http://localhost:8001/health

# Transcribe (replace with your audio file)
curl -X POST \
  -F "file=@test.mp3" \
  -F "diarize=true" \
  -F "response_format=verbose_json" \
  http://localhost:8001/v1/audio/transcriptions \
  | jq .
```

## 6. Use Frontend

Open browser: **http://localhost:8001**

- Drag and drop audio file
- Toggle "Diarization" on
- Click "Transcribe"
- View results with speaker labels

## What's Next?

Read **[CURRENT.md](./CURRENT.md)** for:
- Full feature roadmap (Phase 1, 2, 3)
- Implementation plan with detailed tables
- API schema
- Testing strategy

Try **[mockup.html](./mockup.html)** in browser to see all planned features interactively.

## Common Issues

**"Permission denied" on .env**:
```bash
chmod 600 .env
```

**Models won't download**:
- Check HF_TOKEN is correct
- Verify license accepted on HuggingFace
- Check network/firewall

**Out of VRAM**:
- Close other GPU processes (check `nvidia-smi`)
- RTX 3090 with 24GB is recommended
- Minimum: 8GB VRAM

**Container won't start**:
```bash
# Check GPU available
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check logs
docker compose logs mvp-scribe

# Rebuild from scratch
docker compose down -v
docker compose up -d --build
```

## Files You Can Ignore

- `.env` - your secrets (gitignored)
- `image.png` - Deepgram screenshot for mockup reference
- `.claude/` - Claude Code project metadata
- `frontend/node_modules/` - npm packages

## Development Mode

If you're modifying code:

```bash
# Backend changes
docker compose down && docker compose up -d --build

# Frontend changes (if you set up hot reload)
cd frontend
npm install
npm run dev
```

For backend Python changes, you can mount local code as volume for faster iteration (see docker-compose.yml comments).

## Accessing from Other Machines

**On LAN**:

1. Find server IP:
```bash
hostname -I | awk '{print $1}'
# Example: 192.168.1.10
```

2. Expose port in docker-compose.yml:
```yaml
ports:
  - "20301:8001"
```

3. Access from other machines:
```bash
curl -X POST -F "file=@audio.mp3" -F "diarize=true" \
  http://192.168.1.10:20301/v1/audio/transcriptions
```

**Over Internet** (NOT recommended without SSL + auth):
- Setup nginx reverse proxy with SSL
- Enable API key authentication (Phase 1 feature)
- Add rate limiting (Phase 3 feature)

## Support

Check these in order:
1. **CURRENT.md** - Full documentation
2. **Container logs** - `docker compose logs -f`
3. **GPU status** - `nvidia-smi`
4. **Python imports** - `docker exec mvp-scribe python3 -c "import torchaudio; print('OK')"`

## Success Criteria

You know it's working when:
- ✅ Container starts without errors
- ✅ GPU shows up in logs
- ✅ Health endpoint returns 200
- ✅ Test transcription returns JSON with text
- ✅ Frontend uploads and displays transcript
- ✅ Speaker labels appear when diarization enabled

**Typical successful response** (with diarization):
```json
{
  "text": "full transcript...",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "Hello everyone", "speaker": "SPEAKER_0"}
  ],
  "language": "en",
  "duration": 45.2
}
```
