# RunPod Serverless Deployment

Deploy MVP-Echo Scribe as a RunPod serverless GPU worker. The handler wraps the existing `TranscribeAudioUseCase` — no backend code changes required.

## Build

Build from the **project root** (not from `runpod/`):

```bash
docker build -f runpod/Dockerfile -t mvp-scribe-runpod .
```

## Local GPU Test

```bash
docker run --rm --gpus all \
  -v runpod-cache:/runpod-volume \
  -e HF_TOKEN=$HF_TOKEN \
  mvp-scribe-runpod
```

The handler auto-detects local mode (no `RUNPOD_POD_ID` env var) and processes `test_input.json`, printing the result to stdout.

## Push to Docker Hub

```bash
docker tag mvp-scribe-runpod your-dockerhub-user/mvp-scribe-runpod:latest
docker push your-dockerhub-user/mvp-scribe-runpod:latest
```

## RunPod Setup

### 1. Create a Network Volume

- Region: same as your endpoint (e.g., US-TX-3)
- Size: 20 GB (holds NeMo + Pyannote models)
- Mount path: `/runpod-volume`

### 2. Create a Template

| Setting | Value |
|---|---|
| Container Image | `your-dockerhub-user/mvp-scribe-runpod:latest` |
| Container Disk | 20 GB |
| Volume Mount | `/runpod-volume` |
| Environment Variables | `HF_TOKEN=hf_your_token` |

### 3. Create a Serverless Endpoint

| Setting | Recommended Value |
|---|---|
| Template | (select template from step 2) |
| GPU Type | RTX 3090 / RTX 4090 / A5000 (8GB+ VRAM) |
| Active Workers | 0 (scale to zero) |
| Max Workers | 3 |
| Idle Timeout | 5 seconds |
| Execution Timeout | 600 seconds |

## API Request Format

### Using audio URL

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.mp3",
      "diarize": true,
      "response_format": "verbose_json"
    }
  }'
```

### Using base64-encoded audio

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_base64": "'$(base64 -w0 audio.mp3)'",
      "filename": "audio.mp3",
      "diarize": true,
      "response_format": "verbose_json"
    }
  }'
```

### Async (for long audio)

Use `/run` instead of `/runsync`, then poll `/status/{JOB_ID}`:

```bash
# Submit
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/long-meeting.mp3"}}'

# Poll
curl "https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Input Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_url` | string | — | URL to download audio from (mutually exclusive with `audio_base64`) |
| `audio_base64` | string | — | Base64-encoded audio data |
| `filename` | string | `audio.wav` | Filename hint (used with `audio_base64`) |
| `diarize` | bool | `true` | Enable speaker diarization |
| `response_format` | string | `verbose_json` | Output format: `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `language` | string | auto | Language code (e.g., `en`) |
| `num_speakers` | int | auto | Exact number of speakers (hint for diarization) |
| `min_speakers` | int | — | Minimum expected speakers |
| `max_speakers` | int | — | Maximum expected speakers |
| `word_timestamps` | bool | `false` | Include word-level timestamps |
| `timestamps` | bool | `false` | Include segment timestamps |
| `include_diarization_in_text` | bool | `true` | Include speaker labels in text output |
| `detect_paragraphs` | bool | `false` | Group segments into paragraphs |
| `paragraph_silence_threshold` | float | `0.8` | Silence gap (seconds) for paragraph breaks |
| `remove_fillers` | bool | `false` | Remove filler words (um, uh, etc.) |
| `min_confidence` | float | `0.0` | Filter segments below this confidence |
| `find_replace` | string | — | JSON array of `{"find": "...", "replace": "..."}` rules |
| `text_rules` | string | — | JSON array of text transformation rules |
| `speaker_labels` | string | — | JSON object mapping `speaker_0` → custom names |
| `detect_entities` | bool | `false` | Run spaCy NER on paragraphs |
| `detect_topics` | bool | `false` | Extract topics from transcript |
| `detect_sentiment` | bool | `false` | Run VADER sentiment analysis on paragraphs |
| `chunk_duration` | int | `500` | Audio chunk size in seconds |

## Response Format

```json
{
  "text": "Speaker 1: And so my fellow Americans...",
  "language": "en",
  "duration": 11.0,
  "model": "parakeet-tdt-0.6b-v2",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 11.0,
      "text": "And so my fellow Americans...",
      "speaker": "speaker_0"
    }
  ]
}
```

On failure, the response includes an `error` key:

```json
{
  "error": "Input must include 'audio_url' or 'audio_base64'"
}
```

## Troubleshooting

### Models re-download on every cold start

Ensure the network volume is mounted at `/runpod-volume`. The model cache env vars (`HF_HOME`, `TORCH_HOME`, `NEMO_CACHE_DIR`) point there. Verify the volume is in the same region as your endpoint.

### OOM (Out of Memory) errors

- Use a GPU with 8GB+ VRAM (RTX 3090 recommended)
- Reduce `chunk_duration` (default 500s) for very long audio
- Peak VRAM usage is ~6GB for transcription + diarization combined

### Diarization not working

- Verify `HF_TOKEN` is set — Pyannote requires a HuggingFace token with access to `pyannote/speaker-diarization-3.1`
- Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1

### Import errors on startup

- The torchaudio shim is installed during the Docker build. If you see `ModuleNotFoundError: No module named 'torchaudio'`, rebuild the image
- The pyannote version checker is patched to handle NVIDIA's non-SemVer torch version strings
