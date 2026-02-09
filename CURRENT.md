# MVP-Echo Studio - Session Context

## Current State (2026-02-09)

**Status**: ✅ **FULLY WORKING** - Transcription + Speaker Diarization on GPU
**Location**: `/home/corey/projects/mvp-echo-toolbar/mvp-echo-studio/`
**Container**: `mvp-scribe` running on RTX 3090 (24GB VRAM)
**Ports**: 8001 (internal), mapped to 20301 externally
**Mockup**: `mockup.html` - Interactive feature demo with all proposed features

---

## Executive Summary

**What We Have**: GPU-accelerated transcription service (NeMo Parakeet TDT 0.6B) + speaker diarization (Pyannote 3.1) running in Docker. OpenAI-compatible API. 15s to transcribe 17min audio.

**What We're Building Next**:

**v0.2.0 (Phase 1 - Core Features)**:
- API key authentication (simple JSON file, LAN bypass)
- Paragraph grouping (speaker-aware, silence-based)
- Audio player with click-to-seek timestamps (mockup already shows this)
- Speaker timeline, stats, search UI
- Post-processing: filler removal, find/replace, profanity filter
- Export: SRT, VTT, TXT, JSON
- Speaker hints for better diarization (min/max/exact count)
- Custom speaker labels (rename Speaker 0 → "Alice")

**v0.3.0 (Infrastructure + Multi-File Upload)**:
- **Redis**: Enables multi-file concurrent upload (background job queue) - **very compelling**
- Redis also handles: rate limiting, usage tracking, custom PII pattern storage
- WebSocket progress updates for long files
- Simple regex-based PII redaction (SSN, CC, email, phone + user-uploadable patterns)

**Key Architectural Decisions**:
1. **Redis for everything infrastructure**: Job queue (multi-file) + rate limiting + usage tracking + pattern storage. Single dependency, multiple benefits.
2. **Regex over ML for PII**: Simple, fast, user-configurable. No model downloads. Good enough for MVP.
3. **No heavy AI models for MVP**: Skip entity detection, sentiment until there's proven demand.

**Not Doing for MVP**:
- Channel-per-speaker, inline editing, per-word confidence, entity detection, HIPAA certification
- LLM features (future paid tier)

**Maybe (Discuss First)**:
- Sentiment analysis - need to justify value vs model download overhead

---

## Feature Roadmap

All features organized by implementation phase with detailed planning tables.

### Legend

**Confidence Levels:**
- 🟢 Green: Fully confident, clear implementation path
- 🟡 Yellow: Minor concerns, may need research or testing
- 🔴 Red: Needs significant research or dependencies unclear

**Status:**
- ✅ Available: Implemented and working
- 🔨 Ready: Backend plumbing exists, needs API exposure
- 📋 Planned: Clear spec, ready to implement
- 🔬 Research: Needs investigation

---

## Phase 1: Core Features & Quick Wins

**Goal**: Enhance existing transcription with post-processing and better UX. No new models, all local processing.

| Feature | Dependencies | Goals Alignment | Confidence | Status | Notes | Test Strategy |
|---------|-------------|-----------------|-----------|--------|-------|---------------|
| **API Key Validation** | `backend/api-keys.json` schema | Security, multi-user access | 🟢 | 📋 | Use existing `api-keys.json` format. Middleware checks Bearer token against active keys. | Create test keys, verify 401 on invalid, 200 on valid. Test LAN bypass logic. |
| **Diarization (Min/Max/Exact Speakers)** | Pyannote pipeline already supports params | Accuracy improvement | 🟢 | 🔨 | `Diarizer.diarize()` at `diarization/__init__.py:54` accepts `num_speakers`, just needs API exposure at `api.py:74` | Upload 2-person call, set `num_speakers=2`, verify better accuracy vs auto-detect |
| **Word-Level Timestamps** | NeMo `result.timestamp['word']` | Precision, word-hover UI | 🟢 | 🔨 | Already in model output, needs JSON response field `words: [...]` | Verify word timestamps align with segment timestamps. UI hover test. |
| **Profanity Filter** | `better-profanity` pip package | Content moderation | 🟢 | 📋 | Regex-based, runs on segment text post-transcription. Add toggle to API. | Upload audio with profanity, toggle on, verify asterisk replacement. |
| **Remove Filler Words** | None (pure regex) | Readability | 🟢 | 📋 | Strip "uh", "um", "like", "you know", "I mean", "sort of", "kind of" from `.text` | Upload conversational audio, toggle on, verify fillers removed. Manual review. |
| **Find & Replace** | None (pure regex) | Jargon correction, brand names | 🟢 | 📋 | Accept array of `{find, replace}` rules. Apply regex substitution per segment. | Test with "gonna→going to", verify case-insensitive, whole-word match only. |
| **Detect Paragraphs** | Silence gaps from existing timestamps | Readability, export quality | 🟢 | 📋 | Group consecutive same-speaker segments. Break on speaker change or silence > threshold (default 0.8s). | Upload 5min audio, verify paragraph grouping logical. Check SRT/VTT output. |
| **Detect Utterances** | Existing segment timestamps | Alternative grouping | 🟢 | 📋 | Like paragraphs but ignore speaker labels—pure silence-based grouping. | Upload multi-speaker audio, verify utterances group correctly regardless of diarization. |
| **Trim Audio (Start/End)** | ffmpeg already in `audio.py` | Long file processing | 🟢 | 📋 | Add `-ss {start}` and `-t {duration}` to ffmpeg conversion. Parse MM:SS format from API. | Upload 10min file, trim to 2:00-4:00, verify only that range transcribed. |
| **Confidence Threshold** | `WhisperSegment` has `no_speech_prob`, `avg_logprob` fields in `models.py` | Noise filtering | 🟢 | 🔨 | Filter segments where `1 - no_speech_prob < threshold` or `avg_logprob < threshold`. Fields exist but not used. | Upload audio with silence/noise, set threshold 0.7, verify low-confidence segments dropped. |
| **Custom Speaker Labels (UI)** | None (pure frontend) | UX, export quality | 🟢 | 📋 | Frontend-only: map `SPEAKER_0` → user-defined name in display + exports. | Rename speakers in UI, verify SRT/VTT/TXT use custom names. |

---

## Phase 2: Interactive Features (Frontend-Heavy)

**Goal**: Make the transcript viewer a full-featured workbench. All features work with existing backend data.

| Feature | Dependencies | Goals Alignment | Confidence | Status | Notes | Test Strategy |
|---------|-------------|-----------------|-----------|--------|-------|---------------|
| **Audio Playback with Timestamp Sync** | `<audio>` element, file already in browser | Core UX, verification workflow | 🟢 | 📋 | Create blob URL from uploaded file. Click segment → seek to `segment.start`. Highlight active segment as audio plays. | Upload audio, click timestamp, verify playback starts at correct moment. Verify active highlight follows playback. |
| **Speaker Timeline Visualization** | Segment timestamps | Visual overview, quick navigation | 🟢 | 📋 | SVG or CSS horizontal bar. Each segment is a colored block positioned by `(start/duration)*100%`. Click block → seek audio. | Verify timeline matches transcript. Click blocks to seek. Check color coding. |
| **Speaker Statistics** | Segment timestamps + text | Analytics, meeting insights | 🟢 | 📋 | Sum segment durations per speaker. Count words with `.split(/\s+/)`. Show talk time %, word count, talk-to-listen ratio. | Upload multi-speaker audio. Verify percentages sum to 100%. Verify word counts match. |
| **Transcript Search with Highlighting** | None (pure frontend) | Findability in long transcripts | 🟢 | 📋 | Regex search across all segment text. Highlight matches with `<mark>`. Prev/Next navigation with scroll-to-view. Ctrl+F shortcut. | Search "onboarding" in 60min transcript, verify all matches found. Test prev/next nav. |
| **Export Formats** | Existing segment data | Professional output | 🟢 | 📋 | Generate SRT, VTT, TXT, JSON, DOCX from segments/paragraphs. Use custom speaker labels if set. | Generate all 5 formats, verify timestamps, speaker labels, formatting correct. Open in VLC (SRT/VTT), Word (DOCX). |

---

## Phase 3: Optional Value-Add Features (Lower Priority)

**Goal**: Simple enhancements after core features are solid. Keep it minimal for MVP.

| Feature | Dependencies | Goals Alignment | Confidence | Status | Notes | Test Strategy |
|---------|-------------|-----------------|-----------|--------|-------|---------------|
| **PII Redaction (Regex-Based)** | None (pure regex) + Redis for pattern storage | Privacy, content filtering | 🟢 | 📋 | **Built-in patterns**: SSN (`\d{3}-\d{2}-\d{4}`), credit card (`\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}`), email, phone (US). **User-uploadable patterns**: POST JSON to `/redaction/patterns` with `[{"name": "Medical_Record", "pattern": "MR-\\d+", "replacement": "[MRN]"}]`. Store in Redis key `redaction:patterns`. Pre-compile regexes. Apply all to segment text. | Upload audio with SSN "123-45-6789", phone, email. Verify masked. Upload custom pattern JSON. Test persistence. Verify pre-compiled patterns used. |
| **Language Detection** | Check NeMo output OR `langdetect` (~10ms) | Informational tag | 🟢 | 📋 | **Only if free/trivial**. First check if Parakeet model outputs language field. If yes, pass through. If no, optionally run `langdetect.detect(text)` on full transcript. Skip if any complexity. | Check NeMo response structure. If language field exists, return it. Otherwise evaluate if worth adding langdetect. |
| **Sentiment Analysis** | `transformers` + model (~250MB download) | Call quality scoring | 🟡 | 🔬 | **Discuss value first**. Per-segment POSITIVE/NEGATIVE scoring. Use case: call center QA, customer satisfaction. Trade-off: downloads model, uses GPU memory. May not be worth it for MVP POC. | Pending conversation on use cases. If approved: upload customer service call, verify sentiment useful. Check GPU memory impact. |

---

## Infrastructure & Backend Improvements

| Task | Dependencies | Goals Alignment | Confidence | Status | Notes | Test Strategy |
|------|-------------|-----------------|-----------|--------|-------|---------------|
| **API Key Middleware** | FastAPI dependency injection, `backend/api-keys.json` | Security, access control | 🟢 | 📋 | Read JSON on startup into in-memory dict. Check `Authorization: Bearer {key}` header. Bypass for LAN IPs (192.168.x.x, 10.x.x.x, 127.0.0.1). Log key name on each request. Simple JSON file sufficient for MVP/POC scale. | Add test key, curl with valid/invalid tokens. Verify LAN bypass. Check logs show key name. Test inactive key rejection. |
| **Redis Infrastructure (Multi-Purpose)** | Redis server, `redis-py` client | Scalability foundation | 🟢 | 📋 | **Single Redis instance handles 3 use cases**: (1) Background job queue for concurrent uploads, (2) Rate limiting counters per API key, (3) Usage tracking/analytics. Consolidates infrastructure vs separate solutions. Add redis service to docker-compose. | Start Redis container, verify connectivity. Test job queue, rate limiting, usage logging independently. |
| **Background Job Queue (via Redis)** | Redis + `rq` or `arq` library | Multi-file upload, non-blocking API | 🟢 | 📋 | Upload returns job ID immediately. Client polls `/jobs/{id}` for status. Worker processes jobs from queue. Enables concurrent uploads without blocking. | Upload 3 files concurrently. Verify all process in parallel. Check job status endpoint. Test queue priority. |
| **Rate Limiting (via Redis)** | Redis counters | Abuse prevention, fair usage | 🟢 | 📋 | Use Redis INCR with TTL. Key: `ratelimit:{api_key}:{hour}`. Limit: 100 req/hour for guest keys, unlimited for named keys. Return 429 with `Retry-After` header. | Spam requests with guest key, verify 429 after 100. Check named keys bypass limit. Verify counter resets hourly. |
| **Usage Tracking (via Redis)** | Redis sorted sets or hash | Analytics, quota tracking | 🟢 | 📋 | Log to Redis: `usage:{api_key}` → sorted set with timestamp scores. Store: file_size, duration, features_used, processing_time. Optionally export to JSON daily for archival. | Query usage for a key. Verify all fields logged. Test daily export. Check Redis memory usage. |
| **WebSocket Progress Updates** | FastAPI WebSocket, job queue integration | UX for long files | 🟡 | 📋 | Job worker emits progress events to Redis pubsub. Frontend WebSocket subscribes to job channel. Shows real-time progress: "Converting...", "Transcribing chunk 2/5", "Diarizing...". | Upload 20min file, verify progress updates in real-time. Test disconnect/reconnect. Check message ordering. |
| **Model Caching Strategy** | Existing NeMo/Pyannote models in `/models` volume | Startup time, memory | 🟢 | ✅ | Models already cached in Docker volume. On startup: ~10s to load to GPU. Keep loaded in memory. | Restart container, check load time. Verify no re-download from HuggingFace. |

---

## Redis Infrastructure Deep Dive

**Context**: For MVP, we want multi-file upload support (background job queue). If we're adding Redis for that, we can consolidate rate limiting and usage tracking into the same infrastructure instead of building separate solutions.

### Option 1: Redis for Everything (Recommended)

**Single Redis container handles**:
1. **Job Queue** (`rq` library)
2. **Rate Limiting** (Redis INCR with TTL)
3. **Usage Tracking** (Redis sorted sets or hashes)

**Docker Compose Addition**:
```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: mvp-redis
    restart: unless-stopped
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    networks:
      - mvp-network

volumes:
  redis-data:
```

**Backend Changes**:
```python
# backend/redis_client.py
import redis
from rq import Queue

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
job_queue = Queue(connection=redis_client)

# Rate limiting
def check_rate_limit(api_key: str, limit: int = 100) -> bool:
    """Check if API key is under rate limit (requests per hour)."""
    key = f"ratelimit:{api_key}:{datetime.now().strftime('%Y%m%d%H')}"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, 3600)  # 1 hour TTL
    return count <= limit

# Usage tracking
def log_usage(api_key: str, metadata: dict):
    """Log request metadata for analytics."""
    timestamp = datetime.now().timestamp()
    redis_client.zadd(f"usage:{api_key}", {json.dumps(metadata): timestamp})
    # Optional: daily export to JSON file for long-term storage
```

**Pros**:
- Enables multi-file upload (killer feature)
- Proper rate limiting (atomic, distributed-safe)
- Usage analytics in real-time
- Pub/sub for WebSocket progress updates
- Easy to scale (add more workers)

**Cons**:
- Adds Redis dependency (~15MB image)
- Slightly more complex docker-compose
- Redis persistence needs volume management

**Resource Usage**:
- Redis memory: < 100MB for typical usage (thousands of jobs)
- CPU: negligible
- Disk: append-only file for persistence

---

### Option 2: Hybrid Approach (Simpler for POC)

**Use Redis only for job queue**, keep rate limiting + usage tracking simple:

- **Job Queue**: Redis + `rq` (enables multi-file upload)
- **Rate Limiting**: In-memory dict, reset on server restart (acceptable for POC)
- **Usage Tracking**: Append to JSON file (simple, no query capability)

**Pros**:
- Still get multi-file upload
- Simpler code (no Redis integration for rate/usage)
- Can add Redis rate limiting later if needed

**Cons**:
- Rate limits reset on server restart
- Usage tracking is append-only (no real-time queries)
- Miss out on Redis pub/sub for progress updates

---

### Option 3: No Redis (Simplest, Limited)

**Keep everything in-process**:
- **No background jobs**: Synchronous processing only (one upload at a time)
- **Rate Limiting**: In-memory dict with timestamps
- **Usage Tracking**: JSON file append

**Pros**:
- Zero external dependencies
- Simplest possible deployment
- Fine for single-user or low-traffic POC

**Cons**:
- No concurrent upload support
- Rate limits lost on restart
- No real-time usage analytics

---

### Recommendation: Start with Option 1 (Full Redis)

**Rationale**:
- Multi-file upload is **very compelling** (user's words)
- If we're adding Redis anyway, consolidating rate limiting + usage tracking is minimal extra work (< 2 hours)
- Sets up proper foundation for scaling beyond POC
- Redis is lightweight and well-understood

**Implementation Priority**:
1. **P0 (v0.2.0)**: API key middleware (no Redis needed)
2. **P1 (v0.3.0)**: Add Redis + job queue for multi-file upload
3. **P1 (v0.3.0)**: Add rate limiting via Redis (piggyback on existing infra)
4. **P3 (v0.4.0)**: Add usage tracking via Redis (nice-to-have analytics)

**Fallback**: If Redis proves problematic, can ship v0.2.0 without it (synchronous processing only), add in v0.3.0.

---

## PII Redaction Pattern Upload Workflow

**Goal**: Let users add domain-specific redaction patterns without code changes.

**Example Use Case**: Medical office wants to redact patient IDs and medical record numbers.

**Step 1: User Creates Pattern JSON**
```json
{
  "patterns": [
    {
      "name": "Medical_Record",
      "pattern": "MR-\\d{6}",
      "replacement": "[MRN]"
    },
    {
      "name": "Patient_ID",
      "pattern": "PT\\d{8}",
      "replacement": "[PATIENT_ID]"
    },
    {
      "name": "Account_Number",
      "pattern": "ACCT-[A-Z0-9]{10}",
      "replacement": "[ACCOUNT]"
    }
  ]
}
```

**Step 2: Upload Patterns**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-corey-2026" \
  -d @redaction-patterns.json \
  http://192.168.1.10:20301/redaction/patterns
```

**Response**:
```json
{"status": "success", "pattern_count": 3, "stored_in_redis": true}
```

**Step 3: Use in Transcription**
```bash
curl -X POST \
  -F "file=@medical-consultation.mp3" \
  -F "redact_pii=true" \
  http://192.168.1.10:20301/v1/audio/transcriptions
```

**Result**: All built-in patterns (SSN, email, phone, CC) + custom patterns (MR-*, PT*, ACCT-*) are applied. Transcript text shows:
- "Social Security [SSN]"
- "Medical record [MRN]"
- "Patient ID [PATIENT_ID]"
- "Account [ACCOUNT]"

**Pattern Management**:
- GET `/redaction/patterns` → list all active patterns (built-in + custom)
- POST `/redaction/patterns` → replace custom patterns (overwrites previous)
- DELETE `/redaction/patterns` → clear custom patterns (keep built-in only)

**Redis Storage**:
- Key: `redaction:patterns`
- Value: JSON array of pattern objects
- Persistence: Redis append-only file (survives container restart)
- Per-user patterns (future): `redaction:patterns:{api_key}`

---

## Phase 1 Features: Detailed Implementation Plan

### Audio Pre-Processing

| Feature | Backend Changes | Frontend Changes | API Parameters | Complexity | Test Criteria |
|---------|----------------|------------------|----------------|-----------|---------------|
| **Trim Audio** | `audio.py`: Add `trim_start`, `trim_end` params to `convert_audio_to_wav()`. Pass to ffmpeg as `-ss {start} -t {duration}`. | Add time inputs (MM:SS format) in upload settings. | `trim_start: str, trim_end: str` (optional) | 🟢 Low | Upload 10min file, trim to 1:30-3:00, verify output is 90s. Check timestamps start at 0. |

### Transcription Enhancements

| Feature | Backend Changes | Frontend Changes | API Parameters | Complexity | Test Criteria |
|---------|----------------|------------------|----------------|-----------|---------------|
| **Diarization Speaker Hints** | `diarization/__init__.py`: Pass `min_speakers`, `max_speakers`, `num_speakers` to `Pipeline.apply()`. Already supported by pyannote, just needs param forwarding. | Add 3 number inputs under Diarization toggle. | `min_speakers: int, max_speakers: int, num_speakers: int` (all optional) | 🟢 Low | Upload 3-person meeting, set `num_speakers=3`, compare accuracy vs auto-detect. Verify error if `min > max`. |
| **Word-Level Timestamps** | `transcription.py`: Check if `result.timestamp` has `word` key. If yes, add to `WhisperSegment` model. Return in `verbose_json` response. | Display word timestamps on hover in transcript viewer. | `word_timestamps: bool` (optional) | 🟡 Medium | Verify NeMo output structure first. If available, test word-level alignment accuracy. UI: hover word, see timestamp tooltip. |
| **Punctuation Control** | `transcription.py`: NeMo model outputs punctuated text by default. To disable, strip `.?!,;:` and lowercase. | Toggle: "Punctuation" (default on). | `punctuate: bool` (default true) | 🟢 Low | Toggle off, verify lowercase no-punctuation output. Toggle on, verify natural punctuation. |
| **Smart Format Control** | Similar to punctuation—output may need post-processing to convert "23%" back to "twenty three percent" if disabled. | Toggle: "Smart Format" (default on). | `smart_format: bool` (default true) | 🟡 Medium | Toggle off, verify "twenty three percent" instead of "23%". May need regex reverse-mapping. |

### Post-Processing

| Feature | Backend Changes | Frontend Changes | API Parameters | Complexity | Test Criteria |
|---------|----------------|------------------|----------------|-----------|---------------|
| **Detect Paragraphs** | New function in `transcription.py` or `diarization/__init__.py`. Logic: group consecutive segments by speaker, split on silence > threshold. Return `paragraphs: [{speaker, start, end, text}]` in response. | Toggle + silence threshold slider (0.2-5.0s). Display paragraph view by default. | `detect_paragraphs: bool, paragraph_silence_threshold: float` | 🟢 Low | Upload conversational audio. Verify paragraphs break on speaker change + long pauses. Check threshold slider changes grouping. |
| **Remove Filler Words** | Regex filter in post-processing. Pattern: `\b(uh|um|like|you know|i mean|sort of|kind of)\b` (case-insensitive). Apply to each segment `.text`. | Toggle: "Remove Filler Words". | `remove_fillers: bool` | 🟢 Low | Upload podcast audio (high filler word count). Toggle on, verify all fillers stripped. Check for false positives ("I like this" should not remove "like" if used as verb). |
| **Profanity Filter** | Install `better-profanity`. `profanity.censor(text)` per segment. | Toggle: "Profanity Filter". | `filter_profanity: bool` | 🟢 Low | Upload audio with profanity. Verify censoring with `***`. Test custom word lists if needed. |
| **Confidence Filter** | Filter `segments` where `1 - segment.no_speech_prob < min_confidence` OR `segment.avg_logprob < threshold`. Return only high-confidence segments. | Slider: "Min Confidence" (0.0-1.0). Show filtered segment count. | `min_confidence: float` (0-1) | 🟢 Low | Upload noisy audio or audio with silence. Set threshold 0.6, verify low-confidence segments removed. Check edge case: threshold=0 (no filtering). |
| **Find & Replace** | Accept JSON array of rules. Apply per segment: `re.sub(r'\b{find}\b', replace, text, flags=re.IGNORECASE)`. | UI: Add/remove rules dynamically. Each rule has Find/Replace inputs. | `find_replace: [{"find": str, "replace": str}]` (optional) | 🟢 Low | Add rule "gonna→going to". Verify replacement. Test regex special chars escaped. Test multiple rules applied in order. |
| **Custom Speaker Labels** | Optional: accept `speaker_labels: {"SPEAKER_0": "Alice", ...}` and apply server-side to all outputs. OR keep frontend-only. | Input fields for each detected speaker. Apply to transcript display + exports. | `speaker_labels: dict` (optional, server-side) OR frontend-only | 🟢 Low | Rename Speaker 0 to "Alice". Export SRT. Verify "[Alice]" appears instead of "[SPEAKER_0]". |
---


---

## API Schema Extensions

### Request (Form Data)

```python
class TranscriptionRequest(BaseModel):
    file: UploadFile  # Audio file (MP3, WAV, FLAC, etc.)

    # Existing
    diarize: bool = False
    response_format: Literal["text", "json", "verbose_json", "srt", "vtt"] = "json"

    # Phase 1: Audio Pre-Processing
    trim_start: Optional[str] = None  # "MM:SS" format
    trim_end: Optional[str] = None

    # Phase 1: Diarization Hints
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    num_speakers: Optional[int] = None

    # Phase 1: Transcription
    word_timestamps: bool = False
    punctuate: bool = True
    smart_format: bool = True

    # Phase 1: Post-Processing
    detect_paragraphs: bool = False
    paragraph_silence_threshold: float = 0.8
    detect_utterances: bool = False
    utterance_silence_threshold: float = 0.8
    remove_fillers: bool = False
    filter_profanity: bool = False
    min_confidence: float = 0.0  # 0.0-1.0
    find_replace: Optional[List[Dict[str, str]]] = None  # [{"find": "gonna", "replace": "going to"}]
    speaker_labels: Optional[Dict[str, str]] = None  # {"SPEAKER_0": "Alice", "SPEAKER_1": "Bob"}

    # Phase 3: Optional Features
    redact_pii: bool = False  # Regex-based PII masking (SSN, CC, email, phone + custom patterns)
    language_detection: bool = False  # Only if NeMo outputs it natively (otherwise skip)
```

### Response (verbose_json)

```json
{
  "text": "Full transcript text...",
  "language": "en",
  "duration": 91.2,
  "model": "nvidia/parakeet-tdt-0.6b-v2",

  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Good morning, everyone.",
      "speaker": "SPEAKER_0",  // if diarize=true
      "words": [  // if word_timestamps=true
        {"word": "Good", "start": 0.0, "end": 0.3, "confidence": 0.98},
        {"word": "morning,", "start": 0.32, "end": 0.7, "confidence": 0.95}
      ],
      "confidence": 0.96,  // avg segment confidence
      "no_speech_prob": 0.02
    }
  ],

  "paragraphs": [  // if detect_paragraphs=true
    {
      "speaker": "SPEAKER_0",
      "start": 0.0,
      "end": 6.8,
      "text": "Good morning, everyone. Thank you for joining...",
      "segment_count": 2
    }
  ],

  "utterances": [  // if detect_utterances=true
    {
      "start": 0.0,
      "end": 6.8,
      "text": "Combined text...",
      "speakers": ["SPEAKER_0"]  // may span multiple speakers
    }
  ],

  "statistics": {  // if diarize=true
    "speakers": {
      "SPEAKER_0": {"duration": 64.2, "percentage": 70.4, "word_count": 542},
      "SPEAKER_1": {"duration": 18.5, "percentage": 20.3, "word_count": 156},
      "SPEAKER_2": {"duration": 8.5, "percentage": 9.3, "word_count": 72}
    },
    "total_speakers": 3
  },

  "redactions": {  // if redact_pii=true
    "total_count": 4,
    "by_pattern": {
      "SSN": 1,
      "Email": 1,
      "Phone_US": 2
    }
  }
}
```

---

## Implementation Priority Matrix

**Focus**: MVP features with high value-to-effort ratio. Deprioritize nice-to-haves and future paid features.

| Feature | User Value | Implementation Effort | Priority | Target Release | Notes |
|---------|-----------|----------------------|----------|----------------|-------|
| **API Key Validation** | HIGH | Low (1-2 hours) | P0 | v0.2.0 | Foundation for multi-user access |
| **Diarization Speaker Hints** | HIGH | Low (30 min) | P0 | v0.2.0 | Huge accuracy improvement for known speaker counts |
| **Detect Paragraphs** | HIGH | Low (2-3 hours) | P0 | v0.2.0 | Essential for readable output |
| **Custom Speaker Labels** | HIGH | Low (1 hour frontend) | P0 | v0.2.0 | Professional exports |
| **Audio Playback + Timestamp Sync** | VERY HIGH | Medium (4-6 hours) | P0 | v0.2.0 | **Killer feature** - makes verification effortless |
| **Export Formats (SRT/VTT/TXT/JSON)** | HIGH | Medium (4-5 hours) | P0 | v0.2.0 | Professional output, required for usability |
| **Speaker Timeline Viz** | MEDIUM | Low (2 hours) | P1 | v0.2.0 | Great visual overview |
| **Speaker Statistics** | MEDIUM | Low (1 hour) | P1 | v0.2.0 | Meeting analytics value |
| **Transcript Search** | HIGH | Low (2-3 hours) | P1 | v0.2.0 | Essential for long transcripts |
| **Remove Filler Words** | MEDIUM | Low (1 hour) | P1 | v0.2.0 | Easy readability win |
| **Find & Replace** | MEDIUM | Low (2 hours) | P1 | v0.2.0 | Jargon/brand name correction |
| **Profanity Filter** | MEDIUM | Low (30 min) | P1 | v0.2.0 | Content moderation |
| **Redis Infrastructure** | MEDIUM | Medium (4-5 hours) | P1 | v0.3.0 | Enables job queue + rate limiting + usage tracking |
| **Background Job Queue** | HIGH | Medium (3-4 hours w/ Redis) | P1 | v0.3.0 | **Multi-file upload** - very compelling |
| **Word-Level Timestamps** | MEDIUM | Medium (2-3 hours + verification) | P2 | v0.3.0 | Nice-to-have, depends on NeMo output structure |
| **Trim Audio** | MEDIUM | Low (1-2 hours) | P2 | v0.3.0 | Useful for long files |
| **Confidence Filter** | LOW | Low (1 hour) | P2 | v0.3.0 | Niche use case (noisy audio) |
| **Language Detection** | LOW | Low (30 min if easy) | P2 | v0.3.0 | Only if trivial to add |
| **PII Redaction (Regex)** | MEDIUM | Low (2-3 hours) | P2 | v0.3.0 | Simple regex patterns + user-uploadable custom patterns stored in Redis |
| **Sentiment Analysis** | LOW | Medium (3-4 hours + model) | P3 | Future | Discuss value first - may not need for MVP |
| **WebSocket Progress** | MEDIUM | Medium (3-4 hours w/ Redis) | P3 | v0.4.0 | UX polish for long jobs |
| **Rate Limiting** | LOW | Low (1-2 hours w/ Redis) | P3 | v0.4.0 | Abuse prevention, low priority for POC |
| **Usage Tracking** | LOW | Low (2-3 hours w/ Redis) | P3 | v0.4.0 | Analytics, nice-to-have |

---

## Security & Access Control

### API Key System (Phase 1, P0)

**File**: `backend/api-keys.json`

**Schema**:
```json
{
  "keys": [
    {"key": "sk-corey-2026", "name": "Corey (dev)", "active": true},
    {"key": "sk-alex-2026", "name": "Alex", "active": true},
    {"key": "sk-guest1-{hash}", "name": "Guest 1", "active": true}
  ]
}
```

**Implementation**:
1. Load keys on FastAPI startup into in-memory dict
2. Middleware: Extract `Authorization: Bearer {key}` header
3. If source IP is LAN (192.168.x.x, 10.x.x.x, 127.0.0.1): **bypass** key check
4. If external IP: require valid key, return 401 if missing/invalid/inactive
5. Log key name with each request for analytics

**Test Plan**:
- Curl from LAN without key → should work
- Curl from external IP without key → 401 Unauthorized
- Curl with valid key → 200 OK, logs show key name
- Curl with inactive key → 401
- Curl with non-existent key → 401
- Restart server, verify keys reload correctly

---

## Performance Targets

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Transcription RTF (Real-Time Factor) | 0.015 (17min → 15s) | < 0.02 | Parakeet TDT is very fast |
| Diarization overhead | ~5s for 17min audio | < 10% of audio length | Acceptable for batch processing |
| Memory usage (peak) | 6GB VRAM | < 10GB | Leaves headroom for concurrent requests |
| API response time (3min audio) | ~3-4s | < 5s | Including upload, conversion, inference |
| Concurrent uploads supported | 1 (synchronous) | 3-5 (with job queue) | Future: need Redis + worker pool |

---

## Testing Strategy

### Unit Tests
- `test_audio.py`: Test ffmpeg conversion, chunking, trim, channel split
- `test_transcription.py`: Mock NeMo model, test segment formatting
- `test_diarization.py`: Mock pyannote output, test speaker merging logic
- `test_post_processing.py`: Test filler removal, find/replace, paragraph grouping

### Integration Tests
- `test_api.py`: Upload audio to `/v1/audio/transcriptions`, verify response schema
- Test each feature toggle independently (diarize on/off, paragraphs on/off, etc.)
- Test feature combinations (diarize + paragraphs + fillers)

### End-to-End Tests
- Upload real audio files (30s, 3min, 17min, 60min)
- Verify SRT/VTT playable in VLC
- Verify JSON parseable
- Test with various audio formats (MP3, WAV, FLAC, M4A)
- Test edge cases: mono vs stereo, low bitrate, background noise

### Performance Tests
- Benchmark transcription speed vs audio duration (plot RTF curve)
- VRAM usage profiling (transcription only, diarization only, both, +NER, +sentiment)
- Concurrent upload stress test (if job queue implemented)

---

## Known Issues & Limitations

| Issue | Impact | Workaround | Fix Plan |
|-------|--------|-----------|----------|
| **Parakeet TDT English-only** | Cannot transcribe other languages | Use whisper-large-v3 for multi-language (slower) | Document limitation, or add model swap endpoint |
| **Long audio chunking artifacts** | 500s chunks may cut mid-sentence | Chunk at silence boundaries instead of fixed duration | Implement smart chunking with VAD |
| **Pyannote speaker count limit** | Max ~20 speakers, degrades above 10 | Use `num_speakers` hint for better accuracy | Document limit, warn if >10 speakers detected |
| **No streaming support** | Must upload full file before processing | Implement WebSocket streaming | Phase 3 feature (complex) |
| **Synchronous processing** | Blocks server during long transcriptions | Add job queue (Redis + workers) | Phase 3 feature |
| **No persistent storage** | Transcripts only in API response | Add SQLite DB + transcript history UI | Future consideration |

---

## Development Workflow

### Adding a New Feature

1. **Backend**:
   - Update `models.py` with new request/response fields
   - Add processing logic in appropriate module (`transcription.py`, `diarization/`, etc.)
   - Add API parameter to `api.py` endpoint
   - Write unit test

2. **Frontend**:
   - Add toggle/input in settings panel (left sidebar in mockup)
   - Update API call in `api/transcription.ts`
   - Update transcript viewer to display new data
   - Test in `mockup.html` first for rapid iteration

3. **Testing**:
   - Add to test suite
   - Test with real audio samples
   - Verify export formats include new data
   - Check GPU memory impact

4. **Documentation**:
   - Update this CURRENT.md
   - Add curl example to API Usage section
   - Update mockup if UI changes

---

## Quick Reference Commands

```bash
# Start container
docker compose up -d --build

# View logs
docker compose logs -f mvp-scribe

# Test API (no auth from LAN)
curl -X POST -F "file=@audio.mp3" -F "diarize=true" \
  http://192.168.1.10:20301/v1/audio/transcriptions

# Test with all Phase 1 features
curl -X POST \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "num_speakers=3" \
  -F "detect_paragraphs=true" \
  -F "remove_fillers=true" \
  -F "filter_profanity=false" \
  -F "response_format=verbose_json" \
  http://192.168.1.10:20301/v1/audio/transcriptions

# Check GPU memory
docker exec mvp-scribe nvidia-smi

# Verify torch version (should be NVIDIA's custom build)
docker exec mvp-scribe python3 -c "import torch; print(torch.__version__)"
# Expected: 2.7.0a0+7c8ec84dab.nv25.03

# Rebuild after code changes
docker compose down && docker compose up -d --build

# Interactive shell for debugging
docker exec -it mvp-scribe bash
```

---

## Architecture Decisions

### Why OpenAI-Compatible API?

Allows drop-in replacement for Whisper API users. Same endpoint structure, just add `speaker` field to segments.

### Why Separate Transcription + Diarization?

- Transcription (NeMo Parakeet): Fast, accurate, word-level timestamps
- Diarization (Pyannote): State-of-the-art speaker separation
- Combined: Best of both worlds, ~15s for 17min audio

Alternative: Whisper with diarization (whisperX) is slower and less accurate on GPU for this use case.

### Why Docker?

- NVIDIA NeMo requires specific CUDA/torch versions
- Docker provides reproducible environment
- GPU passthrough works reliably with nvidia-container-toolkit
- Easy deployment (ship docker-compose.yml)

### Why Pure-Python torchaudio Shim?

- Building torchaudio from source: 30+ min, complex dependencies
- Shim: 200 lines, zero compile time, perfect compatibility
- Maintainable: pure Python is easy to debug and extend
- Trade-off: Missing exotic transforms (doesn't matter for pyannote use case)

---

## Migration from Sherpa-ONNX (Previous Implementation)

### What Changed

| Aspect | Sherpa-ONNX (v1) | NeMo Parakeet (v2) | Reason for Change |
|--------|------------------|-------------------|-------------------|
| Model | Whisper Tiny/Base ONNX | Parakeet TDT 0.6B PyTorch | Better accuracy, native punctuation |
| Runtime | ONNX Runtime | NeMo Framework (PyTorch) | More flexible, GPU-optimized |
| Diarization | Not implemented | Pyannote 3.1 on GPU | Core feature requirement |
| Timestamps | Segment-level | Word-level + segment-level | Higher precision |
| Punctuation | None | Native from model | No post-processing needed |
| Container | Custom Python env | NVIDIA NeMo base image | Simplified GPU setup |

### Migration Path (If Reverting)

Sherpa-ONNX is still viable for:
- CPU-only environments (no CUDA)
- Minimal VRAM setups (< 4GB)
- Embedded/edge deployments

Keep sherpa-onnx branch for CPU fallback option.

---

## Mockup → Real Implementation Checklist

The `mockup.html` demonstrates all proposed features interactively. To implement for real:

### Frontend (React + TypeScript)

- [ ] Port mockup CSS to Tailwind classes (use MVP Scale design tokens)
- [ ] Convert feature toggles to React state + settings panel component
- [ ] Implement audio player component with Web Audio API for click-to-seek
- [ ] Build speaker timeline component (color-coded horizontal bar, clickable)
- [ ] Add speaker statistics display (talk time %, word count)
- [ ] Implement transcript search with highlight and prev/next navigation
- [ ] Wire up all export format generators: SRT, VTT, TXT, JSON (client-side)
- [ ] Add loading states, error handling, file size validation, WebSocket progress (Phase 3)

### Backend (FastAPI + Python)

**Phase 1 (v0.2.0)**:
- [ ] Add API key middleware with LAN bypass (reads `backend/api-keys.json`)
- [ ] Extend `/v1/audio/transcriptions` endpoint with Phase 1 params
- [ ] Implement paragraph detection algorithm (speaker grouping + silence threshold)
- [ ] Add filler word removal post-processor (regex on segment text)
- [ ] Add find/replace engine (user-defined rules)
- [ ] Integrate `better-profanity` library for censoring
- [ ] Expose speaker hints to diarization pipeline (`num_speakers`, `min_speakers`, `max_speakers`)
- [ ] Add confidence filtering logic (use existing `no_speech_prob` field)
- [ ] Implement trim audio (ffmpeg `-ss` and `-t` params)
- [ ] Test word-level timestamps from NeMo output (check if available)
- [ ] Update response schema: add `paragraphs`, `utterances`, `statistics` fields
- [ ] Add speaker statistics calculation (duration %, word count per speaker)

**Phase 3 (v0.3.0)**:
- [ ] Add Redis service to docker-compose
- [ ] Implement job queue with `rq` library
- [ ] Create `/jobs/{id}` status endpoint
- [ ] Add rate limiting via Redis INCR
- [ ] Add usage tracking via Redis sorted sets
- [ ] WebSocket progress updates via Redis pub/sub

**Phase 3 (v0.3.0+ - Optional)**:
- [ ] Implement regex-based PII redaction (built-in patterns for SSN, CC, email, phone)
- [ ] Add `/redaction/patterns` endpoint for uploading custom patterns
- [ ] Store custom patterns in Redis, pre-compile on upload
- [ ] Add language detection only if NeMo outputs it natively (otherwise skip)
- [ ] Discuss sentiment analysis value - may skip for MVP

### Testing

- [ ] Create test audio dataset (2-person call, 5-person meeting, noisy audio, clean podcast, stereo call)
- [ ] Write integration tests for each feature
- [ ] Performance benchmark suite (measure RTF, VRAM per feature)
- [ ] Export format validation (VLC playback for SRT/VTT, JSON schema validation)

### Documentation

- [ ] API reference with all parameters
- [ ] Feature guide with examples
- [ ] curl cookbook (common use cases)
- [ ] Python client library examples
- [ ] Performance tuning guide

---

## Appendix: Technical Deep Dives

### A. torchaudio ABI Mismatch Resolution

**Problem**: NVIDIA NeMo ships `torch==2.7.0a0+7c8ec84dab.nv25.3`, but all PyPI torchaudio wheels are ABI-incompatible.

**Solution**: Pure-Python shim in `backend/torchaudio_compat/` implements the subset pyannote-audio uses:
- `info()`, `load()` via soundfile
- `Resample`, `MFCC`, `MelSpectrogram` as torch.nn.Module classes
- `compliance.kaldi` functions copied from official torchaudio v2.7.0 (pure torch, BSD-2 license)

**Verification**:
```bash
docker exec mvp-scribe python3 -c "from pyannote.audio import Pipeline; print('OK')"
```

### B. Paragraph Detection Algorithm (Planned)

**Input**: Array of segments with `{start, end, text, speaker}`

**Output**: Array of paragraphs with `{start, end, text, speaker, segment_ids}`

**Logic**:
1. Sort segments by start time
2. Initialize first paragraph with first segment
3. For each subsequent segment:
   - If speaker changes: close current paragraph, start new one
   - If silence gap > threshold (default 0.8s): close current paragraph, start new one
   - Else: append segment text to current paragraph, extend end time
4. Return paragraph array

**Optional enhancement**: Use sentence tokenizer (nltk.punkt) to avoid breaking mid-sentence.

### C. PII Redaction Implementation (Simple Regex Approach)

**No external libraries, no model downloads.** Pure regex + user-uploadable patterns.

**Built-in Patterns** (shipped with backend):

```python
# backend/pii_patterns.py
DEFAULT_PATTERNS = [
    {"name": "SSN", "pattern": r"\b\d{3}-\d{2}-\d{4}\b", "replacement": "[SSN]"},
    {"name": "Credit_Card", "pattern": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "replacement": "[CREDIT_CARD]"},
    {"name": "Email", "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "replacement": "[EMAIL]"},
    {"name": "Phone_US", "pattern": r"\b(\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "replacement": "[PHONE]"},
    {"name": "IP_Address", "pattern": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "replacement": "[IP]"},
]
```

**User-Uploadable Patterns** (stored in Redis):

Upload JSON to `/redaction/patterns` endpoint:
```json
{
  "patterns": [
    {"name": "Medical_Record", "pattern": "MR-\\d{6}", "replacement": "[MRN]"},
    {"name": "Patient_ID", "pattern": "PT\\d{8}", "replacement": "[PATIENT_ID]"},
    {"name": "Account_Number", "pattern": "ACCT-[A-Z0-9]{10}", "replacement": "[ACCOUNT]"}
  ]
}
```

**Backend Implementation**:
```python
import re
import redis

# Load on startup
redis_client = redis.Redis(host='redis', decode_responses=True)
compiled_patterns = {}

def load_patterns():
    """Load default + user patterns, pre-compile regexes."""
    global compiled_patterns
    patterns = DEFAULT_PATTERNS.copy()

    # Load user patterns from Redis
    user_patterns = redis_client.get("redaction:patterns")
    if user_patterns:
        patterns.extend(json.loads(user_patterns))

    # Pre-compile all patterns
    for p in patterns:
        compiled_patterns[p["name"]] = {
            "regex": re.compile(p["pattern"], re.IGNORECASE),
            "replacement": p["replacement"]
        }

def redact_text(text: str) -> tuple[str, dict]:
    """Apply all patterns to text, return redacted text + counts."""
    redacted = text
    counts = {}

    for name, pattern_data in compiled_patterns.items():
        matches = pattern_data["regex"].findall(redacted)
        if matches:
            counts[name] = len(matches)
            redacted = pattern_data["regex"].sub(pattern_data["replacement"], redacted)

    return redacted, counts
```

**API Endpoints**:
```python
# Upload custom patterns
@app.post("/redaction/patterns")
async def upload_redaction_patterns(patterns: List[PatternDefinition], api_key: str = Depends(verify_api_key)):
    """Upload custom PII regex patterns. Stored in Redis, persisted across requests."""
    redis_client.set("redaction:patterns", json.dumps([p.dict() for p in patterns]))
    load_patterns()  # Recompile
    return {"status": "success", "pattern_count": len(patterns)}

# Get current patterns
@app.get("/redaction/patterns")
async def get_redaction_patterns(api_key: str = Depends(verify_api_key)):
    """Return all active patterns (built-in + custom)."""
    return {"patterns": list(compiled_patterns.keys())}
```

**Testing**:
```bash
# Test built-in patterns
curl -X POST -F "file=@test.mp3" -F "redact_pii=true" http://localhost:20301/v1/audio/transcriptions
# Transcript with "SSN 123-45-6789" → "SSN [SSN]"

# Upload custom patterns
curl -X POST -H "Content-Type: application/json" \
  -d '{"patterns":[{"name":"MRN","pattern":"MR-\\d{6}","replacement":"[MRN]"}]}' \
  http://localhost:20301/redaction/patterns

# Test custom pattern
curl -X POST -F "file=@medical.mp3" -F "redact_pii=true" http://localhost:20301/v1/audio/transcriptions
# "Medical record MR-123456" → "Medical record [MRN]"
```

**Advantages**:
- Zero dependencies (no model downloads)
- Fast (pure regex, pre-compiled)
- Flexible (users add domain-specific patterns)
- Persistent (Redis storage)
- Simple to test and debug

**Limitations**:
- Won't catch names (no NER)
- May have false positives (normal numbers matching patterns)
- Users responsible for pattern quality

### D. Redis Pattern Storage Strategy

**Key**: `redaction:patterns` → JSON string of pattern array

**On startup**:
1. Load DEFAULT_PATTERNS from `pii_patterns.py`
2. Check Redis for `redaction:patterns`
3. Merge + compile all patterns into `compiled_patterns` dict

**On pattern upload**:
1. Validate JSON structure
2. Test-compile each regex pattern (catch syntax errors)
3. Save to Redis `redaction:patterns`
4. Reload + recompile all patterns

**Pattern versioning** (optional future enhancement):
- Store patterns per API key: `redaction:patterns:{api_key}`
- Each user can have custom patterns
- For MVP: single global pattern set is fine

### E. Export Format Specifications

**SRT (SubRip)**:
```
1
00:00:00,000 --> 00:00:03,200
[Speaker 0] Good morning, everyone.

2
00:00:03,400 --> 00:00:06,800
[Speaker 0] I wanted to start by going over the Q4 metrics.
```

**VTT (WebVTT)**:
```
WEBVTT

00:00:00.000 --> 00:00:03.200
<v Speaker 0>Good morning, everyone.

00:00:03.400 --> 00:00:06.800
<v Speaker 0>I wanted to start by going over the Q4 metrics.
```

**TXT (Plain Text with Paragraphs)**:
```
Speaker 0 [0:00 - 0:06]:
Good morning, everyone. Thank you for joining the product review meeting today. I wanted to start by going over the Q4 metrics and then discuss our roadmap for next quarter.

Speaker 1 [0:07 - 0:13]:
Sounds good. Before we dive in, can we also address the customer feedback from last week? There were some recurring themes around the onboarding flow that I think we need to prioritize.
```

**JSON (Verbose)**:
See Response Schema section above.

**DOCX (Word Document)**:
- Each paragraph as a Word paragraph
- Speaker name in bold + color-coded
- Timestamps in gray, smaller font
- Proper spacing between speakers

---

## Deprecated / Not Doing for MVP

**Removed from active roadmap to keep scope focused.**

| Feature | Reason Skipped | Could Reconsider If... |
|---------|---------------|------------------------|
| **Channel per Speaker** | Niche use case (stereo call recordings only). Standard diarization works for 99% of audio. | Call center users specifically request stereo channel assignment. |
| **Inline Transcript Editing** | Users export and edit in their tool of choice (Word, Docs, text editor). Over-engineering the UI. | Users request collaborative editing or version history features. |
| **Per-Word Confidence Scores** | Segment-level confidence is sufficient. Word-level adds complexity with minimal value for MVP. | Quality visualization becomes a priority use case. |
| **Entity Detection (NER)** | Would require downloading transformers models (~400MB+) and GPU memory. Not worth it for MVP POC. | Business intelligence use cases emerge where tagging people/orgs/locations is high value. |
| **HIPAA Compliance (Full)** | Over-engineered for MVP. Simple regex PII redaction is enough. Don't need Presidio, spaCy, comprehensive coverage. | Healthcare customers require certified HIPAA compliance and are willing to pay for it. |

---

## Future Considerations (Post-MVP)

### Near-Term (Could Add Later)
- **DOCX Export**: Word document with formatted paragraphs, speaker colors, timestamps
- **Batch Processing API**: Upload multiple files, get single combined transcript
- **Transcript History**: SQLite DB storing past transcriptions with search/filter
- **Custom Vocabulary Boosting**: Improve recognition of domain-specific terms (medical, legal, technical)
- **Multi-language Support**: Add whisper-large-v3 for 99 languages (slower but more versatile)

### Long-Term (Significant Scope)
- **LLM Integration (Optional Paid Tier)**: Summarization, topic extraction, intent recognition, action items via Claude API or local LLaMA. User provides API key. Cost: $0.01-0.10 per transcript.
- **Live Transcription**: WebSocket audio streaming for real-time transcription (requires different model architecture)
- **Speaker Enrollment**: Upload reference audio per speaker for voice-based identification (pyannote embeddings)
- **Collaboration Features**: Multi-user editing, comments, annotations, version history
- **Integration Plugins**: Zapier, Notion, Google Docs, Slack, CRM webhooks
- **Meeting Assistant**: Auto-detect meeting type, extract decisions, track action items with assignees

---

## Key Contacts & Resources

- **Models**: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- **Documentation**: [NeMo Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html), [Pyannote Docs](https://github.com/pyannote/pyannote-audio)
- **Base Image**: [NGC NeMo Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- **HuggingFace Token**: Required for gated pyannote models (sign license agreement)

---

---

## MVP Philosophy - Simplified Approach

**Keep It Simple**:
- No heavy model downloads unless absolutely necessary
- Regex solutions over ML models where viable
- User-configurable via JSON uploads (patterns, rules)
- Redis for infrastructure, not complexity
- Beautiful frontend (mockup already shows this)
- Focus on the big value features (audio player with click-to-seek, paragraph grouping, multi-file upload)

**What We're NOT Doing for MVP**:
- ❌ HIPAA compliance certification (just simple PII filtering)
- ❌ NER models for entity detection (removed entirely)
- ❌ Complex libraries (Presidio, spaCy, transformers for MVP)
- ❌ Per-word confidence scores (segment-level sufficient)
- ❌ Inline editing (users export and edit elsewhere)
- ❌ Channel-per-speaker (niche)
- ❌ LLM integration (future, not POC)

**What We ARE Doing**:
- ✅ Regex-based PII redaction with user-uploadable custom patterns
- ✅ Multi-file concurrent upload (Redis job queue)
- ✅ Audio playback with timestamp sync (killer feature)
- ✅ Speaker timeline + stats + search
- ✅ Paragraph grouping, filler removal, find/replace
- ✅ Simple API key auth (JSON file, LAN bypass)
- ✅ Export SRT/VTT/TXT/JSON

**Redis Consolidation Decision**:
- Use Redis for: job queue (multi-file upload) + rate limiting + usage tracking + pattern storage
- One dependency, multiple benefits
- Can disable for single-user/LAN-only if needed

**Sentiment Analysis**:
- On hold - need to discuss if value justifies model download
- If yes, add to Phase 3 end
- If no, skip entirely

---

**Last Updated**: 2026-02-09
**Contributors**: Corey (dev), Claude (implementation)
**Version**: v0.1.0 (working baseline) → targeting v0.2.0 (Phase 1 complete) → v0.3.0 (Redis + multi-file)
