# MVP-Echo Scribe - Session Context

## Current State (2026-02-09)

**Status**: ✅ **v0.2.0 IN PROGRESS** - Core features implemented, iterating on UI polish
**Location**: `/home/corey/projects/mvp-echo-scribe/`
**Container**: `mvp-scribe` running on RTX 3090 (24GB VRAM)
**Ports**: 8001 (internal), mapped to 20301 externally
**Mockup**: `mockup.html` - Interactive feature demo with all proposed features

---

## Executive Summary

**What We Have**: GPU-accelerated transcription service (NeMo Parakeet TDT 0.6B) + speaker diarization (Pyannote 3.1) running in Docker. OpenAI-compatible API. 15s to transcribe 17min audio.

**v0.2.0 Progress**:
- ✅ API key authentication (LAN bypass + Bearer token)
- ✅ Paragraph grouping (speaker-aware, silence-based) — **client-side reactive** (threshold slider works live)
- ✅ Audio player with click-to-seek timestamps
- ✅ Speaker timeline, stats, search UI
- ✅ Post-processing: confidence filter — **client-side real-time**
- ✅ **Smart Redaction & PII** (renamed from Text Rules) — JSON-based filler removal, PII redaction, text replacements. Import/Export. Category pills with semantic colors (orange=Fillers, red=PII, cyan=Replace). **Purely client-side** — toggle on/off restores original text instantly.
- ✅ **Redaction highlighting** — redacted markers like `[SSN]`, `[CREDIT CARD]` render in bright rose text
- ✅ Export: SRT, VTT, TXT, JSON — **respects line-by-line vs paragraph view mode**
- ✅ Speaker hints for diarization (min/max/exact count, compact inline layout)
- ✅ Custom speaker labels — **dynamic detection, real-time rename, colors preserved**
- ✅ Two-panel layout matching mockup (Features/Code Sample left, results right)
- ✅ Code Sample panel with live curl/Python generation
- ✅ Separated upload from Run (configure settings before transcribing)
- ✅ **Per-paragraph Entity Detection** — spaCy NER counts per paragraph (PERSON, ORG, GPE, DATE, etc.) with colored pills on paragraph headers. Toggleable per-type. **Post-process capable** — toggle on after transcription, no re-run needed.
- ✅ **Topic Detection** — spaCy noun phrase extraction, most-discussed subjects
- ✅ **Per-paragraph Sentiment Analysis** — VADER sentiment (positive/neutral/negative) per paragraph. Colored pills on paragraph headers. Toggleable per-type. **Post-process capable**.
- ✅ **Settings panel redesign** — removed developer params, unified pill style, consistent hover states, compact layout
- ✅ Audio Intelligence section: Entity Detection, Topic Detection, Sentiment Analysis all live

**v0.3.0 (Infrastructure + Multi-File Upload)**:
- **Redis**: Enables multi-file concurrent upload (background job queue) - **very compelling**
- Redis also handles: rate limiting, usage tracking
- WebSocket progress updates for long files

**Key Architectural Decisions**:
1. **Unified Text Rules over separate features**: Filler removal, find/replace, and PII redaction are all the same operation (pattern match → replacement). Single JSON-based system with import/export replaces three separate mechanisms.
2. **Client-side post-processing for all text rules**: Redaction/rules are applied in the browser, never sent to the backend. Raw transcript stays clean. Toggle on/off works instantly without re-transcribing.
3. **Client-side paragraph detection**: Paragraphs re-detected from segments whenever the silence threshold slider moves. Backend paragraphs used as initial state, but client takes over for reactivity.
4. **Per-paragraph entity/sentiment over top-level aggregation**: Entity counts and sentiment live on each paragraph, not as flat lists above the transcript. More contextual, more navigational.
5. **Post-hoc annotation pattern**: Entity detection and sentiment can be toggled on after transcription via lightweight CPU-only endpoints (`/v1/audio/entities`, `/v1/audio/sentiment`). No re-transcription needed.
6. **VADER for sentiment over transformer models**: Lightweight, no model download, works on conversational text. Runs on CPU in milliseconds.
7. **spaCy for entity/topic detection**: Lightweight (~12MB model), runs on CPU, no GPU needed.
8. **No `better-profanity` dependency**: Profanity filtering = community text rules preset.

**Not Doing for MVP**:
- Channel-per-speaker, inline editing, per-word confidence, HIPAA certification
- LLM features (future paid tier)

**Future: sherpa-onnx migration** — Current stack uses NVIDIA NeMo container (heavy, requires HF token for pyannote). Sherpa-onnx has the same Parakeet TDT 0.6B model as ONNX + built-in diarization. Migration would eliminate NeMo container, HF token, torchaudio shim, and all ABI hacks. Planned for v0.3.0.

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
| **API Key Validation** | `backend/api-keys.json` schema | Security, multi-user access | 🟢 | ✅ | Implemented in `auth.py`. Middleware checks Bearer token against active keys. LAN bypass working. | Create test keys, verify 401 on invalid, 200 on valid. Test LAN bypass logic. |
| **Diarization (Min/Max/Exact Speakers)** | Pyannote pipeline already supports params | Accuracy improvement | 🟢 | ✅ | `num_speakers`, `min_speakers`, `max_speakers` exposed as API params and passed to Pyannote pipeline. | Upload 2-person call, set `num_speakers=2`, verify better accuracy vs auto-detect |
| **Word-Level Timestamps** | NeMo `result.timestamp['word']` | Precision, word-hover UI | 🟢 | 🔨 | Already in model output, needs JSON response field `words: [...]` | Verify word timestamps align with segment timestamps. UI hover test. |
| **Profanity Filter** | `better-profanity` pip package | Content moderation | 🟢 | 📋 | Regex-based, runs on segment text post-transcription. Add toggle to API. | Upload audio with profanity, toggle on, verify asterisk replacement. |
| **Remove Filler Words** | None (pure regex) | Readability | 🟢 | ✅ | `remove_fillers=true` API param. Regex strips filler phrases with word-boundary matching. | Upload conversational audio, toggle on, verify fillers removed. Manual review. |
| **Find & Replace** | None (pure regex) | Jargon correction, brand names | 🟢 | ✅ | `find_replace` JSON param. Escaped regex with whole-word, case-insensitive matching. | Test with "gonna→going to", verify case-insensitive, whole-word match only. |
| **Detect Paragraphs** | Silence gaps from existing timestamps | Readability, export quality | 🟢 | ✅ | `detect_paragraphs=true` + `paragraph_silence_threshold` params. Groups by speaker + silence gap. Returns `paragraphs` array. | Upload 5min audio, verify paragraph grouping logical. Check SRT/VTT output. |
| **Detect Utterances** | Existing segment timestamps | Alternative grouping | 🟢 | 📋 | Like paragraphs but ignore speaker labels—pure silence-based grouping. | Upload multi-speaker audio, verify utterances group correctly regardless of diarization. |
| **Trim Audio (Start/End)** | ffmpeg already in `audio.py` | Long file processing | 🟢 | 📋 | Add `-ss {start}` and `-t {duration}` to ffmpeg conversion. Parse MM:SS format from API. | Upload 10min file, trim to 2:00-4:00, verify only that range transcribed. |
| **Confidence Threshold** | `WhisperSegment` has `no_speech_prob`, `avg_logprob` fields in `models.py` | Noise filtering | 🟢 | ✅ | `min_confidence` param (0.0-1.0). Filters segments where `1 - no_speech_prob < threshold`. | Upload audio with silence/noise, set threshold 0.7, verify low-confidence segments dropped. |
| **Custom Speaker Labels (UI)** | None (pure frontend) | UX, export quality | 🟢 | ✅ | Client-side real-time rename. Dynamic detection of speaker count. Colors preserved via `originalSpeaker` tracking. Auto-enabled when >1 speaker. Exports use renamed labels. | Rename speakers, verify colors persist, verify exports use custom names. |

---

## Phase 2: Interactive Features (Frontend-Heavy)

**Goal**: Make the transcript viewer a full-featured workbench. All features work with existing backend data.

| Feature | Dependencies | Goals Alignment | Confidence | Status | Notes | Test Strategy |
|---------|-------------|-----------------|-----------|--------|-------|---------------|
| **Audio Playback with Timestamp Sync** | `<audio>` element, file already in browser | Core UX, verification workflow | 🟢 | ✅ | `AudioPlayer.tsx` implemented with play/pause, skip, seek bar, click-to-seek from segments. Active segment highlighting in `TranscriptViewer.tsx`. | Upload audio, click timestamp, verify playback starts at correct moment. Verify active highlight follows playback. |
| **Speaker Timeline Visualization** | Segment timestamps | Visual overview, quick navigation | 🟢 | ✅ | `SpeakerTimeline.tsx` — color-coded horizontal bar, click to seek, playhead tracks audio position. | Verify timeline matches transcript. Click blocks to seek. Check color coding. |
| **Speaker Statistics** | Segment timestamps + text | Analytics, meeting insights | 🟢 | ✅ | Backend `compute_speaker_statistics()` + frontend `SpeakerStats.tsx` with per-speaker cards, percentage bars, word counts. | Upload multi-speaker audio. Verify percentages sum to 100%. Verify word counts match. |
| **Transcript Search with Highlighting** | None (pure frontend) | Findability in long transcripts | 🟢 | ✅ | `SearchBar.tsx` with case-insensitive search, match count, highlight in `SpeakerSegment.tsx`. Prev/Next not yet implemented. | Search "onboarding" in 60min transcript, verify all matches found. Test prev/next nav. |
| **Export Formats** | Existing segment data | Professional output | 🟢 | ✅ | SRT/VTT/TXT/JSON export in frontend `export.ts`. **Respects view mode**: paragraph view exports paragraphs, line-by-line exports segments. Custom speaker labels and post-processing applied. DOCX not yet implemented. | Switch view modes, export all 4 formats, verify output matches view. |

---

## Phase 3: Optional Value-Add Features (Lower Priority)

**Goal**: Simple enhancements after core features are solid. Keep it minimal for MVP.

| Feature | Dependencies | Goals Alignment | Confidence | Status | Notes | Test Strategy |
|---------|-------------|-----------------|-----------|--------|-------|---------------|
| **PII Redaction (Regex-Based)** | None (pure regex) + Redis for pattern storage | Privacy, content filtering | 🟢 | 📋 | **Built-in patterns**: SSN (`\d{3}-\d{2}-\d{4}`), credit card (`\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}`), email, phone (US). **User-uploadable patterns**: POST JSON to `/redaction/patterns` with `[{"name": "Medical_Record", "pattern": "MR-\\d+", "replacement": "[MRN]"}]`. Store in Redis key `redaction:patterns`. Pre-compile regexes. Apply all to segment text. | Upload audio with SSN "123-45-6789", phone, email. Verify masked. Upload custom pattern JSON. Test persistence. Verify pre-compiled patterns used. |
| **Language Detection** | Check NeMo output OR `langdetect` (~10ms) | Informational tag | 🟢 | 📋 | **Only if free/trivial**. First check if Parakeet model outputs language field. If yes, pass through. If no, optionally run `langdetect.detect(text)` on full transcript. Skip if any complexity. | Check NeMo response structure. If language field exists, return it. Otherwise evaluate if worth adding langdetect. |
| **Sentiment Analysis** | `vaderSentiment` (pure Python, no model download) | Call quality scoring | 🟢 | ✅ | Per-paragraph positive/neutral/negative via VADER. Colored pills on paragraph headers. Post-process capable. Toggleable per-type. | Upload conversational audio, verify sentiment labels match tone. Toggle on/off. |

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

## Text Rules System (Replaces PII Redaction Workflow)

**Approach**: Unified JSON-based text processing. Filler removal, PII redaction, and text replacements are all the same operation: pattern match → replacement. Users manage rules via JSON import/export. No Redis, no separate endpoints.

**JSON Schema**:
```json
{
  "version": 1,
  "name": "Echo Scribe Text Rules",
  "rules": [
    {
      "name": "Remove 'um'",
      "find": "um",
      "replace": "",
      "isRegex": false,
      "flags": "gi",
      "category": "filler",
      "enabled": true
    },
    {
      "name": "Redact SSN",
      "find": "\\d{3}-\\d{2}-\\d{4}",
      "replace": "[SSN]",
      "isRegex": true,
      "flags": "g",
      "category": "pii",
      "enabled": true
    }
  ]
}
```

**Rule fields**: `name` (label), `find` (pattern), `replace` (substitution, empty=delete), `isRegex` (false=word-boundary literal, true=raw regex), `flags` (regex flags: g/i/m/s), `category` (filler/pii/replace — drives pill filter), `enabled` (toggle).

**UI**: Toggle on/off, category pills with counts (All/Fillers/PII/Replace), status message, Import JSON / Export JSON / Defaults buttons. Pills select which category runs.

**Default ruleset**: 17 rules — 7 fillers, 5 PII regex, 5 text replacements. Ships pre-loaded.

**Custom patterns**: Users export JSON, edit (or ask AI to generate rules), import back. No server-side storage needed.

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
    min_confidence: float = 0.0  # 0.0-1.0
    text_rules: Optional[str] = None  # JSON array of TextRule objects (unified filler/PII/replace)
    speaker_labels: Optional[Dict[str, str]] = None  # {"SPEAKER_0": "Alice", "SPEAKER_1": "Bob"}

    # Audio Intelligence (spaCy)
    detect_entities: bool = False  # NER: PERSON, ORG, GPE, DATE, MONEY, etc.
    detect_topics: bool = False  # Noun phrase frequency extraction

    # Legacy (backward compat, ignored when text_rules present)
    remove_fillers: bool = False
    find_replace: Optional[List[Dict[str, str]]] = None
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

  "entities": [  // if detect_entities=true (spaCy NER)
    {"text": "John", "label": "PERSON", "category": "People", "count": 3},
    {"text": "Acme Corp", "label": "ORG", "category": "Organizations", "count": 2}
  ],

  "topics": [  // if detect_topics=true (noun phrase frequency)
    {"text": "product review", "count": 5},
    {"text": "Q4 metrics", "count": 3}
  ]
}
```

---

## Implementation Priority Matrix

**Focus**: MVP features with high value-to-effort ratio. Deprioritize nice-to-haves and future paid features.

| Feature | User Value | Implementation Effort | Priority | Target Release | Notes |
|---------|-----------|----------------------|----------|----------------|-------|
| **API Key Validation** | HIGH | Low | P0 | v0.2.0 | ✅ Done - `auth.py` with LAN bypass |
| **Diarization Speaker Hints** | HIGH | Low | P0 | v0.2.0 | ✅ Done - `num_speakers`/`min_speakers`/`max_speakers` API params |
| **Detect Paragraphs** | HIGH | Low | P0 | v0.2.0 | ✅ Done - `detect_paragraphs` + `paragraph_silence_threshold` API params |
| **Custom Speaker Labels** | HIGH | Low | P0 | v0.2.0 | ✅ Done - Dynamic detection, real-time rename, colors preserved |
| **Audio Playback + Timestamp Sync** | VERY HIGH | Medium | P0 | v0.2.0 | ✅ Done - `AudioPlayer.tsx` with click-to-seek |
| **Export Formats (SRT/VTT/TXT/JSON)** | HIGH | Medium | P0 | v0.2.0 | ✅ Done - frontend `export.ts` + backend formatters |
| **Speaker Timeline Viz** | MEDIUM | Low | P1 | v0.2.0 | ✅ Done - `SpeakerTimeline.tsx` with click-to-seek and playhead |
| **Speaker Statistics** | MEDIUM | Low | P1 | v0.2.0 | ✅ Done - Backend + `SpeakerStats.tsx` frontend display |
| **Transcript Search** | HIGH | Low | P1 | v0.2.0 | ✅ Done - `SearchBar.tsx` with highlighting |
| **Unified Text Rules** | HIGH | Medium | P1 | v0.2.0 | ✅ Done - Replaces filler removal, find/replace, profanity filter. JSON import/export, per-rule regex flags, category pills. |
| **Entity Detection** | MEDIUM | Low | P1 | v0.2.0 | ✅ Done - Per-paragraph spaCy NER counts. Colored pills on headers. Toggle per-type. Post-hoc capable. |
| **Topic Detection** | MEDIUM | Low | P1 | v0.2.0 | ✅ Done - spaCy noun phrase extraction. Frequency-based topic tags. |
| **Sentiment Analysis** | MEDIUM | Low | P1 | v0.2.0 | ✅ Done - VADER per-paragraph. Colored pills (positive/neutral/negative). Post-hoc capable. |
| **Redis Infrastructure** | MEDIUM | Medium (4-5 hours) | P1 | v0.3.0 | Enables job queue + rate limiting + usage tracking |
| **Background Job Queue** | HIGH | Medium (3-4 hours w/ Redis) | P1 | v0.3.0 | **Multi-file upload** - very compelling |
| **Word-Level Timestamps** | MEDIUM | Medium (2-3 hours + verification) | P2 | v0.3.0 | Nice-to-have, depends on NeMo output structure |
| **Trim Audio** | MEDIUM | Low (1-2 hours) | P2 | v0.3.0 | Useful for long files |
| **Confidence Filter** | LOW | Low (1 hour) | P2 | v0.3.0 | Niche use case (noisy audio) |
| **Language Detection** | LOW | Low (30 min if easy) | P2 | v0.3.0 | Only if trivial to add |
| **PII Redaction (Regex)** | MEDIUM | Low (2-3 hours) | P2 | v0.3.0 | Simple regex patterns + user-uploadable custom patterns stored in Redis |
| **Sentiment Analysis** | MEDIUM | Low | P1 | v0.2.0 | ✅ Done — VADER, per-paragraph, post-hoc capable |
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

# Test API
curl -X POST -H "Authorization: Bearer $API_KEY" \
  -F "file=@audio.mp3" -F "diarize=true" \
  http://192.168.1.10:20301/v1/audio/transcriptions

# Test with all Phase 1 features
curl -X POST -H "Authorization: Bearer $API_KEY" \
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

## Migration: NeMo → Sherpa-ONNX (Planned for v0.3.0)

### Current Stack (NeMo)
| Aspect | Current |
|--------|---------|
| Model | Parakeet TDT 0.6B via NeMo PyTorch |
| Runtime | NeMo Framework (PyTorch) in `nvcr.io/nvidia/nemo:25.04` (~20GB image) |
| Diarization | Pyannote 3.1 (requires HF_TOKEN for gated model) |
| Workarounds | `torchaudio_compat/` ABI shim, pip pinning, pyannote version patches |

### Target Stack (Sherpa-ONNX)
| Aspect | Target |
|--------|--------|
| Model | Same Parakeet TDT 0.6B as ONNX (`sherpa-onnx-nemo-parakeet-tdt-0.6b-v2`) |
| Runtime | ONNX Runtime (slim Python image, no NeMo, no PyTorch) |
| Diarization | sherpa-onnx built-in (no HF token, no pyannote) |
| INT8 available | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` for faster inference |

### What Migration Eliminates
- `nvcr.io/nvidia/nemo:25.04` base image (20GB → slim Python base)
- `HF_TOKEN` requirement (no gated models)
- `torchaudio_compat/` shim (no ABI issues with ONNX Runtime)
- All pip pinning gymnastics in Dockerfile
- Pyannote version patches

### What Needs Verification
- sherpa-onnx diarization quality vs pyannote (pyannote is strong)
- Segment-level timestamp output format from ONNX model
- GPU acceleration via ONNX Runtime CUDA EP performance
- VAD (voice activity detection) — sherpa-onnx has built-in, could replace silence-based chunking

### Files to Touch
- `Dockerfile` — complete rewrite (slim base, sherpa-onnx install)
- `transcription.py` — swap NeMo API for sherpa-onnx `OfflineRecognizer`
- `diarization/` — swap pyannote for sherpa-onnx diarization
- `requirements.txt` — remove NeMo/pyannote deps, add sherpa-onnx
- `docker-compose.yml` — remove NeMo-specific env vars

---

## Mockup → Real Implementation Checklist

The `mockup.html` demonstrates all proposed features interactively. To implement for real:

### Frontend (React + TypeScript)

- [x] Port mockup CSS to Tailwind classes (MVP Scale design tokens)
- [x] Convert feature toggles to React state + SettingsPanel component
- [x] Implement audio player with click-to-seek (AudioPlayer.tsx)
- [x] Build speaker timeline (SpeakerTimeline.tsx - color-coded, clickable, playhead)
- [x] Add speaker statistics display (SpeakerStats.tsx - talk time %, word count, % bars)
- [x] Implement transcript search with highlighting (SearchBar.tsx + SpeakerSegment.tsx)
- [ ] Search prev/next navigation
- [x] Wire up all export format generators: SRT, VTT, TXT, JSON (client-side, view-mode aware)
- [x] Two-panel layout matching mockup (settings left, results right)
- [x] Code Sample panel with live curl/Python generation
- [x] Paragraph view + Line-by-line view with consistent styling
- [x] Client-side post-processing (find/replace, speaker labels, filler removal — real-time)
- [x] Speaker colors preserved on rename via originalSpeaker tracking
- [x] Dynamic speaker label detection (auto-enabled when >1 speaker)
- [ ] WebSocket progress (Phase 3)

### Backend (FastAPI + Python)

**Phase 1 (v0.2.0)**:
- [x] Add API key middleware with LAN bypass (auth.py)
- [x] Extend `/v1/audio/transcriptions` with Phase 1 params
- [x] Implement paragraph detection algorithm (post_processing.py)
- [x] Add filler word removal post-processor (regex)
- [x] Add find/replace engine (user-defined rules)
- [ ] Integrate `better-profanity` library for censoring
- [x] Expose speaker hints to diarization pipeline (num_speakers, min_speakers, max_speakers)
- [x] Add confidence filtering logic (min_confidence param)
- [ ] Implement trim audio (ffmpeg -ss and -t)
- [ ] Test word-level timestamps from NeMo output
- [x] Update response schema: paragraphs, statistics fields
- [x] Add speaker statistics calculation (duration %, word count per speaker)

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

### C. Text Rules Engine (Replaced PII Redaction + Redis Pattern Storage)

**Superseded**: The separate PII regex system and Redis pattern storage have been replaced by the unified Text Rules system. See "Text Rules System" section above.

**Key files**: `frontend/src/utils/text-rule-presets.ts` (defaults), `frontend/src/utils/post-processing.ts` (engine), `backend/post_processing.py` (`apply_text_rules()`), `backend/entity_detection.py` (spaCy NER).

**Two layers of data protection**:
1. **Text Rules** (regex): Pattern-based filler removal, PII redaction, text replacements. User-managed via JSON import/export.
2. **Entity Detection** (spaCy NER): Semantic entity recognition — people, organizations, locations, dates, money. Toggle in Audio Intelligence section.

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
| **Entity Detection (NER)** | ✅ **MOVED TO DONE** — spaCy en_core_web_sm (~12MB, CPU) added for entity + topic detection. | N/A — implemented. |
| **HIPAA Compliance (Full)** | Over-engineered for MVP. Text Rules + spaCy entity detection covers basic PII needs. | Healthcare customers require certified HIPAA compliance and are willing to pay for it. |

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
- ❌ HIPAA compliance certification (Text Rules + spaCy covers basic PII)
- ❌ Per-word confidence scores (segment-level sufficient)
- ❌ Inline editing (users export and edit elsewhere)
- ❌ Channel-per-speaker (niche)
- ❌ LLM integration (future, not POC)
- ~~❌ Sentiment analysis~~ → ✅ **Done** (VADER, lightweight, no ML model needed)

**What We ARE Doing**:
- ✅ Unified Text Rules (JSON-based filler removal, PII redaction, replacements with import/export)
- ✅ spaCy entity detection (PERSON, ORG, GPE, DATE, MONEY) + topic extraction
- ✅ Audio playback with timestamp sync (killer feature)
- ✅ Speaker timeline + stats + search
- ✅ Paragraph grouping, confidence filter
- ✅ Simple API key auth (JSON file, LAN bypass)
- ✅ Export SRT/VTT/TXT/JSON

**Redis Consolidation Decision**:
- Use Redis for: job queue (multi-file upload) + rate limiting + usage tracking
- PII pattern storage no longer needs Redis (Text Rules JSON replaces it)
- Can disable for single-user/LAN-only if needed

---

**Last Updated**: 2026-02-09
**Contributors**: Corey (dev), Claude (implementation)
**Version**: v0.1.0 (working baseline) → **v0.2.0 near-complete** (entity/sentiment/settings done, polish remaining) → v0.3.0 (sherpa-onnx migration + Redis + multi-file)

---

### Implementation Log

**Batch 1 Complete** — Backend post-processing pipeline:
- ✅ `post_processing.py` - paragraph detection, filler removal, find/replace, confidence filtering, speaker stats, speaker labels
- ✅ `models.py` - Paragraph, SpeakerStatistics, Statistics models added to TranscriptionResponse
- ✅ `diarization/__init__.py` - min_speakers/max_speakers/num_speakers forwarded to Pyannote
- ✅ `api.py` - All new Form params wired into post-processing pipeline

**Batch 2 Complete** — Frontend overhaul (mockup alignment):
- ✅ Two-panel layout: settings left (380px), results right
- ✅ `SettingsPanel.tsx` - All feature toggles with API param display
- ✅ `SpeakerTimeline.tsx` - Color-coded bar, click-to-seek, playhead
- ✅ `SpeakerStats.tsx` - Per-speaker cards with % bars and word counts
- ✅ `ParagraphView.tsx` - Paragraph-grouped transcript with search highlighting
- ✅ `CodeSamplePanel.tsx` - Live curl/Python code generation from settings
- ✅ Toolbar: Features/Code Sample tabs + green Run button
- ✅ Upload separated from Run (configure before transcribing)
- ✅ Audio Intelligence section — Entity Detection, Topic Detection, Sentiment Analysis all live

**Batch 3 Complete** — Client-side post-processing (real-time):
- ✅ `post-processing.ts` - Find/replace, speaker labels, filler removal applied client-side instantly
- ✅ `useMemo` recomputes displayed segments/paragraphs on every options change
- ✅ Raw API data preserved as `rawSegments`/`rawParagraphs` (never mutated)
- ✅ `originalSpeaker` field tracks raw speaker ID so colors survive rename

**Batch 4 Complete** — UI polish and consistency:
- ✅ Speaker colors preserved when renamed (via `originalSpeaker` in SpeakerBadge)
- ✅ Speaker labels auto-enabled when >1 speaker detected (no toggle gate)
- ✅ Line-by-line view redesigned: compact rows, speaker + time range + text on same line
- ✅ Consistent visual language between paragraph and line-by-line views (same border, colors, fonts)
- ✅ Full-row click-to-seek on line-by-line (not just timestamp button)
- ✅ Removed duplicate output tabs — simplified to transcript view + export buttons
- ✅ Export respects view mode: paragraph view → paragraph exports, line-by-line → segment exports

### Architecture Decisions Made This Session
- **Client-side post-processing**: Find/replace, speaker labels, and filler removal run in the browser via `useMemo`. No server round-trip needed. Server-side post-processing kept for API/curl consumers.
- **Raw vs display data**: API returns `rawSegments`/`rawParagraphs` (never mutated). `applyPostProcessing()` produces display versions that react to settings changes instantly.
- **originalSpeaker tracking**: When speakers are renamed, the original API ID is preserved in `originalSpeaker` field so `speakerColor()` can resolve the correct color.
- **Upload/Run separation**: File selection and transcription are decoupled. User picks a file, configures settings, then clicks Run. Settings can be changed after transcription and post-processing updates in real-time.
- **View-mode-aware exports**: Export buttons check `viewMode` — paragraph view exports paragraph-grouped data, line-by-line exports individual segments.
- **No output tabs**: Removed Transcript/JSON/SRT/VTT/TXT tab switcher to avoid confusion with export buttons. Right panel always shows transcript view.
- **Subagents over Agent Teams**: For this project's file structure (tight coupling in api.py, models.py), subagents orchestrated by a single session are more effective than Agent Teams. Revisit for v0.3.0 when backend/infra work is more parallelizable.

### Key Files Modified This Session
**Backend**: `api.py`, `models.py`, `diarization/__init__.py`, `post_processing.py` (new)
**Frontend**: `App.tsx`, `types.ts`, `api.ts`, `SettingsPanel.tsx` (new), `CodeSamplePanel.tsx` (new), `SpeakerTimeline.tsx` (new), `SpeakerStats.tsx` (new), `ParagraphView.tsx` (new), `SpeakerSegment.tsx` (rewritten), `SpeakerBadge.tsx`, `ExportBar.tsx`, `Layout.tsx`, `utils/post-processing.ts` (new), `utils/export.ts`

**Batch 5 Complete** — Unified Text Rules system:
- ✅ Replaced separate filler removal toggle + find/replace editor with unified Text Rules
- ✅ JSON-based: import/export single file with all rules across categories
- ✅ Category pills (All/Fillers/PII/Replace) select which rules run
- ✅ Per-rule regex `flags` field: `"gi"` (case-insensitive), `"g"` (case-sensitive), `"gm"` (multiline), etc.
- ✅ Toggle on/off for entire Text Rules system (consistent with other post-processing options)
- ✅ Default ruleset ships with 17 rules: 7 fillers, 5 PII regex, 5 replacements
- ✅ Defaults button resets to built-in ruleset
- ✅ localStorage persistence (versioned key, auto-clears stale data)
- ✅ Backend `apply_text_rules()` with same per-rule flags engine
- ✅ Backward compatible: legacy `remove_fillers`/`find_replace` params still work via API
- ✅ Eliminated need for: `better-profanity` pip dep, Redis PII pattern storage, `/redaction/patterns` endpoint

**Batch 6 Complete** — spaCy Entity Detection + Topic Extraction:
- ✅ Added `spacy` + `en_core_web_sm` model (~12MB) to Docker image
- ✅ `entity_detection.py` — NER extraction (PERSON, ORG, GPE, DATE, MONEY, etc.) + noun-phrase topic extraction
- ✅ `detect_entities` and `detect_topics` API params
- ✅ Entity/topic results in `TranscriptionResponse` (entities array, topics array)
- ✅ Frontend toggles in Audio Intelligence section (no longer grayed out)
- ✅ Color-coded entity tags (purple=people, blue=orgs, green=locations, yellow=dates) + topic tags
- ✅ `redact_entities()` function ready for entity-type-based redaction (wired but not yet exposed in UI)

**Batch 7 Complete** — Per-paragraph entity detection + sentiment analysis + settings redesign:

**Per-Paragraph Entity Detection:**
- ✅ `annotate_paragraphs_with_entities()` in `entity_detection.py` — runs spaCy NER per paragraph, stores `entity_counts` dict (merges GPE+LOC)
- ✅ `entity_counts: Optional[Dict[str, int]]` added to `Paragraph` model
- ✅ Wired into transcription pipeline (when `detect_entities=true`)
- ✅ Removed old top-level `extract_entities()` — per-paragraph counts replace flat entity list
- ✅ `POST /v1/audio/entities` — lightweight post-hoc endpoint, CPU-only, annotates paragraph texts
- ✅ Frontend: colored entity pills on paragraph header row (labeled "Entities" section)
- ✅ Frontend: toggle pills in settings per entity type (13 types with semantic colors)
- ✅ Frontend: `showEntities` prop gates all entity pills when toggle is off
- ✅ Post-hoc: toggle on after transcription triggers API call, merges counts into existing paragraphs

**Per-Paragraph Sentiment Analysis:**
- ✅ `backend/sentiment_analysis.py` — VADER-based, lazy-loaded, CPU-only
- ✅ `SentimentResult` model + `sentiment` field on `Paragraph` model
- ✅ `detect_sentiment` API form param wired into pipeline
- ✅ `POST /v1/audio/sentiment` — lightweight post-hoc endpoint
- ✅ Frontend: colored sentiment pill on paragraph header row (labeled "Sentiment" section, right-aligned)
- ✅ Frontend: toggle pills in settings (Positive=green, Neutral=gray, Negative=red)
- ✅ Post-hoc: toggle on after transcription triggers API call
- ✅ `vaderSentiment` added to `requirements.txt`

**Smart Redaction & PII (renamed from Text Rules):**
- ✅ Renamed section label to "Smart Redaction & PII"
- ✅ Import/Export moved to title row as subtle gray text links
- ✅ Removed Defaults button and "N loaded" counter
- ✅ Category pills got semantic colors (orange=Fillers, red=PII, cyan=Replace, blue=All)
- ✅ Text rules no longer sent to backend — **purely client-side**. Toggle off restores original text instantly.
- ✅ Redacted markers (`[SSN]`, `[CREDIT CARD]`, `[EMAIL]`, etc.) render in bright rose-400 text
- ✅ Shared `renderText()` utility in `utils/highlight-text.tsx` handles both redaction highlighting and search highlighting

**Client-Side Paragraph Detection:**
- ✅ `detectParagraphsClient()` in `utils/post-processing.ts` — mirrors backend algorithm
- ✅ Silence threshold slider is now reactive — paragraphs regroup live as slider moves
- ✅ Entity counts and sentiment carry over from matching raw paragraphs

**Settings Panel Redesign:**
- ✅ Removed all `param=value` monospace developer lines from every toggle
- ✅ Removed description text from obvious toggles (Diarization, Paragraphs, Confidence, Redaction)
- ✅ Kept descriptions for Audio Intelligence features (Entity, Topic, Sentiment)
- ✅ Unified pill style: `PILL_BASE` + `PILL_INACTIVE` shared constants across all sections
- ✅ All interactive elements get `hover:border-mvp-blue` on hover
- ✅ Diarization hints compacted: 3 rows → 1 inline row (`Min [__] Max [__] Exact [__]`)
- ✅ Diarization hints hidden when ≤1 speaker detected
- ✅ Number inputs: `type="text" inputMode="numeric"` — no browser spinner arrows
- ✅ Speaker labels aligned under `pl-[46px]` like all other sub-sections
- ✅ Speaker labels auto-enabled with always-on toggle when speakers detected

**Files Modified:**
- Backend: `entity_detection.py`, `sentiment_analysis.py` (new), `models.py`, `api.py`, `requirements.txt`
- Frontend: `types.ts`, `api.ts`, `App.tsx`, `ParagraphView.tsx`, `SettingsPanel.tsx`, `SpeakerSegment.tsx`, `utils/post-processing.ts`, `utils/highlight-text.tsx` (new)

### Architecture Decisions Made This Session
- **Client-side redaction**: Text rules never sent to backend. Raw data stays clean. Toggle works reactively.
- **Client-side paragraph detection**: Re-detection on threshold change means no server round-trip. Backend paragraphs are initial seed only.
- **Post-hoc annotation pattern**: Lightweight CPU endpoints for entity/sentiment. Toggle on after transcription without re-running GPU pipeline.
- **VADER over transformer sentiment**: Zero model download, millisecond inference, good enough for conversational text classification.
- **Unified visual system**: All toggleable pills share `PILL_BASE`/`PILL_INACTIVE` constants. Consistent sizing, font, hover behavior across entity types, sentiment types, and redaction categories.
- **sherpa-onnx migration planned**: Current NeMo stack works but is heavy. Same Parakeet TDT model available as ONNX with built-in diarization. Migration would eliminate NeMo container, HF token, torchaudio shim. Target: v0.3.0.

### Remaining for v0.2.0
- 📋 Trim audio (ffmpeg `-ss`/`-t`)
- 📋 Search prev/next navigation
- 📋 Word-level timestamps (verify NeMo output structure)
- 📋 Code Sample panel update (reflect new API params: detect_sentiment, etc.)

### Next: v0.3.0 (Infrastructure + Migration)
- 📋 **sherpa-onnx migration** — Replace NeMo container with ONNX Runtime, drop HF token requirement
- 📋 Redis infrastructure (job queue + rate limiting)
- 📋 Multi-file concurrent upload (background jobs)
- 📋 WebSocket progress updates
