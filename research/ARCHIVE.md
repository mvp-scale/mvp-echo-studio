# MVP-Echo Scribe - v0.2.0 Archive

> **Archived**: 2026-02-10
> **Reason**: v0.2.0 feature work complete. CURRENT.md now tracks v0.3.0 hexagonal architecture refactor.
> **Status at archive time**: All v0.2.0 features implemented and working.

---

## v0.2.0 Completed Features

- API key authentication (LAN bypass + Bearer token)
- Paragraph grouping (speaker-aware, silence-based) — client-side reactive (threshold slider works live)
- Audio player with click-to-seek timestamps
- Speaker timeline, stats, search UI
- Post-processing: confidence filter — client-side real-time
- Smart Redaction & PII — JSON-based filler removal, PII redaction, text replacements. Import/Export. Category pills with semantic colors. Purely client-side.
- Redaction highlighting — redacted markers render in bright rose text
- Export: SRT, VTT, TXT, JSON — respects line-by-line vs paragraph view mode
- Speaker hints for diarization (min/max/exact count)
- Custom speaker labels — dynamic detection, real-time rename, colors preserved
- Two-panel layout matching mockup (Features/Code Sample left, results right)
- Code Sample panel with live curl/Python generation
- Separated upload from Run (configure settings before transcribing)
- Per-paragraph Entity Detection — spaCy NER counts per paragraph, toggleable per-type, post-process capable
- Topic Detection — spaCy noun phrase extraction
- Per-paragraph Sentiment Analysis — VADER sentiment, colored pills, post-process capable
- Settings panel redesign — removed developer params, unified pill style, compact layout

---

## Key Architectural Decisions (v0.2.0)

1. **Unified Text Rules over separate features**: Filler removal, find/replace, and PII redaction are all the same operation (pattern match -> replacement). Single JSON-based system with import/export.
2. **Client-side post-processing for all text rules**: Redaction/rules applied in browser. Raw transcript stays clean. Toggle on/off works instantly.
3. **Client-side paragraph detection**: Paragraphs re-detected from segments whenever the silence threshold slider moves.
4. **Per-paragraph entity/sentiment over top-level aggregation**: Entity counts and sentiment live on each paragraph, not as flat lists.
5. **Post-hoc annotation pattern**: Entity detection and sentiment toggled on after transcription via lightweight CPU-only endpoints.
6. **VADER for sentiment over transformer models**: Lightweight, no model download, millisecond inference.
7. **spaCy for entity/topic detection**: Lightweight (~12MB model), CPU-only.

---

## Implementation Log

### Batch 1 — Backend post-processing pipeline
- `post_processing.py` - paragraph detection, filler removal, find/replace, confidence filtering, speaker stats, speaker labels
- `models.py` - Paragraph, SpeakerStatistics, Statistics models added to TranscriptionResponse
- `diarization/__init__.py` - min_speakers/max_speakers/num_speakers forwarded to Pyannote
- `api.py` - All new Form params wired into post-processing pipeline

### Batch 2 — Frontend overhaul (mockup alignment)
- Two-panel layout: settings left (380px), results right
- `SettingsPanel.tsx` - All feature toggles with API param display
- `SpeakerTimeline.tsx` - Color-coded bar, click-to-seek, playhead
- `SpeakerStats.tsx` - Per-speaker cards with % bars and word counts
- `ParagraphView.tsx` - Paragraph-grouped transcript with search highlighting
- `CodeSamplePanel.tsx` - Live curl/Python code generation from settings
- Toolbar: Features/Code Sample tabs + green Run button
- Audio Intelligence section — Entity Detection, Topic Detection, Sentiment Analysis

### Batch 3 — Client-side post-processing (real-time)
- `post-processing.ts` - Find/replace, speaker labels, filler removal applied client-side instantly
- `useMemo` recomputes displayed segments/paragraphs on every options change
- Raw API data preserved as `rawSegments`/`rawParagraphs` (never mutated)
- `originalSpeaker` field tracks raw speaker ID so colors survive rename

### Batch 4 — UI polish and consistency
- Speaker colors preserved when renamed (via `originalSpeaker` in SpeakerBadge)
- Line-by-line view redesigned: compact rows
- Consistent visual language between paragraph and line-by-line views
- Export respects view mode

### Batch 5 — Unified Text Rules system
- Replaced separate filler removal + find/replace with unified Text Rules
- JSON-based: import/export single file with all rules across categories
- Category pills (All/Fillers/PII/Replace)
- Default ruleset ships with 17 rules
- Eliminated need for: `better-profanity` dep, Redis PII pattern storage

### Batch 6 — spaCy Entity Detection + Topic Extraction
- `entity_detection.py` — NER extraction + noun-phrase topic extraction
- `detect_entities` and `detect_topics` API params
- Color-coded entity tags + topic tags in frontend

### Batch 7 — Per-paragraph annotations + settings redesign
- Per-paragraph entity detection with `annotate_paragraphs_with_entities()`
- Per-paragraph sentiment analysis with VADER
- `POST /v1/audio/entities` and `POST /v1/audio/sentiment` post-hoc endpoints
- Smart Redaction & PII renamed from Text Rules
- Settings panel: removed developer params, unified pill style

---

## Files Modified During v0.2.0

**Backend**: `api.py`, `models.py`, `diarization/__init__.py`, `post_processing.py`, `entity_detection.py`, `sentiment_analysis.py`, `requirements.txt`

**Frontend**: `App.tsx`, `types.ts`, `api.ts`, `SettingsPanel.tsx`, `CodeSamplePanel.tsx`, `SpeakerTimeline.tsx`, `SpeakerStats.tsx`, `ParagraphView.tsx`, `SpeakerSegment.tsx`, `SpeakerBadge.tsx`, `ExportBar.tsx`, `Layout.tsx`, `utils/post-processing.ts`, `utils/export.ts`, `utils/highlight-text.tsx`

---

## Feature Roadmap Tables (v0.2.0 snapshot)

### Phase 1: Core Features

| Feature | Status | Notes |
|---------|--------|-------|
| API Key Validation | Done | `auth.py` with LAN bypass |
| Diarization Speaker Hints | Done | `num_speakers`/`min_speakers`/`max_speakers` |
| Word-Level Timestamps | Ready | In model output, needs JSON response field |
| Remove Filler Words | Done | Unified into Text Rules |
| Find & Replace | Done | Unified into Text Rules |
| Detect Paragraphs | Done | Client-side reactive |
| Trim Audio | Planned | ffmpeg `-ss`/`-t` |
| Confidence Filter | Done | Client-side slider |
| Custom Speaker Labels | Done | Dynamic detection, real-time rename |

### Phase 2: Interactive Features

| Feature | Status | Notes |
|---------|--------|-------|
| Audio Playback + Timestamp Sync | Done | `AudioPlayer.tsx` with click-to-seek |
| Speaker Timeline | Done | `SpeakerTimeline.tsx` |
| Speaker Statistics | Done | Backend + `SpeakerStats.tsx` |
| Transcript Search | Done | `SearchBar.tsx` with highlighting |
| Export Formats | Done | SRT/VTT/TXT/JSON, view-mode aware |

### Phase 3: Value-Add

| Feature | Status | Notes |
|---------|--------|-------|
| PII Redaction | Done | Via Text Rules (client-side) |
| Entity Detection | Done | spaCy NER, per-paragraph |
| Topic Detection | Done | spaCy noun phrases |
| Sentiment Analysis | Done | VADER, per-paragraph |

---

## Performance Targets (v0.2.0)

| Metric | Achieved | Notes |
|--------|----------|-------|
| Transcription RTF | 0.015 (17min -> 15s) | Parakeet TDT on RTX 3090 |
| Diarization overhead | ~5s for 17min | Pyannote 3.1 |
| Peak VRAM | ~6GB | Transcription + diarization |
| API response (3min audio) | ~3-4s | Including upload, conversion, inference |

---

## Not Doing for MVP

- Channel-per-speaker, inline editing, per-word confidence, HIPAA certification
- LLM features (future paid tier)
- Live transcription (WebSocket streaming)
- Speaker enrollment (voice-based identification)

---

## Redis Infrastructure Plan (deferred to v0.3.0)

Redis was planned to handle: job queue (multi-file upload), rate limiting, usage tracking, WebSocket progress. Single `redis:7-alpine` container. Using `rq` for job queue, `INCR` with TTL for rate limiting, sorted sets for usage tracking.

---

## Technical Deep Dives

### torchaudio ABI Mismatch
NVIDIA NeMo ships `torch==2.7.0a0+7c8ec84dab.nv25.3` with incompatible C++ ABI. Pure-Python shim in `backend/torchaudio_compat/` implements the subset pyannote-audio needs: `info()`, `load()` via soundfile, `Resample`/`MFCC`/`MelSpectrogram` as torch.nn.Module, Kaldi compliance functions.

### Paragraph Detection Algorithm
Sort segments by start time. Group by speaker continuity and silence gaps > threshold. Returns paragraph array with aggregated text and timestamps.

### Text Rules Engine
Unified JSON schema for filler removal, PII redaction, text replacements. Per-rule regex flags. Category pills for selective application. Import/export. Client-side only — raw transcript never mutated.

### Export Formats
SRT (SubRip), VTT (WebVTT), TXT (plain with paragraphs), JSON (verbose). All respect current view mode (paragraph vs line-by-line). Custom speaker labels applied to exports.

---

**Last Updated**: 2026-02-09 (archived 2026-02-10)
