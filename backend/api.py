import os
import json
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse

try:
    import torch
except ImportError:
    torch = None

from models import (
    WhisperSegment, TranscriptionResponse, ModelInfo, ModelList,
    Paragraph, SpeakerStatistics, Statistics, Topic,
)
from config import get_config, create_ml_adapters, create_audio_adapter, create_infra_adapters
from auth import AuthMiddleware
from use_cases.transcribe import TranscribeAudioUseCase, TranscribeRequest
from entity_detection import extract_topics, annotate_paragraphs_with_entities
from sentiment_analysis import annotate_paragraphs_with_sentiment

logger = logging.getLogger(__name__)

config = get_config()

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

# Adapter instances (set during startup)
_transcription = None
_diarization = None
_audio = None
_use_case = None


def _format_timestamp(seconds: float, always_include_hours: bool = False,
                      decimal_marker: str = ".") -> str:
    hours = int(seconds / 3600)
    seconds = seconds % 3600
    minutes = int(seconds / 60)
    seconds = seconds % 60

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:06.3f}".replace(".", decimal_marker) \
        if decimal_marker == "," else f"{hours_marker}{minutes:02d}:{seconds:06.3f}"


def format_srt(segments: List[WhisperSegment]) -> str:
    srt = ""
    for i, seg in enumerate(segments):
        start = _format_timestamp(seg.start, always_include_hours=True, decimal_marker=",")
        end = _format_timestamp(seg.end, always_include_hours=True, decimal_marker=",")
        text = seg.text.strip().replace("-->", "->")
        speaker = f"[{seg.speaker}] " if seg.speaker else ""
        srt += f"{i + 1}\n{start} --> {end}\n{speaker}{text}\n\n"
    return srt.strip()


def format_vtt(segments: List[WhisperSegment]) -> str:
    vtt = "WEBVTT\n\n"
    for seg in segments:
        start = _format_timestamp(seg.start, always_include_hours=True)
        end = _format_timestamp(seg.end, always_include_hours=True)
        text = seg.text.strip()
        speaker = f"<v {seg.speaker}>" if seg.speaker else ""
        vtt += f"{start} --> {end}\n{speaker}{text}\n\n"
    return vtt.strip()


def create_app() -> FastAPI:
    app = FastAPI(title="MVP-Echo Scribe")

    # Create infra adapters early (lightweight, no GPU/model dependency)
    _infra = create_infra_adapters(config)

    # Auth middleware with rate limiter
    app.add_middleware(AuthMiddleware, rate_limiter=_infra["rate_limiter"])

    allowed_origins = [
        "https://scribe.mvp-scale.com",
        "http://localhost:5173",       # Vite dev server
        "http://localhost:20301",      # Local Docker
        "http://127.0.0.1:20301",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )

    @app.on_event("startup")
    async def startup_event():
        global _transcription, _diarization, _audio, _use_case

        try:
            if torch and torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            elif torch:
                logger.warning("CUDA not available, using CPU")
            else:
                logger.info("PyTorch not installed (using ONNX Runtime)")

            # Create adapters via factory
            _transcription, _diarization = create_ml_adapters(config)
            _audio = create_audio_adapter()

            # Load ML models
            _transcription.load(config.model_id, device=config.asr_provider)
            logger.info(f"Transcription model loaded: {_transcription.model_name()}")

            if _diarization:
                hf_token = config.get_hf_token()
                if hf_token:
                    logger.info("Initializing diarization pipeline...")
                    _diarization.load(access_token=hf_token)
                    logger.info("Diarization pipeline ready")
                else:
                    logger.info("No HF token, diarization disabled")
            else:
                logger.info("No diarization adapter configured")

            # Wire up use case
            _use_case = TranscribeAudioUseCase(
                transcription=_transcription,
                diarization=_diarization,
                audio=_audio,
                progress=_infra["progress"],
            )

            logger.info(f"Engine: {config.engine} | Infra: {config.infra}")

        except Exception as e:
            logger.error(f"Startup error: {e}")

    @app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamps: bool = Form(False),
        vad_filter: bool = Form(False),
        word_timestamps: bool = Form(False),
        diarize: bool = Form(True),
        include_diarization_in_text: Optional[bool] = Form(None),
        # --- Phase 1: Diarization hints ---
        num_speakers: Optional[int] = Form(None),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
        # --- Phase 1: Post-processing ---
        detect_paragraphs_flag: bool = Form(False, alias="detect_paragraphs"),
        paragraph_silence_threshold: float = Form(0.8),
        remove_fillers: bool = Form(False),
        min_confidence: float = Form(0.0),
        find_replace: Optional[str] = Form(None),
        text_rules: Optional[str] = Form(None),
        speaker_labels: Optional[str] = Form(None),
        # --- Audio Intelligence ---
        detect_entities: bool = Form(False),
        detect_topics: bool = Form(False),
        detect_sentiment: bool = Form(False),
        # --- Config JSON (overrides individual params when present) ---
        config_json: Optional[str] = Form(None, alias="config"),
    ):
        if not _use_case or not _transcription or not _transcription.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        # If a config JSON is provided, override individual form params
        if config_json:
            try:
                cfg = json.loads(config_json)
                diarize = cfg.get("diarize", diarize)
                num_speakers = cfg.get("numSpeakers", num_speakers)
                min_speakers = cfg.get("minSpeakers", min_speakers)
                max_speakers = cfg.get("maxSpeakers", max_speakers)
                detect_paragraphs_flag = cfg.get("detectParagraphs", detect_paragraphs_flag)
                paragraph_silence_threshold = cfg.get("paragraphSilenceThreshold", paragraph_silence_threshold)
                min_confidence = cfg.get("minConfidence", min_confidence)
                detect_entities = cfg.get("detectEntities", detect_entities)
                detect_topics = cfg.get("detectTopics", detect_topics)
                detect_sentiment = cfg.get("detectSentiment", detect_sentiment)
                if cfg.get("speakerLabels"):
                    speaker_labels = json.dumps(cfg["speakerLabels"])
                if cfg.get("textRulesEnabled") and cfg.get("textRules"):
                    text_rules = json.dumps(cfg["textRules"])
                    logger.info(f"Config: loaded {len(cfg['textRules'])} text rules")
                logger.info("Config JSON applied")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")

        # Numeric parameter validation
        if not (0.0 <= temperature <= 2.0):
            raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 2.0")
        if not (0.0 <= min_confidence <= 1.0):
            raise HTTPException(status_code=400, detail="min_confidence must be between 0.0 and 1.0")
        if not (0.1 <= paragraph_silence_threshold <= 10.0):
            raise HTTPException(status_code=400, detail="paragraph_silence_threshold must be between 0.1 and 10.0")
        for name, val in [("num_speakers", num_speakers), ("min_speakers", min_speakers), ("max_speakers", max_speakers)]:
            if val is not None and not (1 <= val <= 20):
                raise HTTPException(status_code=400, detail=f"{name} must be between 1 and 20")
        if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
            raise HTTPException(status_code=400, detail="min_speakers must be <= max_speakers")

        filename = file.filename or "audio"
        logger.info(f"Transcription: {filename}, format={response_format}, diarize={diarize}, engine={config.engine}")

        try:
            # Save uploaded file
            temp_dir = Path(config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(filename).suffix or ".wav"
            temp_file = temp_dir / f"upload_{os.urandom(8).hex()}{suffix}"
            with open(temp_file, "wb") as f:
                total = 0
                while True:
                    chunk = await file.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        temp_file.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="File exceeds 500 MB limit")
                    f.write(chunk)

            # Resolve include_diarization_in_text default
            use_in_text = include_diarization_in_text if include_diarization_in_text is not None else config.include_diarization_in_text

            # Build request and delegate to use case
            req = TranscribeRequest(
                audio_path=str(temp_file),
                filename=filename,
                language=language,
                word_timestamps=word_timestamps,
                response_format=response_format,
                timestamps=timestamps,
                diarize=diarize,
                include_diarization_in_text=use_in_text,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                detect_paragraphs_flag=detect_paragraphs_flag,
                paragraph_silence_threshold=paragraph_silence_threshold,
                remove_fillers=remove_fillers,
                min_confidence=min_confidence,
                find_replace=find_replace,
                text_rules=text_rules,
                speaker_labels=speaker_labels,
                detect_entities=detect_entities,
                detect_topics=detect_topics,
                detect_sentiment=detect_sentiment,
                chunk_duration=config.chunk_duration,
            )

            response, all_segments, full_text = _use_case.execute(req)

            if response_format == "json":
                return response.dict()
            elif response_format == "text":
                return PlainTextResponse(full_text)
            elif response_format == "srt":
                return PlainTextResponse(format_srt(all_segments))
            elif response_format == "vtt":
                return PlainTextResponse(format_vtt(all_segments))
            elif response_format == "verbose_json":
                return response.dict()
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {response_format}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/audio/entities")
    async def annotate_entities(paragraphs: List[dict] = Body(...)):
        """Annotate paragraph texts with entity counts (lightweight, CPU-only)."""
        if len(paragraphs) > 500:
            raise HTTPException(status_code=400, detail="Too many paragraphs (max 500)")
        try:
            annotate_paragraphs_with_entities(paragraphs)
            return [p.get("entity_counts") for p in paragraphs]
        except Exception as e:
            logger.error(f"Entity annotation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/audio/sentiment")
    async def annotate_sentiment(paragraphs: List[dict] = Body(...)):
        """Annotate paragraph texts with sentiment (lightweight, CPU-only)."""
        if len(paragraphs) > 500:
            raise HTTPException(status_code=400, detail="Too many paragraphs (max 500)")
        try:
            annotate_paragraphs_with_sentiment(paragraphs)
            return [p.get("sentiment") for p in paragraphs]
        except Exception as e:
            logger.error(f"Sentiment annotation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        gpu_mem = None
        cuda_available = False
        gpu_name = None

        if torch and torch.cuda.is_available():
            cuda_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024),
            }

        return {
            "status": "ok",
            "engine": config.engine,
            "model_loaded": _transcription is not None and _transcription.is_loaded(),
            "model_id": config.model_id,
            "model_name": _transcription.model_name() if _transcription else None,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_memory": gpu_mem,
            "diarization_available": _diarization is not None and _diarization.is_loaded(),
            "asr_provider": config.asr_provider,
        }

    @app.get("/v1/models")
    async def list_models():
        return ModelList(data=[
            ModelInfo(
                id="whisper-1", created=1677649963, owned_by="parakeet",
                root="whisper-1", permission=[]
            )
        ])

    # Mount frontend static files (if built)
    static_dir = Path("/app/static").resolve()
    if static_dir.exists():
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file_path = (static_dir / full_path).resolve()
            # Prevent path traversal outside static directory
            if file_path.is_relative_to(static_dir) and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(static_dir / "index.html")

    return app
