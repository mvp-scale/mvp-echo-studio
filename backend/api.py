import os
import json
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
import torch

from models import (
    WhisperSegment, TranscriptionResponse, ModelInfo, ModelList,
    Paragraph, SpeakerStatistics, Statistics, Topic,
)
from audio import convert_audio_to_wav, split_audio_into_chunks
from transcription import load_model, format_srt, format_vtt, transcribe_audio_chunk
from diarization import Diarizer
from post_processing import (
    detect_paragraphs, remove_filler_words, find_and_replace,
    apply_text_rules, filter_by_confidence, compute_speaker_statistics,
    apply_speaker_labels,
)
from entity_detection import extract_topics, annotate_paragraphs_with_entities
from sentiment_analysis import annotate_paragraphs_with_sentiment
from config import get_config
from auth import AuthMiddleware

logger = logging.getLogger(__name__)

asr_model = None
diarizer_instance = None
config = get_config()


def create_app() -> FastAPI:
    app = FastAPI(title="MVP-Echo Studio")

    # Auth middleware (LAN bypass + Bearer token)
    app.add_middleware(AuthMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        global asr_model, diarizer_instance

        try:
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, using CPU")

            asr_model = load_model(config.model_id)
            logger.info(f"Model {config.model_id} loaded")

            # Initialize diarizer once at startup (pipeline load is slow)
            hf_token = config.get_hf_token()
            if hf_token:
                logger.info("Initializing diarization pipeline...")
                diarizer_instance = Diarizer(access_token=hf_token)
                logger.info("Diarization pipeline ready")
            else:
                logger.info("No HF token, diarization disabled")

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
    ):
        global asr_model, diarizer_instance

        if not asr_model:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        filename = file.filename or "audio"
        logger.info(f"Transcription: {filename}, format={response_format}, diarize={diarize}")

        try:
            temp_dir = Path(config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            suffix = Path(filename).suffix or ".wav"
            temp_file = temp_dir / f"upload_{os.urandom(8).hex()}{suffix}"
            with open(temp_file, "wb") as f:
                content = await file.read()
                f.write(content)

            wav_file = convert_audio_to_wav(str(temp_file))
            audio_chunks = split_audio_into_chunks(wav_file, chunk_duration=config.chunk_duration)

            # Diarization (uses pre-loaded pipeline, now with speaker hints)
            diarization_result = None
            if diarize and diarizer_instance and diarizer_instance.pipeline:
                logger.info("Running speaker diarization")
                diarization_result = diarizer_instance.diarize(
                    wav_file,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                logger.info(f"Found {diarization_result.num_speakers} speakers")

            # Transcribe chunks
            all_text = []
            all_segments = []

            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Processing chunk {i + 1}/{len(audio_chunks)}")
                chunk_text, chunk_segments = transcribe_audio_chunk(
                    asr_model, chunk_path, language=language, word_timestamps=word_timestamps
                )

                if i > 0:
                    offset = i * config.chunk_duration
                    for seg in chunk_segments:
                        seg.start += offset
                        seg.end += offset

                all_text.append(chunk_text)
                all_segments.extend(chunk_segments)

            full_text = " ".join(all_text)

            # Merge diarization
            if diarizer_instance and diarization_result and diarization_result.segments:
                all_segments = diarizer_instance.merge_with_transcription(diarization_result, all_segments)

            # --- Post-processing pipeline ---

            # 1. Confidence filtering (before text transforms)
            if min_confidence > 0.0:
                all_segments = filter_by_confidence(all_segments, min_confidence)

            # 2. Text rules (unified) or legacy filler/find-replace fallback
            if text_rules:
                try:
                    parsed_rules = json.loads(text_rules)
                    # Accept bare array or envelope with rules array
                    if isinstance(parsed_rules, dict) and "rules" in parsed_rules:
                        parsed_rules = parsed_rules["rules"]
                    if isinstance(parsed_rules, list):
                        all_segments = apply_text_rules(all_segments, parsed_rules)
                except json.JSONDecodeError:
                    logger.warning("Invalid text_rules JSON, skipping")
            else:
                # Legacy fallback: separate filler removal and find/replace
                if remove_fillers:
                    all_segments = remove_filler_words(all_segments)
                if find_replace:
                    try:
                        rules = json.loads(find_replace)
                        if isinstance(rules, list):
                            all_segments = find_and_replace(all_segments, rules)
                    except json.JSONDecodeError:
                        logger.warning("Invalid find_replace JSON, skipping")

            # 4. Custom speaker labels
            labels_map = None
            if speaker_labels:
                try:
                    labels_map = json.loads(speaker_labels)
                    if isinstance(labels_map, dict):
                        all_segments = apply_speaker_labels(all_segments, labels_map)
                except json.JSONDecodeError:
                    logger.warning("Invalid speaker_labels JSON, skipping")

            # Rebuild full text after post-processing
            use_in_text = include_diarization_in_text if include_diarization_in_text is not None else config.include_diarization_in_text

            if diarize and diarization_result and diarization_result.segments and use_in_text:
                previous_speaker = None
                seen = set()
                text_parts = []
                for seg in all_segments:
                    if seg.speaker and seg.speaker != "unknown":
                        if seg.speaker != previous_speaker:
                            # Use custom label as-is, or format raw speaker ID
                            if labels_map and seg.speaker in labels_map.values():
                                display = seg.speaker
                            elif seg.speaker.startswith("speaker_"):
                                parts = seg.speaker.split("_")
                                try:
                                    num = int(parts[-1]) + 1
                                    display = f"Speaker {num}"
                                except (ValueError, IndexError):
                                    display = seg.speaker
                            else:
                                display = seg.speaker
                            text_parts.append(f"{display}: {seg.text.strip()}")
                            seen.add(seg.speaker)
                            previous_speaker = seg.speaker
                        else:
                            text_parts.append(seg.text.strip())
                    else:
                        text_parts.append(seg.text.strip())
                        previous_speaker = None
                full_text = " ".join(text_parts)
            else:
                full_text = " ".join(seg.text.strip() for seg in all_segments)

            # Compute duration from audio (last segment end time)
            duration = all_segments[-1].end if all_segments else 0.0

            # 5. Paragraph detection
            paragraphs_data = None
            if detect_paragraphs_flag and all_segments:
                raw_paragraphs = detect_paragraphs(all_segments, paragraph_silence_threshold)
                # Annotate paragraphs with per-paragraph entity counts
                if detect_entities:
                    annotate_paragraphs_with_entities(raw_paragraphs)
                # Annotate paragraphs with sentiment
                if detect_sentiment:
                    annotate_paragraphs_with_sentiment(raw_paragraphs)
                paragraphs_data = [Paragraph(**p) for p in raw_paragraphs]

            # 6. Speaker statistics
            statistics_data = None
            if diarize and diarization_result and diarization_result.segments:
                raw_stats = compute_speaker_statistics(all_segments, duration)
                if raw_stats:
                    statistics_data = Statistics(
                        speakers={k: SpeakerStatistics(**v) for k, v in raw_stats["speakers"].items()},
                        total_speakers=raw_stats["total_speakers"],
                    )

            # 7. Topic extraction (spaCy noun phrases)
            topics_data = None
            if detect_topics and all_segments:
                raw_topics = extract_topics(all_segments)
                if raw_topics:
                    topics_data = [Topic(**t) for t in raw_topics]

            response = TranscriptionResponse(
                text=full_text,
                segments=all_segments if timestamps or response_format == "verbose_json" else None,
                language=language,
                duration=duration,
                model="parakeet-tdt-0.6b-v2",
                paragraphs=paragraphs_data,
                statistics=statistics_data,
                topics=topics_data,
            )

            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if wav_file != str(temp_file) and os.path.exists(wav_file):
                os.unlink(wav_file)
            for chunk in audio_chunks:
                if chunk != wav_file and os.path.exists(chunk):
                    os.unlink(chunk)

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
        try:
            annotate_paragraphs_with_entities(paragraphs)
            return [p.get("entity_counts") for p in paragraphs]
        except Exception as e:
            logger.error(f"Entity annotation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/audio/sentiment")
    async def annotate_sentiment(paragraphs: List[dict] = Body(...)):
        """Annotate paragraph texts with sentiment (lightweight, CPU-only)."""
        try:
            annotate_paragraphs_with_sentiment(paragraphs)
            return [p.get("sentiment") for p in paragraphs]
        except Exception as e:
            logger.error(f"Sentiment annotation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        global asr_model, diarizer_instance
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024),
            }
        return {
            "status": "ok",
            "model_loaded": asr_model is not None,
            "model_id": config.model_id,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory": gpu_mem,
            "diarization_available": diarizer_instance is not None and diarizer_instance.pipeline is not None,
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
