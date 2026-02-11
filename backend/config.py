import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_MODEL_ID_NEMO = "nvidia/parakeet-tdt-0.6b-v2"
DEFAULT_MODEL_ID_SHERPA = "/models/sherpa-onnx"
DEFAULT_CHUNK_DURATION = 500


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.host = os.environ.get("HOST", DEFAULT_HOST)
        self.port = int(os.environ.get("PORT", DEFAULT_PORT))
        self.debug = os.environ.get("DEBUG", "0") == "1"
        self.temperature = float(os.environ.get("TEMPERATURE", "0.0"))
        chunk_env = os.environ.get("CHUNK_DURATION", "").strip()
        self.chunk_duration = int(chunk_env) if chunk_env else DEFAULT_CHUNK_DURATION
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        self.enable_diarization = os.environ.get("ENABLE_DIARIZATION", "true").lower() == "true"
        self.include_diarization_in_text = os.environ.get("INCLUDE_DIARIZATION_IN_TEXT", "true").lower() == "true"
        self.temp_dir = os.environ.get("TEMP_DIR", "/tmp/parakeet")
        self.engine = os.environ.get("ENGINE", "nemo").lower()
        self.infra = os.environ.get("INFRA", "local").lower()
        self.asr_provider = os.environ.get("ASR_PROVIDER", "cuda").lower()

        # Model ID defaults to engine-appropriate value if not explicitly set
        model_id_env = os.environ.get("MODEL_ID", "").strip()
        if model_id_env:
            self.model_id = model_id_env
        elif self.engine == "sherpa":
            self.model_id = DEFAULT_MODEL_ID_SHERPA
        else:
            self.model_id = DEFAULT_MODEL_ID_NEMO

        # Sherpa handles long audio internally with 80s sub-chunks, so the
        # outer use-case-level chunking is unnecessary overhead.  Set a large
        # default to effectively disable it (NeMo still needs 500s chunks for
        # VRAM management).
        if self.engine == "sherpa" and not chunk_env:
            self.chunk_duration = 86400  # 24h — effectively no outer split
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def get_hf_token(self) -> Optional[str]:
        return self.hf_token

    def as_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "chunk_duration": self.chunk_duration,
            "enable_diarization": self.enable_diarization,
            "include_diarization_in_text": self.include_diarization_in_text,
            "has_hf_token": self.hf_token is not None,
            "engine": self.engine,
            "infra": self.infra,
            "asr_provider": self.asr_provider,
        }


config = Config()


def get_config() -> Config:
    return config


def create_ml_adapters(cfg: Config):
    """Create transcription and diarization adapters based on ENGINE env var.

    Uses lazy imports so unused frameworks are never loaded.
    """
    engine = cfg.engine

    if engine == "nemo":
        from adapters.nemo.transcription import NeMoTranscriptionAdapter
        from adapters.nemo.diarization import PyannoteDiarizationAdapter
        transcription = NeMoTranscriptionAdapter()
        diarization = PyannoteDiarizationAdapter()
    elif engine == "sherpa":
        from adapters.sherpa.transcription import SherpaTranscriptionAdapter
        from adapters.nemo.diarization import PyannoteDiarizationAdapter
        transcription = SherpaTranscriptionAdapter()
        diarization = PyannoteDiarizationAdapter()
    else:
        raise ValueError(f"Unknown ENGINE: {engine!r}. Valid options: nemo, sherpa")

    diar_name = type(diarization).__name__ if diarization else "built-in"
    logger.info(f"ML adapters: engine={engine}, transcription={type(transcription).__name__}, diarization={diar_name}")
    return transcription, diarization


def create_audio_adapter():
    """Create the audio processing adapter (always FFmpeg)."""
    from adapters.ffmpeg.audio import FFmpegAudioAdapter
    return FFmpegAudioAdapter()


def create_infra_adapters(cfg: Config):
    """Create infrastructure adapters based on INFRA env var."""
    from adapters.local.sync_job import SyncJobAdapter
    from adapters.local.sliding_window_rate_limiter import SlidingWindowRateLimiter
    from adapters.local.log_progress import LogProgressAdapter
    from adapters.local.json_key_store import JsonFileKeyStore
    from adapters.local.json_file_usage import JsonFileUsageTracker

    infra = cfg.infra

    if infra == "local":
        adapters = {
            "job_queue": SyncJobAdapter(),
            "rate_limiter": SlidingWindowRateLimiter(max_requests=10, window_seconds=60),
            "progress": LogProgressAdapter(),
            "key_store": JsonFileKeyStore("/data/api-keys.json"),
            "usage": JsonFileUsageTracker("/models/usage.json"),
        }
    elif infra == "redis":
        # Redis adapters will be implemented in Phase 5
        raise NotImplementedError("Redis infrastructure adapters not yet implemented (planned for v0.4.0)")
    else:
        raise ValueError(f"Unknown INFRA: {infra!r}. Valid options: local, redis")

    logger.info(f"Infra adapters: {infra} -> {', '.join(type(v).__name__ for v in adapters.values())}")
    return adapters
