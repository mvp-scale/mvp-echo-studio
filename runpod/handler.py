"""RunPod Serverless Handler for MVP-Echo Scribe.

Loads models once at worker boot, then processes transcription jobs via
RunPod's serverless input/output format. Wraps TranscribeAudioUseCase.execute()
directly — no auth, rate limiting, or HTTP layer needed.

Usage:
    # RunPod serverless (automatic)
    CMD ["python", "-u", "/app/handler.py"]

    # Local testing (reads test_input.json)
    python handler.py
"""

import base64
import json
import logging
import os
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import runpod

from config import Config, create_ml_adapters, create_audio_adapter
from adapters.local.log_progress import LogProgressAdapter
from use_cases.transcribe import TranscribeAudioUseCase, TranscribeRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("runpod.handler")

# ---------------------------------------------------------------------------
# Model loading (once per cold start)
# ---------------------------------------------------------------------------
logger.info("Loading models...")
_cfg = Config()
_transcription, _diarization = create_ml_adapters(_cfg)
_audio = create_audio_adapter()

_transcription.load(_cfg.model_id, device=_cfg.asr_provider)
if _diarization:
    _diarization.load(access_token=_cfg.get_hf_token())

_use_case = TranscribeAudioUseCase(
    transcription=_transcription,
    diarization=_diarization,
    audio=_audio,
    progress=LogProgressAdapter(),
)
logger.info("Models loaded — ready for jobs")


# ---------------------------------------------------------------------------
# Helper: download or decode audio to a temp file
# ---------------------------------------------------------------------------
def _fetch_audio(job_input: dict, job_dir: str) -> tuple[str, str]:
    """Return (local_path, filename) from audio_url or audio_base64."""
    if "audio_url" in job_input:
        url = job_input["audio_url"]
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "audio.wav"
        dest = os.path.join(job_dir, filename)
        logger.info(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
        return dest, filename

    if "audio_base64" in job_input:
        raw = base64.b64decode(job_input["audio_base64"])
        filename = job_input.get("filename", "audio.wav")
        dest = os.path.join(job_dir, filename)
        with open(dest, "wb") as f:
            f.write(raw)
        return dest, filename

    raise ValueError("Input must include 'audio_url' or 'audio_base64'")


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(event: dict) -> dict:
    """Process a single transcription job."""
    job_input = event.get("input", {})
    job_dir = tempfile.mkdtemp(prefix="runpod_scribe_")

    try:
        audio_path, filename = _fetch_audio(job_input, job_dir)

        req = TranscribeRequest(
            audio_path=audio_path,
            filename=filename,
            language=job_input.get("language"),
            word_timestamps=job_input.get("word_timestamps", False),
            response_format=job_input.get("response_format", "verbose_json"),
            timestamps=job_input.get("timestamps", False),
            diarize=job_input.get("diarize", True),
            include_diarization_in_text=job_input.get("include_diarization_in_text", True),
            num_speakers=job_input.get("num_speakers"),
            min_speakers=job_input.get("min_speakers"),
            max_speakers=job_input.get("max_speakers"),
            detect_paragraphs_flag=job_input.get("detect_paragraphs", False),
            paragraph_silence_threshold=job_input.get("paragraph_silence_threshold", 0.8),
            remove_fillers=job_input.get("remove_fillers", False),
            min_confidence=job_input.get("min_confidence", 0.0),
            find_replace=job_input.get("find_replace"),
            text_rules=job_input.get("text_rules"),
            speaker_labels=job_input.get("speaker_labels"),
            detect_entities=job_input.get("detect_entities", False),
            detect_topics=job_input.get("detect_topics", False),
            detect_sentiment=job_input.get("detect_sentiment", False),
            chunk_duration=job_input.get("chunk_duration", 500),
        )

        response, _segments, _full_text = _use_case.execute(req)
        return response.dict()

    except Exception as e:
        logger.error(f"Job failed: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

    finally:
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Local testing: read test_input.json if it exists
    test_file = Path(__file__).parent / "test_input.json"
    if test_file.exists() and "RUNPOD_POD_ID" not in os.environ:
        logger.info(f"Local test mode — reading {test_file}")
        with open(test_file) as f:
            test_event = json.load(f)
        result = handler(test_event)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)

    runpod.serverless.start({"handler": handler})
