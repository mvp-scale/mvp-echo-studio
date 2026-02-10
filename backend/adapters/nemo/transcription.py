"""NeMoTranscriptionAdapter — wraps NeMo Parakeet TDT for transcription."""

import logging
from typing import Optional

import soundfile as sf

from domain.models import TranscriptSegment
from ports.transcription import TranscriptionPort

logger = logging.getLogger(__name__)


class NeMoTranscriptionAdapter(TranscriptionPort):
    def __init__(self):
        self._model = None
        self._model_id: str = ""

    def load(self, model_id: str = "nvidia/parakeet-tdt-0.6b-v2", device: str = "cuda") -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        logger.info(f"Loading NeMo model {model_id}")
        self._model_id = model_id
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

        if device == "cuda" and torch.cuda.is_available():
            # cuDNN auto-tuning: benchmarks convolution algorithms on first run,
            # then caches the fastest for subsequent calls with same input shapes.
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")

            self._model = self._model.cuda()

            # Half-precision inference: halves memory bandwidth, speeds up compute.
            # RTX 3090 has full fp16/bf16 tensor core support.
            self._model = self._model.half()
            logger.info(f"Model loaded on GPU (fp16): {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, running on CPU")

        self._model.eval()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> tuple[str, list[TranscriptSegment]]:
        import torch

        if not self._model:
            return "", []

        try:
            audio_duration = self._get_audio_duration(audio_path)

            with torch.inference_mode():
                output = self._model.transcribe([audio_path], timestamps=True)

            if not output:
                logger.warning(f"No transcription for {audio_path}")
                return "", []

            result = output[0]
            text = result.text if hasattr(result, "text") else str(result)

            if not text or not text.strip():
                return "", []

            segments: list[TranscriptSegment] = []

            if hasattr(result, "timestamp") and result.timestamp and "segment" in result.timestamp:
                for stamp in result.timestamp["segment"]:
                    seg_text = stamp.get("segment", "")
                    if seg_text:
                        segments.append(TranscriptSegment(
                            start=stamp["start"],
                            end=stamp["end"],
                            text=seg_text,
                        ))

            if segments:
                logger.info(f"Transcribed with {len(segments)} timestamped segments")
                return text, segments

            logger.info(f"Transcribed (no segment timestamps): {len(text)} chars")
            segments = [TranscriptSegment(start=0.0, end=audio_duration, text=text)]
            return text, segments

        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}", exc_info=True)
            return "", []

    def model_name(self) -> str:
        return self._model_id.split("/")[-1] if self._model_id else "parakeet-tdt-0.6b-v2"

    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return 0.0
