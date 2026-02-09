"""Minimal torchaudio shim for pyannote.audio 3.3.2 + torch-pitch-shift.

NVIDIA NeMo 25.04 ships torch 2.7.0a0 (custom build) whose C++ ABI is
incompatible with ALL pre-built torchaudio wheels on PyPI.  This package
provides the exact torchaudio API surface that pyannote.audio 3.3.2 and
torch-pitch-shift 1.2.x actually use at import and inference time,
implemented with soundfile + torch (no C++ extensions, no ABI issues).

APIs provided (verified against source):
  pyannote.audio/core/io.py:
    torchaudio.info(path, backend=)
    torchaudio.load(path, backend=, frame_offset=, num_frames=)
    torchaudio.list_audio_backends()
    torchaudio.AudioMetaData
    torchaudio.functional.resample()
  torch_pitch_shift/main.py:
    torchaudio.__version__
    torchaudio.transforms.Resample
    torchaudio.transforms.TimeStretch

Sources:
  https://github.com/pyannote/pyannote-audio/blob/3.3.2/pyannote/audio/core/io.py
  https://github.com/KentoNishi/torch-pitch-shift/blob/master/torch_pitch_shift/main.py
"""

__version__ = "2.7.0+shim"

import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass

# Submodules — must be importable as `import torchaudio.transforms` etc.
from . import transforms  # noqa: F401
from . import functional  # noqa: F401
from . import backend  # noqa: F401
from . import compliance  # noqa: F401


@dataclass
class AudioMetaData:
    """Mirrors torchaudio.AudioMetaData (removed in torchaudio >=2.9)."""
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def list_audio_backends() -> list:
    """Return available audio backends.  Always returns ['soundfile']."""
    return ["soundfile"]


def set_audio_backend(backend_name: str) -> None:
    """No-op for compatibility (pyannote calls this with 'soundfile')."""
    pass


def get_audio_backend() -> str:
    return "soundfile"


def info(filepath, backend=None, **kwargs) -> AudioMetaData:
    """Return audio file metadata (mirrors torchaudio.info)."""
    i = sf.info(str(filepath))
    return AudioMetaData(
        sample_rate=i.samplerate,
        num_frames=i.frames,
        num_channels=i.channels,
    )


def load(filepath, frame_offset=0, num_frames=-1, backend=None, **kwargs):
    """Load audio as (waveform, sample_rate) (mirrors torchaudio.load).

    Returns:
        waveform: torch.FloatTensor of shape (channels, samples)
        sample_rate: int
    """
    stop = frame_offset + num_frames if num_frames > 0 else None
    data, sample_rate = sf.read(
        str(filepath),
        start=frame_offset,
        stop=stop,
        dtype="float32",
        always_2d=True,
    )
    # soundfile: (frames, channels) -> torchaudio: (channels, frames)
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sample_rate
