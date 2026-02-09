"""torchaudio.backend.common shim.

Provides AudioMetaData imported by pyannote.audio/core/io.py:
    from torchaudio.backend.common import AudioMetaData
"""

from dataclasses import dataclass


@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"
