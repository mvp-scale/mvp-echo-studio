"""torchaudio.transforms shim.

Provides Resample, TimeStretch, MelSpectrogram, and MFCC.
Sources verified against:
  pyannote.audio/core/io.py - Resample
  pyannote.audio/models/embedding/xvector.py line 28 - MFCC
  torch_pitch_shift/main.py - Resample, TimeStretch
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class Resample(nn.Module):
    """Resample waveform between sample rates.

    Used by:
      - pyannote.audio/core/io.py (via torchaudio.functional.resample)
      - torch_pitch_shift/main.py line 33
    """

    def __init__(self, orig_freq: int = 16000, new_freq: int = 16000):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        ratio = self.new_freq / self.orig_freq
        n_samples = round(waveform.shape[-1] * ratio)
        needs_batch = waveform.dim() == 2
        if needs_batch:
            waveform = waveform.unsqueeze(0)
        resampled = torch.nn.functional.interpolate(
            waveform.float(), size=n_samples, mode="linear", align_corners=False
        )
        if needs_batch:
            resampled = resampled.squeeze(0)
        return resampled


class TimeStretch(nn.Module):
    """Stub for torchaudio.transforms.TimeStretch.

    torch-pitch-shift imports this at module level (main.py line 40) but
    only calls it inside pitch_shift(). The diarization pipeline never
    calls pitch_shift(), so this stub only needs to survive import.
    If actually called, it raises NotImplementedError with guidance.
    """

    def __init__(self, fixed_rate=None, n_freq=201, hop_length=None):
        super().__init__()
        self.fixed_rate = fixed_rate
        self.n_freq = n_freq
        self.hop_length = hop_length

    def forward(self, complex_specgrams, overriding_rate=None):
        raise NotImplementedError(
            "TimeStretch is a stub in the torchaudio shim. "
            "Install real torchaudio to use pitch_shift()."
        )


class MelSpectrogram(nn.Module):
    """Compute mel spectrogram from waveform.

    Used by pyannote.audio/models/embedding/xvector.py (via MFCC).
    Implements the same interface as torchaudio.transforms.MelSpectrogram.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_mels: int = 128,
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        norm: Optional[str] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_mels = n_mels
        # Pre-compute mel filterbank
        self.register_buffer("fb", self._mel_filterbank())

    def _hz_to_mel(self, freq: float) -> float:
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_filterbank(self) -> torch.Tensor:
        low_mel = self._hz_to_mel(self.f_min)
        high_mel = self._hz_to_mel(self.f_max)
        mel_points = torch.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()
        n_freqs = self.n_fft // 2 + 1
        fb = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
            low, center, high = bins[i], bins[i + 1], bins[i + 2]
            if center > low:
                fb[i, low:center] = torch.linspace(0, 1, center - low + 1)[:-1] if center > low else torch.tensor([])
            if high > center:
                fb[i, center:high] = torch.linspace(1, 0, high - center + 1)[:-1] if high > center else torch.tensor([])
        return fb  # (n_mels, n_freqs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=waveform.device, dtype=waveform.dtype)
        stft = torch.stft(
            waveform, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=window, center=self.center, pad_mode=self.pad_mode, return_complex=True,
        )
        spec = stft.abs().pow(self.power)  # (..., n_freqs, time)
        fb = self.fb.to(device=spec.device, dtype=spec.dtype)
        mel = torch.matmul(fb, spec)  # (..., n_mels, time)
        return mel


class MFCC(nn.Module):
    """Compute MFCC features from waveform.

    Used by pyannote.audio/models/embedding/xvector.py line 28:
        from torchaudio.transforms import MFCC
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        log_mels: bool = False,
        melkwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.log_mels = log_mels
        melkwargs = melkwargs or {}
        self.MelSpectrogram = MelSpectrogram(sample_rate=sample_rate, **melkwargs)
        n_mels = melkwargs.get("n_mels", 128)
        # DCT matrix
        self.register_buffer("dct_mat", self._dct_matrix(n_mfcc, n_mels))

    @staticmethod
    def _dct_matrix(n_mfcc: int, n_mels: int) -> torch.Tensor:
        n = torch.arange(float(n_mels))
        k = torch.arange(float(n_mfcc)).unsqueeze(1)
        dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
        return dct  # (n_mfcc, n_mels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.MelSpectrogram(waveform)
        if self.log_mels:
            mel = torch.log(mel.clamp(min=1e-10))
        else:
            mel = torch.log(mel.clamp(min=1e-10) + 1)
        # mel: (..., n_mels, time) → mfcc: (..., n_mfcc, time)
        dct = self.dct_mat.to(device=mel.device, dtype=mel.dtype)
        mfcc = torch.matmul(dct, mel)
        return mfcc
