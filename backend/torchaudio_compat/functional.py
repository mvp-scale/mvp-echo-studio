"""torchaudio.functional shim.

Provides:
  resample() - pyannote.audio/core/io.py line 217
  create_dct() - used by torchaudio.compliance.kaldi._get_dct_matrix
"""

import math
from typing import Optional

import torch
from torch import Tensor


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample a waveform tensor from orig_freq to new_freq.

    Args:
        waveform: (channels, samples) float tensor
        orig_freq: original sample rate
        new_freq: target sample rate

    Returns:
        Resampled waveform tensor of shape (channels, new_samples)
    """
    if orig_freq == new_freq:
        return waveform
    ratio = new_freq / orig_freq
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


def create_dct(n_mfcc: int, n_mels: int, norm: Optional[str]) -> Tensor:
    """Create a DCT transformation matrix with shape (n_mels, n_mfcc).

    Copied from torchaudio v2.7.0 (pure torch, no C++ deps).
    Used by torchaudio.compliance.kaldi._get_dct_matrix.
    """
    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)
    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()
