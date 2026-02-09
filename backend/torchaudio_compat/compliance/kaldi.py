"""Kaldi-compatible feature extraction (fbank, mfcc, spectrogram).

Copied from torchaudio v2.7.0 src/torchaudio/compliance/kaldi.py.
This is 100% pure Python + torch with zero C++ dependencies.
The only change: 'import torchaudio' replaced with import from our functional shim.

License: BSD-2-Clause (PyTorch)
Source: https://github.com/pytorch/audio/blob/v2.7.0/src/torchaudio/compliance/kaldi.py
"""

import math
from typing import Tuple

import torch
from torch import Tensor

# Instead of 'import torchaudio' we import create_dct from our shim
from torchaudio.functional import create_dct as _create_dct

__all__ = [
    "get_mel_banks",
    "inverse_mel_scale",
    "inverse_mel_scale_scalar",
    "mel_scale",
    "mel_scale_scalar",
    "spectrogram",
    "fbank",
    "mfcc",
    "vtln_warp_freq",
    "vtln_warp_mel_freq",
]

EPSILON = torch.tensor(torch.finfo(torch.float).eps)
MILLISECONDS_TO_SECONDS = 0.001

HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]


def _get_epsilon(device, dtype):
    return EPSILON.to(device=device, dtype=dtype)


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    assert waveform.dim() == 1
    num_samples = waveform.size(0)
    strides = (window_shift * waveform.stride(0), waveform.stride(0))

    if snip_edges:
        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            pad_left = reversed_waveform[-pad:]
            waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
        else:
            waveform = torch.cat((waveform[-pad:], pad_right), dim=0)

    sizes = (m, window_size)
    return waveform.as_strided(sizes, strides)


def _feature_window_function(
    window_type: str, window_size: int, blackman_coeff: float, device: torch.device, dtype: int,
) -> Tensor:
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, device=device, dtype=dtype)
    elif window_type == POVEY:
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, device=device, dtype=dtype)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = torch.arange(window_size, device=device, dtype=dtype)
        return (
            blackman_coeff
            - 0.5 * torch.cos(a * window_function)
            + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
        ).to(device=device, dtype=dtype)
    else:
        raise Exception("Invalid window type " + window_type)


def _get_log_energy(strided_input: Tensor, epsilon: Tensor, energy_floor: float) -> Tensor:
    device, dtype = strided_input.device, strided_input.dtype
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()
    if energy_floor == 0.0:
        return log_energy
    return torch.max(log_energy, torch.tensor(math.log(energy_floor), device=device, dtype=dtype))


def _get_waveform_and_window_properties(
    waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient,
) -> Tuple[Tensor, int, int, int]:
    channel = max(channel, 0)
    assert channel < waveform.size(0), f"Invalid channel {channel} for size {waveform.size(0)}"
    waveform = waveform[channel, :]
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), f"choose a window size {window_size} that is [2, {len(waveform)}]"
    assert 0 < window_shift, "`window_shift` must be greater than 0"
    assert padded_window_size % 2 == 0
    assert 0.0 <= preemphasis_coefficient <= 1.0
    assert sample_frequency > 0
    return waveform, window_shift, window_size, padded_window_size


def _get_window(
    waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
    snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient,
) -> Tuple[Tensor, Tensor]:
    device, dtype = waveform.device, waveform.dtype
    epsilon = _get_epsilon(device, dtype)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        rand_gauss = torch.randn(strided_input.shape, device=device, dtype=dtype)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)
        strided_input = strided_input - row_means

    if raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)

    if preemphasis_coefficient != 0.0:
        offset_strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (1, 0), mode="replicate"
        ).squeeze(0)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    window_function = _feature_window_function(
        window_type, window_size, blackman_coeff, device, dtype
    ).unsqueeze(0)
    strided_input = strided_input * window_function

    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0
        ).squeeze(0)

    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)

    return strided_input, signal_log_energy


def _subtract_column_mean(tensor: Tensor, subtract_mean: bool) -> Tensor:
    if subtract_mean:
        col_means = torch.mean(tensor, dim=0).unsqueeze(0)
        tensor = tensor - col_means
    return tensor


def inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def inverse_mel_scale(mel_freq: Tensor) -> Tensor:
    return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, freq):
    assert vtln_low_cutoff > low_freq
    assert vtln_high_cutoff < high_freq
    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l
    Fh = scale * h
    assert l > low_freq and h < high_freq
    scale_left = (Fl - low_freq) / (l - low_freq)
    scale_right = (high_freq - Fh) / (high_freq - h)
    res = torch.empty_like(freq)
    outside_low_high_freq = torch.lt(freq, low_freq) | torch.gt(freq, high_freq)
    before_l = torch.lt(freq, l)
    before_h = torch.lt(freq, h)
    after_h = torch.ge(freq, h)
    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h] = scale * freq[before_h]
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high_freq] = freq[outside_low_high_freq]
    return res


def vtln_warp_mel_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, mel_freq):
    return mel_scale(
        vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor,
                       inverse_mel_scale(mel_freq))
    )


def get_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq, vtln_low, vtln_high, vtln_warp_factor):
    assert num_bins > 3
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq)
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    if vtln_high < 0.0:
        vtln_high += nyquist
    assert vtln_warp_factor == 1.0 or (
        (low_freq < vtln_low < high_freq) and (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high)
    )
    bin = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta
    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
        center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
        right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)
    center_freqs = inverse_mel_scale(center_mel)
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    if vtln_warp_factor == 1.0:
        bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
    else:
        bins = torch.zeros_like(up_slope)
        up_idx = torch.gt(mel, left_mel) & torch.le(mel, center_mel)
        down_idx = torch.gt(mel, center_mel) & torch.lt(mel, right_mel)
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]
    return bins, center_freqs


def spectrogram(
    waveform, blackman_coeff=0.42, channel=-1, dither=0.0, energy_floor=1.0,
    frame_length=25.0, frame_shift=10.0, min_duration=0.0, preemphasis_coefficient=0.97,
    raw_energy=True, remove_dc_offset=True, round_to_power_of_two=True,
    sample_frequency=16000.0, snip_edges=True, subtract_mean=False, window_type=POVEY,
) -> Tensor:
    device, dtype = waveform.device, waveform.dtype
    epsilon = _get_epsilon(device, dtype)
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)
    if len(waveform) < min_duration * sample_frequency:
        return torch.empty(0)
    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
        snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient)
    fft = torch.fft.rfft(strided_input)
    power_spectrum = torch.max(fft.abs().pow(2.0), epsilon).log()
    power_spectrum[:, 0] = signal_log_energy
    power_spectrum = _subtract_column_mean(power_spectrum, subtract_mean)
    return power_spectrum


def fbank(
    waveform, blackman_coeff=0.42, channel=-1, dither=0.0, energy_floor=1.0,
    frame_length=25.0, frame_shift=10.0, high_freq=0.0, htk_compat=False,
    low_freq=20.0, min_duration=0.0, num_mel_bins=23, preemphasis_coefficient=0.97,
    raw_energy=True, remove_dc_offset=True, round_to_power_of_two=True,
    sample_frequency=16000.0, snip_edges=True, subtract_mean=False, use_energy=False,
    use_log_fbank=True, use_power=True, vtln_high=-500.0, vtln_low=100.0,
    vtln_warp=1.0, window_type=POVEY,
) -> Tensor:
    """Kaldi-compatible fbank feature extraction (pure torch)."""
    device, dtype = waveform.device, waveform.dtype
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)
    if len(waveform) < min_duration * sample_frequency:
        return torch.empty(0, device=device, dtype=dtype)
    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
        snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient)
    spectrum = torch.fft.rfft(strided_input).abs()
    if use_power:
        spectrum = spectrum.pow(2.0)
    mel_energies, _ = get_mel_banks(
        num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq, vtln_low, vtln_high, vtln_warp)
    mel_energies = mel_energies.to(device=device, dtype=dtype)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)
    mel_energies = torch.mm(spectrum, mel_energies.T)
    if use_log_fbank:
        mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()
    if use_energy:
        signal_log_energy = signal_log_energy.unsqueeze(1)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
        else:
            mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)
    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies


def _get_dct_matrix(num_ceps: int, num_mel_bins: int) -> Tensor:
    dct_matrix = _create_dct(num_mel_bins, num_mel_bins, "ortho")
    dct_matrix[:, 0] = math.sqrt(1 / float(num_mel_bins))
    dct_matrix = dct_matrix[:, :num_ceps]
    return dct_matrix


def _get_lifter_coeffs(num_ceps: int, cepstral_lifter: float) -> Tensor:
    i = torch.arange(num_ceps)
    return 1.0 + 0.5 * cepstral_lifter * torch.sin(math.pi * i / cepstral_lifter)


def mfcc(
    waveform, blackman_coeff=0.42, cepstral_lifter=22.0, channel=-1, dither=0.0,
    energy_floor=1.0, frame_length=25.0, frame_shift=10.0, high_freq=0.0,
    htk_compat=False, low_freq=20.0, num_ceps=13, min_duration=0.0, num_mel_bins=23,
    preemphasis_coefficient=0.97, raw_energy=True, remove_dc_offset=True,
    round_to_power_of_two=True, sample_frequency=16000.0, snip_edges=True,
    subtract_mean=False, use_energy=False, vtln_high=-500.0, vtln_low=100.0,
    vtln_warp=1.0, window_type=POVEY,
) -> Tensor:
    """Kaldi-compatible MFCC feature extraction (pure torch)."""
    assert num_ceps <= num_mel_bins
    device, dtype = waveform.device, waveform.dtype
    feature = fbank(
        waveform=waveform, blackman_coeff=blackman_coeff, channel=channel, dither=dither,
        energy_floor=energy_floor, frame_length=frame_length, frame_shift=frame_shift,
        high_freq=high_freq, htk_compat=htk_compat, low_freq=low_freq, min_duration=min_duration,
        num_mel_bins=num_mel_bins, preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy, remove_dc_offset=remove_dc_offset,
        round_to_power_of_two=round_to_power_of_two, sample_frequency=sample_frequency,
        snip_edges=snip_edges, subtract_mean=False, use_energy=use_energy,
        use_log_fbank=True, use_power=True, vtln_high=vtln_high, vtln_low=vtln_low,
        vtln_warp=vtln_warp, window_type=window_type)
    if use_energy:
        signal_log_energy = feature[:, num_mel_bins if htk_compat else 0]
        mel_offset = int(not htk_compat)
        feature = feature[:, mel_offset:(num_mel_bins + mel_offset)]
    dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins).to(dtype=dtype, device=device)
    feature = feature.matmul(dct_matrix)
    if cepstral_lifter != 0.0:
        lifter_coeffs = _get_lifter_coeffs(num_ceps, cepstral_lifter).unsqueeze(0)
        feature *= lifter_coeffs.to(device=device, dtype=dtype)
    if use_energy:
        feature[:, 0] = signal_log_energy
    if htk_compat:
        energy = feature[:, 0].unsqueeze(1)
        feature = feature[:, 1:]
        if not use_energy:
            energy *= math.sqrt(2)
        feature = torch.cat((feature, energy), dim=1)
    feature = _subtract_column_mean(feature, subtract_mean)
    return feature
