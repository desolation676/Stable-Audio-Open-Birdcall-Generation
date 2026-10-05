import torch
import torchaudio.functional as F
import numpy as np
import math


def remove_dc_offset(waveform):
    return waveform - waveform.mean(dim=-1, keepdim=True)


def detect_clipping(waveform, threshold=0.999, max_clip_ratio=0.0005, min_consecutive=3):
    abs_wave = waveform.abs()
    clipped_samples = abs_wave >= threshold
    clip_ratio = clipped_samples.float().mean().item()

    if clip_ratio > max_clip_ratio:
        return True, clip_ratio

    if min_consecutive > 1:
        kernel = torch.ones(1, 1, min_consecutive, device=waveform.device, dtype=torch.float32)
        mask = clipped_samples.reshape(1, 1, -1).float()
        consecutive_runs = torch.nn.functional.conv1d(mask, kernel)
        if (consecutive_runs >= min_consecutive).any().item():
            return True, clip_ratio

    return False, clip_ratio


def highpass(waveform, sr, cutoff_freq=200.0):
    return F.highpass_biquad(waveform, sample_rate=sr, cutoff_freq=cutoff_freq)


def band_limit(waveform, sr, bandlimit_sr=32000, target_sr=44100):
    params = dict(lowpass_filter_width=64, rolloff=0.95,
              resampling_method="sinc_interp_kaiser", beta=14.77)
    if sr != bandlimit_sr:
        waveform = F.resample(waveform, sr, bandlimit_sr, **params)
    return F.resample(waveform, bandlimit_sr, target_sr, **params)

def peak_normalize(waveform, peak_dbfs=-1.0):
    peak = waveform.abs().max().clamp_min(1e-8)
    gain = 10 ** (peak_dbfs / 20) / peak
    return waveform * gain, 20 * math.log10(gain.item())

def to_mono(waveform):
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    return waveform


def stft_power(mono, sr, n_fft=1024):
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=mono.device)
    spec = torch.stft(mono, n_fft=n_fft, hop_length=hop, window=window,
                      return_complex=True, center=True)
    power = spec.abs().pow(2)
    freqs = torch.linspace(0, sr / 2, power.shape[0], device=mono.device)
    return power, freqs, hop


def band_energy(power, freqs, frame_start, frame_end, low, high, q=0.9):
    band_mask = (freqs >= low) & (freqs <= high)
    if frame_end <= frame_start or not band_mask.any():
        return 0.0
    per_frame = power[band_mask, frame_start:frame_end].mean(dim=0)
    if per_frame.numel() == 0:
        return 0.0
    return torch.quantile(per_frame, q).item()

def call_energy(power, freqs, bg_bins, frame_start, frame_end, low, high):
    band_mask = (freqs >= low) & (freqs <= high)
    if frame_end <= frame_start or not band_mask.any():
        return 0.0
    excess = (power[band_mask, frame_start:frame_end] - bg_bins[band_mask, None]).clamp_min(0)
    return excess.sum().item()

def db_margin(energy_a, energy_b):
    return 10.0 * np.log10(max(energy_a, 1e-12) / max(energy_b, 1e-12))


def loudness_normalize(clip, target_rms=0.05):
    rms = clip.pow(2).mean().sqrt().clamp_min(1e-8)
    return (clip * (target_rms / rms)).clamp(-1.0, 1.0)

def frames(e, sr, hop, n_frames):
    fs = max(0, int(round(e["start_s"] * sr)) // hop)
    fe = max(fs + 1, min(int(round(e["end_s"] * sr)) // hop, n_frames))
    return fs, fe

def free_starts(busy_cs, length, n_frames):
    # All start frames s where [s, s+length) overlaps no annotation
    if length >= n_frames:
        return torch.empty(0, dtype=torch.long)
    starts = torch.arange(0, n_frames - length + 1)
    return starts[(busy_cs[starts + length] - busy_cs[starts]) == 0]

def null_snr(profile, bg, starts, length, n, q=0.9, generator=None):
    if len(starts) == 0 or n <= 0:
        return torch.empty(0)
    pick = starts[torch.randint(len(starts), (n,), generator=generator)]
    windows = profile.unfold(0, length, 1)[pick]
    ev = torch.quantile(windows, q, dim=1)
    return 10.0 * torch.log10(ev.clamp_min(1e-12) / max(bg, 1e-12))

def annotated_cumsum(spans, n_frames):
    busy = torch.zeros(n_frames, dtype=torch.long)
    for fs, fe in spans:
        busy[fs:fe] = 1
    return torch.cat([torch.zeros(1, dtype=torch.long), busy.cumsum(0)])