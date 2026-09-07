import torch
import torchaudio.functional as F
import numpy as np


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


def resample(waveform, sr, target_sr=44100):
    if sr != target_sr:
        waveform = F.resample(waveform, sr, target_sr)
    return waveform


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


def db_margin(energy_a, energy_b):
    return 10.0 * np.log10(max(energy_a, 1e-12) / max(energy_b, 1e-12))


def loudness_normalize(clip, target_rms=0.05):
    rms = clip.pow(2).mean().sqrt().clamp_min(1e-8)
    return (clip * (target_rms / rms)).clamp(-1.0, 1.0)

def frames(e, sr, hop, n_frames):
    fs = max(0, int(round(e["start_s"] * sr)) // hop)
    fe = max(fs + 1, min(int(round(e["end_s"] * sr)) // hop, n_frames))
    return fs, fe
