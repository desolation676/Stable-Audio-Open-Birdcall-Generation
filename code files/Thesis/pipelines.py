import json
import torch
import pandas as pd
import torchaudio.functional as F
import soundfile as sf
from scipy.io import wavfile
from abc import ABC, abstractmethod
from pathlib import Path
from DSP_helpers import *


class AudioPipeline(ABC):
    def __init__(self, cache_dir, target_sr=44100, highpass_hz=200.0, n_fft=1024):
        self.target_sr = target_sr
        self.highpass_hz = highpass_hz
        self.n_fft = n_fft
        self.save_dir = Path(cache_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_params(self):
        """Returns custom name for saved file including preprocessing params"""
        return f"sr{self.target_sr}_hp{self.highpass_hz:g}_mono"

    def _save_paths(self, path):
        """Returns save path for wav and json file"""
        stem = f"{Path(path).stem}_{self._save_params()}"
        return self.save_dir / f"{stem}.wav", self.save_dir / f"{stem}.meta.json"

    def preprocess_raw(self, path):
        """
        Performs raw preprocessing on audio file .flac or .mp3
        Saves finished wav and json
        """
        wav_path, meta_path = self._save_paths(path)

        if wav_path.exists() and meta_path.exists():
            _, data = wavfile.read(wav_path)
            waveform = torch.from_numpy(data.copy()).float()
            meta = json.loads(meta_path.read_text())
            meta["cached"] = True
            return waveform, meta, wav_path

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = to_mono(torch.from_numpy(data).t())
        waveform = remove_dc_offset(waveform)
        clipped, clip_ratio = detect_clipping(waveform)
        waveform = highpass(waveform, sr, self.highpass_hz)
        waveform = resample(waveform, sr, self.target_sr)

        wavfile.write(wav_path, self.target_sr, waveform.numpy().astype(np.float32))
        meta = {
            "native_sr": sr,
            "upsampled": sr < self.target_sr,
            "clip_ratio": clip_ratio,
            "clipped": bool(clipped),
        }
        meta_path.write_text(json.dumps(meta))
        meta["cached"] = False
        return waveform, meta, wav_path

    @abstractmethod
    def extract_events(self, path):
        """Preprocesses annotations into standardized events"""

    @abstractmethod
    def gate(self, power, freqs, hop, event):
        """ Dominance gate, returns true or false"""

    def gating_config(self):
        """Records gate configs used"""
        return {"target_sr": self.target_sr, "highpass_hz": self.highpass_hz,
                "n_fft": self.n_fft, "pipeline": type(self).__name__}

    def precompute_events(self, path):
        """
        Precomputes energies for extracted events
        """
        # preprocess files and store preprocessed version
        waveform, meta, wav_path = self.preprocess_raw(path)
        # get all events, annotations for soundscapes, snippets for XC
        # list of events for file, start time, end time, species + other info
        events = self.extract_events(path)
        if not events:
            return []

        # compute power for file
        power, freqs, hop = stft_power(waveform, self.target_sr, self.n_fft)
        n_samples = waveform.shape[-1]
        rows = []

        # collects all important information, checks if gate is passed and deletes pow to free memory
        for i, e in enumerate(events):
            fs = max(0, int(round(e["start_s"] * self.target_sr)) // hop)
            fe = max(fs + 1, min(int(round(e["end_s"] * self.target_sr)) // hop,
                                 power.shape[-1]))
            rows.append({
                "event_id": f"{Path(path).stem}:{i}",
                "wav_path": str(wav_path),
                "source_path": str(path),
                "n_samples": int(n_samples),
                "start_s": float(e["start_s"]),
                "end_s": float(e["end_s"]),
                "low_hz": float(e["low_hz"]),
                "high_hz": float(e["high_hz"]),
                "species": e["species"],
                "band_energy": band_energy(power, freqs, fs, fe, e["low_hz"], e["high_hz"]),
                "passes_gate": bool(self.gate(power, freqs, hop, e)),
                "native_sr": meta.get("native_sr"),
                "upsampled": meta.get("upsampled"),
                "clip_ratio": meta.get("clip_ratio"),
                "clipped": meta.get("clipped"),
            })

        del power
        return rows


class SoundscapePipeline(AudioPipeline):
    def __init__(self, annotations_csv, dominance_margin_db=3.0,
                 floor_percentile=60, **kwargs):
        super().__init__(**kwargs)
        self.annotations = self.load_annotations(annotations_csv)
        self.dominance_margin_db = dominance_margin_db
        self.floor_percentile = floor_percentile

    def load_annotations(self, annotations_csv):
        """Load annotations csv into df"""
        df = pd.read_csv(annotations_csv)
        return {
            filename: group.drop(columns=["Filename"]).to_dict(orient="records")
            for filename, group in df.groupby("Filename")
        }

    def extract_events(self, path):
        """Extract events from annotation df for one audio file"""
        return [{
            "start_s": r["Start Time (s)"],
            "end_s": r["End Time (s)"],
            "low_hz": r["Low Freq (Hz)"],
            "high_hz": r["High Freq (Hz)"],
            "species": r["Species eBird Code"],
        } for r in self.annotations.get(Path(path).name, [])]

    def gate(self, power, freqs, hop, e):
        if e["end_s"] - e["start_s"] < self.min_dur_s:
            return False  # 0.00 s annotations exist in this CSV

        # make band mask for freq range given in annotation
        band_mask = (freqs >= e["low_hz"]) & (freqs <= e["high_hz"])
        if not band_mask.any():
            return False
        # convert it to frames
        fs, fe = frames(e, self.target_sr, hop, power.shape[-1])

        # temporal SNR: this band now vs this band normally
        profile = power[band_mask].mean(dim=0)  # (T,) whole file
        # calc background noise of entire file using background_percentile of power
        bg = torch.quantile(profile, self.background_percentile / 100).item()
        # calc power above background percentile during event duration
        ev = torch.quantile(profile[fs:fe], self.event_percentile / 100).item()
        # check if event is louder than background noise
        if db_margin(ev, bg) < self.snr_margin_db:
            return False

       # Spectral SNR: local spectral check
        if self.local_margin_db is not None:
            # calc bandwidth and nyquist limit
            w = e["high_hz"] - e["low_hz"]
            nyq = self.target_sr / 2
            rivals = []
            # fetch frequency bands below and above the given one
            for lo, hi in ((e["low_hz"] - w, e["low_hz"]), (e["high_hz"], e["high_hz"] + w)):
                if lo < 0 or hi > nyq:
                    continue
                m = (freqs >= lo) & (freqs <= hi)
                # calc power for rival freq bands during event
                if m.any():
                    rivals.append(power[m, fs:fe].mean().item())
            # check if mean event freq band power is bigger than max power of rival bands
            if rivals and db_margin(power[band_mask, fs:fe].mean().item(),
                                    max(rivals)) < self.local_margin_db:
                return False

        return True

    def gating_config(self):
        """ Add additional configs """
        cfg = super().gating_config()
        cfg.update(dominance_margin_db=self.dominance_margin_db,
                   floor_percentile=self.floor_percentile)
        return cfg


class XenoCantoPipeline(AudioPipeline):
    def __init__(self, file_species, **kwargs):
        super().__init__(**kwargs)
        self.file_species = file_species

    def extract_active_chips(self, waveform):
        raise NotImplementedError

    def extract_events(self, path):
        raise NotImplementedError

    def gate(self, power, freqs, hop, event):
        raise NotImplementedError
