import hashlib
import json
import torch
import pandas as pd
import torchaudio.functional as F
import soundfile as sf
from scipy.io import wavfile
from abc import ABC, abstractmethod
from pathlib import Path
from DSP_helpers import *

def get_class_mapping(path):
    with open(path, "r", encoding="utf-8") as file:
        class_mappings = json.load(file)
    return class_mappings

class AudioPipeline(ABC):
    def __init__(self, cache_dir, target_sr=44100, highpass_hz=200.0, bandlimit_sr=32000, min_native_sr=32000, peak_dbfs = -1.0):
        self.target_sr = target_sr
        self.highpass_hz = highpass_hz
        self.bandlimit_sr = bandlimit_sr
        self.min_native_sr = min_native_sr
        self.peak_dbfs = peak_dbfs

        self.save_dir = Path(cache_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.skipped = {"low_native_sr": [], "too_short": []}

    def _save_params(self):
        """Returns custom name for saved file including preprocessing params"""
        #todo add all params
        return (f"sr{self.target_sr}_hp{self.highpass_hz:g}_bl{self.bandlimit_sr}"
                f"_pk{self.peak_dbfs:g}_mono_v2")

    def _save_paths(self, path):
        """Returns save path for wav and json file"""
        stem = f"{Path(path).stem}_{self._save_params()}"
        return self.save_dir / f"{stem}.wav", self.save_dir / f"{stem}.meta.json"

    def native_sr_pass(self, path):
        # skip files below 32 khz
        sr = sf.info(path).samplerate
        if sr < self.min_native_sr:
            self.skipped["low_native_sr"].append(str(path))
            return False
        return True


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

        #decode
        data, sr = sf.read(path, dtype="float32", always_2d=True)

        raw = torch.from_numpy(data).t()
        # clip
        clipped, clip_ratio = detect_clipping(raw)
        # mono, dc removal, highpass
        waveform = remove_dc_offset(to_mono(raw))
        waveform = highpass(waveform, sr, self.highpass_hz)
        # resample -> 32 khz -> 44.1 khz
        waveform = band_limit(waveform,  sr, self.bandlimit_sr, self.target_sr)
        waveform, gain_db = peak_normalize(waveform, self.peak_dbfs)

        wavfile.write(wav_path, self.target_sr, waveform.numpy().astype(np.float32))
        meta = {
            "native_sr": sr,
            "n_channels": int(raw.shape[0]),
            "upsampled": sr < self.target_sr,
            "bandwidth_hz": self.bandlimit_sr / 2,
            "gain_db": gain_db,
            "clip_ratio": clip_ratio,
            "clipped": bool(clipped),
        }
        meta_path.write_text(json.dumps(meta))
        meta["cached"] = False
        return waveform, meta, wav_path

    def row(self, path, wav_path, meta, n_samples, i, e, band_energy=float("nan"), call_energy=float("nan"),
            passes_gate=True,  snr_db=float("nan"), gate_p=float("nan")):
        """
        Row building for parquets. Passes gate default true for XC, same for band_energy
        """
        return {
            "event_id": f"{Path(path).stem}:{i}",
            "wav_path": str(wav_path),
            "source_path": str(path),
            "n_samples": int(n_samples),
            "start_s": float(e["start_s"]),
            "end_s": float(e["end_s"]),
            "low_hz": float(e["low_hz"]),
            "high_hz": float(e["high_hz"]),
            "species": e["species"],
            "band_energy": float(band_energy),
            "call_energy": float(call_energy),
            "passes_gate": bool(passes_gate),
            "snr_db": float(snr_db),
            "gate_p":  float(gate_p),
            "native_sr": meta.get("native_sr"),
            "n_channels": meta.get("n_channels"),
            "upsampled": meta.get("upsampled"),
            "bandwidth_hz": meta.get("bandwidth_hz"),
            "gain_db": meta.get("gain_db"),
            "clip_ratio": meta.get("clip_ratio"),
            "clipped": meta.get("clipped"),
        }
    @abstractmethod
    def extract_events(self, path):
        """Preprocesses annotations into standardized events"""

    def gating_config(self):
        """Records gate configs used"""
        return {"target_sr": self.target_sr, "highpass_hz": self.highpass_hz,
                "pipeline": type(self).__name__}

    @abstractmethod
    def precompute_events(self, path):
        """Compute rows and build them with self.row()"""


class SoundscapePipeline(AudioPipeline):
    def __init__(self, annotations_csv, n_fft=1024, gate_threshold=None,
                 background_percentile=25, event_percentile=90,
                 min_dur_s=0.05, band_energy_q=0.9, n_null=100, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.annotations = self.load_annotations(annotations_csv)
        #  allowed false-pass rate, None = no gate
        self.gate_alpha = gate_threshold
        self.n_fft = n_fft
        self.background_percentile = background_percentile
        self.event_percentile = event_percentile
        self.min_dur_s = min_dur_s
        self.band_energy_q = band_energy_q
        self.seed = seed
        self.n_null = n_null

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

    def gate(self, power, freqs, hop, e, busy_cs, generator):
        nan = float("nan")

        # skip events below min duration
        if e["end_s"] - e["start_s"] < self.min_dur_s:
            return False, nan, nan

        # make band mask for freq range given in annotation
        band_mask = (freqs >= e["low_hz"]) & (freqs <= e["high_hz"])
        if not band_mask.any():
            return False, nan, nan
        # mean power per frame inside band over whole file
        profile = power[band_mask].mean(dim=0)

        # convert it to frames
        n_frames = power.shape[-1]
        fs, fe = frames(e, self.target_sr, hop, n_frames)

        # temporal SNR: this band now vs this band normally,
        # calc background noise of entire file using background_percentile of power
        bg = torch.quantile(profile, self.background_percentile / 100).item()
        # calc power above background percentile during event duration
        ev = torch.quantile(profile[fs:fe], self.event_percentile / 100).item()
        snr_db = db_margin(ev, bg)

        if self.gate_alpha is None:
            return True, snr_db, nan

        # check same band but no annotations
        # get starts, pick free windows and calc snr
        starts = free_starts(busy_cs, fe-fs, n_frames)
        null = null_snr(profile, bg, starts, fe-fs, self.n_null,  self.event_percentile / 100, generator)

        #  No annotation free positions available
        if len(null) == 0:
            return True, snr_db, nan

        p_value = (1 + int((null >= snr_db).sum())) / (1 + len(null))
        return p_value <=  self.gate_alpha, snr_db, p_value
    def precompute_events(self, path):
        """
        Precomputes energies for extracted events
        """
        # get all events, annotations for soundscapes, snippets for XC
        # list of events for file, start time, end time, species + other info
        events = self.extract_events(path)
        # drop empty files or files with sr < 32000
        if not events or not self.native_sr_pass(path):
            return []

        # preprocess files and store preprocessed version
        waveform, meta, wav_path = self.preprocess_raw(path)

        # compute power for file + background bins
        power, freqs, hop = stft_power(waveform, self.target_sr, self.n_fft)
        # too many elems for torch.percentile
        bg_bins  = torch.from_numpy(np.percentile(power.cpu().numpy(), self.background_percentile, axis=1)).to(power.dtype)
        n_samples, n_frames = waveform.shape[-1], power.shape[-1]
        rows = []

        # frames covered by annotations
        busy_cs = annotated_cumsum([frames(e, self.target_sr, hop, n_frames) for e in events], n_frames)
        seed = int(hashlib.md5(f"{self.seed}:{Path(path).name}".encode()).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)

        # collects all important information, checks if gate is passed and deletes pow to free memory
        for i, e in enumerate(events):
            fs, fe = frames(e, self.target_sr, hop, n_frames)
            passes, snr_db, p_value = self.gate(power, freqs, hop,  e, busy_cs, generator)
            rows.append(self.row(path, wav_path, meta, n_samples, i, e,
                                 band_energy = band_energy(power, freqs, fs, fe, e["low_hz"], e["high_hz"], q=self.band_energy_q),
                                 call_energy=call_energy(power, freqs, bg_bins, fs, fe, e["low_hz"], e["high_hz"]),
                                 passes_gate=passes, snr_db=snr_db, gate_p=p_value)
            )

        del power
        return rows

    def gating_config(self):
        cfg = super().gating_config()
        cfg.update(n_fft=self.n_fft,
                   gate_alpha=self.gate_alpha,
                   seed=self.seed,
                   background_percentile=self.background_percentile,
                   event_percentile=self.event_percentile,
                   min_dur_s=self.min_dur_s,
                   band_energy_q=self.band_energy_q)
        return cfg


class XenoCantoPipeline(AudioPipeline):
    def __init__(self, class_mapping, min_dur_s=1.0, **kwargs):
        super().__init__(**kwargs)
        self.class_mapping = get_class_mapping(class_mapping)
        self.min_dur_s = min_dur_s

    def extract_events(self, path):
        species = Path(path).parent.name
        if sf.info(path).duration < self.min_dur_s:
            self.skipped["too_short"].append(str(path))
            return []
        else:
            return [{
                "start_s": 0.0,
                "end_s": None, #filled when file is read
                "low_hz": 0.0,
                "high_hz": self.bandlimit_sr /2,
                "species": species,
            }]

    def  precompute_events(self, path):
        events = self.extract_events(path)
        if not events:
            return []

        waveform, meta, wav_path = self.preprocess_raw(path)
        n_samples = int(waveform.shape[-1])

        rows = []
        for i, e in enumerate(events):
            e  = {**e, "end_s": n_samples / self.target_sr}
            rows.append(self.row(path, wav_path, meta, n_samples, i, e))
        return rows

    def  gating_config(self):
        cfg = super().gating_config()
        cfg.update(mode="whole_file", min_dur_s =self.min_dur_s)
        return cfg