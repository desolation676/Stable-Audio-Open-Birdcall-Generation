import numpy as np
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler
from DSP_helpers import db_margin, loudness_normalize
from pipelines import get_class_mapping
import torch
from scipy.io import wavfile
import math



class BirdDataset(Dataset):
    def __init__(self, parquet_path, class_mapping_path, sample_size=524288, target_sr=44100, target_channels=2
                 , jitter=0.0, overlap_margin_db=3.0):
        self.sr = target_sr
        self.sample_size = sample_size
        self.target_channels = target_channels
        # implement specific prompt?
        self.class_mapping = get_class_mapping(class_mapping_path)
        self.window_s = sample_size / target_sr
        self.jitter = jitter

        df = pd.read_parquet(parquet_path)
        # rival = same file different species event but all events for now
        # rivals dont exist for XC as they are all separate files. No extra handling needed
        self.rivals_by_file = {k: v for k, v in df.groupby("wav_path")}

        anchors = df[df.passes_gate & df.species.isin(self.class_mapping)]
        print(f"gate:   {len(anchors)}")
        anchors = self.dedup(anchors)
        print(f"dedup:  {len(anchors)}")
        anchors = self.veto_overlaps(anchors, overlap_margin_db)
        print(f"veto:   {len(anchors)}")
        self.anchors = anchors.reset_index(drop=True)

    def start_range(self, a):
        # int cause we only work with full samples
        s = int(round(a.start_s * self.sr))
        e =  int(round(a.end_s * self.sr))

        # safe indexing
        if getattr(a, "dataset", None) == "xc":
            lo, hi = min(s, e - self.sample_size), max(s, e - self.sample_size)
        else:
            j = int(round(self.jitter * self.sr))
            c = (s + e) // 2 - self.sample_size // 2
            lo, hi = c - j, c + j
        last = max(0, int(a.n_samples) - self.sample_size)
        lo, hi = max(0, min(lo, last)), max(0, min(hi, last))
        return lo, max(lo, hi)

    def dedup(self, anchors):
        """Combine overlapping or back to back events of same species into (sample sized) window"""
        keep = []
        for _, grp in anchors.groupby(["wav_path", "species"], sort=False):
            last_end = -float("inf")
            for _, e in grp.sort_values("start_s").iterrows():
                if e.start_s > last_end:
                    keep.append(e.event_id)
                    last_end = (e.start_s + e.end_s) / 2 + self.window_s / 2
        return anchors[anchors.event_id.isin(keep)]

    def veto_overlaps(self, anchors, overlap_margin_db):
        """Checks overlapping events of different species """
        keep = []
        for _, a in anchors.iterrows():
            lo, hi = self.start_range(a)
            window_start = lo / self.sr
            window_end = (hi + self.sample_size) / self.sr

            r = self.rivals_by_file[a.wav_path]
            # not same event, not same species, rival not finished before window start, rival starts during active window
            r = r[(r.event_id != a.event_id) & (r.species != a.species)
                  & (r.end_s > window_start) & (r.start_s < window_end)]
            # check if anchor is  higher energy as all rivals on band
            if r.empty or all(db_margin(a.band_energy, x) >= overlap_margin_db for x in r.band_energy):
                keep.append(a.event_id)
        return anchors[anchors.event_id.isin(keep)]

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        a =  self.anchors.iloc[idx]
        lo, hi = self.start_range(a)
        start = lo if hi <= lo else lo + int(torch.randint(0, hi - lo + 1, (1,)).item())

        _, data  = wavfile.read(a.wav_path, mmap=True)
        clip = torch.from_numpy(np.asarray(data[start:start + self.sample_size]).copy()).float()
        # normalize loudness before pad
        clip = loudness_normalize(clip)
        # check size, pad with 0
        n_real = clip.numel()
        if n_real  < self.sample_size:
            clip = torch.nn.functional.pad(clip, (0, self.sample_size - n_real))

        # padding mask shows which samples are real and which are padding for training wrapper
        padding_mask = torch.ones(self.sample_size)
        padding_mask[n_real:] = 0.0

        clip = clip.unsqueeze(0)

        # convert to stereo
        if self.target_channels > 1:
            clip = clip.repeat(self.target_channels, 1)


        assert clip.shape[-1] == self.sample_size
        # todo consider other second start option
        # expects tuple
        return clip, {
            "species_id": self.class_mapping[a.species]["id"],
            "seconds_start": math.floor(start / self.sr),
            "seconds_total": math.ceil(a.n_samples / self.sr),
            "padding_mask": [padding_mask],
            "prompt": "A field recording of a bird singing in nature, stereo audio",
            "event_id": a.event_id,
        }

def species_sampler(ds, alpha=0.5):
    """Wrapper for balanced WeightedRandomSampler, alpha defines aggressiveness"""
    counts = ds.anchors.species.value_counts()
    w = ds.anchors.species.map(lambda s: counts[s] ** -alpha).to_numpy()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(ds), replacement=True)