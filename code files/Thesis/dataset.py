import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
from DSP_helpers import db_margin, loudness_normalize
import torch
from scipy.io import wavfile

def get_class_mapping(path):
    with open(path, "r", encoding="utf-8") as file:
        class_mappings = json.load(file)
    return class_mappings


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
        self.rivals_by_file = {k: v for k, v in df.groupby("wav_path")}

        anchors = df[df.passes_gate & df.species.isin(self.class_mapping)]
        print(f"gate:   {len(anchors)}")
        anchors = self.dedup(anchors)
        print(f"dedup:  {len(anchors)}")
        anchors = self.veto_overlaps(anchors, overlap_margin_db)
        print(f"veto:   {len(anchors)}")
        self.anchors = anchors.reset_index(drop=True)

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
            mid = (a.start_s + a.end_s) / 2
            window_start = mid - self.window_s / 2 - self.jitter
            window_end = mid + self.window_s / 2 + self.jitter

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
        mid = (a.start_s + a.end_s) / 2

        if self.jitter:
            mid += (torch.rand(1).item()  * 2 - 1) * self.jitter

        # int cause we only work with full samples
        start = int(round((mid - self.window_s /2) *  self.sr))
        # safe indexing
        start = max(0, min(start, max(0, a.n_samples - self.sample_size)))

        _, data  = wavfile.read(a.wav_path, mmap=True)
        clip = torch.from_numpy(np.asarray(data[start:start + self.sample_size]).copy()).float()
        # check size, pad with 0
        if clip.numel() < self.sample_size:
            clip = torch.nn.functional.pad(clip, (0, self.sample_size - clip.numel()))
        clip = clip.unsqueeze(0)

        # convert to stereo
        if self.target_channels > 1:
            clip = clip.repeat(self.target_channels, 1)

        clip = loudness_normalize(clip)
        assert clip.shape[-1] == self.sample_size
        # todo consider other second start option
        seconds_start = start / self.sr

        return {
            "audio": clip,
            "species_id": self.class_mapping[a.species],
            "seconds_start": float(seconds_start),
            "seconds_total": float(a.n_samples / self.sr),
            "prompt": "A field recording of a bird singing in nature, stereo audio",
            "event_id": a.event_id,
        }

    #  todo train test split
