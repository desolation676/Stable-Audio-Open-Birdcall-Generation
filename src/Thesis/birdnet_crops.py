import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# birdnet wants 3s clips
SEG_S = 3.0


def peak_norm(x, dbfs=-1.0):
    p = np.abs(x).max()
    return x * (10 ** (dbfs / 20) / p) if p > 0 else x


def read_mono(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(1), sr


def fit(x, n):
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def max_energy_start(x, n, hop):
    if len(x) <= n:
        return 0
    # calc energy and find window of length n with highest energy
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, len(x) - n + 1, hop)
    return int(starts[np.argmax(cs[starts + n] - cs[starts])])


def centred_start(start_s, end_s, sr, n, total):
    c = int(round((start_s + end_s) / 2 * sr))
    return int(np.clip(c - n // 2, 0, max(0, total - n)))


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def crop_real(args):
    from dataset import BirdDataset

    for split in args.splits:
        # can just use our anchors from BirdDataset
        ds = BirdDataset(args.parquet_path, args.class_mapping_path, split=split,
                         target_channels=1, random_crop=False, cache_dir=args.cache_dir,
                         gate_alpha="manifest")

        out = Path(args.out) / "real" / split
        rows, cache = [], {}

        for a in tqdm(ds.anchors.itertuples(), total=len(ds.anchors), desc=f"real/{split}"):
            if a.wav_path not in cache:
                cache = {a.wav_path: read_mono(a.wav_path)}
            x, sr = cache[a.wav_path]
            n = int(SEG_S + sr)

            # we use annotations for SS and max energy for XC to find the best 3s call window
            if a.dataset == "soundscape":
                s, rule = centred_start(a.start_s, a.end_s, sr, n, len(x)), "annotation_centre"
            else:
                s, rule = max_energy_start(x, n, int(0.1 * sr)), "max_rms"

            clip = peak_norm(fit(x[s:s + n], n))
            f = out / a.species / f"{safe(a.event_id)}.wav"
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(f, clip, sr)
            rows.append({"file": f"{a.species}/{f.name}", "species": a.species, "dataset": a.dataset,
                         "recording": Path(a.source_path).stem, "event_id": a.event_id,
                         "crop_start_s": s / sr, "rule": rule, "split": split})

        m = pd.DataFrame(rows)
        m.to_csv(out / "manifest.csv", index=False)
        print(f"\n{split}: {len(m)} crops -> {out}")
        print(m.groupby(["species", "dataset"]).recording.nunique().unstack(fill_value=0)
              .rename(columns=lambda c: f"rec_{c}").to_string())


def crop_synth(args):
    for gd in map(Path, args.gen_dirs):
        out = Path(args.out) / "synth" / gd.name
        rows = []
        files = sorted(p for p in gd.rglob("*.wav") if p.parent != gd)
        for p in tqdm(files, desc=f"synth/{gd.name}"):
            species = p.parent.name
            x, sr = read_mono(p)
            n = int(SEG_S * sr)
            s = max_energy_start(x, n, int(0.1 * sr))
            f = out / species / p.name
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(f, peak_norm(fit(x[s:s + n], n)), sr)
            rows.append({"file": f"{species}/{p.name}", "species": species, "dataset": "synthetic",
                         "recording": f"{gd.name}/{p.stem}", "crop_start_s": s / sr,
                         "rule": "max_rms", "set": gd.name})
        m = pd.DataFrame(rows)
        m.to_csv(out / "manifest.csv", index=False)
        print(f"{gd.name}: {len(m)} crops, {m.species.nunique()} species -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("--parquet-path", required=True)
    r.add_argument("--class-mapping-path", required=True)
    r.add_argument("--cache-dir", default=None, help="only needed if wav paths in the parquet are from another machine")
    r.add_argument("--splits", nargs="+", default=["train", "val"])
    r.add_argument("--final", action="store_true")
    r.add_argument("--out", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--gen-dirs", nargs="+", required=True, help="generation folders with one subfolder per species")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    crop_real(args) if args.cmd == "real" else crop_synth(args)


if __name__ == "__main__":
    main()
