
import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from dataset import BirdDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-path", required=True)
    ap.add_argument("--class-mapping-path", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-size", type=int, default=524288)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--overlap-margin-db", type=float, default=3.0)
    ap.add_argument("--cache-dir", default=None, help="where the cached wavs live on THIS machine")
    ap.add_argument("--final", action="store_true",
                    help="required for --split test: confirms CFG/steps are already fixed on val")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        raise SystemExit("Refusing to export test before model selection is done. "
                         "Choose CFG/steps on val first, then rerun with --final.")

    ds = BirdDataset(parquet_path=args.parquet_path, class_mapping_path=args.class_mapping_path,
                     split=args.split, sample_size=args.sample_size, target_sr=args.sample_rate,
                     target_channels=1, overlap_margin_db=args.overlap_margin_db,
                     random_crop=False, cache_dir=args.cache_dir,
                     gate_alpha="manifest")   # same gated reference set for every model (gate and no-gate)

    out = Path(args.out)
    rows = []
    for i in range(len(ds)):
        clip, meta = ds[i]
        a = ds.anchors.iloc[i]
        n_real = int(meta["padding_mask"][0].sum().item())
        audio = clip[0, :n_real].numpy()             # drop zero padding of short files

        species_dir = out / a.species
        species_dir.mkdir(parents=True, exist_ok=True)
        fname = meta["event_id"].replace("/", "_").replace(":", "_") + ".wav"
        sf.write(species_dir / fname, audio, args.sample_rate)

        rows.append({"file": f"{a.species}/{fname}", "event_id": meta["event_id"],
                     "species": a.species, "species_id": meta["species_id"],
                     "dataset": a.dataset, "source_path": a.source_path,
                     "duration_s": n_real / args.sample_rate})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    (out / "export_config.json").write_text(json.dumps({**vars(args), "counts": ds.counts}, indent=2))

    print(f"{len(manifest)} clips -> {out}")
    print(manifest.groupby(["dataset", "species"]).size().unstack(0).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
