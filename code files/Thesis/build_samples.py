
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipelines import SoundscapePipeline, XenoCantoPipeline


def build_dataset_parquets(pipeline, source_paths, out_path, dataset, resume=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # get the gate config
    cfg = pipeline.gating_config()
    cfg["dataset"] = dataset
    # hashed configs to filter out rows from prev runs
    cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    rows, done = [], set()
    # check if it was already computed
    if resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        wanted = {str(p) for p in source_paths}
        prev = prev[prev.source_path.isin(wanted) & (prev.cfg_hash == cfg_hash)]
        rows = prev.to_dict(orient="records")
        done = set(prev.source_path)
        print(f"resuming: {len(done)} files already current")

    # else precompute events for all files and append information to list
    for path in tqdm(source_paths, desc=dataset):
        if str(path) in done:
            continue
        try:
            for r in pipeline.precompute_events(path):
                r["dataset"] = dataset
                r["cfg_hash"] = cfg_hash
                r["event_id"] = f"{dataset}/{r['event_id']}"
                rows.append(r)
        except Exception as exc:
            print(f"  SKIP {Path(path).name}: {type(exc).__name__}: {exc}")

    # save as parquet
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)

    # save config json
    cfg["cfg_hash"] = cfg_hash
    out_path.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2))

    # check gate configuration
    n_pass = int(df.passes_gate.sum()) if len(df) else 0
    print(f"\n{len(df)} events, {n_pass} pass the gate "
          f"({n_pass / max(len(df), 1):.1%}), {df.species.nunique() if len(df) else 0} species")
    if len(df) and not 0.05 < n_pass / len(df) < 0.95:
        print("  WARNING: gate is near-degenerate -- calibrate the thresholds")
    return df


def merge_parquets(shard_paths, out_path):
    # read parquets from each dataset and merge them into one for the dataloader
    df = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
    dupes = int(df.event_id.duplicated().sum())
    assert dupes == 0, f"{dupes} duplicate event_ids across shards"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"merged {len(df)} events from {len(shard_paths)} shards -> {out_path}")
    return df


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["soundscape", "xc"])
    ap.add_argument("--audio-dir", nargs="+")
    ap.add_argument("--annotations")
    ap.add_argument("--class-mapping")
    ap.add_argument("--cache-dir", default="cache/")
    ap.add_argument("--glob", default="*.flac")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge_parquets", nargs="+")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--snr_margin_db", type=float, default=3.0)
    ap.add_argument("--min_dur_s", type=float, default=0.05)
    ap.add_argument("--background_percentile", type=float, default=25)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.merge_parquets:
        merge_parquets(args.merge_parquets, args.out)
        return

    paths = sorted(p for d in args.audio_dir for p in Path(args.audio_dir).glob(args.glob))
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} files")

    if args.dataset == "soundscape":
        pipe = SoundscapePipeline(
            annotations_csv=args.annotations,
            cache_dir=args.cache_dir,
            snr_margin_db=args.snr_margin_db,
            background_percentile=args.background_percentile,
            min_dur_s=args.min_dur_s
        )
    else:
        pipe = XenoCantoPipeline(
            class_mapping=args.class_mapping,
            cache_dir=args.cache_dir,
        )

    build_dataset_parquets(pipe, paths, args.out, args.dataset, resume=not args.no_resume)

if __name__ == "__main__":
    main()