import argparse
from pathlib import Path
 
import pandas as pd
import torch
 
from DSP_helpers import stft_power, db_margin
from pipelines import SoundscapePipeline
 
 
def score_event(pipe, power, freqs, hop, e):
    """Same quantities gate() computes, but nothing is thresholded -- every
    event gets a score so you can see the whole distribution."""
    band_mask = (freqs >= e["low_hz"]) & (freqs <= e["high_hz"])
    if not band_mask.any():
        return None
 
    fs = max(0, int(round(e["start_s"] * pipe.target_sr)) // hop)
    fe = max(fs + 1, min(int(round(e["end_s"] * pipe.target_sr)) // hop, power.shape[-1]))
 
    profile = power[band_mask].mean(dim=0)
    ev_frames = profile[fs:fe]
 
    bgs = {p: torch.quantile(profile, p / 100).item() for p in (10, 25, 50)}
    ev90 = torch.quantile(ev_frames, 0.9).item()
    evmean = ev_frames.mean().item()
 
    row = {
        "file": None, "species": e["species"],
        "dur_s": e["end_s"] - e["start_s"], "n_frames": fe - fs,
        # the metric gate() thresholds on, at each candidate background level
        **{f"snr_p{p}_db": db_margin(ev90, bg) for p, bg in bgs.items()},
        # what the old mean-based band_energy would have scored, for contrast
        "snr_mean_p25_db": db_margin(evmean, bgs[25]),
    }
 
    # local spectral check, scored but not applied
    w = e["high_hz"] - e["low_hz"]
    nyq = pipe.target_sr / 2
    rivals = []
    for lo, hi in ((e["low_hz"] - w, e["low_hz"]), (e["high_hz"], e["high_hz"] + w)):
        if lo < 0 or hi > nyq:
            continue
        m = (freqs >= lo) & (freqs <= hi)
        if m.any():
            rivals.append(power[m, fs:fe].mean().item())
    row["local_db"] = (db_margin(power[band_mask, fs:fe].mean().item(), max(rivals))
                       if rivals else float("nan"))
    return row
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--cache-dir", default="cache/")
    ap.add_argument("--glob", default="*.flac")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--min-dur-s", type=float, default=0.05)
    ap.add_argument("--out", default="gate_diag_v2.csv")
    args = ap.parse_args()
 
    pipe = SoundscapePipeline(annotations_csv=args.annotations, cache_dir=args.cache_dir)
    paths = sorted(Path(args.audio_dir).glob(args.glob))[:args.limit]
 
    rows = []
    for p in paths:
        waveform, meta, _ = pipe.preprocess_raw(p)
        events = pipe.extract_events(p)
        if not events:
            continue
        power, freqs, hop = stft_power(waveform, pipe.target_sr, pipe.n_fft)
        for e in events:
            r = score_event(pipe, power, freqs, hop, e)
            if r:
                r["file"] = p.name
                rows.append(r)
        del power
 
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
 
    n_short = int((df.dur_s < args.min_dur_s).sum())
    df = df[df.dur_s >= args.min_dur_s]
    print(f"\n{len(df)} events ({n_short} dropped as shorter than "
          f"{args.min_dur_s}s) -> {args.out}\n")
 
    print("temporal SNR (event p90 vs background), dB:")
    print(df[["snr_p10_db", "snr_p25_db", "snr_p50_db", "snr_mean_p25_db"]]
          .describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
 
    print("\npass rate by snr_margin_db, background = p25:")
    for m in [0, 3, 6, 9, 12, 15, 20]:
        keep = df[df.snr_p25_db >= m]
        n_ok = (keep.species.value_counts() >= 20).sum()
        print(f"  >= {m:>2} dB : {len(keep) / len(df):>6.1%}  "
              f"({keep.species.nunique()}/{df.species.nunique()} species kept, "
              f"{n_ok} with >=20 events)")
 
    print("\nlocal spectral margin, dB (not applied):")
    print(df.local_db.describe(percentiles=[.1, .5, .9]).round(2).to_string())
 
 
if __name__ == "__main__":
    main()