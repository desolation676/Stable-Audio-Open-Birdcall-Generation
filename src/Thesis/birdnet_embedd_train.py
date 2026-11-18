import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from birdnetlib.analyzer import Analyzer
from birdnetlib import RecordingBuffer

KS = [4, 8, 16, 32, "all"]
synth_all = 100
synth_ratios = [1, 2]
re = 5
c_values = [0.01, 0.1, 1.0, 10.0]


# Load and embed
def load_birdnet():
    # get rid of birdnet prints (prints a lot)
    with contextlib.redirect_stdout(io.StringIO()):
        return Analyzer


def birdnet_embedding(wav_path, analyzer):
    audio, sr = sf.read(wav_path, dtype="float32")
    # gens are already at that rate but reals not
    audio_48k = resample_poly(audio, 48000, sr).astype(np.float32)

    recording = RecordingBuffer(analyzer, audio_48k, 48000)
    with contextlib.redirect_stdout(io.StringIO()):
        recording.extract_embeddings()

    # should only be one for 3s clip
    vectors = [chunk["embeddings"] for chunk in recording.embeddings]
    return np.mean(vectors, axis=0)


def embed_folder(folder, analyzer):
    folder = Path(folder)
    # each set has a manifest, like used for training SAO
    manifest = pd.read_csv(folder / "manifest.csv")
    out_file = folder / "birdnet_emb.npz"

    embeddings = [birdnet_embedding(folder / f, analyzer) for f in manifest.file]
    np.savez(out_file, emb=np.asarray(embeddings, dtype=np.float32))
    print(f"{len(embeddings)}  embeddings saved to {out_file}")


def cmd_embed(args):
    analyzer = load_birdnet()
    for manifest in sorted(Path(args.crops).rglob("manifest.csv")):
        embed_folder(manifest.parent, analyzer)


# Sample
def load_folder(folder):
    folder = Path(folder)
    manifest = pd.read_csv(folder / "manifest.csv")
    embeddings = np.load(folder / "birdnet_emb.npz")["emb"]
    return manifest, embeddings


def sample_real(train, k, rng):
    if k == "all":
        return list(train.index)

    rows = []
    # choose k random crops from recordings per species
    for species, crops in train.groupby("species"):
        recordings = rng.permutation(crops.recording.unique())[:k]
        for rec in recordings:
            crops_of_rec = crops.index[crops.recording == rec]
            rows.append(rng.choice(crops_of_rec))

    return rows


def sample_synthetic(synth, clips_per_species, rng):
    rows = []
    for species, clips in synth.groupby("species"):
        n = min(clips_per_species.get(species, 0), len(clips))
        rows.extend(rng.choice(clips.index, n, replace=False))
    return rows


def build_training_sets(k, rng, train, train_emb, synth_sets):
    # get indices, fetch embeddings and species
    real_rows = sample_real(train, k, rng)
    real_X = train_emb[real_rows]
    real_y = train.species.to_numpy()[real_rows]
    # some might not have enough samples for k
    real_per_species = pd.Series(real_y).value_counts().to_dict()

    sets = [{"condition": "real", "synth_set": "-", "X": real_X, "y": real_y}]

    for name, (synth, synth_emb) in synth_sets.item():
        # take the exact ratio not just k times ratio
        if k == "all":
            ratios = {f"real+synth{synth_all}":
                          {sp: synth_all for sp in real_per_species}}
        else:
            ratios = {f"real+synth{r}x": {sp: r * n for sp, n in real_per_species.items()}
                      for r in synth_ratios}

        # append synth samples according to ratio or all
        for condition, counts in ratios.items():
            rows = sample_synthetic(synth, counts, rng)
            sets.append({"condition": condition, "synth_set": name,
                         "X": np.vstack([real_X, synth_emb[rows]]),
                         "y": np.concatenate([real_y, synth.species.to_numpy()[rows]])})

        # and the synth only case
        rows = sample_synthetic(synth, real_per_species, rng)
        sets.append({"condition": "synth_only", "synth_set": name,
                     "X": synth_emb[rows], "y": synth.species.to_numpy()[rows]})
    return sets


# train & score
def train_classifier(X, y, X_val, y_val):
    best_f1, best_C, best_model = -1, None, None
    for c in c_values:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
        model.fit(X, y)
        f1 = f1_score(y_val, model.predict(X_val), average="macro")
        if f1 > best_f1:
            best_f1, best_C, best_model = f1, c, model

    return best_model, best_C


def score(model, X, y):
    pred = model.predict(X)
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "balanced_acc": float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))}


def run(args):
    crops = Path(args.crops)
    train, train_emb = load_folder(crops / "real" / "train")
    val, val_emb = load_folder(crops / "real" / "val")
    val_y = val.species.to_numpy()

    if args.final:
        test, test_emb = load_folder(crops / "real" / "test")
        eval_name, eval_X, eval_y = "test", test_emb, test.species.to_numpy()
    else:
        eval_name, eval_X, eval_y = "val", val_emb, val_y

    synth_sets = {name: load_folder(crops / "synth" / name) for name in args.synth}
    ks = [k if k == "all" else int(k) for k in args.ks]
    results = []

    for k in ks:
        for repeat in range(re):
            rng = np.random.default_rng(1000 * repeat + (0 if k == "all" else k))
            for ts in build_training_sets(k, rng, train, train_emb, synth_sets):
                model, c = train_classifier(ts["X"], ts["y"], val_emb, val_y)
                s = score(model, eval_X, eval_y)
                results.append({"k": k, "repeat": repeat, "condition": ts["condition"],
                                "synth_set": ts["synth_set"], "n_train": len(ts["y"]), "C": c,
                                f"{eval_name}_macro_f1": s["macro_f1"],
                                f"{eval_name}_balanced_acc": s["balanced_acc"]})

                print(f"k={str(k):>3}  repeat={repeat}  {ts['condition']:18s} {ts['synth_set']:40s} "
                      f"n={len(ts['y']):5d}  {eval_name} macro F1 = {s['macro_f1']:.3f}", flush=True)

    results = pd.DataFrame(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / f"birdnet_train_{eval_name}_runs.csv", index=False)

    metric = f"{eval_name}_macro_f1"
    summary = (results.groupby(["k", "condition", "synth_set"], sort=False)[metric]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(out / f"downstream_{eval_name}_summary.csv")
    print("\n" + summary.to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--crops", required=True)

    r = sub.add_parser("run")
    r.add_argument("--crops", required=True)
    r.add_argument("--synth", nargs="+", default=[])
    r.add_argument("--ks", nargs="+", default=[str(k) for k in KS])
    r.add_argument("--out", default="../../results/birdnet_train")
    r.add_argument("--final", action="store_true")

    args = ap.parse_args()

    if args.cmd == "embed":
        cmd_embed(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
