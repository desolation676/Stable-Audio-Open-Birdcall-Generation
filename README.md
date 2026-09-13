# Birdcall Generation with Stable Audio Open

Fine-tuning [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
to generate species-conditioned bird vocalisations, and measuring whether the result is
actually recognisable as the target species.

This repository covers two consecutive pieces of work:

|        | Phase 1 — Practical Work                                | Phase 2 — Master's thesis                          |
|--------|---------------------------------------------------------|----------------------------------------------------|
| Status | Complete                                                | In progress                                        |
| Topic  | Birdcall sample generation using per-species recordings | Multi species Birdcall soundscape generation       |
| Data   | Xeno-Canto, plus additional data from Kaggle            | Annotated ARU soundscapes, unified with Xeno-Canto |
| Code   | [`src/PR/`](src/PR)                            | [`src/Thesis/`](src/Thesis)               |

Phase 1 established that species conditioning works, and how. Phase 2 moves from clean,
single-species recordings to continuous field audio, where the events have to be found and
isolated before anything can be trained on them.

---

# Phase 1 — Practical Work

Stable Audio Open conditions on a T5 text embedding, and bird species names carry almost no
useful signal in that embedding. This phase compares six ways of injecting species
information and evaluates each with an external classifier.

**Headline result:** routing a learned species embedding through cross-attention reaches
**87.6% BirdNET species accuracy** against **90.5% for real recordings**, while text-prompt
conditioning alone reaches **0.2%**.

## Data

Xeno-Canto recordings supplemented with additional data from Kaggle: targeted,
single-species clips, one species per file, largely free of overlapping vocalisations.
Training used the `audio_dir` loader from `stable-audio-tools` with random cropping, on a
pre-materialised preprocessed corpus.

## Evaluation

Two metrics:

- **BirdNET accuracy**: generated clips are passed to [BirdNET](https://birdnet.cornell.edu/),
  Cornell's bird sound classifier, and scored on whether it identifies the intended species.
  This measures *controllability*. Real recordings from the corpus score 90.46% under the
  same procedure, which is the practical ceiling.
- **FAD**: Fréchet Audio Distance against the real corpus. This measures *realism*, and investigates if a model can produce convincing bird-like audio that is the
  wrong species, or the right species with obvious artefacts.

21 species, 30 generated clips per species per configuration.

## Conditioning variants

| Variant | Text prompt      | Species enters via                        |
|---|------------------|-------------------------------------------|
| `baseline` | specific         | prompt only                               |
| `general` | generic          | global conditioning|
| `specific` | species-specific | global conditioning |
| `no_prompt` | none             | cross-attention    |
| `general_CA` | generic          | cross-attention    |
| `specific_CA` | species-specific | cross-attention    |

Model configs for each variant are in [`checkpoints/`](checkpoints).

## Results

BirdNET accuracy (%), with FAD in parentheses. CFG = classifier-free guidance scale.

| Variant | Steps | CFG 3 | CFG 6 | CFG 9 |
|---|---|---|---|---|
| **Real recordings** | — | | 90.46 | |
| `baseline` | — | 0.16 (2.3e-4) | 0.16 (1.6e-3) | 0.63 (2.2e-3) |
| `general` | 6000 | 5.87 (4.0e-4) | 5.87 (5.0e-4) | 6.67 (3.8e-4) |
| `specific` | 6000 | 52.86 (2.1e-4) | 63.49 (1.4e-4) | 67.30 (1.3e-4) |
| `specific` | 10000 | 55.56 (5.1e-4) | 71.75 (4.2e-4) | 72.86 (3.5e-4) |
| `no_prompt` | 6000 | 45.56 (4.1e-4) | 54.13 (5.0e-4) | 55.24 (5.0e-4) |
| `no_prompt` | 10000 | 62.54 (5.8e-4) | 68.73 (4.7e-4) | 69.21 (6.0e-4) |
| `specific_CA` | 6000 | 47.46 (3.4e-4) | 66.98 (2.2e-4) | 70.63 (1.9e-4) |
| `specific_CA` | 10000 | 64.92 (4.0e-4) | 79.37 (2.4e-4) | 82.70 (1.8e-4) |
| `general_CA` | 6000 | 58.89 (4.6e-4) | 76.67 (2.6e-4) | 82.38 (2.0e-4) |
| **`general_CA`** | **10000** | 72.86 (2.1e-4) | **87.62 (1.4e-4)** | 87.46 (1.5e-4) |

Per-species breakdowns for every row are in [`results.txt`](results.txt).

### Discussion

**The text prompt cannot carry species identity.** The baseline never exceeds 0.63%. T5
embeddings of species names do not give the model anything to condition on, so it produces
generic bird-like audio.

**How the species ID is injected matters more than anything else.** `general` and
`general_CA` differ only in whether `species_id` is a scalar passed to global conditioning
or an integer routed to a learned embedding in cross-attention — 6.67% versus 82.38% at the
same step count.

**Putting the species name in the text prompt hurts once cross-attention works.** The text channel appears to compete with the
embedding rather than reinforce it.

**Accuracy rises with CFG throughout, but FAD does not improve monotonically**. The best configuration overall is
CFG 6 at 10000 steps, where both metrics peak together.

---

# Phase 2 - WORK IN PROGRESS — Master's thesis 

Phase 1 trained on clean, targeted recordings.
Continuous autonomous recording unit (ARU) soundscapes, low signal-to-noise, overlapping
species, long stretches of nothing. The thesis addresses the data problem that creates
locating usable vocalisation events in continuous audio and turning them into training
windows without silently poisoning the labels.

The two corpora are unified behind one interface:

- **Annotated soundscapes** — continuous ARU recordings with per-event annotations (start,
  end, frequency band, eBird species code). Realistic, noisy, frequently overlapping.
- **Xeno-Canto** — reused from Phase 1, now routed through the same pipeline so that both
  corpora produce identical manifest rows.

## Pipeline

**1. Decode and cache** — [`pipelines.py: AudioPipeline.preprocess_raw`](src/Thesis/pipelines.py)

**2. Event manifest with gating** — [`build_samples.py`](src/Thesis/build_samples.py)

**3. Dynamic windowing at training time  + Xeno Canto data** — [`dataset.py: BirdDataset`](src/Thesis/dataset.py)



## Usage

...

## Open points

...

## Repository structure

```
scr/PR/           Phase 1 — practical work
  data_exploration.ipynb     Corpus statistics
  data_preprocessing.ipynb   Preprocessing for the audio_dir loader
  model_setup.py             Model loading
  model_expand.py            Conditioning expansion for the variants
  custom_metadata.py         Metadata hook for stable-audio-tools
  model_evaluation.ipynb     BirdNET scoring and FAD
results.txt              Phase 1 — full per-species results, every configuration
checkpoints/             Phase 1 — one model config per conditioning variant

scr/Thesis/       Phase 2 — master's thesis
  DSP_helpers.py             Filtering, resampling, STFT power, band energy, dB margins
  pipelines.py               AudioPipeline ABC, SoundscapePipeline, XenoCantoPipeline
  build_samples.py           CLI: builds and merges Parquet event manifests
  dataset.py                 BirdDataset: dedup, overlap veto, dynamic windowing

class_mappings/          Species code to class index mappings
train_scripts/           Training launchers (local and RunPod)
stable-audio-tools/      Vendored copy of the Stability AI training framework
```

Training uses `stable-audio-tools`; see `train_scripts/`.
