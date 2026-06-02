# Data-Speech — Nepali Fork

**`dataspeech-nepali`** is a Nepali-adapted version of [HuggingFace's Data-Speech](https://github.com/huggingface/dataspeech).

It replaces the English `g2p` phonemizer with an `espeak-ng` Nepali backend (with a pure-Python Devanagari syllable-count fallback), adds Nepali-language LLM prompt templates, and ships a new script to compute dataset-calibrated bin edges from your own Nepali speech data.

Everything else — GPU enrichments (pitch, SNR, reverberation, squim), `metadata_to_text.py`, the TGI inference path — is unchanged from upstream, so this fork is a drop-in replacement for any Nepali TTS pipeline.

Its primary use is to reproduce the annotation method from Dan Lyth and Simon King's research paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912), and to prepare Nepali datasets for fine-tuning [Parler-TTS](https://github.com/huggingface/parler-tts).

---

## 📖 Quick Index

- [What's different from upstream](#whats-different-from-upstream)
- [Requirements](#set-up)
- [Annotating a Nepali dataset to fine-tune Parler-TTS](#annotating-a-nepali-dataset-to-fine-tune-parler-tts)
- [Annotating Nepali datasets from scratch](#annotating-nepali-datasets-from-scratch)
- [Using Data-Speech to filter your speech datasets](#using-data-speech-to-filter-your-speech-datasets)
- [❓ FAQ](#faq)
- [Logs](#logs)

---

## What's different from upstream

### New files

| File | Purpose |
|---|---|
| `main_nepali.py` | Nepali entry point (default `--text_column_name transcription`, adds `--language ne` flag, Nepali-labelled print output) |
| `scripts/run_prompt_creation_nepali.py` | Prompt creation with **Nepali-language LLM prompts** and a `--prompt_language` switch (`ne` / `en`) |
| `scripts/compute_bin_edges_nepali.py` | Computes percentile bin edges from **your actual Nepali data** instead of reusing English LibriTTS-R defaults |
| `examples/tags_to_annotations/v01_text_bins_nepali.json` | Nepali keyword label vocabulary (Devanagari) |
| `examples/tags_to_annotations/v01_bin_edges_nepali.json` | Starter Nepali bin edges (derived from OpenSLR-43/54 statistics; replace with your own via `compute_bin_edges_nepali.py`) |
| `examples/run_nepali_pipeline.sh` | Full end-to-end shell script for the four-step Nepali pipeline |
| `nepali/01_run_annotation.sh` | Step 1 helper |
| `nepali/02_run_metadata_to_text.sh` | Step 3 helper |
| `nepali/03_run_prompt_creation.sh` | Step 4 helper |

### Modified files

**`dataspeech/cpu_enrichments/rate.py`** — the core change.

The upstream version depends on the English `g2p` library:
```python
from g2p import make_g2p
transducer = make_g2p('eng', 'eng-ipa')
phonemes = transducer(text).output_string
```

This fork replaces it with a two-tier Nepali phonemizer (same public `rate_apply()` API, so all downstream scripts work unchanged):

1. **`espeak-ng` via subprocess** — calls `espeak-ng -v ne --ipa` for accurate Nepali IPA. Used when `espeak-ng` is installed.
2. **Devanagari syllable counter** — pure Python fallback; counts syllable nuclei from Devanagari Unicode ranges. Zero extra dependencies, no crash if `espeak-ng` is absent.

**`main_nepali.py`** (separate file, `main.py` untouched):
- Default `--text_column_name` changed from `"text"` → `"transcription"` (standard in Nepali TTS datasets)
- Added `--language ne` flag
- Print statements prefixed with `[Nepali]` for clarity

**Why Nepali needs its own bin edges:** Nepali syllable rate (~4–6 syllables/sec) is much lower than the English phoneme rate (~10–14/sec) that the upstream LibriTTS-R bins were calibrated on. Using the English bins would collapse all Nepali utterances into the bottom 1–2 speed categories. Pitch distributions also differ. `compute_bin_edges_nepali.py` recomputes percentile-based edges from your actual data.

---

## Set-up

```sh
git clone https://github.com/anil-titung-tamang/dataspeech-nepali.git dataspeech
cd dataspeech
pip install -r requirements.txt

# Install espeak-ng with Nepali voice data (recommended for accurate speaking rate)
sudo apt-get install -y espeak-ng espeak-ng-data

# Verify Nepali voice is available
espeak-ng --voices | grep " ne "
# If not found, the pipeline falls back to Devanagari syllable counting automatically.
```

---

## Annotating a Nepali dataset to fine-tune Parler-TTS

In the following examples we annotate a Nepali single-speaker TTS dataset in order to fine-tune [Parler-TTS Mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1).

The pipeline has four steps:
1. Annotate the dataset with continuous acoustic variables
2. Compute Nepali-calibrated bin edges from your data
3. Map continuous annotations to Nepali text keyword bins
4. Generate Nepali natural-language descriptions with an LLM

> **Dataset note:** Your dataset must be on the HuggingFace Hub with at least an `audio` column and a `transcription` column. See the [FAQ](#how-do-i-upload-a-local-nepali-dataset-to-the-hub) if you have local files.

### Step 1 — Annotate acoustic features

`main_nepali.py` computes speaking rate (using the Nepali phonemizer), pitch, SNR, reverberation, and optionally SI-SDR/PESQ/STOI.

```sh
python main_nepali.py "YOUR_HF_HANDLE/nepali-tts-dataset" \
  --configuration "default" \
  --text_column_name "transcription" \
  --audio_column_name "audio" \
  --cpu_num_workers 4 \
  --num_workers_per_gpu_for_pitch 2 \
  --rename_column \
  --repo_id "YOUR_HF_HANDLE/nepali-tts-tags" \
  --apply_squim_quality_estimation
```

The script scales automatically to every GPU available. The resulting dataset is pushed to `YOUR_HF_HANDLE/nepali-tts-tags` on the HuggingFace Hub with new columns: `speaking_rate`, `phonemes`, `utterance_pitch_mean`, `utterance_pitch_std`, `snr`, `c50`, `si-sdr`, `pesq`, `stoi`.

### Step 2 — Compute Nepali-calibrated bin edges

Run this **once** on your annotated dataset to produce percentile-based bin edges calibrated to your actual speech distribution. This is important because Nepali speaking rates and pitch ranges differ substantially from the English LibriTTS-R data that the upstream default bins were derived from.

```sh
python scripts/compute_bin_edges_nepali.py "YOUR_HF_HANDLE/nepali-tts-tags" \
  --configuration "default" \
  --split "train" \
  --output_path "./examples/tags_to_annotations/v01_bin_edges_nepali.json" \
  --n_bins 7 \
  --cpu_num_workers 4
```

This overwrites `examples/tags_to_annotations/v01_bin_edges_nepali.json` with edges computed from your data. A starter file with hand-tuned defaults (derived from OpenSLR-43/54 statistics) is already provided if you want to skip this step.

### Step 3 — Map continuous tags to Nepali text keyword bins

```sh
python ./scripts/metadata_to_text.py "YOUR_HF_HANDLE/nepali-tts-tags" \
    --repo_id "YOUR_HF_HANDLE/nepali-tts-tags" \
    --configuration "default" \
    --cpu_num_workers 4 \
    --leading_split_for_bins "train" \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges_nepali.json" \
    --path_to_text_bins "./examples/tags_to_annotations/v01_text_bins_nepali.json" \
    --avoid_pitch_computation \
    --apply_squim_quality_estimation
```

This adds Nepali keyword label columns such as `speaking_rate`, `noise`, `reverberation`, `speech_monotony`. The Nepali keyword vocabulary used is (from `v01_text_bins_nepali.json`):

| Feature | Nepali labels (slowest/quietest/lowest → fastest/loudest/highest) |
|---|---|
| Speaking rate | धेरै बिस्तारै, बिस्तारै, अलिकति बिस्तारै, मध्यम गति, अलिकति छिटो, छिटो, धेरै छिटो |
| Noise | अत्यन्त कोलाहलपूर्ण, धेरै कोलाहलपूर्ण, कोलाहलपूर्ण, अलिकति कोलाहलपूर्ण, लगभग शान्त, धेरै स्पष्ट |
| Reverberation | धेरै टाढाको आवाज, टाढाको आवाज, अलिकति टाढाको आवाज, अलिकति नजिकको आवाज, धेरै नजिकको आवाज |
| Speech monotony | धेरै एकरस, एकरस, अलिकति भावपूर्ण र सजिव, भावपूर्ण र सजिव, धेरै भावपूर्ण र सजिव |
| Pitch | धेरै कम पिच, कम पिच, अलिकति कम पिच, मध्यम पिच, अलिकति उच्च पिच, उच्च पिच, धेरै उच्च पिच |

### Step 4 — Generate Nepali natural-language descriptions

`run_prompt_creation_nepali.py` sends Nepali-language prompts to an LLM, producing a `text_description` column in Devanagari. This is the conditioning input Parler-TTS will be trained on.

Pass `--is_single_speaker` and `--speaker_name` for a named single-speaker dataset:

```sh
python ./scripts/run_prompt_creation_nepali.py \
  --speaker_name "Sunita" \
  --is_single_speaker \
  --is_new_speaker_prompt \
  --prompt_language "ne" \
  --dataset_name "YOUR_HF_HANDLE/nepali-tts-tags" \
  --dataset_config_name "default" \
  --model_name_or_path "google/gemma-2-2b-it" \
  --per_device_eval_batch_size 16 \
  --attn_implementation "sdpa" \
  --output_dir "./tmp_nepali_prompts" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "YOUR_HF_HANDLE/nepali-tts-tagged" \
  --preprocessing_num_workers 4 \
  --dataloader_num_workers 4
```

**`--model_name_or_path`** must point to a model with strong Devanagari support. Recommended options:

| Model | Notes |
|---|---|
| `google/gemma-2-2b-it` | Fits on T4 (4-bit), Apache 2.0 |
| `google/gemma-2-9b-it` | Better quality, needs A100/L4 |
| `Qwen/Qwen2.5-3B-Instruct` | Strong Nepali/Hindi, smaller footprint |
| `Qwen/Qwen2.5-7B-Instruct` | Best quality at manageable size |

**`--prompt_language`**: `ne` (default) generates Nepali Devanagari descriptions. Pass `en` to fall back to the upstream English prompts.

The generated descriptions will look like:
> *सुनिता मध्यम गतिमा र धेरै स्पष्ट आवाजमा बोल्नुहुन्छ। आवाज अलिकति नजिकको छ।*

For a multi-speaker Nepali dataset, remove `--is_single_speaker` and `--speaker_name`. To associate specific names to speaker IDs, pass `--speaker_id_column "speaker_id" --speaker_ids_to_name_json ./examples/prompt_creation/speaker_ids_to_names.json`.

### Running the full pipeline in one go

```sh
# Edit HF_HANDLE, DATASET_NAME, and LLM_MODEL at the top of the script
bash examples/run_nepali_pipeline.sh
```

---

## Annotating Nepali datasets from scratch

The following section explains the steps in more detail for users who want to understand what's happening under the hood, or who want to apply the pipeline to a large multi-speaker Nepali corpus.

There are four steps:
1. Annotate the speech dataset with continuous acoustic variables
2. Compute Nepali-calibrated bin edges from your data
3. Map continuous annotations to Nepali text keyword bins
4. Generate Nepali natural-language descriptions

### Step 1 — Predict annotations

`main_nepali.py` generates speaking rate, SNR, reverberation, PESQ, SI-SDR, and pitch for every utterance. Speaking rate is computed as `len(phonemes) / utterance_duration`, where `phonemes` comes from the Nepali phonemizer (espeak-ng → Devanagari syllable fallback).

```sh
python main_nepali.py "YOUR_HF_HANDLE/nepali-tts-dataset" \
  --configuration "default" \
  --output_dir ./tmp_nepali/ \
  --text_column_name "transcription" \
  --audio_column_name "audio" \
  --cpu_num_workers 4 \
  --rename_column \
  --apply_squim_quality_estimation
```

New columns added to the dataset:

| Column | Description |
|---|---|
| `speaking_rate` | Nepali phonemes (or syllables) per second |
| `phonemes` | Phoneme/syllable string used to compute speaking rate |
| `utterance_pitch_mean` | Mean pitch (Hz) |
| `utterance_pitch_std` | Pitch standard deviation |
| `snr` | Speech-to-noise ratio |
| `c50` | Reverberation (C50) |
| `si-sdr` | Scale-Invariant SDR (proxy noise measure) |
| `pesq` | Perceptual speech quality |
| `stoi` | Short-time objective intelligibility |

Use `python main_nepali.py --help` to see all available arguments.

### Step 2 — Compute Nepali-calibrated bin edges

Unlike the upstream English pipeline, you should not reuse the LibriTTS-R bin edges for Nepali. Run `compute_bin_edges_nepali.py` once on your annotated data:

```sh
python scripts/compute_bin_edges_nepali.py "YOUR_HF_HANDLE/nepali-tts-tags" \
  --configuration "default" \
  --split "train" \
  --output_path "./examples/tags_to_annotations/v01_bin_edges_nepali.json" \
  --n_bins 7
```

The script computes percentile-based edges (trimming 1% extremes on each side, matching upstream methodology) and renames columns to match `metadata_to_text.py` expectations (`snr` → `noise`, `c50` → `reverberation`, `si-sdr` → `sdr`).

If you don't have data yet, a starter `v01_bin_edges_nepali.json` is included with hand-tuned defaults based on OpenSLR-43/54 statistics.

### Step 3 — Map continuous annotations to Nepali text keyword bins

```sh
python ./scripts/metadata_to_text.py "YOUR_HF_HANDLE/nepali-tts-tags" \
  --repo_id "YOUR_HF_HANDLE/nepali-tts-tags" \
  --configuration "default" \
  --cpu_num_workers 4 \
  --leading_split_for_bins "train" \
  --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges_nepali.json" \
  --path_to_text_bins "./examples/tags_to_annotations/v01_text_bins_nepali.json" \
  --avoid_pitch_computation \
  --apply_squim_quality_estimation
```

You can pass multiple datasets separated by `"+"` just as in the upstream pipeline.

### Step 4 — Generate Nepali natural-language descriptions

#### 4.1 Accelerate inference (recommended)

```sh
python ./scripts/run_prompt_creation_nepali.py \
  --dataset_name "YOUR_HF_HANDLE/nepali-tts-tags" \
  --dataset_config_name "default" \
  --model_name_or_path "google/gemma-2-9b-it" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --output_dir "./tmp_nepali_prompts" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "YOUR_HF_HANDLE/nepali-tts-tagged" \
  --is_new_speaker_prompt \
  --prompt_language "ne" \
  --preprocessing_num_workers 4 \
  --dataloader_num_workers 4
```

For multi-GPU machines:

```sh
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=4 \
  scripts/run_prompt_creation_nepali.py \
  --dataset_name "YOUR_HF_HANDLE/nepali-tts-tags" \
  --dataset_config_name "default" \
  --model_name_or_path "google/gemma-2-9b-it" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --output_dir "./tmp_nepali_prompts" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "YOUR_HF_HANDLE/nepali-tts-tagged" \
  --is_new_speaker_prompt \
  --prompt_language "ne" \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 8
```

#### 4.2 TGI inference (high-throughput clusters)

The upstream `scripts/run_prompt_creation_llm_swarm.py` can be used unchanged. It relies on [TGI](https://huggingface.co/docs/text-generation-inference/en/index) and [LLM-Swarm](https://github.com/huggingface/llm-swarm):

```sh
pip install git+https://github.com/huggingface/llm-swarm.git

python scripts/run_prompt_creation_llm_swarm.py \
  --dataset_name "YOUR_HF_HANDLE/nepali-tts-tags" \
  --dataset_config_name "default" \
  --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
  --num_instances 1 \
  --output_dir "./" \
  --push_to_hub \
  --hub_dataset_id "YOUR_HF_HANDLE/nepali-tts-tagged"
```

Note that this path uses the upstream English prompts. For Nepali output, use the Accelerate path (`run_prompt_creation_nepali.py`) with `--prompt_language ne`.

---

## Using Data-Speech to filter your speech datasets

Data-Speech can also be used to filter Nepali datasets on quality or speech characteristics before training. For example:

1. Run Step 1 (`main_nepali.py`) to compute SNR, reverberation, and speaking rate.
2. Filter to retain only samples above an SNR threshold or within a target speaking-rate range.

This is useful for cleaning scraped Nepali web audio, removing utterances with high background noise, or building a subset of slow/fast speech for speed-robust TTS training.

---

## FAQ

### What kind of datasets do I need?

A HuggingFace `datasets`-compatible dataset with at least one `audio` column and a `transcription` column (Devanagari text). A `gender` and `speaker_id` column are also needed if you want per-speaker pitch computation. See the [datasets docs](https://huggingface.co/docs/datasets/v2.17.0/en/index) for details.

### How do I upload a local Nepali dataset to the Hub?

If you have a folder of `.wav` files and a `metadata.csv` with columns `file_name` and `transcription`:

```python
import pandas as pd
from datasets import Dataset, Audio

df = pd.read_csv("/path/to/metadata.csv")
df["audio"] = df["file_name"].apply(lambda f: f"/path/to/wavs/{f}")

dataset = Dataset.from_pandas(df[["audio", "transcription"]])
dataset = dataset.cast_column("audio", Audio())
dataset.push_to_hub("YOUR_HF_HANDLE/nepali-tts-dataset")
```

### Can I use this with an English dataset?

Yes. Use the original `main.py` and `scripts/run_prompt_creation.py` from this repo (both are unchanged from upstream). Or use `run_prompt_creation_nepali.py` with `--prompt_language en` to get English descriptions from the same script.

### What if `espeak-ng` doesn't have the Nepali voice?

The pipeline falls back silently to pure-Python Devanagari syllable counting. The speaking rate values will be slightly less accurate (syllables rather than IPA phonemes) but the pipeline will complete without errors and the quality difference is generally small for Nepali.

### Why not just use the upstream English bin edges?

Nepali syllable rate (~4–6 syllables/sec) is roughly half the English phoneme rate (~10–14/sec) that the upstream LibriTTS-R bins were derived from. Using English bins would place almost all Nepali utterances in the bottom speed bins, making the keyword labels meaningless. The same issue applies to pitch (Nepali female speakers average higher pitch than English audiobook speakers). `compute_bin_edges_nepali.py` solves this by computing bins from your actual data's percentile distribution.

---

## Logs

- **[2024]**: Initial Nepali fork
  - Replaced `g2p` English phonemizer in `rate.py` with `espeak-ng` Nepali backend + Devanagari syllable fallback
  - Added `main_nepali.py` with Nepali defaults
  - Added `run_prompt_creation_nepali.py` with full Devanagari prompt templates
  - Added `compute_bin_edges_nepali.py` for dataset-specific bin calibration
  - Added `v01_text_bins_nepali.json` and starter `v01_bin_edges_nepali.json`
  - Added `examples/run_nepali_pipeline.sh` end-to-end pipeline script

- **[August 2024]** (upstream): Updated Data-Speech for Parler-TTS v1
  - New measures: PESQ and SI-SDR
  - Improved prompts
  - Speaker consistency and accent support

- **[April 2024]** (upstream): First release of Data-Speech

---

## Acknowledgements

This fork builds on [Data-Speech](https://github.com/huggingface/dataspeech) by Yoach Lacombe, Vaibhav Srivastav, and Sanchit Gandhi at HuggingFace.

Special thanks to:
- Dan Lyth and Simon King for [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912)
- The maintainers of [datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [brouhaha](https://github.com/marianne-m/brouhaha-vad), [penn](https://github.com/interactiveaudiolab/penn), [espeak-ng](https://github.com/espeak-ng/espeak-ng), [accelerate](https://huggingface.co/docs/accelerate/en/index), and [transformers](https://huggingface.co/docs/transformers/index)

## Citation

```
@misc{lacombe-etal-2024-dataspeech,
 
  title = {Data-Speech-nepali},
  year = {2026},
  howpublished = {\url{https://github.com/ylacombe/dataspeech}}
}
```

```
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

### TODOs

- [ ] Nepali accent classification training script
- [ ] Nepali accent classification inference script
- [ ] Multilingual speaking rate estimation (generalise beyond Nepali)
- [x] Nepali-aware speaking rate estimation (espeak-ng + Devanagari fallback)
- [x] Dataset-specific bin edge computation (`compute_bin_edges_nepali.py`)
- [x] Nepali-language LLM prompt templates
- [ ] Add more Nepali annotation categories
- [ ] (long term) Benchmark for best audio dataset format
- [ ] (long term) Compatibility with streaming