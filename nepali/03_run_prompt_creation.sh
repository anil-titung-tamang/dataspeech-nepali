#!/bin/bash
# Step 3: Generate English natural-language descriptions from text bins.
# Descriptions are always in English (they go into Parler-TTS description tokenizer).
# Nepali text (transcription) goes into the prompt tokenizer (byt5-small) separately.
# Replace YOUR_HF_HANDLE and "Sita" with your actual handle and speaker name.

cd ../../  # run from dataspeech root

python ./scripts/run_prompt_creation.py \
  --speaker_name "Sita" \
  --is_single_speaker \
  --dataset_name "YOUR_HF_HANDLE/nepali-tts-tags" \
  --output_dir "./tmp_nepali" \
  --dataset_config_name "default" \
  --model_name_or_path "google/gemma-2b-it" \
  --per_device_eval_batch_size 12 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 2 \
  --push_to_hub \
  --hub_dataset_id "YOUR_HF_HANDLE/nepali-tts-tagged" \
  --preprocessing_num_workers 2