#!/bin/bash
# Step 1: Annotate your Nepali dataset with acoustic features.
# Replace YOUR_HF_HANDLE with your HuggingFace username.
# Replace nepali-tts-dataset with your actual dataset repo name.

cd ../../  # run from dataspeech root

python main_nepali.py "Titung/cc100-nepali-tts-10k" \
  --configuration "default" \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 2 \
  --num_workers_per_gpu_for_pitch 2 \
  --rename_column \
  --repo_id "Titung/nepali-tts-tags"