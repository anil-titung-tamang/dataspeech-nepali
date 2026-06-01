#!/bin/bash
# Step 1: Annotate your Nepali dataset with acoustic features.
# Replace YOUR_HF_HANDLE with your HuggingFace username.
# Replace nepali-tts-dataset with your actual dataset repo name.

cd ../../  # run from dataspeech root

python main_nepali.py "YOUR_HF_HANDLE/nepali-tts-dataset" \
  --configuration "default" \
  --text_column_name "transcription" \
  --audio_column_name "audio" \
  --cpu_num_workers 2 \
  --num_workers_per_gpu_for_pitch 2 \
  --rename_column \
  --repo_id "YOUR_HF_HANDLE/nepali-tts-tags"