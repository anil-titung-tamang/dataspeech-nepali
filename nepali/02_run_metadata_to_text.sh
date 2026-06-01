#!/bin/bash
# Step 2: Convert continuous acoustic annotations to text bins.
# Uses pre-computed v01_bin_edges.json (language-agnostic acoustic bins).
# Replace YOUR_HF_HANDLE with your HuggingFace username.

cd ../../  # run from dataspeech root

python ./scripts/metadata_to_text.py \
    "YOUR_HF_HANDLE/nepali-tts-tags" \
    --repo_id "YOUR_HF_HANDLE/nepali-tts-tags" \
    --configuration "default" \
    --cpu_num_workers 2 \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation