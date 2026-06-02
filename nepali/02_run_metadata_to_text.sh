#!/bin/bash

cd ../../  # run from dataspeech root

python ./scripts/metadata_to_text.py \
    "Titung/nepali-tts-tags" \
    --repo_id "Titung/nepali-tts-tags" \
    --configuration "default" \
    --cpu_num_workers 2 \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation