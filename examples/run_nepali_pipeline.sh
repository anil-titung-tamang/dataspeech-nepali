set -euo pipefail


HF_HANDLE="Titung" # your HuggingFace username
DATASET_NAME="${HF_HANDLE}/cc100-nepali-tts-10k"   # your raw Nepali dataset on HF Hub
TAGS_REPO="${HF_HANDLE}/nepali-tts-tags"          # where to push annotated tags
TAGGED_REPO="${HF_HANDLE}/nepali-tts-tagged"      # where to push final descriptions

# LLM for prompt creation — must support Nepali (Devanagari) well.
# Recommended options (choose one):
#   google/gemma-2-9b-it        — good multilingual coverage, Apache 2.0
#   meta-llama/Llama-3.1-8B-Instruct  — strong multilingual, needs HF token
#   Qwen/Qwen2.5-7B-Instruct   — strong Nepali/Hindi coverage
LLM_MODEL="google/gemma-2-9b-it"

CPU_WORKERS=4
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

echo "==========================================="
echo " Nepali DataSpeech Pipeline"
echo "  Dataset : ${DATASET_NAME}"
echo "  GPUs    : ${GPU_COUNT}"
echo "==========================================="


# Install espeak-ng Nepali support (run once)


echo "[Step 0] Checking espeak-ng..."
if ! command -v espeak-ng &>/dev/null; then
    echo "  Installing espeak-ng..."
    sudo apt-get install -y espeak-ng espeak-ng-data
else
    echo "  espeak-ng already installed."
fi
# Verify Nepali voice is available
if espeak-ng --voices | grep -q " ne "; then
    echo "  Nepali (ne) voice available."
else
    echo "  WARNING: Nepali voice not found. Falling back to Devanagari syllable count."
fi


# Annotate with acoustic features


echo "[Step 1] Annotating acoustic features..."
python main_nepali.py "${DATASET_NAME}" \
    --configuration "default" \
    --text_column_name "transcription" \
    --audio_column_name "audio" \
    --cpu_num_workers "${CPU_WORKERS}" \
    --rename_column \
    --repo_id "${TAGS_REPO}" \
    --apply_squim_quality_estimation


# Compute Nepali-calibrated bin edges from YOUR data


echo "[Step 2] Computing Nepali bin edges from annotated dataset..."
python scripts/compute_bin_edges_nepali.py "${TAGS_REPO}" \
    --configuration "default" \
    --split "train" \
    --output_path "./examples/tags_to_annotations/v01_bin_edges_nepali.json" \
    --n_bins 7 \
    --cpu_num_workers "${CPU_WORKERS}"


# Map continuous tags → text keyword bins

echo "[Step 3] Mapping continuous annotations to text bins..."
python ./scripts/metadata_to_text.py "${TAGS_REPO}" \
    --repo_id "${TAGS_REPO}" \
    --configuration "default" \
    --cpu_num_workers "${CPU_WORKERS}" \
    --leading_split_for_bins "train" \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges_nepali.json" \
    --path_to_text_bins "./examples/tags_to_annotations/v01_text_bins_nepali.json" \
    --avoid_pitch_computation \
    --apply_squim_quality_estimation


# Generate Nepali natural-language descriptions


echo "[Step 4] Generating Nepali text descriptions with LLM..."

if [ "${GPU_COUNT}" -ge 2 ]; then
    LAUNCH_CMD="accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=${GPU_COUNT}"
else
    LAUNCH_CMD="python"
fi

${LAUNCH_CMD} scripts/run_prompt_creation_nepali.py \
    --dataset_name "${TAGS_REPO}" \
    --dataset_config_name "default" \
    --model_name_or_path "${LLM_MODEL}" \
    --per_device_eval_batch_size 16 \
    --attn_implementation "sdpa" \
    --output_dir "./tmp_nepali_prompts" \
    --load_in_4bit \
    --push_to_hub \
    --hub_dataset_id "${TAGGED_REPO}" \
    --is_new_speaker_prompt \
    --prompt_language ne \
    --preprocessing_num_workers "${CPU_WORKERS}" \
    --dataloader_num_workers "${CPU_WORKERS}"


echo "==========================================="
echo " Pipeline complete!"
echo "  Tags repo    : https://huggingface.co/datasets/${TAGS_REPO}"
echo "  Tagged repo  : https://huggingface.co/datasets/${TAGGED_REPO}"
echo "==========================================="