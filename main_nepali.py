
import os
import gc

# Must be set BEFORE torch is imported — prevents CUDA memory fragmentation OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import load_dataset, load_from_disk, Audio
from multiprocess import set_start_method
from dataspeech import rate_apply, pitch_apply, snr_apply, squim_apply
import torch
import argparse


def save_stage(dataset, output_dir, stage_name):
# Save a dataset stage to disk so we can resume if the next stage crashes.
    if output_dir:
        path = os.path.join(output_dir, f"stage_{stage_name}")
        print(f"[Nepali] 💾 Saving {stage_name} checkpoint → {path}")
        dataset.save_to_disk(path)


def load_stage(output_dir, stage_name):
#    Return a previously saved stage dataset, or None if not found.
    if output_dir:
        path = os.path.join(output_dir, f"stage_{stage_name}")
        if os.path.isdir(path):
            print(f"[Nepali] ⏩ Resuming from checkpoint: {stage_name}")
            return load_from_disk(path)
    return None


if __name__ == "__main__":
    set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Annotate a Nepali TTS dataset with acoustic features."
    )
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--configuration", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Save annotated dataset AND per-stage checkpoints here.")
    parser.add_argument("--repo_id", default=None, type=str)
    parser.add_argument("--audio_column_name", default="audio", type=str)
    parser.add_argument("--text_column_name", default="transcription", type=str)
    parser.add_argument("--rename_column", action="store_true")
    parser.add_argument("--language", default="ne", type=str)
    parser.add_argument("--cpu_num_workers", default=1, type=int)
    parser.add_argument("--cpu_writer_batch_size", default=1000, type=int)
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Samples per GPU batch. Default 1 to stay within T4 VRAM.")
    parser.add_argument("--penn_batch_size", default=4096, type=int)
    parser.add_argument("--num_workers_per_gpu_for_pitch", default=1, type=int,
                        help="Workers per GPU for pitch. Default 1 — do NOT set >1 on T4.")
    parser.add_argument("--num_workers_per_gpu_for_snr", default=1, type=int)
    parser.add_argument("--apply_squim_quality_estimation", action="store_true")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int)
    args = parser.parse_args()

    # If output_dir given, create it (used for both final save and stage checkpoints)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Nepali] Loading dataset: {args.dataset_name}")
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration,
                               num_proc=args.cpu_num_workers)
    else:
        dataset = load_dataset(args.dataset_name, num_proc=args.cpu_num_workers)

    audio_column_name = "audio" if args.rename_column else args.audio_column_name
    text_column_name  = "text"  if args.rename_column else args.text_column_name

    if args.rename_column:
        dataset = dataset.rename_columns(
            {args.audio_column_name: "audio", args.text_column_name: "text"}
        )

    # Stage 1: SQUIM (SI-SDR, PESQ, STOI)
    if args.apply_squim_quality_estimation:
        squim_dataset = load_stage(args.output_dir, "squim")
        if squim_dataset is None:
            print("[Nepali] Computing SI-SDR, PESQ, STOI...")
            squim_dataset = dataset.map(
                squim_apply,
                batched=True,
                batch_size=args.batch_size,
                with_rank=True if torch.cuda.device_count() > 0 else False,
                num_proc=(torch.cuda.device_count() * args.num_workers_per_gpu_for_squim
                          if torch.cuda.device_count() > 0 else args.cpu_num_workers),
                remove_columns=[audio_column_name],
                fn_kwargs={"audio_column_name": audio_column_name},
            )
            save_stage(squim_dataset, args.output_dir, "squim")
        # Free VRAM before pitch
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 2: Pitch
    pitch_dataset = load_stage(args.output_dir, "pitch")
    if pitch_dataset is None:
        print("[Nepali] Computing pitch...")
        pitch_dataset = dataset.cast_column(
            audio_column_name, Audio(sampling_rate=16_000)
        ).map(
            pitch_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=True if torch.cuda.device_count() > 0 else False,
            num_proc=(torch.cuda.device_count() * args.num_workers_per_gpu_for_pitch
                      if torch.cuda.device_count() > 0 else args.cpu_num_workers),
            remove_columns=[audio_column_name],
            fn_kwargs={"audio_column_name": audio_column_name,
                       "penn_batch_size": args.penn_batch_size},
        )
        save_stage(pitch_dataset, args.output_dir, "pitch")
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 3: SNR / Reverberation
    snr_dataset = load_stage(args.output_dir, "snr")
    if snr_dataset is None:
        print("[Nepali] Computing SNR and reverberation...")
        snr_dataset = dataset.map(
            snr_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=True if torch.cuda.device_count() > 0 else False,
            num_proc=(torch.cuda.device_count() * args.num_workers_per_gpu_for_snr
                      if torch.cuda.device_count() > 0 else args.cpu_num_workers),
            remove_columns=[audio_column_name],
            fn_kwargs={"audio_column_name": audio_column_name},
        )
        save_stage(snr_dataset, args.output_dir, "snr")
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 4: Speaking Rate
    rate_dataset = load_stage(args.output_dir, "rate")
    if rate_dataset is None:
        print("[Nepali] Computing speaking rate (Nepali phoneme-aware)...")
        if "speech_duration" in snr_dataset[next(iter(snr_dataset.keys()))].features:
            rate_dataset = snr_dataset.map(
                rate_apply,
                with_rank=False,
                num_proc=args.cpu_num_workers,
                writer_batch_size=args.cpu_writer_batch_size,
                fn_kwargs={"audio_column_name": audio_column_name,
                           "text_column_name": text_column_name},
            )
        else:
            rate_dataset = dataset.map(
                rate_apply,
                with_rank=False,
                num_proc=args.cpu_num_workers,
                writer_batch_size=args.cpu_writer_batch_size,
                remove_columns=[audio_column_name],
                fn_kwargs={"audio_column_name": audio_column_name,
                           "text_column_name": text_column_name},
            )
        save_stage(rate_dataset, args.output_dir, "rate")

    # Merge all stages
    print("[Nepali] Merging all stages...")
    for split in dataset.keys():
        dataset[split] = (
            pitch_dataset[split]
            .add_column("snr",  snr_dataset[split]["snr"])
            .add_column("c50",  snr_dataset[split]["c50"])
        )
        if "speech_duration" in snr_dataset[split].features:
            dataset[split] = dataset[split].add_column(
                "speech_duration", snr_dataset[split]["speech_duration"]
            )
        dataset[split] = (
            dataset[split]
            .add_column("speaking_rate", rate_dataset[split]["speaking_rate"])
            .add_column("phonemes",      rate_dataset[split]["phonemes"])
        )
        if args.apply_squim_quality_estimation:
            dataset[split] = (
                dataset[split]
                .add_column("stoi",  squim_dataset[split]["stoi"])
                .add_column("si-sdr", squim_dataset[split]["sdr"])
                .add_column("pesq",  squim_dataset[split]["pesq"])
            )

    if args.output_dir:
        print(f"[Nepali] Saving final merged dataset → {args.output_dir}/final")
        dataset.save_to_disk(os.path.join(args.output_dir, "final"))

    if args.repo_id:
        print(f"[Nepali] Pushing to Hub: {args.repo_id}")
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)

    print("[Nepali] ✅ Annotation complete!")