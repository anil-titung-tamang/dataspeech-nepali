import os
import gc

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import load_dataset, load_from_disk, Audio, DatasetDict
from multiprocess import set_start_method
from dataspeech import rate_apply, pitch_apply, snr_apply, squim_apply
import torch
import argparse


def save_stage(dataset, output_dir, stage_name):
    if output_dir:
        path = os.path.join(output_dir, f"stage_{stage_name}")
        print(f"[Nepali] 💾 Saving {stage_name} checkpoint → {path}")
        dataset.save_to_disk(path)


def load_stage(output_dir, stage_name):
    if not output_dir:
        return None
    path = os.path.join(output_dir, f"stage_{stage_name}")
    if os.path.isdir(path):
        try:
            ds = load_from_disk(path)
            print(f"[Nepali] ⏩ Resuming from checkpoint: {stage_name}")
            return ds
        except Exception as e:
            print(f"[Nepali] ⚠️  Checkpoint {stage_name} found but failed to load ({e}), recomputing.")
            return None
    return None


if __name__ == "__main__":
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--configuration", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--repo_id", default=None, type=str)
    parser.add_argument("--audio_column_name", default="audio", type=str)
    parser.add_argument("--text_column_name", default="transcription", type=str)
    parser.add_argument("--rename_column", action="store_true")
    parser.add_argument("--language", default="ne", type=str)
    parser.add_argument("--cpu_num_workers", default=1, type=int)
    parser.add_argument("--cpu_writer_batch_size", default=1000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--penn_batch_size", default=4096, type=int)
    parser.add_argument("--num_workers_per_gpu_for_pitch", default=1, type=int)
    parser.add_argument("--num_workers_per_gpu_for_snr", default=1, type=int)
    parser.add_argument("--apply_squim_quality_estimation", action="store_true")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int)
    args = parser.parse_args()

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

    num_gpu_workers = torch.cuda.device_count()

    # ── Stage 1: SQUIM ───────────────────────────────────────────────────────
    squim_dataset = None
    if args.apply_squim_quality_estimation:
        squim_dataset = load_stage(args.output_dir, "squim")
        if squim_dataset is None:
            print("[Nepali] Computing SI-SDR, PESQ, STOI...")
            squim_dataset = dataset.cast_column(
                audio_column_name, Audio(sampling_rate=16_000)
            ).map(
                squim_apply,
                batched=True,
                batch_size=args.batch_size,
                with_rank=num_gpu_workers > 0,
                num_proc=max(num_gpu_workers * args.num_workers_per_gpu_for_squim, 1),
                remove_columns=[audio_column_name],
                fn_kwargs={"audio_column_name": audio_column_name},
            )
            save_stage(squim_dataset, args.output_dir, "squim")
        gc.collect()
        torch.cuda.empty_cache()

    # ── Stage 2: PITCH ───────────────────────────────────────────────────────
    pitch_dataset = load_stage(args.output_dir, "pitch")
    if pitch_dataset is None:
        print("[Nepali] Computing pitch...")
        pitch_dataset = dataset.cast_column(
            audio_column_name, Audio(sampling_rate=16_000)
        ).map(
            pitch_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=num_gpu_workers > 0,
            num_proc=max(num_gpu_workers * args.num_workers_per_gpu_for_pitch, 1),
            remove_columns=[audio_column_name],
            fn_kwargs={"audio_column_name": audio_column_name,
                       "penn_batch_size": args.penn_batch_size},
        )
        save_stage(pitch_dataset, args.output_dir, "pitch")
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 3: SNR ─────────────────────────────────────────────────────────
    snr_dataset = load_stage(args.output_dir, "snr")
    if snr_dataset is None:
        print("[Nepali] Computing SNR and reverberation...")
        snr_dataset = dataset.cast_column(
            audio_column_name, Audio(sampling_rate=16_000)
        ).map(
            snr_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=num_gpu_workers > 0,
            num_proc=max(num_gpu_workers * args.num_workers_per_gpu_for_snr, 1),
            remove_columns=[audio_column_name],
            fn_kwargs={"audio_column_name": audio_column_name},
        )
        save_stage(snr_dataset, args.output_dir, "snr")
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 4: SPEAKING RATE ───────────────────────────────────────────────
    rate_dataset = load_stage(args.output_dir, "rate")
    if rate_dataset is None:
        print("[Nepali] Computing speaking rate...")
        # rate_apply works on snr_dataset because it already has speech_duration
        rate_dataset = snr_dataset.map(
            rate_apply,
            with_rank=False,
            num_proc=args.cpu_num_workers,
            writer_batch_size=args.cpu_writer_batch_size,
            fn_kwargs={"audio_column_name": audio_column_name,
                       "text_column_name": text_column_name},
        )
        save_stage(rate_dataset, args.output_dir, "rate")

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("[Nepali] Merging all stage columns...")
    final = DatasetDict()
    for split in dataset.keys():
        base = pitch_dataset[split]

        # Add SNR columns
        for col in ["snr", "c50", "speech_duration"]:
            if col in snr_dataset[split].column_names:
                base = base.add_column(col, snr_dataset[split][col])

        # Add rate columns
        for col in ["speaking_rate", "phonemes"]:
            if col in rate_dataset[split].column_names:
                base = base.add_column(col, rate_dataset[split][col])

        # Add squim columns
        if squim_dataset is not None:
            col_map = {"stoi": "stoi", "sdr": "si-sdr", "pesq": "pesq"}
            for src_col, dst_col in col_map.items():
                if src_col in squim_dataset[split].column_names:
                    base = base.add_column(dst_col, squim_dataset[split][src_col])

        final[split] = base

    if args.output_dir:
        final_path = os.path.join(args.output_dir, "final")
        print(f"[Nepali] 💾 Saving final dataset → {final_path}")
        final.save_to_disk(final_path)

    if args.repo_id:
        print(f"[Nepali] Pushing to Hub: {args.repo_id}")
        if args.configuration:
            final.push_to_hub(args.repo_id, args.configuration)
        else:
            final.push_to_hub(args.repo_id)

    print("[Nepali] ✅ Annotation complete!")