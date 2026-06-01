"""
main_nepali.py — Nepali-adapted entry point for Data-Speech annotation.

Changes from original main.py:
  - Default text_column_name changed to "transcription" (common in Nepali datasets)
  - Added --language flag (documents intent; rate.py handles Nepali phonemes internally)
  - rate.py uses espeak-ng Nepali backend or Devanagari character fallback automatically
  - Print statements label Nepali-specific steps for clarity

Usage example (Colab / terminal):
    python main_nepali.py "YOUR_HF_HANDLE/nepali-tts-dataset" \\
      --configuration "default" \\
      --text_column_name "transcription" \\
      --audio_column_name "audio" \\
      --cpu_num_workers 2 \\
      --num_workers_per_gpu_for_pitch 2 \\
      --rename_column \\
      --repo_id "YOUR_HF_HANDLE/nepali-tts-tags"
"""

from datasets import load_dataset, Audio
from multiprocess import set_start_method
from dataspeech import rate_apply, pitch_apply, snr_apply, squim_apply
import torch
import argparse


if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Annotate a Nepali TTS dataset with acoustic features (speaking rate, pitch, SNR, reverberation)."
    )

    parser.add_argument("dataset_name", type=str,
                        help="HuggingFace Hub path or name of your Nepali dataset.")
    parser.add_argument("--configuration", default=None, type=str,
                        help="Dataset configuration to use, if necessary.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="If specified, save the annotated dataset to disk at this path.")
    parser.add_argument("--repo_id", default=None, type=str,
                        help="If specified, push the annotated dataset to this HuggingFace Hub repo.")
    parser.add_argument("--audio_column_name", default="audio", type=str,
                        help="Column name of the audio in your dataset.")
    parser.add_argument("--text_column_name", default="transcription", type=str,
                        help="Column name of the Nepali text transcription. Default: 'transcription'.")
    parser.add_argument("--rename_column", action="store_true",
                        help="Rename audio/text columns to 'audio'/'text' (useful for merging datasets later).")
    parser.add_argument("--language", default="ne", type=str,
                        help="Language code (default: 'ne' for Nepali). rate.py uses this internally via espeak-ng.")
    parser.add_argument("--cpu_num_workers", default=1, type=int,
                        help="Number of CPU workers for non-GPU transformations.")
    parser.add_argument("--cpu_writer_batch_size", default=1000, type=int,
                        help="writer_batch_size for CPU transformations.")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Samples passed per GPU worker batch.")
    parser.add_argument("--penn_batch_size", default=4096, type=int,
                        help="Batch size for pitch (PENN) estimation.")
    parser.add_argument("--num_workers_per_gpu_for_pitch", default=1, type=int,
                        help="Workers per GPU for pitch estimation.")
    parser.add_argument("--num_workers_per_gpu_for_snr", default=1, type=int,
                        help="Workers per GPU for SNR/reverberation estimation.")
    parser.add_argument("--apply_squim_quality_estimation", action="store_true",
                        help="Also compute SI-SNR, STOI, PESQ via torchaudio-squim.")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int,
                        help="Workers per GPU for squim estimation.")

    args = parser.parse_args()

    print(f"[Nepali] Loading dataset: {args.dataset_name}")
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration, num_proc=args.cpu_num_workers)
    else:
        dataset = load_dataset(args.dataset_name, num_proc=args.cpu_num_workers)

    audio_column_name = "audio" if args.rename_column else args.audio_column_name
    text_column_name = "text" if args.rename_column else args.text_column_name
    if args.rename_column:
        dataset = dataset.rename_columns({args.audio_column_name: "audio", args.text_column_name: "text"})

    if args.apply_squim_quality_estimation:
        print("[Nepali] Computing SI-SDR, PESQ, STOI...")
        squim_dataset = dataset.map(
            squim_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=True if torch.cuda.device_count() > 0 else False,
            num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_squim if torch.cuda.device_count() > 0 else args.cpu_num_workers,
            remove_columns=[audio_column_name],
            fn_kwargs={"audio_column_name": audio_column_name},
        )

    print("[Nepali] Computing pitch...")
    pitch_dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=16_000)).map(
        pitch_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count() > 0 else False,
        num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_pitch if torch.cuda.device_count() > 0 else args.cpu_num_workers,
        remove_columns=[audio_column_name],
        fn_kwargs={"audio_column_name": audio_column_name, "penn_batch_size": args.penn_batch_size},
    )

    print("[Nepali] Computing SNR and reverberation...")
    snr_dataset = dataset.map(
        snr_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count() > 0 else False,
        num_proc=torch.cuda.device_count() * args.num_workers_per_gpu_for_snr if torch.cuda.device_count() > 0 else args.cpu_num_workers,
        remove_columns=[audio_column_name],
        fn_kwargs={"audio_column_name": audio_column_name},
    )

    print("[Nepali] Computing speaking rate (Nepali phoneme-aware via espeak-ng or Devanagari fallback)...")
    if "speech_duration" in snr_dataset[next(iter(snr_dataset.keys()))].features:
        rate_dataset = snr_dataset.map(
            rate_apply,
            with_rank=False,
            num_proc=args.cpu_num_workers,
            writer_batch_size=args.cpu_writer_batch_size,
            fn_kwargs={"audio_column_name": audio_column_name, "text_column_name": text_column_name},
        )
    else:
        rate_dataset = dataset.map(
            rate_apply,
            with_rank=False,
            num_proc=args.cpu_num_workers,
            writer_batch_size=args.cpu_writer_batch_size,
            remove_columns=[audio_column_name],
            fn_kwargs={"audio_column_name": audio_column_name, "text_column_name": text_column_name},
        )

    for split in dataset.keys():
        dataset[split] = (
            pitch_dataset[split]
            .add_column("snr", snr_dataset[split]["snr"])
            .add_column("c50", snr_dataset[split]["c50"])
        )
        if "speech_duration" in snr_dataset[split]:
            dataset[split] = dataset[split].add_column("speech_duration", snr_dataset[split]["speech_duration"])
        dataset[split] = (
            dataset[split]
            .add_column("speaking_rate", rate_dataset[split]["speaking_rate"])
            .add_column("phonemes", rate_dataset[split]["phonemes"])
        )
        if args.apply_squim_quality_estimation:
            dataset[split] = (
                dataset[split]
                .add_column("stoi", squim_dataset[split]["stoi"])
                .add_column("si-sdr", squim_dataset[split]["sdr"])
                .add_column("pesq", squim_dataset[split]["pesq"])
            )

    if args.output_dir:
        print(f"[Nepali] Saving annotated dataset to disk: {args.output_dir}")
        dataset.save_to_disk(args.output_dir)
    if args.repo_id:
        print(f"[Nepali] Pushing annotated dataset to HuggingFace Hub: {args.repo_id}")
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)

    print("[Nepali] Annotation complete!")