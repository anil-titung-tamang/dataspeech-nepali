
import argparse
import json
import numpy as np
from datasets import load_dataset


CONTINUOUS_COLUMNS = [
    "speaking_rate",
    "snr",
    "c50",
    "utterance_pitch_mean",
    "utterance_pitch_std",
]

# For squim columns (optional — present only if --apply_squim_quality_estimation was used)
SQUIM_COLUMNS = ["si-sdr", "pesq", "stoi"]

# Fraction of extremes to trim before computing histogram edges (same as upstream)
TRIM_FRACTION = 0.01


def percentile_edges(values: np.ndarray, n_bins: int) -> list:
    """Return n_bins-1 percentile-based edges, trimming extreme outliers."""
    lo = np.percentile(values, TRIM_FRACTION * 100)
    hi = np.percentile(values, (1 - TRIM_FRACTION) * 100)
    trimmed = values[(values >= lo) & (values <= hi)]
    edges = np.percentile(trimmed, np.linspace(0, 100, n_bins + 1)[1:-1])
    return [round(float(e), 4) for e in edges]


def main():
    parser = argparse.ArgumentParser(description="Compute Nepali bin edges from annotated dataset.")
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--configuration", default=None)
    parser.add_argument("--split", default="train", help="Dataset split to use for computing bins.")
    parser.add_argument("--output_path", default="./examples/tags_to_annotations/v01_bin_edges_nepali.json")
    parser.add_argument("--n_bins", type=int, default=7, help="Number of bins (labels) per feature.")
    parser.add_argument("--cpu_num_workers", type=int, default=4)
    args = parser.parse_args()

    print(f"[Nepali] Loading dataset: {args.dataset_name}")
    dataset = load_dataset(
        args.dataset_name,
        args.configuration,
        split=args.split,
        num_proc=args.cpu_num_workers,
    )

    bin_edges = {}

    for col in CONTINUOUS_COLUMNS + SQUIM_COLUMNS:
        if col not in dataset.features:
            print(f"  Column '{col}' not found — skipping.")
            continue
        values = np.array(dataset[col], dtype=float)
        # Remove NaN / inf
        values = values[np.isfinite(values)]
        if len(values) == 0:
            print(f"  Column '{col}' has no finite values — skipping.")
            continue

        # Pitch: compute separately for male/female if gender column exists
        if col == "utterance_pitch_mean" and "gender" in dataset.features:
            genders = dataset["gender"]
            male_vals = values[[i for i, g in enumerate(genders) if g == "male"]]
            female_vals = values[[i for i, g in enumerate(genders) if g == "female"]]
            if len(male_vals) > args.n_bins:
                bin_edges["utterance_pitch_mean_male"] = percentile_edges(male_vals, args.n_bins)
            if len(female_vals) > args.n_bins:
                bin_edges["utterance_pitch_mean_female"] = percentile_edges(female_vals, args.n_bins)
            continue

        bin_edges[col] = percentile_edges(values, args.n_bins)
        print(f"  {col}: {bin_edges[col]}")

    # Rename snr → noise and si-sdr → sdr to match metadata_to_text.py expectations
    if "snr" in bin_edges:
        bin_edges["noise"] = bin_edges.pop("snr")
    if "si-sdr" in bin_edges:
        bin_edges["sdr"] = bin_edges.pop("si-sdr")
    if "c50" in bin_edges:
        bin_edges["reverberation"] = bin_edges.pop("c50")

    bin_edges["_comment"] = [
        f"Auto-generated from {args.dataset_name} ({args.split} split).",
        "Overwrite v01_bin_edges_nepali.json with this file.",
        f"n_bins={args.n_bins}, trim_fraction={TRIM_FRACTION}",
    ]

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(bin_edges, f, ensure_ascii=False, indent=2)

    print(f"\n[Nepali] Bin edges written to: {args.output_path}")
    print("Replace examples/tags_to_annotations/v01_bin_edges_nepali.json with this file.")
    print("Then run metadata_to_text.py with --path_to_bin_edges pointing to this file.")


if __name__ == "__main__":
    main()