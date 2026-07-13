import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
MODES = ["similarity", "fixed", "random"]


def load_summary(mode: str):
    candidate_paths = [
        RESULTS_ROOT / mode / "cider_summary.csv",
        RESULTS_ROOT / "cider_summary.csv",
    ]

    for summary_path in candidate_paths:
        if not summary_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        last_row = rows[-1]
        label = mode
        if summary_path.parent == RESULTS_ROOT and summary_path.name == "cider_summary.csv":
            label = f"{mode} (legacy root results)"

        return {
            "mode": label,
            "samples": int(last_row["num_samples"]),
            "cider": float(last_row["cider"]),
            "summary_path": str(summary_path),
        }

    return None


def main():
    summaries = []
    for mode in MODES:
        summary = load_summary(mode)
        if summary is not None:
            summaries.append(summary)

    print("=" * 60)
    print("Experiment summary")
    print("=" * 60)

    if not summaries:
        print("No experiment results found.")
        return

    for item in summaries:
        print(f"Mode: {item['mode']}")
        print(f"  Samples: {item['samples']}")
        print(f"  Final CIDEr: {item['cider']:.4f}")
        print(f"  Summary: {item['summary_path']}")

    print("=" * 60)
    best = max(summaries, key=lambda x: x["cider"])
    print(f"Best mode: {best['mode']} ({best['cider']:.4f})")


if __name__ == "__main__":
    main()
