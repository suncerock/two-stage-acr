from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WEIGHTS = {"SWD": 0.094, "BPSD": 0.360, "BSQD": 0.546}
METRICS = ["WCSR/majmin", "WCSR/tetrads"]
CONFIGS = ["Chord-S", "Joint-S", "MPE+Chord-S", "MPE+Chord-F"]
MODELS = ["CRNN", "Conformer"]
DATA = {
    "CRNN": {
        "Chord-S": {"SWD": (77.69, 67.19, 94.57), "BPSD": (80.09, 70.24, 95.10), "BSQD": (75.06, 61.04, 88.70)},
        "Joint-S": {"SWD": (80.21, 69.69, 94.71), "BPSD": (81.58, 72.33, 95.08), "BSQD": (75.41, 62.02, 88.77)},
        "MPE+Chord-S": {"SWD": (81.14, 71.55, 95.49), "BPSD": (82.61, 73.66, 95.85), "BSQD": (77.45, 64.22, 89.69)},
        "MPE+Chord-F": {"SWD": (82.33, 73.30, 95.90), "BPSD": (83.53, 74.92, 96.36), "BSQD": (78.82, 65.98, 90.12)},
    },
    "Conformer": {
        "Chord-S": {"SWD": (77.04, 65.53, 93.39), "BPSD": (80.04, 69.15, 94.28), "BSQD": (75.54, 60.47, 88.26)},
        "Joint-S": {"SWD": (77.90, 65.88, 93.61), "BPSD": (81.28, 70.58, 94.47), "BSQD": (76.28, 61.60, 88.39)},
        "MPE+Chord-S": {"SWD": (79.32, 69.13, 94.72), "BPSD": (81.80, 71.43, 95.16), "BSQD": (77.45, 63.09, 88.69)},
        "MPE+Chord-F": {"SWD": (80.93, 71.21, 95.09), "BPSD": (83.08, 73.26, 95.78), "BSQD": (78.75, 64.97, 89.28)},
    },
}


def weighted_scores(model_name: str) -> dict[str, tuple[float, float, float]]:
    scores: dict[str, tuple[float, float]] = {}
    for config_name, datasets in DATA[model_name].items():
        aggregate = []
        for metric_idx in range(len(METRICS)):
            metric_score = sum(
                WEIGHTS[dataset_name] * datasets[dataset_name][metric_idx]
                for dataset_name in WEIGHTS
            )
            aggregate.append(metric_score)
        scores[config_name] = tuple(aggregate)
    return scores


def main() -> None:
    weighted = {model_name: weighted_scores(model_name) for model_name in MODELS}
    x = np.arange(len(MODELS))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    colors = {
        "Chord-S": "#0072B2",
        "Joint-S": "#E69F00",
        "MPE+Chord-S": "#009E73",
        "MPE+Chord-F": "#CC79A7",
    }
    y_limits = [(60, 83), (60, 72)]

    for metric_idx, metric_name in enumerate(METRICS):
        ax = axes[metric_idx]
        bar_sets = []
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONFIGS))

        for offset, config_name in zip(offsets, CONFIGS):
            values = [weighted[model_name][config_name][metric_idx] for model_name in MODELS]
            bars = ax.bar(x + offset, values, width, label=config_name, color=colors[config_name])
            bar_sets.append(bars)

        ax.set_title(metric_name)
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_ylim(*y_limits[metric_idx])
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

        for bars in bar_sets:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.35,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # axes[0].set_ylabel("Weighted score")
    # fig.suptitle("Table 2: Weighted Aggregation Across Datasets")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_dir = Path("inference_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "table2_weighted_barplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()