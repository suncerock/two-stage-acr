from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["WCSR/majmin", "WCSR/tetrads"]
ROWS = [
    ("Chord-S", (77.12, 64.93)),
    ("Joint-S", (78.08, 66.45)),
    ("MPE+Chord-F", (80.85, 69.89)),
    ("Use BasicPitch", (77.35, 65.17)),
    ("Binarized MPE", (80.61, 69.59)),
    ("Use Score pitch repr.", (81.88, 71.07)),
]
COLORS = {
    "Chord-S": "#0072B2",
    "Joint-S": "#E69F00",
    "MPE+Chord-F": "#CC79A7",
    "Use BasicPitch": "#D55E00",
    "Binarized MPE": "#009E73",
    "Use Score pitch repr.": "#56B4E9",
}


def main() -> None:
    x = np.arange(len(ROWS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    y_limits = [(60, 85), (60, 85)]

    for metric_idx, metric_name in enumerate(METRICS):
        ax = axes[metric_idx]
        values = [row_values[metric_idx] for _, row_values in ROWS]
        colors = [COLORS[label] for label, _ in ROWS]
        bars = ax.bar(x, values, color=colors, width=0.72)

        ax.axvline(1.5, color="#666666", linestyle="--", linewidth=1)
        ax.text(0.5, y_limits[metric_idx][1] - 1.4, "References", ha="center", va="bottom", fontsize=10, color="#444444")
        ax.text(3.75, y_limits[metric_idx][1] - 1.4, "MPE+Chord-F ablations", ha="center", va="bottom", fontsize=10, color="#444444")

        ax.set_title(metric_name)
        ax.set_ylabel("Weighted score")
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _ in ROWS], rotation=20, ha="right")
        ax.set_ylim(*y_limits[metric_idx])
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.12,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[label]) for label, _ in ROWS]
    legend_labels = [label for label, _ in ROWS]
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir = Path("inference_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "table3_barplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()