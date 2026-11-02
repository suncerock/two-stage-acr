from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


METRICS = ["WCSR/majmin", "WCSR/tetrads"]
SYSTEMS = ["Chord-S", "Joint-S", "MPE+Chord-F"]
VARIANTS = ["(C)", "(P)"]
DATA = {
    "Chord-S": {"(C)": (65.04, 45.78), "(P)": (77.79, 58.62)},
    "Joint-S": {"(C)": (64.77, 46.11), "(P)": (78.49, 59.10)},
    "MPE+Chord-F": {"(C)": (70.00, 50.62), "(P)": (79.11, 60.59)},
}
COLORS = {
    "Chord-S": "#0072B2",
    "Joint-S": "#E69F00",
    "MPE+Chord-F": "#CC79A7",
}


def main() -> None:
    x = np.arange(len(SYSTEMS))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharex=True)
    y_limits = [(44, 81), (44, 81)]

    for metric_idx, metric_name in enumerate(METRICS):
        ax = axes[metric_idx]
        c_values = [DATA[system]["(C)"][metric_idx] for system in SYSTEMS]
        p_values = [DATA[system]["(P)"][metric_idx] for system in SYSTEMS]
        facecolors = [COLORS[system] for system in SYSTEMS]

        c_bars = ax.bar(
            x - width / 2,
            c_values,
            width,
            color=facecolors,
            edgecolor="#444444",
            linewidth=1.3,
            linestyle="--",
            hatch="//",
            label="(C)",
        )
        p_bars = ax.bar(
            x + width / 2,
            p_values,
            width,
            color=facecolors,
            edgecolor="#444444",
            linewidth=0.9,
            label="(P)",
        )

        ax.set_title(metric_name)
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(SYSTEMS)
        ax.set_ylim(*y_limits[metric_idx])
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

        for bars in (c_bars, p_bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.2,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    system_handles = [Patch(facecolor=COLORS[system], edgecolor="#444444", label=system) for system in SYSTEMS]
    variant_handles = [
        Patch(facecolor="#FFFFFF", edgecolor="#444444", linewidth=1.3, linestyle="--", hatch="//", label="(C)"),
        Patch(facecolor="#FFFFFF", edgecolor="#444444", linewidth=0.9, label="(P)"),
    ]

    fig.legend(system_handles, SYSTEMS, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.legend(variant_handles, ["Classical", "Pop"], loc="upper right", bbox_to_anchor=(0.98, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir = Path("inference_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "table4_barplot.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()