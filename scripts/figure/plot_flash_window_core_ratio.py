from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = ROOT_DIR / ".matplotlib"
XDG_CACHE_HOME = ROOT_DIR / ".cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch

from plot_style import legend_box_kwargs, style_legend_frame


CSV_PATH = ROOT_DIR / ".csv" / "flash_window_core_ratio.csv"
FIGURE_DIR = ROOT_DIR / "figures"
OUT_PATH = FIGURE_DIR / "flash_window_core_ratio.pdf"

WIDTH_CM = 3.0
HEIGHT_CM = 4.15
FONT_SIZE_PT = 6.4
OUTPUT_BBOX = Bbox.from_bounds(0.0582, 0.3258, 81.6376 / 72.0, 83.7395 / 72.0)
TOKENS = [256, 512, 768, 1024]
TOKEN_LABELS = ["256", "512", "768", "1024"]
GROUP_STEP = 0.54


def cm_to_inch(value: float) -> float:
    return value / 2.54


def load_ratios() -> dict[str, dict[int, float]]:
    ratios: dict[str, dict[int, float]] = {"Rocket": {}, "BOOM": {}}
    with CSV_PATH.open() as csv_file:
        for row in csv.DictReader(csv_file):
            if row["model"] != "opt":
                continue
            ratios[row["core"]][int(row["tokens"])] = float(row["flash_window_ratio"])
    return ratios


def boxed_legend(ax: plt.Axes) -> None:
    handles = [
        Patch(facecolor="#6B6B6B", edgecolor="black", linewidth=0.55, label="Rocket"),
        Patch(facecolor="#B279A2", edgecolor="black", linewidth=0.55, hatch="//", label="BOOM"),
    ]
    legend = ax.legend(
        handles=handles,
        ncols=1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.42),
        handleheight=0.80,
        **legend_box_kwargs("two", columnspacing=0.0),
    )
    style_legend_frame(legend)


def apply_compact_style(ax: plt.Axes, x: np.ndarray) -> None:
    ax.axhline(1.0, color="#D43F35", linestyle="--", linewidth=0.6, zorder=2)
    ax.set_ylabel("F/W ratio", labelpad=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(TOKEN_LABELS)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_xlim(x[0] - 0.32, x[-1] + 0.32)
    ax.set_ylim(0.0, 3.0)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(axis="y", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.formatter.use_mathtext": True,
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
            "hatch.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ratios = load_ratios()
    x = np.arange(len(TOKENS)) * GROUP_STEP
    width = 0.14

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(left=0.32, right=0.99, top=0.74, bottom=0.30)

    ax.bar(
        x - width / 2,
        [ratios["Rocket"][token] for token in TOKENS],
        width,
        label="Rocket",
        color="#6B6B6B",
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        [ratios["BOOM"][token] for token in TOKENS],
        width,
        label="BOOM",
        color="#B279A2",
        edgecolor="black",
        linewidth=0.45,
        hatch="//",
        zorder=3,
    )

    apply_compact_style(ax, x)
    boxed_legend(ax)

    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches=OUTPUT_BBOX, pad_inches=0.0)
    plt.close(fig)

    for token in TOKENS:
        print(f"{token} tok Rocket Flash/Window: {ratios['Rocket'][token]:.4f}")
        print(f"{token} tok BOOM Flash/Window: {ratios['BOOM'][token]:.4f}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
