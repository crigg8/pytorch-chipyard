from __future__ import annotations

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
import pandas as pd
from matplotlib.patches import Patch

from plot_style import legend_box_kwargs, style_legend_frame


CSV_PATH = ROOT_DIR / ".csv" / "flex_prefill.csv"
FIGURE_DIR = ROOT_DIR / "figures"

WIDTH_CM = 3.0
HEIGHT_CM = 4.15
FONT_SIZE_PT = 6.4

MODEL_ORDER = ["opt", "pythia"]
MODEL_LABELS = {
    "opt": "OPT-125M",
    "pythia": "Pythia-160M",
}
OUTPUTS = {
    "opt": "flex_prefill_opt.pdf",
    "pythia": "flex_prefill_pythia.pdf",
}

TOKENS = [256, 512, 768, 1024]
TOKEN_LABELS = ["256", "512", "768", "1024"]
GROUP_STEP = 0.54

SERIES = [
    ("SDPA", "sdpa_cycles", "#4C78A8", ""),
    ("Flash", "flash_cycles", "#F58518", "////"),
    ("Window", "window_cycles", "#54A24B", r"\\\\"),
]


def cm_to_inch(value: float) -> float:
    return value / 2.54


def load_cycles() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH).sort_values(["model", "tokens"]).reset_index(drop=True)
    for col in ["sdpa_cycles", "flash_cycles", "window_cycles"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 1e9
    return df


def boxed_legend(ax: plt.Axes) -> None:
    by_label = {
        label: Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=0.55,
            hatch=hatch,
        )
        for label, _, color, hatch in SERIES
    }
    legend_order = ["SDPA", "Window", "Flash"]
    legend = ax.legend(
        [by_label[label] for label in legend_order],
        legend_order,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.42),
        handleheight=0.80,
        **legend_box_kwargs("two"),
    )
    style_legend_frame(legend)


def apply_compact_style(
    ax: plt.Axes,
    x: np.ndarray,
    ylabel: str | None,
    show_y_ticklabels: bool = True,
    reserve_y_axis_space: bool = False,
) -> None:
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=0.5)
    elif reserve_y_axis_space:
        ax.set_ylabel("Cycle(B)", labelpad=0.5, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(TOKEN_LABELS)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_xlim(x[0] - 0.32, x[-1] + 0.32)
    ax.set_ylim(0.0, 60.0)
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(axis="y", labelleft=show_y_ticklabels or reserve_y_axis_space)
    if reserve_y_axis_space and not show_y_ticklabels:
        ax.tick_params(axis="y", labelcolor="white")
    ax.grid(axis="y", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")


def draw_model(df: pd.DataFrame, model: str, ylabel: str | None) -> None:
    model_df = df[df["model"] == model].set_index("tokens").reindex(TOKENS)
    x = np.arange(len(TOKENS)) * GROUP_STEP
    width = 0.12
    adjust = {"left": 0.32, "right": 0.99, "top": 0.74, "bottom": 0.30}

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(**adjust)

    for idx, (label, column, color, hatch) in enumerate(SERIES):
        offset = (idx - (len(SERIES) - 1) / 2) * width
        ax.bar(
            x + offset,
            model_df[column],
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            hatch=hatch,
            zorder=3,
        )

    reserve_y_axis_space = ylabel is None
    apply_compact_style(
        ax,
        x,
        ylabel,
        show_y_ticklabels=(ylabel is not None),
        reserve_y_axis_space=reserve_y_axis_space,
    )
    boxed_legend(ax)

    FIGURE_DIR.mkdir(exist_ok=True)
    out_path = FIGURE_DIR / OUTPUTS[model]
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_path}")
    plt.close(fig)


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
            "legend.title_fontsize": 5.5,
            "hatch.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = load_cycles()
    draw_model(df, "opt", "Cycle(B)")
    draw_model(df, "pythia", None)


if __name__ == "__main__":
    main()
