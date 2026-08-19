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
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from plot_style import legend_box_kwargs, style_legend_frame


CSV_PATH = ROOT_DIR / ".csv" / "cnn_result.csv"
FIGURE_DIR = ROOT_DIR / "figures"

ROCKET_WIDTH_CM = 3.0
BACKEND_WIDTH_CM = 2.725
HEIGHT_CM = 4.15
FONT_SIZE_PT = 6.4

MODEL_ORDER = ["ResNet", "AlexNet", "MobileNet", "SqueezeNet"]
MODEL_LABELS = {
    "ResNet": "ResNet50",
    "AlexNet": "AlexNet",
    "MobileNet": "MobileNetV2",
    "SqueezeNet": "SqueezeNet",
}

GROUP_STEP = 0.54
LABEL_ROTATION_DEG = 25
LABEL_X_SHIFT_PT = 7.0
ROCKET_ADJUST = {
    "left": 0.32,
    "right": 0.99,
    "top": 0.66,
    "bottom": 0.30,
}
BACKEND_ADJUST = {
    "left": 0.25,
    "right": 0.99,
    "top": 0.66,
    "bottom": 0.30,
}

ROCKET_CONFIGS = [
    ("4-core", "4-core", "#4C78A8"),
    ("8-core", "8-core", "#F58518"),
    ("16-core", "16-core", "#54A24B"),
]

SATURN_CONFIGS = [
    ("2 Saturn", "2-core", "#9D755D"),
    ("4 Saturn", "4-core", "#72B7B2"),
]

GEMMINI_CONFIGS = [
    ("Dual Gemmini", "2-core", "#B279A2"),
    ("Quad Gemmini", "4-core", "#E45756"),
]


def cm_to_inch(value: float) -> float:
    return value / 2.54


def load_cycles() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH).replace({"x": np.nan, "": np.nan})
    df = df[df["variant"] == "baseline"].copy()
    for column in df.columns[3:]:
        df[column] = pd.to_numeric(df[column], errors="coerce") / 1e9
    return df


def apply_compact_style(fig: plt.Figure, ax: plt.Axes, x: np.ndarray, ylabel: str | None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=LABEL_ROTATION_DEG,
        ha="right",
        rotation_mode="anchor",
    )
    label_offset = mtransforms.ScaledTranslation(
        LABEL_X_SHIFT_PT / 72.0,
        0.0,
        fig.dpi_scale_trans,
    )
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + label_offset)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_xlim(x[0] - 0.33, x[-1] + 0.33)
    ax.grid(axis="y", color="#BDBDBD", linestyle="--", linewidth=0.45, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")


def boxed_legend(ax: plt.Axes, ncols: int, y: float) -> None:
    legend = ax.legend(
        ncols=ncols,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        **legend_box_kwargs("one"),
    )
    style_legend_frame(legend)


def draw_rocket(ax: plt.Axes, df: pd.DataFrame, x: np.ndarray) -> None:
    hardware_df = df[df["device"] == "Rocket"].set_index("model")
    baseline = hardware_df["4-core"].reindex(MODEL_ORDER)
    width = 0.14

    for idx, (column, label, color) in enumerate(ROCKET_CONFIGS):
        values = baseline / hardware_df[column].reindex(MODEL_ORDER)
        offset = (idx - (len(ROCKET_CONFIGS) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )

    ax.set_ylim(0.0, 4.0)
    ax.set_yticks(np.arange(0.0, 4.1, 1.0))
    ax.set_yticklabels(["0.0", "1.0", "2.0", "3.0", ""])
    boxed_legend(ax, ncols=3, y=1.28)


def draw_backend(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: np.ndarray,
    device: str,
    configs: list[tuple[str, str, str]],
    baseline_column: str,
) -> None:
    hardware_df = df[df["device"] == device].set_index("model")
    baseline = hardware_df[baseline_column].reindex(MODEL_ORDER)
    width = 0.14
    max_speedup = 1.0

    for idx, (column, label, color) in enumerate(configs):
        values = baseline / hardware_df[column].reindex(MODEL_ORDER)
        max_speedup = max(max_speedup, float(np.nanmax(values.to_numpy())))
        offset = (idx - (len(configs) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )

    ax.set_ylim(0.0, max(2.2, max_speedup * 1.18))
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    boxed_legend(ax, ncols=2, y=1.28)


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = load_cycles()
    x = np.arange(len(MODEL_ORDER)) * GROUP_STEP
    FIGURE_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(ROCKET_WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(**ROCKET_ADJUST)
    draw_rocket(ax, df, x)
    apply_compact_style(fig, ax, x, "Norm. Perf.")
    out_path = FIGURE_DIR / "cnn_result_rocket.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {out_path}")
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(BACKEND_WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(**BACKEND_ADJUST)
    draw_backend(ax, df, x, "Saturn", SATURN_CONFIGS, "2 Saturn")
    apply_compact_style(fig, ax, x, None)
    out_path = FIGURE_DIR / "cnn_result_saturn.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {out_path}")
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(cm_to_inch(BACKEND_WIDTH_CM), cm_to_inch(HEIGHT_CM)),
        dpi=220,
    )
    fig.subplots_adjust(**BACKEND_ADJUST)
    draw_backend(ax, df, x, "Gemmini", GEMMINI_CONFIGS, "Dual Gemmini")
    apply_compact_style(fig, ax, x, None)
    out_path = FIGURE_DIR / "cnn_result_gemmini.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
