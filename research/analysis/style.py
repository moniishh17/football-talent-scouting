"""Shared matplotlib style (validated categorical order from the dataviz palette; light surface)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]       # blue, orange, aqua, magenta (fixed order)
POS_COLOR = {"DEF": SERIES[0], "MID": SERIES[1], "FWD": SERIES[2], "GK": SERIES[3]}
SEQ = "Blues"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 160,
        "font.family": "DejaVu Sans", "axes.axisbelow": True,
    })


def save(fig, path, caption=None):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if caption:
        fig.text(0.01, -0.13, caption, fontsize=7, color=INK2, ha="left", va="top", wrap=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
