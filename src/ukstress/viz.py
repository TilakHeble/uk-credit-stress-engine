"""Shared matplotlib style so every project chart reads as one system."""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # primary series
ORANGE = "#eb6834"  # comparison series
GREY = "#a8a7a2"  # context / de-emphasised
AQUA = "#1baf7a"
YELLOW = "#eda100"
PORTFOLIO_COLOURS = {  # fixed per portfolio across every chart
    "credit_card": BLUE,
    "other_consumer": ORANGE,
    "mortgage": AQUA,
    "corporate": YELLOW,
}
RECESSIONS = [("1990Q3", "1991Q3"), ("2008Q2", "2009Q2"), ("2020Q1", "2020Q2")]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 24,
            "axes.labelcolor": TEXT_2,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "axes.axisbelow": True,
            "text.parse_math": False,
        }
    )


def subtitle(ax, text: str) -> None:
    """Grey one-line subtitle under a left-aligned title."""
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9.5, va="bottom")


def save(fig, name: str) -> None:
    from ukstress.config import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def shade_recessions(ax, color: str = "#efeeea") -> None:
    """Shade UK recessions (two or more quarters of falling real GDP) on a quarterly time axis."""
    import pandas as pd

    for start, end in RECESSIONS:
        ax.axvspan(pd.Period(start).start_time, pd.Period(end).end_time, color=color, zorder=0)


def money(decimals: int = 0):
    """Tick formatter that writes -£1,000 as −£1,000 (minus before the currency sign)."""
    from matplotlib.ticker import FuncFormatter

    def fmt(x, _pos=None):
        sign = "\u2212" if x < 0 else ""
        return f"{sign}£{abs(x):,.{decimals}f}"

    return FuncFormatter(fmt)


def fig_title(fig, title: str, sub: str | None = None) -> None:
    """Left-aligned figure title (+ optional grey subtitle) for multi-panel figures.

    Positions are in inches, so title and subtitle never collide at any figure height.
    """
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.08 / h, ha="left", va="top", fontweight="bold", fontsize=13)
    top = 1 - 0.45 / h
    if sub:
        fig.text(0.01, 1 - 0.40 / h, sub, ha="left", va="top", color=TEXT_2, fontsize=9.5)
        top = 1 - 0.62 / h
    fig.tight_layout(rect=(0, 0, 1, top))
