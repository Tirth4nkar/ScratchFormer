"""
plots.py — Reusable visualization helpers.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from my_project.config import FIGURES_DIR


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {path}")
