"""Pitch plotting helpers (mplsoccer, StatsBomb 120 x 80 frame)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch


def pitch(ax=None, figsize=(8, 5.5), **kwargs):
    p = Pitch(pitch_type="statsbomb", line_zorder=2, **kwargs)
    if ax is None:
        fig, ax = p.draw(figsize=figsize)
    else:
        p.draw(ax=ax)
    return p, ax


def grid_heatmap(grid: np.ndarray, ax=None, title: str | None = None, cmap="viridis", annotate=False,
                 fmt="{:.2f}", vmin=None, vmax=None, colorbar=True):
    """Plot a (ny, nx) grid (as returned by ``count_grid``) over the pitch."""
    p, ax = pitch(ax=ax)
    ny, nx = grid.shape
    stat = p.bin_statistic(np.array([0.0]), np.array([0.0]), statistic="count", bins=(nx, ny))
    stat["statistic"] = grid
    pcm = p.heatmap(stat, ax=ax, cmap=cmap, edgecolors="#22312b", vmin=vmin, vmax=vmax)
    if annotate:
        stat["statistic"] = np.vectorize(lambda v: fmt.format(v))(grid)
        p.label_heatmap(stat, ax=ax, color="white", fontsize=6, ha="center", va="center",
                        str_format="{}")
    if colorbar:
        plt.colorbar(pcm, ax=ax, shrink=0.7)
    if title:
        ax.set_title(title)
    return ax
