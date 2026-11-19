"""Discretise the pitch into the grid used by xT (Singh, 2019 uses 16 x 12 cells)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.clean import PITCH_X, PITCH_Y


def cell_index(x, y, nx: int = 16, ny: int = 12):
    """Return (ix, iy) integer cell coordinates for StatsBomb positions (arrays or scalars)."""
    ix = np.clip((np.asarray(x) / PITCH_X * nx).astype(int), 0, nx - 1)
    iy = np.clip((np.asarray(y) / PITCH_Y * ny).astype(int), 0, ny - 1)
    return ix, iy


def add_cells(df: pd.DataFrame, nx: int = 16, ny: int = 12) -> pd.DataFrame:
    """Add start/end cell columns (``cell``, ``end_cell`` are flat ids ``iy * nx + ix``)."""
    out = df.copy()
    out["ix"], out["iy"] = cell_index(out["x"], out["y"], nx, ny)
    out["cell"] = out["iy"] * nx + out["ix"]
    has_end = out["end_x"].notna() & out["end_y"].notna()
    ex, ey = cell_index(out["end_x"].fillna(0), out["end_y"].fillna(0), nx, ny)
    out["end_cell"] = np.where(has_end, ey * nx + ex, -1)
    return out


def count_grid(x, y, nx: int = 16, ny: int = 12, weights=None) -> np.ndarray:
    """2-D histogram of positions on the grid, shape (ny, nx) so it plots as an image."""
    ix, iy = cell_index(x, y, nx, ny)
    grid = np.zeros((ny, nx))
    np.add.at(grid, (iy, ix), 1 if weights is None else np.asarray(weights))
    return grid
