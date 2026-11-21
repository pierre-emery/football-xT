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

MOVE_TYPES = ("Pass", "Carry", "Dribble")
MOVE_COLUMNS = ("match_id", "competition_name", "type", "x", "y", "end_x", "end_y", "success", "set_piece")


def move_cells(actions: pd.DataFrame, nx: int = 16, ny: int = 12, open_play: bool = True) -> pd.DataFrame:
    """Moves (passes, carries, dribbles) used by xT, with their start / end cells.

    Drops passes with an ambiguous outcome and, if open_play, set pieces. A successful
    dribble has no end location in StatsBomb (the next carry moves the ball): it stays in its cell.
    """
    keep = [c for c in MOVE_COLUMNS if c in actions.columns]   # light copy: only what xT needs
    m = actions.loc[actions["type"].isin(MOVE_TYPES) & ~actions["ambiguous_outcome"], keep]
    if open_play:
        m = m[~m["set_piece"]]
    m = add_cells(m, nx, ny)
    drib_ok = m["type"].eq("Dribble") & m["success"].eq(True)
    m.loc[drib_ok, "end_cell"] = m.loc[drib_ok, "cell"]
    return m


def action_probabilities(actions: pd.DataFrame, nx: int = 16, ny: int = 12,
                         open_play: bool = True) -> dict[str, np.ndarray]:
    """Per-cell quantities of the xT recursion, as (ny, nx) grids (count-based estimates).

    The ball carrier in cell z shoots (prob. s) or moves the ball (prob. m = 1 - s). A move
    succeeds with prob. move_success (then the ball goes to z' with T[z, z']) or the ball is
    lost (p_loss = m * (1 - move_success)). A shot scores with prob. g (penalties excluded).
    Absorbing states: goal, missed shot, ball lost. Cells without data are NaN.
    """
    moves = move_cells(actions, nx, ny, open_play)
    shots = actions.loc[actions["type"].eq("Shot") & actions["sub_type"].ne("Penalty"),
                        ["x", "y", "is_goal", "set_piece"]]
    if open_play:
        shots = shots[~shots["set_piece"]]
    n_moves = count_grid(moves["x"], moves["y"], nx, ny)
    n_ok = count_grid(moves["x"], moves["y"], nx, ny, weights=moves["success"].astype(float))
    n_shots = count_grid(shots["x"], shots["y"], nx, ny)
    n_goals = count_grid(shots["x"], shots["y"], nx, ny, weights=shots["is_goal"].astype(float))
    n_act = n_moves + n_shots
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(n_act > 0, n_shots / n_act, np.nan)
        succ = np.where(n_moves > 0, n_ok / n_moves, np.nan)
        g = np.where(n_shots > 0, n_goals / n_shots, np.nan)
    return {"n_actions": n_act, "n_shots": n_shots, "n_moves": n_moves, "s": s, "m": 1 - s,
            "move_success": succ, "p_loss": (1 - s) * (1 - succ), "g": g}


def transition_counts(moves: pd.DataFrame, nx: int = 16, ny: int = 12) -> np.ndarray:
    """Counts N[z, z'] of successful moves from z to z' (flat ids, shape (nx*ny, nx*ny)).

    moves comes from move_cells. Dividing row z by the number of moves started in z
    (successful or not) gives T[z, z'] of the classic xT (rows then sum to move_success).
    """
    ok = moves[moves["success"].eq(True) & (moves["end_cell"] >= 0)]
    N = np.zeros((nx * ny, nx * ny))
    np.add.at(N, (ok["cell"].to_numpy(int), ok["end_cell"].to_numpy(int)), 1)
    return N