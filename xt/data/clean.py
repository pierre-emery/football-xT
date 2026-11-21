"""Turn raw StatsBomb events into a tidy table of on-ball actions for xT / xG.

One row per event of the types we care about (passes, carries, dribbles, shots, and the
ball-losing events that end possessions). Coordinates use the StatsBomb frame:
a 120 x 80 pitch, origin at the top-left corner, and **every team attacks left to right**
(x = 120 is the opponent goal line), so no direction flipping is needed.

Columns
-------
match_id, event_id, index, period, minute, second, possession, possession_team, team, player,
position, type, play_pattern, x, y, end_x, end_y, under_pressure,
success      : nullable bool, did the action keep the ball (pass completed, dribble won, carry not
               lost, shot scored); NA for non-move events
ambiguous_outcome : bool, pass outcome is "Unknown" or "Injury Clearance" (not a real turnover)
outcome      : raw StatsBomb outcome name (NaN for completed passes)
sub_type     : pass/shot type (Corner, Free Kick, Throw-in, Penalty, Open Play...)
body_part, pass_height, pass_cross, pass_through_ball, pass_switch
set_piece    : bool, action is a set piece restart (corner, free kick, throw-in, goal kick,
               kick off, penalty)
shot_xg      : StatsBomb xG of the shot (baseline to beat for our own xG)
is_goal      : bool, shot outcome is Goal
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import REPO_ROOT, load_events, load_matches, download_events

PITCH_X, PITCH_Y = 120.0, 80.0

MOVE_TYPES = ("Pass", "Carry", "Dribble")
KEEP_TYPES = MOVE_TYPES + ("Shot", "Dispossessed", "Miscontrol", "Ball Recovery", "Interception",
                           "Clearance", "Foul Won")
SET_PIECE_PASS_TYPES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

GOAL_X, POST_Y1, POST_Y2 = 120.0, 36.0, 44.0   # goal line and posts in the StatsBomb frame
FF_COLUMNS = ("ff_n_opp_triangle", "ff_gk_in_triangle", "ff_gk_dist_goal", "ff_gk_dist_shooter",
              "ff_n_opp_close", "ff_visible_angle", "ff_visible_frac")


def _in_triangle(p, a, b, c):
    """True if point p is inside triangle (a, b, c) (sign of the three cross products)."""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3 = cross(p, a, b), cross(p, b, c), cross(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def shot_freeze_frame_features(loc, freeze_frame, close_radius: float = 2.0,
                               body_half_width: float = 0.5, gk_half_width: float = 1.0) -> dict:
    """Features of the defensive situation at a shot, from its StatsBomb freeze frame.

    * ff_n_opp_triangle: opponents (goalkeeper excluded) inside the triangle shooter-posts;
    * ff_gk_in_triangle: the goalkeeper is inside that triangle;
    * ff_gk_dist_goal / ff_gk_dist_shooter: goalkeeper distance to the goal centre / to the shooter;
    * ff_n_opp_close: opponents within close_radius units of the shooter (pressure);
    * ff_visible_angle (degrees) and ff_visible_frac: part of the goal opening angle not hidden
      by opponents in front of the ball. Each opponent hides an angular interval of half-width
      atan(half_width / distance). Same spirit as the Kos angle (Hassani & Lotfi), not their formula.
    Returns NaN features if there is no freeze frame.
    """
    out = dict.fromkeys(FF_COLUMNS, np.nan)
    if not freeze_frame or loc is None or len(loc) < 2:
        return out
    x, y = float(loc[0]), float(loc[1])
    p1, p2 = (GOAL_X, POST_Y1), (GOAL_X, POST_Y2)
    a1 = np.arctan2(POST_Y1 - y, GOAL_X - x)
    a2 = np.arctan2(POST_Y2 - y, GOAL_X - x)
    lo, hi = min(a1, a2), max(a1, a2)

    n_tri, n_close, gk, blocked = 0, 0, None, []
    for pl in freeze_frame:
        if pl.get("teammate", False):
            continue
        qx, qy = pl["location"][:2]
        is_gk = (pl.get("position") or {}).get("name") == "Goalkeeper"
        d = float(np.hypot(qx - x, qy - y))
        if is_gk:
            gk = (qx, qy, d)
        elif _in_triangle((qx, qy), (x, y), p1, p2):
            n_tri += 1
        if d <= close_radius:
            n_close += 1
        if qx > x and d > 0:   # only players in front of the ball can hide the goal
            aq = np.arctan2(qy - y, qx - x)
            w = np.arctan((gk_half_width if is_gk else body_half_width) / d)
            blocked.append((max(aq - w, lo), min(aq + w, hi)))

    # length of the union of the blocked intervals inside [lo, hi]
    blocked = sorted(iv for iv in blocked if iv[1] > iv[0])
    covered, cur_lo, cur_hi = 0.0, None, None
    for b_lo, b_hi in blocked:
        if cur_hi is None or b_lo > cur_hi:
            if cur_hi is not None:
                covered += cur_hi - cur_lo
            cur_lo, cur_hi = b_lo, b_hi
        else:
            cur_hi = max(cur_hi, b_hi)
    if cur_hi is not None:
        covered += cur_hi - cur_lo

    total = hi - lo
    out.update({
        "ff_n_opp_triangle": n_tri,
        "ff_gk_in_triangle": float(gk is not None and _in_triangle(gk[:2], (x, y), p1, p2)),
        "ff_gk_dist_goal": np.hypot(gk[0] - GOAL_X, gk[1] - (POST_Y1 + POST_Y2) / 2) if gk else np.nan,
        "ff_gk_dist_shooter": gk[2] if gk else np.nan,
        "ff_n_opp_close": n_close,
        "ff_visible_angle": np.degrees(max(total - covered, 0.0)),
        "ff_visible_frac": max(total - covered, 0.0) / total if total > 0 else np.nan,
    })
    return out

def _name(d, key):
    v = d.get(key)
    return v.get("name") if isinstance(v, dict) else v


def events_to_actions(events: list[dict], match_id: int) -> pd.DataFrame:
    """Flatten the raw events of one match into the tidy action table (see module docstring)."""
    rows = []
    for e in events:
        etype = e["type"]["name"]
        if etype not in KEEP_TYPES:
            continue
        loc = e.get("location") or [np.nan, np.nan]
        row = {
            "match_id": match_id,
            "event_id": e["id"],
            "index": e["index"],
            "period": e["period"],
            "minute": e["minute"],
            "second": e["second"],
            "possession": e["possession"],
            "possession_team": _name(e, "possession_team"),
            "team": _name(e, "team"),
            "player": _name(e, "player"),
            "position": _name(e, "position"),
            "type": etype,
            "play_pattern": _name(e, "play_pattern"),
            "x": loc[0],
            "y": loc[1],
            "end_x": np.nan,
            "end_y": np.nan,
            "under_pressure": bool(e.get("under_pressure", False)),
            "outcome": None,
            "sub_type": None,
            "body_part": None,
            "pass_height": None,
            "pass_cross": False,
            "pass_through_ball": False,
            "pass_switch": False,
            "shot_xg": np.nan,
            "is_goal": False,
            "shot_first_time": False,
            "shot_technique": None,
            **dict.fromkeys(FF_COLUMNS, np.nan),
        }
        if etype == "Pass":
            p = e["pass"]
            row["end_x"], row["end_y"] = p["end_location"][:2]
            row["outcome"] = _name(p, "outcome")  # None = completed
            row["sub_type"] = _name(p, "type")
            row["body_part"] = _name(p, "body_part")
            row["pass_height"] = _name(p, "height")
            row["pass_cross"] = bool(p.get("cross", False))
            row["pass_through_ball"] = bool(p.get("through_ball", False)) or _name(p, "technique") == "Through Ball"
            row["pass_switch"] = bool(p.get("switch", False))
        elif etype == "Carry":
            row["end_x"], row["end_y"] = e["carry"]["end_location"][:2]
        elif etype == "Dribble":
            row["outcome"] = _name(e["dribble"], "outcome")
        elif etype == "Shot":
            s = e["shot"]
            end = s.get("end_location") or [np.nan, np.nan]
            row["end_x"], row["end_y"] = end[0], end[1]
            row["outcome"] = _name(s, "outcome")
            row["sub_type"] = _name(s, "type")
            row["body_part"] = _name(s, "body_part")
            row["shot_xg"] = s.get("statsbomb_xg", np.nan)
            row["is_goal"] = row["outcome"] == "Goal"
            row["shot_first_time"] = bool(s.get("first_time", False))
            row["shot_technique"] = _name(s, "technique")
            row.update(shot_freeze_frame_features(e.get("location"), s.get("freeze_frame")))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("index").reset_index(drop=True)

    # --- success flag -------------------------------------------------------------------
    success = pd.Series(False, index=df.index)
    is_pass = df["type"].eq("Pass")
    success[is_pass] = df.loc[is_pass, "outcome"].isna()
    is_drib = df["type"].eq("Dribble")
    success[is_drib] = df.loc[is_drib, "outcome"].eq("Complete")
    success[df["type"].eq("Shot")] = df.loc[df["type"].eq("Shot"), "is_goal"]
    # A carry is "lost" when the next kept event is a Dispossessed/Miscontrol by the same player.
    is_carry = df["type"].eq("Carry")
    nxt_type = df["type"].shift(-1)
    nxt_player = df["player"].shift(-1)
    lost = nxt_type.isin(["Dispossessed", "Miscontrol"]) & nxt_player.eq(df["player"])
    success[is_carry] = ~lost[is_carry]
    # Only meaningful for moves and shots; NA for the other (defensive / ball-losing) events.
    df["success"] = success.astype("boolean")
    df.loc[~df["type"].isin(MOVE_TYPES + ("Shot",)), "success"] = pd.NA
    # Pass outcomes "Unknown" and "Injury Clearance" are not real turnovers: flag them so
    # models can drop them.
    df["ambiguous_outcome"] = df["type"].eq("Pass") & df["outcome"].isin(["Unknown", "Injury Clearance"])

    df["set_piece"] = df["sub_type"].isin(SET_PIECE_PASS_TYPES | {"Penalty", "Free Kick", "Corner"})
    return df


def clean_actions(df: pd.DataFrame, drop_shootouts: bool = True) -> pd.DataFrame:
    """Basic cleaning shared by every model.

    * drop penalty shoot-outs (period 5): not open play, would bias xG and xT;
    * drop rows without a start location;
    * clip coordinates into the pitch (a few StatsBomb events sit at 120.1 / 80.1).
    """
    out = df
    if drop_shootouts:
        out = out[out["period"] < 5]
    out = out.dropna(subset=["x", "y"]).copy()
    for c, hi in (("x", PITCH_X), ("end_x", PITCH_X), ("y", PITCH_Y), ("end_y", PITCH_Y)):
        out[c] = out[c].clip(0, hi)
    return out.reset_index(drop=True)


def build_actions(competition_id: int, season_id: int, max_matches: int | None = None,
                  n_workers: int = 8, save: bool = True, progress: bool = True) -> pd.DataFrame:
    """Download every match of a competition-season, flatten, clean and (optionally) save it.

    The result is written to ``data/processed/actions_<competition_id>_<season_id>.parquet``
    and re-read from there on the next call.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_matches is None else f"_n{max_matches}"
    out_path = PROCESSED_DIR / f"actions_{competition_id}_{season_id}{suffix}.parquet"
    if save and out_path.exists():
        return pd.read_parquet(out_path)

    matches = load_matches(competition_id, season_id)
    if max_matches is not None:
        matches = matches.head(max_matches)
    failed = download_events(matches["match_id"], n_workers=n_workers, progress=progress)
    if failed:
        print(f"[bxt] {len(failed)} matches failed to download and are skipped: {failed[:10]}")

    frames = []
    for mid in matches["match_id"]:
        if mid in failed:
            continue
        frames.append(events_to_actions(load_events(mid), mid))
    actions = clean_actions(pd.concat(frames, ignore_index=True))
    meta = matches[["match_id", "competition_name", "season_name", "match_date"]]
    actions = actions.merge(meta, on="match_id", how="left")
    if save:
        actions.to_parquet(out_path, index=False)
    return actions


def load_actions(pairs: list[tuple[int, int]], **kwargs) -> pd.DataFrame:
    """Concatenate :func:`build_actions` over several (competition_id, season_id) pairs."""
    return pd.concat([build_actions(c, s, **kwargs) for c, s in pairs], ignore_index=True)


__all__ = ["events_to_actions", "clean_actions", "build_actions", "load_actions", "PITCH_X", "PITCH_Y",
           "MOVE_TYPES", "PROCESSED_DIR", "shot_freeze_frame_features", "FF_COLUMNS"]


if __name__ == "__main__":  # pragma: no cover
    import sys
    c, s = int(sys.argv[1]), int(sys.argv[2])
    print(build_actions(c, s).shape, "->", Path(PROCESSED_DIR))
