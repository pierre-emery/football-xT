"""One row per possession, to describe how possessions start and end (and later validate xT)."""
from __future__ import annotations

import pandas as pd

ON_BALL = ("Pass", "Carry", "Dribble", "Shot", "Dispossessed", "Miscontrol", "Clearance", "Foul Won")


def _end_type(last: pd.Series) -> str:
    t = last["type"]
    if t == "Shot":
        return "goal" if last["is_goal"] else "shot (no goal)"
    if t in ("Pass", "Carry", "Dribble") and last["success"] is not None and not bool(last["success"]):
        return "failed move"
    if t in ("Dispossessed", "Miscontrol"):
        return "dispossessed / miscontrol"
    if t == "Clearance":
        return "clearance"
    if t == "Foul Won":
        return "foul won"
    return "other (out, foul, end of half...)"


def possession_table(actions: pd.DataFrame) -> pd.DataFrame:
    """Summarise each StatsBomb possession (match_id, possession), using only the on-ball events
    of the team in possession. Own goals are not shots, so they are not counted in has_goal."""
    cols = ["match_id", "possession", "index", "team", "competition_name", "type", "x", "minute", "second",
            "set_piece", "success", "is_goal", "shot_xg"]
    a = actions.loc[actions["team"].eq(actions["possession_team"]) & actions["type"].isin(ON_BALL), cols]
    a = a.sort_values(["match_id", "index"])
    a = a.assign(t=a["minute"] * 60 + a["second"],
                 is_move_or_shot=a["type"].isin(["Pass", "Carry", "Dribble", "Shot"]),
                 is_pass=a["type"].eq("Pass"), is_shot=a["type"].eq("Shot"),
                 success_obj=a["success"].astype(object).where(a["success"].notna(), None))
    g = a.groupby(["match_id", "possession"], sort=False)
    out = g.agg(team=("team", "first"), competition_name=("competition_name", "first"),
                n_actions=("is_move_or_shot", "sum"), n_passes=("is_pass", "sum"),
                start_x=("x", "first"), t0=("t", "min"), t1=("t", "max"),
                set_piece_start=("set_piece", "first"),
                has_shot=("is_shot", "any"), has_goal=("is_goal", "any"),
                xg=("shot_xg", "sum"))
    last = g.tail(1).set_index(["match_id", "possession"])
    last = last.assign(success=last["success_obj"])
    out["end_type"] = [_end_type(r) for _, r in last.reindex(out.index).iterrows()]
    out["duration"] = out.pop("t1") - out.pop("t0")
    return out.reset_index()