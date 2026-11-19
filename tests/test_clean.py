"""Unit tests for the event -> action flattening (no network needed)."""
import pandas as pd

from bxt.data.clean import clean_actions, events_to_actions
from bxt.features import add_cells, count_grid


def _ev(i, etype, player="A", loc=(60.0, 40.0), period=1, **extra):
    e = {"id": f"e{i}", "index": i, "period": period, "minute": 0, "second": i, "possession": 1,
         "possession_team": {"name": "T"}, "team": {"name": "T"}, "player": {"name": player},
         "position": {"name": "CM"}, "type": {"name": etype}, "play_pattern": {"name": "Regular Play"},
         "location": list(loc)}
    e.update(extra)
    return e


EVENTS = [
    _ev(1, "Pass", **{"pass": {"end_location": [70, 30], "height": {"name": "Ground Pass"}}}),
    _ev(2, "Pass", **{"pass": {"end_location": [90, 30], "outcome": {"name": "Incomplete"}}}),
    _ev(3, "Carry", **{"carry": {"end_location": [75, 35]}}),
    _ev(4, "Dispossessed"),
    _ev(5, "Carry", player="B", **{"carry": {"end_location": [80, 40]}}),
    _ev(6, "Dribble", player="B", **{"dribble": {"outcome": {"name": "Complete"}}}),
    _ev(7, "Shot", player="B", loc=(110, 40), **{"shot": {"end_location": [120, 40, 1], "statsbomb_xg": 0.3,
                                                          "outcome": {"name": "Goal"}, "type": {"name": "Open Play"}}}),
    _ev(8, "Shot", player="B", loc=(108, 40), period=5, **{"shot": {"end_location": [120, 40], "statsbomb_xg": 0.78,
                                                                    "outcome": {"name": "Saved"}, "type": {"name": "Penalty"}}}),
    _ev(9, "Pressure"),  # not kept
]

def test_success_flags():
    df = events_to_actions(EVENTS, match_id=1).set_index("index")
    assert 9 not in df.index
    assert df.loc[1, "success"] and not df.loc[2, "success"]
    assert not df.loc[3, "success"], "carry followed by Dispossessed of same player is lost"
    assert df.loc[5, "success"]
    assert df.loc[6, "success"]
    assert df.loc[7, "is_goal"] and df.loc[7, "shot_xg"] == 0.3
    assert pd.isna(df.loc[4, "success"])


def test_clean_drops_shootout_and_clips():
    ev = EVENTS + [_ev(10, "Carry", loc=(120.4, 80.2), **{"carry": {"end_location": [121, 81]}})]
    df = clean_actions(events_to_actions(ev, match_id=1))
    assert (df.period < 5).all()
    assert df.x.max() <= 120 and df.y.max() <= 80 and df.end_x.max() <= 120


def test_grid():
    df = add_cells(clean_actions(events_to_actions(EVENTS, match_id=1)))
    assert df.cell.between(0, 191).all()
    assert (df.loc[df.type.eq("Dribble"), "end_cell"] == -1).all()
    g = count_grid([0, 119.9, 60], [0, 79.9, 40])
    assert g.shape == (12, 16) and g.sum() == 3 and g[0, 0] == 1 and g[11, 15] == 1
