from src.dashboard.core import TriageRecord, priority_score, rank_tiles


def test_priority_and_rank():
    assert priority_score(0.5, 0.8) == 0.4
    ranked = rank_tiles([TriageRecord("a",0.5,0.5,0.25), TriageRecord("b",0.9,0.9,0.81)])
    assert ranked[0].tile_id == "b"
