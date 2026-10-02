import numpy as np

from src.evaluation.metrics import binary_iou_f1, damage_scores, triage_recall


def test_metrics_finite():
    target = np.array([[[0,1],[3,4]], [[0,2],[2,4]]])
    pred = target.copy()
    assert binary_iou_f1(target>0, target>0)["building_f1"] == 1.0
    s = damage_scores(pred, target)
    assert s["damage_macro_f1"] == 1.0
    r = triage_recall(np.array([0.1,0.9]), target, [0.5,1.0])
    assert r["top_50_severe_pixel_recall"] > 0


def test_summarize_predictions_priority_is_per_sample():
    import torch
    from src.evaluation.metrics import summarize_predictions
    b = torch.zeros(2, 2, 4, 4)
    b[:, 1] = 2.0
    d = torch.zeros(2, 5, 4, 4)
    d[0, 4] = 3.0
    d[1, 1] = 3.0
    out = summarize_predictions(b, d)
    assert out["priority"].shape == (2,)
    assert out["priority"][0] > out["priority"][1]
