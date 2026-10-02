import torch

from src.training.losses import ReliefLensLoss


def test_loss_backward():
    out = {"building_logits": torch.randn(1,2,16,16,requires_grad=True), "damage_logits": torch.randn(1,5,16,16,requires_grad=True)}
    batch = {"building": torch.randint(0,2,(1,16,16)), "damage": torch.randint(0,5,(1,16,16))}
    loss = ReliefLensLoss([0.2,1,1.5,2,2.5])(out,batch)["total"]
    loss.backward()
    assert torch.isfinite(loss)


def test_focal_loss_uses_unweighted_pt_and_class_weight_separately():
    from src.training.losses import FocalCrossEntropy
    logits = torch.tensor([[[[2.0]], [[0.0]], [[0.0]]]])
    target = torch.tensor([[[1]]])
    loss = FocalCrossEntropy([1.0, 2.0, 3.0], gamma=2.0)(logits, target)
    p = torch.softmax(logits[0, :, 0, 0], dim=0)[1]
    expected = 2.0 * (1.0 - p).pow(2.0) * (-torch.log(p))
    assert torch.allclose(loss, expected, atol=1e-6)
