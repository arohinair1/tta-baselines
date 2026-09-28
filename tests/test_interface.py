import torch
from tta_wrapper.losses import TTALossParts
from tta_wrapper import PTTEA


class DummySegModel(torch.nn.Module):
    """Small segmentation model with BatchNorm so PTTEA can adapt it."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(self.conv(x))


class DummyEnergyModel(torch.nn.Module):
    """Small differentiable energy model operating on segmentation probabilities."""

    def forward(self, probs):
        return probs[:, 1:2]


def make_wrapper():
    torch.manual_seed(0)
    return PTTEA(
        DummySegModel(),
        DummyEnergyModel(),
        device="cpu",
    )


def test_predict():
    tta = make_wrapper()
    image = torch.randn(1, 1, 16, 16)

    result = tta.predict(image)

    assert result.pred.shape == (16, 16)
    assert result.probs.shape == (1, 3, 16, 16)
    assert result.confidence.shape == (16, 16)


def test_energy():
    tta = make_wrapper()
    image = torch.randn(1, 1, 16, 16)

    energy = tta.energy(image)

    assert energy.shape == (1, 1, 16, 16)


def test_loss():
    tta = make_wrapper()
    image = torch.randn(1, 1, 16, 16)

    parts = tta.loss(image, strategy="none")

    assert parts.total.ndim == 0
    assert parts.energy.ndim == 0
    assert parts.pseudolabel is None


def test_adapt():
    tta = make_wrapper()
    image = torch.randn(1, 1, 16, 16)

    result = tta.adapt(
        image,
        strategy="none",
        num_iterations=2,
        lr=1e-3,
        variant="pttea",
    )

    assert result.pred.shape == (16, 16)
    assert result.probs.shape == (1, 3, 16, 16)
    assert result.iters_run == 2
    assert len(result.losses) == 2

def test_custom_loss():
    tta = make_wrapper()
    image = torch.randn(1, 1, 16, 16)

    calls = {"count": 0}

    def custom_loss(
        seg_logits,
        energy_model,
        strategy="none",
        warped_label=None,
        pixel_weights=None,
    ):
        calls["count"] += 1

        # Simple differentiable loss depending on the segmentation model.
        loss = (seg_logits ** 2).mean()

        return TTALossParts(
            total=loss,
            energy=loss,
            pseudolabel=None,
        )

    result = tta.adapt(
        image,
        loss_fn=custom_loss,
        strategy="none",
        num_iterations=2,
        lr=1e-3,
        variant="pttea",
    )

    assert calls["count"] == 2
    assert result.iters_run == 2
    assert len(result.losses) == 2
