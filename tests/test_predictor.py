import torch

from vjepa_ac.predictor import Predictor, block_causal_mask
from vjepa_ac.variations import MODELS


def test_block_causal_mask_exact():
    mask = block_causal_mask(2, 2)
    expected = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_forward_shape():
    cfg = MODELS["tiny"]
    model = Predictor(cfg, max_T=3)
    z = torch.randn(2, 3, cfg.n_patches, cfg.d_state)
    a = torch.randn(2, 3, cfg.d_action)
    out = model(z, a)
    assert out.shape == (2, 3, cfg.n_patches, cfg.d_state)


def test_causality():
    torch.manual_seed(0)
    cfg = MODELS["tiny"]
    model = Predictor(cfg, max_T=3).eval()
    z = torch.randn(1, 3, cfg.n_patches, cfg.d_state)
    a = torch.randn(1, 3, cfg.d_action)
    with torch.no_grad():
        out1 = model(z, a)
        z2 = z.clone()
        z2[:, 2] += 10.0
        a2 = a.clone()
        a2[:, 2] += 10.0
        out2 = model(z2, a2)
    assert torch.allclose(out1[:, :2], out2[:, :2], atol=1e-5)
    assert not torch.allclose(out1[:, 2], out2[:, 2], atol=1e-5)
