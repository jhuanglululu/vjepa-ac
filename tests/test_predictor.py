import torch

from vjepa_ac.predictor import Predictor, RMSNorm, block_causal_mask
from vjepa_ac.variations import ModelConfig

TINY = ModelConfig(d_state=32, patch_grid=4, d_model=64, d_ff=256, n_heads=4, n_layers=2)


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
    cfg = TINY
    model = Predictor(cfg, max_T=3)
    z = torch.randn(2, 3, cfg.n_patches, cfg.d_state)
    a = torch.randn(2, 3, cfg.d_action)
    out = model(z, a)
    assert out.shape == (2, 3, cfg.n_patches, cfg.d_state)


def test_causality():
    torch.manual_seed(0)
    cfg = TINY
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


def test_rmsnorm_matches_torch_in_fp32():
    torch.manual_seed(0)
    norm = RMSNorm(16, eps=1e-6)
    ref = torch.nn.RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        ref.weight.copy_(norm.weight)
    x = torch.randn(4, 16)
    assert torch.allclose(norm(x), ref(x), atol=1e-6)


def test_rmsnorm_stays_in_bf16():
    norm = RMSNorm(8, eps=1e-6)
    out = norm(torch.randn(2, 8).bfloat16())
    assert out.dtype == torch.bfloat16


def test_rope_rotates_by_frame_not_slot():
    from vjepa_ac.predictor import RoPE

    rope = RoPE(d_head=4, max_T=3)
    x = torch.randn(1, 1, 1, 4).expand(1, 1, 2, 4).contiguous()
    same_frame = rope(x, torch.tensor([1, 1]))
    assert torch.equal(same_frame[0, 0, 0], same_frame[0, 0, 1])
    diff_frame = rope(x, torch.tensor([1, 2]))
    assert not torch.allclose(diff_frame[0, 0, 0], diff_frame[0, 0, 1])
