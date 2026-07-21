import torch

from vjepa_ac.compressor import Compressor, IDHead, ReconHead
from vjepa_ac.cpredictor import CPredictor
from vjepa_ac.variations import MODELS


def test_compressor_shapes_flat_and_windowed():
    comp = Compressor(d_latent=32, n_tokens=4, d_c=16, n_heads=4)
    assert comp(torch.randn(3, 16, 32)).shape == (3, 4, 16)
    assert comp(torch.randn(2, 5, 16, 32)).shape == (2, 5, 4, 16)


def test_id_head_shape():
    idh = IDHead(d_c=16, d_out=7)
    c = torch.randn(3, 4, 16)
    assert idh(c, c).shape == (3, 7)
    cw = torch.randn(2, 5, 4, 16)
    assert idh(cw, cw).shape == (2, 5, 7)


def test_recon_head_shape():
    recon = ReconHead(d_latent=32, n_patches=16, d_c=16, n_heads=4)
    assert recon(torch.randn(3, 4, 16)).shape == (3, 16, 32)


def test_cpredictor_encode_and_forward():
    mc = MODELS["tiny-c"]
    model = CPredictor(mc, max_T=4)
    z = torch.randn(2, 4, mc.comp_patches, mc.comp_d_latent)
    tokens = model.encode(z)
    assert tokens.shape == (2, 4, mc.n_patches, mc.d_state)
    a = torch.randn(2, 4, mc.d_action)
    assert model(tokens, a).shape == tokens.shape


def test_cpredictor_stats_travel_in_state_dict():
    mc = MODELS["tiny-c"]
    model = CPredictor(mc, max_T=4)
    model.set_stats(torch.full((mc.d_state,), 2.0), torch.full((mc.d_state,), 3.0))
    clone = CPredictor(mc, max_T=4)
    clone.load_state_dict(model.state_dict())
    z = torch.randn(1, 2, mc.comp_patches, mc.comp_d_latent)
    assert torch.allclose(model.encode(z), clone.encode(z))
    assert torch.allclose(clone.c_mean, torch.full((mc.d_state,), 2.0))
