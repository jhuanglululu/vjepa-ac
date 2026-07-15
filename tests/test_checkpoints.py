import pytest
import torch
from safetensors.torch import save_file

from vjepa_ac.checkpoints import (
    flatten_optim_state,
    load_model_weights,
    prune_checkpoints,
    unflatten_optim_state,
)


def make_stepped_optim():
    lin = torch.nn.Linear(4, 4)
    optim = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    lin(torch.randn(2, 4)).sum().backward()
    optim.step()
    return optim


def test_optim_state_roundtrip():
    optim = make_stepped_optim()
    sd = optim.state_dict()
    tensors, param_groups = flatten_optim_state(sd)
    restored = unflatten_optim_state(tensors, param_groups)
    assert restored["param_groups"] == sd["param_groups"]
    assert set(restored["state"].keys()) == set(sd["state"].keys())
    for idx, state in sd["state"].items():
        for k, v in state.items():
            assert torch.equal(restored["state"][idx][k], v)


def test_load_model_weights_requires_prefixed_layout(tmp_path):
    w = {"lin.weight": torch.randn(2, 2), "lin.bias": torch.randn(2)}
    prefixed = tmp_path / "prefixed.safetensors"
    save_file(
        {f"model.{k}": v for k, v in w.items()} | {"optim.0.exp_avg": torch.zeros(2)},
        str(prefixed),
    )
    loaded = load_model_weights(prefixed)
    assert set(loaded.keys()) == set(w.keys())
    for k in w:
        assert torch.equal(loaded[k], w[k])

    raw = tmp_path / "raw.safetensors"
    save_file(w, str(raw))
    with pytest.raises(AssertionError):
        load_model_weights(raw)


def test_prune_keeps_best(tmp_path):
    best = []
    for step, val in [(10, 0.5), (20, 0.3), (30, 0.4), (40, 0.2)]:
        (tmp_path / f"{step}.safetensors").touch()
        (tmp_path / f"{step}.json").touch()
        best.append((val, step))
    kept = prune_checkpoints(tmp_path, best, keep=2)
    assert kept == [(0.2, 40), (0.3, 20)]
    remaining = {p.name for p in tmp_path.glob("*.safetensors")}
    assert remaining == {"40.safetensors", "20.safetensors"}
