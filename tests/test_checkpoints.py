import pytest
import torch
from safetensors.torch import save_file

from vjepa_ac.checkpoints import load_model_weights


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
