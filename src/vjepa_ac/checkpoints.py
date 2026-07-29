import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CKPT_ROOT = Path("checkpoints")
COMPRESSOR_PATH = CKPT_ROOT / "compressor.safetensors"
CURRENT_PATH = CKPT_ROOT / "current.safetensors"


def save_current(model: torch.nn.Module, sidecar: dict) -> None:
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.{k}": v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    tmp = CURRENT_PATH.with_name(CURRENT_PATH.name + ".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, CURRENT_PATH)
    with open(CURRENT_PATH.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)


def load_model_weights(path: str | Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("model.")]
        assert keys, f"{path} has no 'model.' tensors"
        return {k[len("model.") :]: f.get_tensor(k) for k in keys}
