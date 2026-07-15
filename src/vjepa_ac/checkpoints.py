import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CKPT_ROOT = Path(os.environ.get("VJEPA_CKPT_DIR", "checkpoints"))


def checkpoint_dir(model: str, training: str, seed: int) -> Path:
    return CKPT_ROOT / model / training / str(seed)


def atomic_save_file(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)


def flatten_optim_state(optim_sd: dict) -> tuple[dict[str, torch.Tensor], list]:
    tensors = {}
    for idx, state in optim_sd["state"].items():
        for k, v in state.items():
            assert isinstance(v, torch.Tensor)
            tensors[f"optim.{idx}.{k}"] = v.detach().cpu().contiguous()
    return tensors, optim_sd["param_groups"]


def unflatten_optim_state(tensors: dict[str, torch.Tensor], param_groups: list) -> dict:
    state: dict[int, dict] = {}
    for key, v in tensors.items():
        _, idx, name = key.split(".", 2)
        state.setdefault(int(idx), {})[name] = v
    return {"state": state, "param_groups": param_groups}


def rng_tensors() -> dict[str, torch.Tensor]:
    out = {"rng.torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        for i, s in enumerate(torch.cuda.get_rng_state_all()):
            out[f"rng.cuda_{i}"] = s
    return out


def restore_rng(tensors: dict[str, torch.Tensor]) -> None:
    torch.set_rng_state(tensors["rng.torch_cpu"])
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            key = f"rng.cuda_{i}"
            if key in tensors:
                torch.cuda.set_rng_state(tensors[key], i)


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    sched_state: dict,
    val_loss: float,
    run_info: dict,
) -> Path:
    tensors = {f"model.{k}": v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    optim_tensors, param_groups = flatten_optim_state(optim.state_dict())
    tensors.update(optim_tensors)
    tensors.update(rng_tensors())

    path = ckpt_dir / f"{step}.safetensors"
    atomic_save_file(tensors, path)
    sidecar = {
        "step": step,
        "val_loss": val_loss,
        "param_groups": param_groups,
        "sched": sched_state,
        **run_info,
    }
    with open(ckpt_dir / f"{step}.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    shutil.copyfile(path, ckpt_dir / "current.safetensors.tmp")
    os.replace(ckpt_dir / "current.safetensors.tmp", ckpt_dir / "current.safetensors")
    shutil.copyfile(ckpt_dir / f"{step}.json", ckpt_dir / "current.json")
    return path


def load_checkpoint(ckpt_dir: Path) -> tuple[dict[str, torch.Tensor], dict] | None:
    path = ckpt_dir / "current.safetensors"
    sidecar_path = ckpt_dir / "current.json"
    if not (path.exists() and sidecar_path.exists()):
        return None
    with safe_open(str(path), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    return tensors, sidecar


def split_checkpoint_tensors(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model_sd, optim_t, rng_t = {}, {}, {}
    for k, v in tensors.items():
        if k.startswith("model."):
            model_sd[k[len("model.") :]] = v
        elif k.startswith("optim."):
            optim_t[k] = v
        elif k.startswith("rng."):
            rng_t[k] = v
    return model_sd, optim_t, rng_t


def load_model_weights(path: str | Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("model.")]
        assert keys, f"{path} has no 'model.' tensors"
        return {k[len("model.") :]: f.get_tensor(k) for k in keys}


def prune_checkpoints(ckpt_dir: Path, best: list, keep: int) -> list:
    best = sorted(best, key=lambda x: x[0])
    for _, step in best[keep:]:
        for suffix in (".safetensors", ".json"):
            p = ckpt_dir / f"{step}{suffix}"
            if p.exists():
                p.unlink()
    return best[:keep]
