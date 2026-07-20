import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors import safe_open

from .variations import ModelConfig

HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
DATASET_ID = "nvidia/Cosmos3-DROID"
DATASET_SPLIT = "success"
CAMERAS = {
    "ext1": "observation.image.exterior_image_1_left",
    "ext2": "observation.image.exterior_image_2_left",
    "wrist": "observation.image.wrist_image_left",
}
IMG_SIZE = 256
SPLIT_SEED = 0

CACHE_DIR = os.environ.get("VJEPA_CACHE_DIR", "./latent_cache")


def cache_paths(cache_dir: str | None = None) -> tuple[str, str]:
    d = cache_dir or CACHE_DIR
    return os.path.join(d, "latents.safetensors"), os.path.join(d, "cache.json")


@dataclass
class LatentCache:
    latents: Any
    actions: torch.Tensor
    states: torch.Tensor
    episodes: list[list[int]]
    action_dim: int
    state_dim: int
    meta: dict = field(default_factory=dict)


def load_cache(cache_dir: str | None = None) -> LatentCache:
    latents_path, meta_path = cache_paths(cache_dir)
    assert os.path.exists(latents_path), (
        f"no latent cache at {latents_path} -- run scripts/prepare_cache.py first"
    )
    with open(meta_path) as f:
        meta = json.load(f)
    cache = safe_open(latents_path, framework="pt", device="cpu")
    assert "state" in cache.keys(), (
        f"cache at {latents_path} has no states -- rebuild with scripts/prepare_cache.py"
    )
    return LatentCache(
        latents=cache.get_slice("latents"),
        actions=cache.get_tensor("actions").float(),
        states=cache.get_tensor("state").float(),
        episodes=meta["episodes"],
        action_dim=meta["action_dim"],
        state_dim=meta["state_dim"],
        meta=meta,
    )


def synthetic_cache(config: ModelConfig, seed: int = 0) -> LatentCache:
    gen = torch.Generator().manual_seed(seed)
    n_episodes, ep_len = 6, 16
    n = n_episodes * ep_len
    P, D, A = config.n_patches, config.d_state, config.d_action

    actions = torch.rand(n, A, generator=gen) * 2 - 1
    dynamics = torch.randn(A, P * D, generator=gen) * 0.2

    latents = torch.empty(n, P, D)
    states = torch.empty(n, A)
    episodes = []
    for e in range(n_episodes):
        a, b = e * ep_len, (e + 1) * ep_len
        episodes.append([a, b])
        z = torch.randn(P, D, generator=gen) * 0.5
        s = torch.zeros(A)
        for t in range(a, b):
            latents[t] = z
            states[t] = s
            step = (actions[t] @ dynamics).reshape(P, D)
            z = z + step + torch.randn(P, D, generator=gen) * 0.01
            s = s + actions[t]

    return LatentCache(
        latents=latents,
        actions=actions,
        states=states,
        episodes=episodes,
        action_dim=A,
        state_dim=A,
        meta={"synthetic": True},
    )


def split_episodes(
    episodes: list[list[int]], val_frac: float, seed: int = SPLIT_SEED
) -> tuple[list[list[int]], list[list[int]]]:
    eps = sorted(episodes)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(eps), generator=gen).tolist()
    n_val = max(1, round(val_frac * len(eps)))
    val = sorted(eps[i] for i in perm[:n_val])
    train = sorted(eps[i] for i in perm[n_val:])
    return train, val


def window_span(T: int, stride: int) -> int:
    return (T - 1) * stride + 1


def window_starts(episodes: list[list[int]], T: int, stride: int) -> torch.Tensor:
    span = window_span(T, stride)
    starts = []
    for a, b in episodes:
        starts.extend(range(a, b - span + 1))
    return torch.tensor(starts, dtype=torch.long)


class Conditioner:
    def __init__(self, states: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        wd = states[1:] - states[:-1]
        wd = torch.remainder(wd + math.pi, 2 * math.pi) - math.pi
        self.delta_cumsum = torch.cat([torch.zeros(1, states.shape[1]), wd.cumsum(0)])
        self.states = states
        self.mean = mean
        self.std = std

    def features(self, idx: torch.Tensor, stride: int) -> torch.Tensor:
        delta = self.delta_cumsum[idx + stride] - self.delta_cumsum[idx]
        grip = self.states[idx + stride][..., -1:]
        return torch.cat([delta[..., :-1], grip], dim=-1)

    def windows(self, starts: torch.Tensor, T: int, stride: int) -> torch.Tensor:
        idx = starts[:, None] + torch.arange(T - 1) * stride
        feats = (self.features(idx, stride) - self.mean) / self.std
        out = torch.zeros(len(starts), T, feats.shape[-1])
        out[:, :-1] = feats
        return out

    def stats(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_conditioner(states: torch.Tensor, episodes: list[list[int]], stride: int) -> Conditioner:
    dim = states.shape[1]
    cond = Conditioner(states, torch.zeros(dim), torch.ones(dim))
    idx = []
    for a, b in episodes:
        idx.extend(range(a, b - stride))
    feats = cond.features(torch.tensor(idx, dtype=torch.long), stride)
    cond.mean = feats.mean(0)
    cond.std = feats.std(0).clamp(min=1e-6)
    return cond


def load_conditioner(states: torch.Tensor, stats: dict) -> Conditioner:
    return Conditioner(states, torch.tensor(stats["mean"]), torch.tensor(stats["std"]))


def gather(
    cache: LatentCache,
    cond: Conditioner,
    starts: torch.Tensor,
    T: int,
    stride: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = window_span(T, stride)
    z = torch.stack([cache.latents[int(s) : int(s) + span][::stride] for s in starts])
    a = cond.windows(starts, T, stride)
    return z.to(device).float(), a.to(device)
