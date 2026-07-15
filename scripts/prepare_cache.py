import copy
import json
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings(
    "ignore", message="The video decoding and encoding capabilities of torchvision"
)

import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoVideoProcessor

from vjepa_ac import data
from vjepa_ac.device import get_device, pick_free_gpus
from vjepa_ac.variations import TRAININGS

free_gpus = pick_free_gpus()
enc_devices = [f"cuda:{g}" for g in free_gpus[:4]] or [get_device()]
NUM_WORKERS = min(64, os.cpu_count() or 8)
DECODE_BATCH = 32


@torch.no_grad()
def encode_batch(frames, enc, dev, mean, std):
    x = frames.to(dev).float()
    if x.shape[-2:] != (data.IMG_SIZE, data.IMG_SIZE):
        x = F.interpolate(x, data.IMG_SIZE, mode="bilinear", align_corners=False, antialias=True)
    x = (x - mean) / std
    clip = x.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
        tok = enc(clip).last_hidden_state
    return tok.reshape(x.shape[0], tok.shape[1], 1024).half().cpu()


def check_health(latents, actions, episodes, T, stride):
    print("\n=== cache health check ===")
    N, P, D = latents.shape
    print(f"frames {N} | patches {P} | dim {D} | episodes {len(episodes)}")
    warns = []

    nonfinite = 0
    for i in range(0, N, 1024):
        nonfinite += (~torch.isfinite(latents[i : i + 1024])).sum().item()
    print(f"non-finite latent elements: {nonfinite}")
    if nonfinite:
        warns.append(f"{nonfinite} non-finite latent values (NaN/Inf)")

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(N, generator=g)[: min(N, 4096)]
    s = latents[idx].float()
    fn = s.norm(dim=(1, 2))
    print(
        f"latent abs-mean {s.abs().mean():.4f} | std {s.std():.4f} | "
        f"min {s.min():.3f} | max {s.max():.3f}"
    )
    print(f"per-frame L2 norm: mean {fn.mean():.2f} | std {fn.std():.2f}")
    if fn.mean() > 0 and fn.std() / fn.mean() < 1e-3:
        warns.append("per-frame latent norms nearly identical (possible encoder collapse)")

    sel = episodes[:: max(1, len(episodes) // 50)][:50]
    next_mse, sf_mse = [], []
    for a, b in sel:
        if b - a < 2:
            continue
        z = latents[a:b].float()
        next_mse.append(((z[1:] - z[:-1]) ** 2).mean().item())
        sf_mse.append(((z[0] - z[-1]) ** 2).mean().item())
    if next_mse:
        nm = sum(next_mse) / len(next_mse)
        sf = sum(sf_mse) / len(sf_mse)
        print(
            f"per-step Δ mse {nm:.4f} | start→final mse {sf:.4f} | signal ratio {sf / max(nm, 1e-9):.1f}"
        )
        if nm < 1e-4:
            warns.append("per-step latent change ~0 (static or duplicate frames)")

    a_all = actions.float()
    a_std = a_all.std(dim=0)
    a_min = a_all.min(dim=0).values
    a_max = a_all.max(dim=0).values
    print("action per-dim [min, max] std:")
    for d in range(a_all.shape[1]):
        print(f"  dim {d}: [{a_min[d]:+.3f}, {a_max[d]:+.3f}] std {a_std[d]:.4f}")
    dead = [d for d in range(a_all.shape[1]) if a_std[d] < 1e-6]
    if dead:
        warns.append(f"action dims with zero variance: {dead}")

    span = data.window_span(T, stride)
    windows = sum(max(0, (b - a) - span + 1) for a, b in episodes)
    short = sum(1 for a, b in episodes if (b - a) < span)
    print(
        f"training windows (T={T}, stride={stride}, span={span}): {windows} | "
        f"episodes shorter than span: {short}/{len(episodes)}"
    )
    if windows == 0:
        warns.append(f"no training windows: all episodes shorter than span={span}")

    print("--- verdict ---")
    if warns:
        print("UNHEALTHY:")
        for w in warns:
            print(f"  ! {w}")
    else:
        print("HEALTHY: cache passed all checks")
    return not warns


if __name__ == "__main__":
    os.makedirs(data.CACHE_DIR, exist_ok=True)
    print(f"encoding on: {enc_devices} | decode workers: {NUM_WORKERS}")

    datasets = [LeRobotDataset(r, video_backend="pyav") for r in data.DATASET_IDS]
    cams = [data.CAMERA_KEY or ds.meta.camera_keys[0] for ds in datasets]
    dims = [ds[0]["action"].shape[-1] for ds in datasets]
    sdims = [ds[0]["observation.state"].shape[-1] for ds in datasets]
    for r, c, d in zip(data.DATASET_IDS, cams, dims):
        print(
            f"  {r}: camera_key={c} action_dim={d} frames={len(datasets[data.DATASET_IDS.index(r)])}"
        )
    assert len(set(cams)) == 1, f"camera key mismatch across datasets: {cams}"
    assert len(set(dims)) == 1, f"action dim mismatch across datasets: {dims}"
    assert len(set(sdims)) == 1, f"state dim mismatch across datasets: {sdims}"
    cam, action_dim, state_dim = cams[0], dims[0], sdims[0]
    n_frames = sum(len(ds) for ds in datasets)
    print(f"camera_key={cam} action_dim={action_dim} frames={n_frames} datasets={len(datasets)}")

    m = AutoModel.from_pretrained(data.HF_REPO)
    processor = AutoVideoProcessor.from_pretrained(data.HF_REPO)
    base_enc = m.encoder.eval().requires_grad_(False)
    encoders = [copy.deepcopy(base_enc).to(dev) for dev in enc_devices]
    means = [torch.tensor(processor.image_mean, device=dev).view(1, 3, 1, 1) for dev in enc_devices]
    stds = [torch.tensor(processor.image_std, device=dev).view(1, 3, 1, 1) for dev in enc_devices]
    del m, base_enc

    latents = torch.empty(n_frames, 256, 1024, dtype=torch.float16)
    actions = torch.empty(n_frames, action_dim, dtype=torch.float32)
    states = torch.empty(n_frames, state_dim, dtype=torch.float32)
    episodes = []
    off = 0
    ng = len(enc_devices)
    pbar = tqdm(total=n_frames, desc="encoding", unit="frame")

    def collate(samples):
        f = torch.stack([s[cam] for s in samples])
        a = torch.stack([s["action"] for s in samples]).float()
        st = torch.stack([s["observation.state"] for s in samples]).float()
        return f, a, st

    def encode_and_write(frames, acts, sts, offset, gi):
        z = encode_batch(frames, encoders[gi], enc_devices[gi], means[gi], stds[gi])
        b = z.shape[0]
        latents[offset : offset + b] = z
        actions[offset : offset + b] = acts
        states[offset : offset + b] = sts
        pbar.update(b)

    pools = [ThreadPoolExecutor(max_workers=1) for _ in enc_devices]

    for repo, ds in zip(data.DATASET_IDS, datasets):
        base, L = off, len(ds)
        ep = [int(e) for e in ds.hf_dataset["episode_index"]]
        first, last = {}, {}
        for i, e in enumerate(ep):
            first.setdefault(e, i)
            last[e] = i + 1
        for e in first:
            episodes.append([base + first[e], base + last[e]])

        loader = DataLoader(
            ds,
            batch_size=DECODE_BATCH,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate,
            drop_last=False,
            prefetch_factor=2 if NUM_WORKERS else None,
        )
        local, gi, pending = base, 0, deque()
        for frames, acts, sts in loader:
            fut = pools[gi % ng].submit(encode_and_write, frames, acts, sts, local, gi % ng)
            pending.append(fut)
            local += frames.shape[0]
            gi += 1
            if len(pending) >= 2 * ng:
                pending.popleft().result()
        for fut in pending:
            fut.result()
        off += L

    for p in pools:
        p.shutdown()
    pbar.close()

    save_file({"latents": latents, "actions": actions, "state": states}, data.LATENTS_PATH)
    with open(data.CACHE_META, "w") as fjs:
        json.dump(
            {
                "datasets": data.DATASET_IDS,
                "camera_key": cam,
                "action_dim": action_dim,
                "state_dim": state_dim,
                "n_frames": n_frames,
                "episodes": episodes,
                "encoder": data.HF_REPO,
                "img_size": data.IMG_SIZE,
            },
            fjs,
        )
    print(f"saved {n_frames} frame latents ({len(episodes)} episodes) -> {data.LATENTS_PATH}")
    tc = TRAININGS["full"]
    healthy = check_health(latents, actions, episodes, tc.T, tc.stride)
    print(
        "now run: uv run scripts/train.py --model base --training full"
        if healthy
        else "fix the cache before training"
    )
