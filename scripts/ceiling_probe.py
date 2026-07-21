import argparse

import torch
import torch.nn.functional as F

from vjepa_ac import data
from vjepa_ac.device import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--train-pairs", type=int, default=20000)
    p.add_argument("--val-pairs", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def pair_starts(episodes, stride):
    idx = []
    for a, b in episodes:
        idx.extend(range(a, b - stride))
    return torch.tensor(idx, dtype=torch.long)


def subsample(idx, n, seed):
    if len(idx) <= n:
        return idx
    g = torch.Generator().manual_seed(seed)
    return idx[torch.randperm(len(idx), generator=g)[:n]]


def main():
    args = parse_args()
    device = args.device or get_device()
    cache = data.load_cache(args.cache_dir)
    train_eps, val_eps = data.split_episodes(cache.episodes, args.val_frac)
    cond = data.fit_conditioner(cache.states, train_eps, args.stride)
    s = args.stride

    train_idx = subsample(pair_starts(train_eps, s), args.train_pairs, 0)
    val_idx = subsample(pair_starts(val_eps, s), args.val_pairs, 1)
    lat = cache.latents
    shape = lat.get_shape() if hasattr(lat, "get_shape") else lat.shape
    D_all = shape[1] * shape[2]
    k = cache.state_dim + 1
    print(
        f"camera {cache.meta.get('camera', '?')} | stride {s} | "
        f"{len(train_idx)} train / {len(val_idx)} val pairs | latent dim {D_all} | {device}"
    )

    def features(idx):
        f = ((cond.features(idx, s) - cond.mean) / cond.std).to(device)
        return torch.cat([f, torch.ones(len(f), 1, device=device)], dim=1)

    def deltas(idx):
        z0 = torch.stack([cache.latents[int(t) : int(t) + 1] for t in idx]).squeeze(1)
        z1 = torch.stack([cache.latents[int(t) + s : int(t) + s + 1] for t in idx]).squeeze(1)
        return (z1 - z0).reshape(len(idx), -1).to(device).float()

    xtx = torch.zeros(k, k, device=device, dtype=torch.float64)
    xty = torch.zeros(k, D_all, device=device)
    for i in range(0, len(train_idx), args.batch_size):
        bi = train_idx[i : i + args.batch_size]
        x = features(bi)
        xtx += (x.T @ x).double()
        xty += x.T @ deltas(bi)
    reg = args.ridge * len(train_idx) * torch.eye(k, device=device, dtype=torch.float64)
    w = torch.linalg.solve(xtx + reg, xty.double()).float()
    w_drift = torch.zeros_like(w)
    w_drift[-1] = w[-1]

    losses = {"copy": 0.0, "drift": 0.0, "linear": 0.0}
    var_tot, var_res = 0.0, 0.0
    for i in range(0, len(val_idx), args.batch_size):
        bi = val_idx[i : i + args.batch_size]
        x = features(bi)
        d = deltas(bi)
        zero = torch.zeros_like(d)
        n = len(bi)
        losses["copy"] += F.smooth_l1_loss(zero, d).item() * n
        losses["drift"] += F.smooth_l1_loss(x @ w_drift, d).item() * n
        losses["linear"] += F.smooth_l1_loss(x @ w, d).item() * n
        var_tot += (d**2).sum().item()
        var_res += ((d - x @ w) ** 2).sum().item()
    for key in losses:
        losses[key] /= len(val_idx)

    copy, drift, lin = losses["copy"], losses["drift"], losses["linear"]
    print(f"\n=== held-out latent-delta loss (smooth L1), stride {s} ===")
    print(f"copy (predict no change):        {copy:.4f}")
    print(f"+ mean drift (intercept only):   {drift:.4f}  ({(drift - copy) / copy * 100:+.2f}%)")
    print(f"+ linear from conditioning:      {lin:.4f}  ({(lin - copy) / copy * 100:+.2f}%)")
    print(f"\naction-attributable ceiling (linear floor): {(drift - lin) / drift * 100:+.2f}%")
    print(f"latent-delta variance explained: {(1 - var_res / var_tot) * 100:.2f}%")
    print(
        "\nhow to read: 'action-attributable' is the loss reduction actions buy beyond\n"
        "unconditional drift for a linear map -- a nonlinear predictor can beat it, but\n"
        "if this is ~1-2% the +10% shuffled-actions criterion is above the ceiling and\n"
        "needs revising (or the objective needs reweighting toward motion dims)"
    )


if __name__ == "__main__":
    main()
