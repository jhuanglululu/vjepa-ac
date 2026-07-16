import argparse
import json
import os

import torch
from torch import nn
from tqdm.auto import tqdm

from vjepa_ac import data
from vjepa_ac.device import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--eval-pairs", type=int, default=4096)
    p.add_argument("--d-proj", type=int, default=32)
    p.add_argument("--d-hidden", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="records/diagnostics/stride_gate.json")
    return p.parse_args()


class Probe(nn.Module):
    def __init__(self, d_latent, n_patches, d_proj, d_hidden, d_out):
        super().__init__()
        self.proj = nn.Linear(d_latent, d_proj, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_patches * d_proj, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z0, z1):
        h = torch.cat([self.proj(z0).flatten(1), self.proj(z1 - z0).flatten(1)], dim=1)
        return self.mlp(h)


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


def r2_per_dim(pred, y):
    var = ((y - y.mean(0)) ** 2).mean(0)
    mse = ((pred - y) ** 2).mean(0)
    valid = var > 1e-8
    r2 = (1 - mse[valid] / var[valid]).clamp(min=-1.0)
    return r2, valid


def main():
    args = parse_args()
    device = args.device or get_device()

    cache = data.load_cache()
    latents = cache.latents
    N, P, D = latents.get_shape() if hasattr(latents, "get_shape") else latents.shape
    train_eps, val_eps = data.split_episodes(cache.episodes, args.val_frac)
    print(
        f"{len(train_eps)} train / {len(val_eps)} val episodes | "
        f"target: normalized conditioning features (dim {cache.state_dim}) | device {device}"
    )

    def gather_z(idx):
        z = torch.stack([latents[int(t) : int(t) + 1] for t in idx])
        return z.squeeze(1).to(device).float()

    rows = []
    for s in sorted(args.strides):
        train_idx = pair_starts(train_eps, s)
        val_idx = pair_starts(val_eps, s)
        if len(train_idx) < args.batch_size or len(val_idx) < 8:
            print(f"stride {s}: not enough pairs, skipping")
            continue
        cond = data.fit_conditioner(cache.states, train_eps, s)

        def targets(idx):
            return ((cond.features(idx, s) - cond.mean) / cond.std).to(device)

        torch.manual_seed(0)
        probe = Probe(D, P, args.d_proj, args.d_hidden, cache.state_dim).to(device)
        optim = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        for _ in tqdm(range(args.steps), desc=f"stride {s}", unit="step", leave=False):
            bi = train_idx[torch.randint(0, len(train_idx), (args.batch_size,))]
            loss = ((probe(gather_z(bi), gather_z(bi + s)) - targets(bi)) ** 2).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        probe.eval()
        scores = {}
        details = {}
        with torch.no_grad():
            for name, idx in [
                ("train", subsample(train_idx, args.eval_pairs, 1)),
                ("held", subsample(val_idx, args.eval_pairs, 2)),
            ]:
                preds, ys = [], []
                for i in range(0, len(idx), 256):
                    bi = idx[i : i + 256]
                    preds.append(probe(gather_z(bi), gather_z(bi + s)).cpu())
                    ys.append(targets(bi).cpu())
                r2, valid = r2_per_dim(torch.cat(preds), torch.cat(ys))
                scores[name] = r2.mean().item() if valid.any() else float("nan")
                details[name] = r2.tolist()
        rows.append(
            {
                "stride": s,
                "r2": scores,
                "r2_per_dim": details,
                "n_train_pairs": len(train_idx),
                "n_held_pairs": len(val_idx),
            }
        )
        print(f"stride {s:>2} | R2 train {scores['train']:+.3f} | held {scores['held']:+.3f}")

    print(f"\n=== conditioning extractability by stride ({len(val_eps)} held-out episodes) ===")
    print(f"{'stride':>6} | {'train R2':>8} | {'held R2':>8}")
    for r in rows:
        print(f"{r['stride']:>6} | {r['r2']['train']:>+8.3f} | {r['r2']['held']:>+8.3f}")

    passing = [r["stride"] for r in rows if r["r2"]["held"] >= args.threshold]
    if passing:
        print(
            f"\nverdict: train at stride {passing[0]} "
            f"(smallest with held R2 >= {args.threshold}; passing: {passing})\n"
            f"  uv run scripts/train.py --model base --training full --stride {passing[0]}"
        )
    else:
        print(
            f"\nverdict: no stride reaches held R2 >= {args.threshold} -- "
            "the encoder does not expose the conditioning signal at these strides; "
            "training is unlikely to become action-sensitive"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "threshold": args.threshold,
                "val_frac": args.val_frac,
                "steps": args.steps,
                "passing_strides": passing,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
