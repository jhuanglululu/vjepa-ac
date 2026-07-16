import argparse
import json
import os

import torch
from torch import nn
from tqdm.auto import tqdm

from vjepa_ac import data
from vjepa_ac.device import get_device
from vjepa_ac.schedule import make_scheduler

D_MODEL = 768
N_HEADS = 8
PATCH_BLOCKS = 2
HEAD_BLOCKS = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--eval-pairs", type=int, default=4096)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--no-preload", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="records/diagnostics/stride_gate.json")
    return p.parse_args()


class MlpBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.down(nn.functional.silu(self.up(self.norm(x))))


class Probe(nn.Module):
    def __init__(self, d_latent, n_patches, d_out):
        super().__init__()
        self.inp = nn.Linear(2 * d_latent, D_MODEL)
        self.pos = nn.Parameter(torch.randn(n_patches, D_MODEL) * 0.02)
        self.patch_mlp = nn.Sequential(*[MlpBlock(D_MODEL) for _ in range(PATCH_BLOCKS)])
        self.query = nn.Parameter(torch.randn(1, D_MODEL) * 0.02)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.head_mlp = nn.Sequential(*[MlpBlock(D_MODEL) for _ in range(HEAD_BLOCKS)])
        self.out = nn.Linear(D_MODEL, d_out)

    def forward(self, z0, z1):
        x = torch.cat([z0, z1 - z0], dim=-1)
        h = self.patch_mlp(self.inp(x) + self.pos)
        q = self.query.expand(x.shape[0], -1, -1)
        v, _ = self.attn(q, h, h, need_weights=False)
        return self.out(self.head_mlp(v.squeeze(1)))


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


def r2_vec(pred, y):
    var = ((y - y.mean(0)) ** 2).mean(0)
    mse = ((pred - y) ** 2).mean(0)
    r2 = (1 - mse / var.clamp(min=1e-8)).clamp(min=-1.0)
    return torch.where(var > 1e-8, r2, torch.full_like(r2, torch.nan))


def main():
    args = parse_args()
    device = args.device or get_device()

    cache = data.load_cache()
    latents = cache.latents
    N, P, D = latents.get_shape() if hasattr(latents, "get_shape") else latents.shape
    train_eps, val_eps = data.split_episodes(cache.episodes, args.val_frac)

    lat = None
    if not args.no_preload:
        try:
            lat = latents[0:N].to(device)
        except torch.cuda.OutOfMemoryError:
            print("latents do not fit on device, falling back to mmap gathers")
    print(
        f"{len(train_eps)} train / {len(val_eps)} val episodes | "
        f"target: normalized conditioning features (dim {cache.state_dim}) | "
        f"preloaded: {lat is not None} | device {device}"
    )

    def gather_raw(idx):
        if lat is not None:
            return lat[idx.to(lat.device)].float()
        z = torch.stack([latents[int(t) : int(t) + 1] for t in idx])
        return z.squeeze(1).to(device).float()

    stat_idx = subsample(pair_starts(train_eps, 1), 4096, 0)
    zs = gather_raw(stat_idx)
    z_mu = zs.mean(dim=(0, 1))
    z_sd = zs.std(dim=(0, 1)).clamp(min=1e-6)
    del zs

    def gather_z(idx):
        return (gather_raw(idx) - z_mu) / z_sd

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

        eval_sets = {
            "train": subsample(train_idx, args.eval_pairs, 1),
            "held": subsample(val_idx, args.eval_pairs, 2),
        }

        @torch.no_grad()
        def evaluate(probe):
            probe.eval()
            out = {}
            for name, idx in eval_sets.items():
                preds, ys = [], []
                for i in range(0, len(idx), 256):
                    bi = idx[i : i + 256]
                    preds.append(probe(gather_z(bi), gather_z(bi + s)).cpu())
                    ys.append(targets(bi).cpu())
                r2 = r2_vec(torch.cat(preds), torch.cat(ys))
                out[name] = {
                    "all": r2.nanmean().item(),
                    "motion": r2[:-1].nanmean().item(),
                    "per_dim": r2.tolist(),
                }
            probe.train()
            return out

        torch.manual_seed(0)
        probe = Probe(D, P, cache.state_dim).to(device)
        optim = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = make_scheduler(optim, args.warmup, args.steps)
        n_params = sum(p_.numel() for p_ in probe.parameters())

        best = None
        for step in tqdm(range(1, args.steps + 1), desc=f"stride {s}", unit="step", leave=False):
            bi = train_idx[torch.randint(0, len(train_idx), (args.batch_size,))]
            loss = ((probe(gather_z(bi), gather_z(bi + s)) - targets(bi)) ** 2).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            if step % args.eval_every == 0 or step == args.steps:
                scores = evaluate(probe)
                if best is None or scores["held"]["motion"] > best["held"]["motion"]:
                    best = {**scores, "step": step}
        rows.append(
            {
                "stride": s,
                "n_params": n_params,
                "n_train_pairs": len(train_idx),
                "n_held_pairs": len(val_idx),
                **best,
            }
        )
        print(
            f"stride {s:>2} | train {best['train']['motion']:+.3f} | "
            f"held motion {best['held']['motion']:+.3f} | held all {best['held']['all']:+.3f} | "
            f"best@{best['step']}"
        )

    print(
        f"\n=== conditioning extractability by stride "
        f"({rows[0]['n_params'] / 1e6:.1f}M probe, {len(val_eps)} held-out episodes) ==="
        if rows
        else "\nno strides evaluated"
    )
    print(f"{'stride':>6} | {'train R2':>8} | {'held motion':>11} | {'held all':>8}")
    for r in rows:
        print(
            f"{r['stride']:>6} | {r['train']['motion']:>+8.3f} | "
            f"{r['held']['motion']:>+11.3f} | {r['held']['all']:>+8.3f}"
        )

    passing = [r["stride"] for r in rows if r["held"]["motion"] >= args.threshold]
    if passing:
        print(
            f"\nverdict: train at stride {passing[0]} "
            f"(smallest with held motion R2 >= {args.threshold}; passing: {passing})\n"
            f"  uv run scripts/train.py --model base --training full --stride {passing[0]}"
        )
    else:
        print(
            f"\nverdict: no stride reaches held motion R2 >= {args.threshold} -- "
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
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "passing_strides": passing,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
