import argparse
import json
import math
import os
import statistics
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from vjepa_ac import data
from vjepa_ac.device import get_device
from vjepa_ac.schedule import make_scheduler
from vjepa_ac.variations import TRAININGS

D_MODEL = 768
N_HEADS = 8
PATCH_BLOCKS = 2
HEAD_BLOCKS = 2
MIN_TRAIN_R2 = 0.5


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
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--no-preload", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", default=None)
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
    r2 = 1 - mse / var.clamp(min=1e-8)
    return torch.where(var > 1e-8, r2, torch.full_like(r2, torch.nan))


def motion_r2(pred, y):
    return r2_vec(pred, y)[:-1].nanmean().item()


def bootstrap_stds(pair_preds, base_preds, ys_groups, n_boot, seed):
    n = len(ys_groups)
    g = torch.Generator().manual_seed(seed)
    pair_samples, margin_samples = [], []
    for _ in range(n_boot):
        ids = torch.randint(0, n, (n,), generator=g).tolist()
        yy = torch.cat([ys_groups[i] for i in ids])
        pm = motion_r2(torch.cat([pair_preds[i] for i in ids]), yy)
        bm = motion_r2(torch.cat([base_preds[i] for i in ids]), yy)
        pair_samples.append(pm)
        margin_samples.append(pm - bm)
    return statistics.stdev(pair_samples), statistics.stdev(margin_samples)


def main():
    args = parse_args()
    device = args.device or get_device()

    cache = data.load_cache(args.cache_dir)
    camera = cache.meta.get("camera", "cache")
    out_path = args.out or f"records/diagnostics/stride_gate_{camera}.json"
    latents = cache.latents
    N, P, D = latents.get_shape() if hasattr(latents, "get_shape") else latents.shape
    train_eps, val_eps = data.split_episodes(cache.episodes, args.val_frac)

    lat = None
    if not args.no_preload:
        try:
            lat = latents[0:N].to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print("latents do not fit on device, falling back to mmap gathers")
    print(
        f"camera: {camera} | {len(train_eps)} train / {len(val_eps)} val episodes | "
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

    disjoint = len(val_eps) >= 2
    if disjoint:
        sel_eps, test_eps = val_eps[0::2], val_eps[1::2]
        print(
            f"{len(sel_eps)} selection / {len(test_eps)} test episodes (disjoint) | "
            f"{args.seeds} seeds x (pair + z0-only) probes per stride"
        )
    else:
        sel_eps = test_eps = val_eps
        print(
            "WARNING: only one held-out episode -- selection and test sets coincide, "
            "scores are optimistic"
        )

    T_full = TRAININGS["full"].T
    rows: list[dict[str, Any]] = []
    for s in sorted(args.strides):
        train_idx = pair_starts(train_eps, s)
        sel_idx = pair_starts(sel_eps, s)
        test_groups = [pair_starts([ep], s) for ep in test_eps]
        test_groups = [g for g in test_groups if len(g) > 0]
        if len(train_idx) < args.batch_size or len(sel_idx) < 8 or len(test_groups) < 2:
            print(f"stride {s}: not enough pairs/episodes, skipping")
            continue
        cond = data.fit_conditioner(cache.states, train_eps, s)

        def targets(idx):
            return ((cond.features(idx, s) - cond.mean) / cond.std).to(device)

        train_sub = subsample(train_idx, args.eval_pairs, 1)
        sel_sub = subsample(sel_idx, args.eval_pairs, 2)
        ys_train = targets(train_sub).cpu()
        ys_sel = targets(sel_sub).cpu()
        ys_groups = [targets(g).cpu() for g in test_groups]
        y_all = torch.cat(ys_groups)

        @torch.no_grad()
        def batch_predict(probe, idx, ablate):
            preds = []
            for i in range(0, len(idx), 256):
                bi = idx[i : i + 256]
                z0 = gather_z(bi)
                z1 = z0 if ablate else gather_z(bi + s)
                preds.append(probe(z0, z1).cpu())
            return torch.cat(preds)

        def train_probe(seed, ablate):
            torch.manual_seed(2 * seed + int(ablate))
            probe = Probe(D, P, cache.state_dim).to(device)
            optim = torch.optim.AdamW(
                probe.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            sched = make_scheduler(optim, args.warmup, args.steps)
            label = f"stride {s} seed {seed} {'z0-only' if ablate else 'pair'}"
            best: Any = None
            for step in tqdm(range(1, args.steps + 1), desc=label, unit="step", leave=False):
                bi = train_idx[torch.randint(0, len(train_idx), (args.batch_size,))]
                z0 = gather_z(bi)
                z1 = z0 if ablate else gather_z(bi + s)
                loss = ((probe(z0, z1) - targets(bi)) ** 2).mean()
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
                if step % args.eval_every == 0 or step == args.steps:
                    probe.eval()
                    sel_m = motion_r2(batch_predict(probe, sel_sub, ablate), ys_sel)
                    if best is None or sel_m > best["sel"]:
                        best = {
                            "sel": sel_m,
                            "step": step,
                            "train": motion_r2(batch_predict(probe, train_sub, ablate), ys_train),
                            "preds": [batch_predict(probe, g, ablate) for g in test_groups],
                        }
                    probe.train()
            return best

        seed_rows: list[dict[str, Any]] = []
        for seed in range(args.seeds):
            pair = train_probe(seed, ablate=False)
            base = train_probe(seed, ablate=True)
            pair_m = motion_r2(torch.cat(pair["preds"]), y_all)
            base_m = motion_r2(torch.cat(base["preds"]), y_all)
            pair_bstd, margin_bstd = bootstrap_stds(
                pair["preds"], base["preds"], ys_groups, args.bootstrap, seed
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "pair_test_motion": pair_m,
                    "base_test_motion": base_m,
                    "margin": pair_m - base_m,
                    "pair_boot_std": pair_bstd,
                    "margin_boot_std": margin_bstd,
                    "pair_train_motion": pair["train"],
                    "base_train_motion": base["train"],
                    "pair_per_dim": r2_vec(torch.cat(pair["preds"]), y_all).tolist(),
                    "pair_best_step": pair["step"],
                    "base_best_step": base["step"],
                }
            )

        def agg(key):
            vals = [r[key] for r in seed_rows]
            se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            return statistics.mean(vals), se

        pair_point, pair_seed_se = agg("pair_test_motion")
        base_point, _ = agg("base_test_motion")
        margin_point, margin_seed_se = agg("margin")
        train_point, _ = agg("pair_train_motion")
        pair_ep_se = statistics.mean(r["pair_boot_std"] for r in seed_rows)
        margin_ep_se = statistics.mean(r["margin_boot_std"] for r in seed_rows)
        pair_se = math.sqrt(pair_ep_se**2 + pair_seed_se**2)
        margin_se = math.sqrt(margin_ep_se**2 + margin_seed_se**2)

        full_windows = len(data.window_starts(cache.episodes, T_full, s))
        passed = pair_point - pair_se >= args.threshold and margin_point - margin_se >= args.margin
        probe_limited = train_point < MIN_TRAIN_R2
        rows.append(
            {
                "stride": s,
                "pair_test_motion": pair_point,
                "pair_se": pair_se,
                "base_test_motion": base_point,
                "margin": margin_point,
                "margin_se": margin_se,
                "pair_train_motion": train_point,
                "passed": passed,
                "probe_limited": probe_limited,
                "full_T_windows": full_windows,
                "n_test_episodes": len(test_groups),
                "n_test_pairs": int(sum(len(g) for g in test_groups)),
                "seeds": seed_rows,
            }
        )
        print(
            f"stride {s:>2} | pair {pair_point:+.3f} ±{pair_se:.3f} | z0-only {base_point:+.3f} | "
            f"margin {margin_point:+.3f} ±{margin_se:.3f} | train {train_point:+.3f} | "
            f"windows(T={T_full}) {full_windows} | {'PASS' if passed else 'fail'}"
            + (" (probe underfit)" if probe_limited else "")
        )

    print(
        f"\n=== motion extractable from latent pairs, beyond z0 alone "
        f"({args.seeds} seeds, bootstrap over {len(test_eps)} test episodes) ==="
    )
    print(
        f"{'stride':>6} | {'pair test R2':>14} | {'z0-only':>8} | {'margin':>14} | "
        f"{'train R2':>8} | {'windows':>8} | verdict"
    )
    for r in rows:
        print(
            f"{r['stride']:>6} | {r['pair_test_motion']:>+7.3f} ±{r['pair_se']:.3f} | "
            f"{r['base_test_motion']:>+8.3f} | {r['margin']:>+7.3f} ±{r['margin_se']:.3f} | "
            f"{r['pair_train_motion']:>+8.3f} | {r['full_T_windows']:>8} | "
            + ("PASS" if r["passed"] else "fail")
            + (" (probe underfit)" if r["probe_limited"] else "")
        )

    eligible = [r for r in rows if r["passed"] and r["full_T_windows"] > 0]
    if eligible:
        pick = eligible[0]
        print(
            f"\nverdict: train at stride {pick['stride']} -- smallest where both "
            f"pair R2 - SE >= {args.threshold} and margin - SE >= {args.margin} "
            f"(passing: {[r['stride'] for r in eligible]})\n"
            f"  uv run scripts/train.py --model base --training full --stride {pick['stride']}"
        )
        if pick["full_T_windows"] < 5000:
            print(
                f"  caution: only {pick['full_T_windows']} training windows exist at "
                f"T={T_full} stride {pick['stride']} -- expect heavy sample reuse"
            )
    else:
        print(
            f"\nverdict: no usable stride (pair R2 - SE >= {args.threshold} and margin - SE >= {args.margin})"
        )
        blocked = [r["stride"] for r in rows if r["passed"] and r["full_T_windows"] == 0]
        failed = [r for r in rows if not r["passed"]]
        limited = [r["stride"] for r in failed if r["probe_limited"]]
        conclusive = [r["stride"] for r in failed if not r["probe_limited"]]
        if blocked:
            print(
                f"strides {blocked} PASSED the signal gate but have zero training windows at "
                f"T={T_full} -- the episodes are too short for that span; train with a "
                "smaller T or a smaller stride"
            )
        if conclusive:
            print(
                f"strides {conclusive} failed with the probe fitting its training set, so "
                "those latent pairs genuinely lack transferable motion signal beyond what z0 "
                "already carries -- training there is unlikely to become action-sensitive"
            )
        if limited:
            print(
                f"strides {limited} are INCONCLUSIVE: the probe never fit the training set "
                f"(train R2 < {MIN_TRAIN_R2}), so low test scores there reflect probe "
                "capacity/optimization, not the latents -- rerun with more steps or a "
                "different lr before concluding anything"
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "camera": camera,
                "threshold": args.threshold,
                "margin": args.margin,
                "val_frac": args.val_frac,
                "steps": args.steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seeds": args.seeds,
                "bootstrap": args.bootstrap,
                "sel_test_disjoint": disjoint,
                "n_sel_episodes": len(sel_eps),
                "n_test_episodes": len(test_eps),
                "passing_strides": [r["stride"] for r in eligible],
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
