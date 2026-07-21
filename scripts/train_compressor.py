import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm.auto import tqdm

from vjepa_ac import data
from vjepa_ac.checkpoints import checkpoint_dir
from vjepa_ac.compressor import Compressor, IDHead, ReconHead
from vjepa_ac.device import get_device
from vjepa_ac.records import RecordWriter
from vjepa_ac.schedule import make_scheduler
from vjepa_ac.variations import MODELS


MODEL = "base-c16"
STEPS = 3000
WARMUP = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
RECON_WEIGHT = 0.1
EVAL_EVERY = 500
EVAL_PAIRS = 4096
VAL_FRAC = 0.1
RIDGE = 1e-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
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


def motion_r2(pred, y):
    var = ((y - y.mean(0)) ** 2).mean(0)
    mse = ((pred - y) ** 2).mean(0)
    r2 = torch.where(var > 1e-8, 1 - mse / var.clamp(min=1e-8), torch.nan)
    return r2[:-1].nanmean().item(), r2


def main():
    args = parse_args()
    mc = MODELS[MODEL]
    device = get_device()
    s = args.stride
    torch.manual_seed(args.seed)

    cache = data.load_cache()
    lat = cache.latents
    shape = lat.get_shape() if hasattr(lat, "get_shape") else lat.shape
    assert (shape[1], shape[2]) == (mc.comp_patches, mc.comp_d_latent)
    train_eps, val_eps = data.split_episodes(cache.episodes, VAL_FRAC)
    cond = data.fit_conditioner(cache.states, train_eps, s)
    train_idx = pair_starts(train_eps, s)
    val_idx = subsample(pair_starts(val_eps, s), EVAL_PAIRS, 1)
    print(
        f"{MODEL} phase 1 | stride {s} | {len(train_idx)} train / {len(val_idx)} val pairs | "
        f"tokens {mc.n_patches}x{mc.d_state} | recon weight {RECON_WEIGHT} | {device}"
    )

    def gather_z(idx):
        z = torch.stack([lat[int(t) : int(t) + 1] for t in idx])
        return z.squeeze(1).to(device).float()

    def targets(idx):
        return ((cond.features(idx, s) - cond.mean) / cond.std).to(device)

    comp = Compressor(mc.comp_d_latent, mc.n_patches, mc.d_state, mc.comp_heads).to(device)
    idh = IDHead(mc.d_state, cache.state_dim).to(device)
    recon = ReconHead(mc.comp_d_latent, mc.comp_patches, mc.d_state, mc.comp_heads).to(device)
    params = [p for m in (comp, idh, recon) for p in m.parameters()]
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = make_scheduler(optim, WARMUP, STEPS)
    n_params = sum(p.numel() for p in params)
    print(f"parameters: {n_params:,} (incl. throwaway recon head)")

    training_name = f"comp-s{s}"
    record = RecordWriter(MODEL, training_name, args.seed)
    record.meta(MODEL, training_name, args.seed, {"phase1": vars(args)})

    @torch.no_grad()
    def evaluate():
        comp.eval()
        idh.eval()
        preds, stds = [], []
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i : i + 256]
            c0 = comp(gather_z(bi))
            c1 = comp(gather_z(bi + s))
            preds.append(idh(c0, c1).cpu())
            stds.append(c0.std(dim=(0, 1)).mean().item())
        comp.train()
        idh.train()
        r2, per_dim = motion_r2(torch.cat(preds), targets(val_idx).cpu())
        return r2, per_dim, sum(stds) / len(stds)

    best: Any = None
    for step in tqdm(range(1, STEPS + 1), desc="phase 1", unit="step"):
        bi = train_idx[torch.randint(0, len(train_idx), (BATCH_SIZE,))]
        z0 = gather_z(bi)
        c0 = comp(z0)
        c1 = comp(gather_z(bi + s))
        id_loss = F.mse_loss(idh(c0, c1), targets(bi))
        recon_loss = F.smooth_l1_loss(recon(c0), z0)
        loss = id_loss + RECON_WEIGHT * recon_loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        sched.step()
        if step % 100 == 0:
            record.step(step, loss.item(), float(sched.get_last_lr()[0]), 0.0, 0.0)
        if step % EVAL_EVERY == 0 or step == STEPS:
            r2, per_dim, tok_std = evaluate()
            record.eval(step, -r2, loss.item())
            tqdm.write(
                f"step {step:>5} | id {id_loss.item():.4f} | recon {recon_loss.item():.4f} | "
                f"val motion R2 {r2:+.3f} | token std {tok_std:.3f}"
            )
            if best is None or r2 > best["r2"]:
                best = {
                    "r2": r2,
                    "per_dim": [round(v, 4) for v in per_dim.tolist()],
                    "step": step,
                    "comp": {k: v.detach().cpu().clone() for k, v in comp.state_dict().items()},
                    "idh": {k: v.detach().cpu().clone() for k, v in idh.state_dict().items()},
                }
    record.close()

    comp.load_state_dict(best["comp"])
    idh.load_state_dict(best["idh"])
    comp.eval()

    out_dir = checkpoint_dir(MODEL, training_name, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compressor.safetensors"
    tensors = {f"compressor.{k}": v for k, v in best["comp"].items()}
    tensors |= {f"id_head.{k}": v for k, v in best["idh"].items()}
    save_file(tensors, str(out_path))
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(
            {
                "model": MODEL,
                "stride": s,
                "seed": args.seed,
                "phase1": vars(args),
                "val_motion_r2": best["r2"],
                "per_dim_r2": best["per_dim"],
                "best_step": best["step"],
                "conditioning": cond.stats(),
            },
            f,
            indent=2,
        )
    print(f"saved best (step {best['step']}) -> {out_path}")

    @torch.no_grad()
    def tokens_flat(idx):
        return comp(gather_z(idx)).reshape(len(idx), -1)

    stat = subsample(train_idx, 2048, 2)
    ct = tokens_flat(stat)
    t_sd = ct.std(0).clamp(min=1e-6)

    k = cache.state_dim + 1
    tr = subsample(train_idx, 20000, 3)
    xtx = torch.zeros(k, k, device=device, dtype=torch.float64)
    xty = torch.zeros(k, ct.shape[1], device=device)
    with torch.no_grad():
        for i in range(0, len(tr), 256):
            bi = tr[i : i + 256]
            x = torch.cat([targets(bi), torch.ones(len(bi), 1, device=device)], dim=1)
            d = (tokens_flat(bi + s) - tokens_flat(bi)) / t_sd
            xtx += (x.T @ x).double()
            xty += x.T @ d
        reg = RIDGE * len(tr) * torch.eye(k, device=device, dtype=torch.float64)
        w = torch.linalg.solve(xtx + reg, xty.double()).float()
        w_drift = torch.zeros_like(w)
        w_drift[-1] = w[-1]
        losses = {"copy": 0.0, "drift": 0.0, "linear": 0.0}
        for i in range(0, len(val_idx), 256):
            bi = val_idx[i : i + 256]
            x = torch.cat([targets(bi), torch.ones(len(bi), 1, device=device)], dim=1)
            d = (tokens_flat(bi + s) - tokens_flat(bi)) / t_sd
            n = len(bi)
            losses["copy"] += F.smooth_l1_loss(torch.zeros_like(d), d).item() * n
            losses["drift"] += F.smooth_l1_loss(x @ w_drift, d).item() * n
            losses["linear"] += F.smooth_l1_loss(x @ w, d).item() * n
    copy, drift, lin = (losses[key] / len(val_idx) for key in ("copy", "drift", "linear"))
    ceiling = (drift - lin) / drift * 100

    print("\n=== gates ===")
    print(
        f"gate 1 (held-out ID motion R2 >= 0.2): {best['r2']:+.3f} "
        f"{'PASS' if best['r2'] >= 0.2 else 'FAIL'}"
    )
    print(f"  per-dim R2: {best['per_dim']}")
    print(
        f"gate 2 (C-space linear action ceiling >= +2%): {ceiling:+.2f}% "
        f"{'PASS' if ceiling >= 2 else 'FAIL'} "
        f"(copy {copy:.4f} | drift {drift:.4f} | linear {lin:.4f})"
    )
    if best["r2"] >= 0.2 and ceiling >= 2:
        print(
            f"\nboth gates pass -> uv run scripts/train.py --model {MODEL} "
            f"--training c-full --seed {args.seed} --no-rollout"
        )
    else:
        print(
            "\ngate failed -- iterate on phase 1 (RECON_WEIGHT, STEPS, LR at the top of "
            "this script) before "
            "spending phase-2 GPU time"
        )


if __name__ == "__main__":
    main()
