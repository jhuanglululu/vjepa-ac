import argparse
import bisect
import json
import os
import statistics
from pathlib import Path

import torch
from tqdm.auto import tqdm

from vjepa_ac import data
from vjepa_ac.checkpoints import load_model_weights
from vjepa_ac.cpredictor import CPredictor
from vjepa_ac.device import get_device
from vjepa_ac.predictor import Predictor
from vjepa_ac.variations import ModelConfig, TrainingConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="weights/model.safetensors")
    p.add_argument("--windows", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 15])
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or get_device()

    sidecar_path = Path(args.checkpoint).with_suffix(".json")
    assert sidecar_path.exists(), (
        f"no sidecar at {sidecar_path} -- evaluate needs the config and "
        "conditioning stats saved next to the checkpoint"
    )
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    assert "conditioning" in sidecar, f"{sidecar_path} has no conditioning stats"
    mc = ModelConfig(**sidecar["config"]["model"])
    tc = TrainingConfig(**sidecar["config"]["training"])
    T, stride = tc.T, tc.stride
    assert max(args.horizons) <= T - 1
    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint) or ".", "eval_results.json"
    )

    cache = data.load_cache()
    assert cache.state_dim == mc.d_action
    cond = data.load_conditioner(cache.states, sidecar["conditioning"])

    episodes = sorted(cache.episodes)
    ep_starts = [a for a, b in episodes]
    _, val_eps = data.split_episodes(cache.episodes, tc.val_frac)
    val_starts = data.window_starts(val_eps, T, stride)

    gen = torch.Generator().manual_seed(1)
    sel = torch.randperm(len(val_starts), generator=gen)[: args.windows]
    eval_starts = val_starts[sel]
    print(
        f"evaluating {len(eval_starts)} windows from {len(val_eps)} held-out episodes | "
        f"T {T} | stride {stride}"
    )

    model_cls = CPredictor if mc.compressor else Predictor
    model = model_cls(mc, T).to(device).eval()
    model.load_state_dict(load_model_weights(args.checkpoint))
    print(f"loaded {args.checkpoint}" + (" (token space)" if mc.compressor else ""))

    def forward(states, acts):
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(states, acts).float()
        return model(states, acts).float()

    @torch.no_grad()
    def to_space(z):
        if not isinstance(model, CPredictor):
            return z
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.encode(z).float()
        return model.encode(z).float()

    @torch.no_grad()
    def rollout(z, a):
        s = z.clone()
        for t in range(T - 1):
            pred = forward(s, a)
            s[:, t + 1] = s[:, t] + pred[:, t]
        return s

    def episode_of(s):
        i = bisect.bisect_right(ep_starts, s) - 1
        return episodes[i]

    @torch.no_grad()
    def retrieve_frame(zh, a0, b0):
        q = zh.reshape(1, -1)
        best_d, best_i = float("inf"), a0
        for i in range(a0, b0, 256):
            j = min(i + 256, b0)
            chunk = to_space(cache.latents[i:j].to(device).float()).reshape(j - i, -1)
            d = torch.cdist(q, chunk)[0]
            m = int(d.argmin())
            if float(d[m]) < best_d:
                best_d, best_i = float(d[m]), i + m
        return best_i

    variants = ["model", "zero_actions", "shuffled_actions"]
    l1_sum = {v: {k: 0.0 for k in args.horizons} for v in variants + ["copy_first"]}
    n_windows = 0
    retr = {k: {"hits": 0, "offsets": []} for k in args.horizons}
    copy_retr = {k: {"hits": 0, "offsets": []} for k in args.horizons}

    for i in tqdm(range(0, len(eval_starts), args.batch_size), desc="eval", unit="batch"):
        sb = eval_starts[i : i + args.batch_size]
        z, a = data.gather(cache, cond, sb, T, stride, device)
        z = to_space(z)
        B = len(sb)
        preds = {
            "model": rollout(z, a),
            "zero_actions": rollout(z, torch.zeros_like(a)),
            "shuffled_actions": rollout(z, torch.roll(a, 1, dims=0)),
        }
        for k in args.horizons:
            for v in variants:
                l1_sum[v][k] += (preds[v][:, k] - z[:, k]).abs().mean(dim=(1, 2)).sum().item()
            l1_sum["copy_first"][k] += (z[:, 0] - z[:, k]).abs().mean(dim=(1, 2)).sum().item()
        n_windows += B

        for bi in range(B):
            s0 = int(sb[bi])
            a0, b0 = episode_of(s0)
            for k in args.horizons:
                target = s0 + k * stride
                ri = retrieve_frame(preds["model"][bi, k], a0, b0)
                retr[k]["hits"] += int(ri == target)
                retr[k]["offsets"].append(ri - target)
                ci = retrieve_frame(z[bi, 0], a0, b0)
                copy_retr[k]["hits"] += int(ci == target)
                copy_retr[k]["offsets"].append(ci - target)

    l1 = {v: {k: l1_sum[v][k] / n_windows for k in args.horizons} for v in l1_sum}

    print(f"\n=== rollout latent L1 vs ground truth ({n_windows} windows, stride {stride}) ===")
    print(
        f"{'h':>3} | {'copy-first':>10} | {'model':>10} | {'zero-act':>10} | "
        f"{'shuf-act':>10} | {'model/copy':>10}"
    )
    for k in args.horizons:
        print(
            f"{k:>3} | {l1['copy_first'][k]:>10.4f} | {l1['model'][k]:>10.4f} | "
            f"{l1['zero_actions'][k]:>10.4f} | {l1['shuffled_actions'][k]:>10.4f} | "
            f"{l1['model'][k] / max(l1['copy_first'][k], 1e-9):>10.3f}"
        )

    kmax = max(args.horizons)
    sens = (l1["shuffled_actions"][kmax] - l1["model"][kmax]) / max(l1["model"][kmax], 1e-9)
    print(f"\naction sensitivity @h={kmax}: shuffled is {sens * 100:+.1f}% worse than true actions")

    print(f"\n=== frame retrieval within episode ({n_windows} windows) ===")
    print(
        f"{'h':>3} | {'top1 acc':>8} | {'med offset':>10} | {'copy top1':>9} | {'copy offset':>11}"
    )
    for k in args.horizons:
        mo = statistics.median(retr[k]["offsets"])
        co = statistics.median(copy_retr[k]["offsets"])
        print(
            f"{k:>3} | {retr[k]['hits'] / n_windows:>8.3f} | {mo:>10.1f} | "
            f"{copy_retr[k]['hits'] / n_windows:>9.3f} | {co:>11.1f}"
        )

    results = {
        "checkpoint": args.checkpoint,
        "model": sidecar.get("model"),
        "training": sidecar.get("training"),
        "seed": sidecar.get("seed"),
        "stride": stride,
        "n_windows": n_windows,
        "horizons": args.horizons,
        "rollout_l1": {v: {str(k): l1[v][k] for k in args.horizons} for v in l1},
        "action_sensitivity": {str(kmax): sens},
        "retrieval": {
            str(k): {
                "top1_acc": retr[k]["hits"] / n_windows,
                "median_offset": statistics.median(retr[k]["offsets"]),
                "copy_top1_acc": copy_retr[k]["hits"] / n_windows,
                "copy_median_offset": statistics.median(copy_retr[k]["offsets"]),
            }
            for k in args.horizons
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
