# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "safetensors",
#     "tqdm",
# ]
# ///
import vjepa_common as C

import argparse, bisect, json, os, statistics

import torch
from safetensors import safe_open
from tqdm.auto import tqdm

from predictor import Predictor

T = C.T
tc = C.training_config

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=os.path.join(C.OUTPUT_DIR, "predictor_final.safetensors"))
parser.add_argument("--windows", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 15])
parser.add_argument("--device", default=C.train_device)
parser.add_argument("--out", default=os.path.join(C.OUTPUT_DIR, "eval_results.json"))
args = parser.parse_args()
device = args.device
assert max(args.horizons) <= T - 1

meta = json.load(open(C.CACHE_META))
action_dim = meta["action_dim"]
cfg = dict(C.predictor_config)
cfg["d_action"] = action_dim

cache = safe_open(C.LATENTS_PATH, framework="pt", device="cpu")
latents = cache.get_slice("latents")
actions = cache.get_tensor("actions")

episodes = sorted(meta["episodes"])
ep_starts = [a for a, b in episodes]

starts = []
for a, b in episodes:
    starts.extend(range(a, b - T + 1))
starts = torch.tensor(starts, dtype=torch.long)
gen = torch.Generator().manual_seed(0)
perm = torch.randperm(len(starts), generator=gen)
val_n = min(tc["val_windows"], len(starts) // 10)
val_starts = starts[perm[:val_n]]

gen2 = torch.Generator().manual_seed(1)
sel = torch.randperm(len(val_starts), generator=gen2)[: args.windows]
eval_starts = val_starts[sel]
print(f"evaluating {len(eval_starts)} val windows (split leaks into train at window level; see notes)")

model = Predictor(cfg).to(device).eval()
with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
    model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
print(f"loaded {args.checkpoint}")

_arangeT = torch.arange(T)


def gather(sb):
    z = torch.stack([latents[int(s) : int(s) + T] for s in sb]).to(device).float()
    flat = (sb[:, None] + _arangeT).reshape(-1)
    a = actions[flat].reshape(len(sb), T, action_dim).to(device).float()
    return z, a


def forward(states, acts):
    if device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(states, acts).float()
    return model(states, acts).float()


@torch.no_grad()
def rollout(z, a):
    s = z.clone()
    acts = [a[:, t] for t in range(T)]
    for t in range(T - 1):
        pred = forward([s[:, i] for i in range(T)], acts)
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
        chunk = latents[i:j].to(device).float().reshape(j - i, -1)
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
    z, a = gather(sb)
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
            ri = retrieve_frame(preds["model"][bi, k], a0, b0)
            retr[k]["hits"] += int(ri == s0 + k)
            retr[k]["offsets"].append(ri - (s0 + k))
            ci = retrieve_frame(z[bi, 0], a0, b0)
            copy_retr[k]["hits"] += int(ci == s0 + k)
            copy_retr[k]["offsets"].append(ci - (s0 + k))

l1 = {v: {k: l1_sum[v][k] / n_windows for k in args.horizons} for v in l1_sum}

print(f"\n=== rollout latent L1 vs ground truth ({n_windows} windows) ===")
print(f"{'h':>3} | {'copy-first':>10} | {'model':>10} | {'zero-act':>10} | {'shuf-act':>10} | {'model/copy':>10}")
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
print(f"{'h':>3} | {'top1 acc':>8} | {'med offset':>10} | {'copy top1':>9} | {'copy offset':>11}")
for k in args.horizons:
    mo = statistics.median(retr[k]["offsets"])
    co = statistics.median(copy_retr[k]["offsets"])
    print(
        f"{k:>3} | {retr[k]['hits'] / n_windows:>8.3f} | {mo:>10.1f} | "
        f"{copy_retr[k]['hits'] / n_windows:>9.3f} | {co:>11.1f}"
    )

results = {
    "checkpoint": args.checkpoint,
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
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {args.out}")
