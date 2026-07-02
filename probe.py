# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "safetensors",
#     "tqdm",
# ]
# ///
import vjepa_common as C

import argparse, json, math, os

import torch
from torch import nn
from safetensors import safe_open
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--val-frac", type=float, default=0.2)
parser.add_argument("--holdout-frac", type=float, default=0.15)
parser.add_argument("--val-pairs", type=int, default=4096)
parser.add_argument("--d-proj", type=int, default=32)
parser.add_argument("--d-hidden", type=int, default=512)
parser.add_argument("--target", choices=["action", "dstate", "state"], default="action")
parser.add_argument("--state-path", default=os.path.join(C.CACHE_DIR, "state.safetensors"))
parser.add_argument("--device", default=C.train_device)
parser.add_argument("--out", default=os.path.join(C.OUTPUT_DIR, "probe_results.json"))
args = parser.parse_args()
device = args.device

meta = json.load(open(C.CACHE_META))
action_dim = meta["action_dim"]
W, H = C.predictor_config["latent_size"]
P = W * H
D = C.predictor_config["d_state"]

cache = safe_open(C.LATENTS_PATH, framework="pt", device="cpu")
latents = cache.get_slice("latents")
if args.target in ("dstate", "state"):
    if "state" in cache.keys():
        S = cache.get_tensor("state").float()
    else:
        assert os.path.exists(args.state_path), "no state in cache -- run check_actions.py first"
        with safe_open(args.state_path, framework="pt", device="cpu") as f:
            S = f.get_tensor("state").float()
    action_dim = S.shape[1]

if args.target == "dstate":
    wd = S[1:] - S[:-1]
    wd = torch.remainder(wd + math.pi, 2 * math.pi) - math.pi
    base = torch.cat([wd, torch.zeros(1, action_dim)])
    cs = torch.cat([torch.zeros(1, action_dim), base.cumsum(0)])
    raw_targets = lambda idx, k: (base[idx], cs[idx + k] - cs[idx])
    half_names = ("first", "sum")
    print(f"target: executed motion (wrap-corrected dstate, {action_dim} dims)")
elif args.target == "state":
    raw_targets = lambda idx, k: (S[idx], S[idx + k])
    half_names = ("pose t", "pose t+k")
    print(f"target: absolute proprio pose ({action_dim} dims)")
else:
    actions = cache.get_tensor("actions").float()
    cs = torch.cat([torch.zeros(1, action_dim), actions.cumsum(0)])
    raw_targets = lambda idx, k: (actions[idx], cs[idx + k] - cs[idx])
    half_names = ("first", "sum")
    print("target: commanded actions")

episodes = sorted(meta["episodes"])
gen = torch.Generator().manual_seed(0)
ep_perm = torch.randperm(len(episodes), generator=gen).tolist()
n_val_eps = max(1, round(args.val_frac * len(episodes)))
val_eps = [episodes[i] for i in ep_perm[:n_val_eps]]
train_eps = [episodes[i] for i in ep_perm[n_val_eps:]]
print(f"{len(train_eps)} train / {len(val_eps)} val episodes")


def pairs_for(eps, k):
    idx = []
    for a, b in eps:
        idx.extend(range(a, b - k))
    return torch.tensor(idx, dtype=torch.long)


def split_pairs(eps, k):
    tr, ho = [], []
    for a, b in eps:
        h = max(k + 1, round(args.holdout_frac * (b - a)))
        sp = b - h
        tr.extend(range(a, max(a, sp - k)))
        ho.extend(range(sp, b - k))
    return torch.tensor(tr, dtype=torch.long), torch.tensor(ho, dtype=torch.long)


def gather_z(idx):
    z = torch.stack([latents[int(t) : int(t) + 1] for t in idx])
    return z.squeeze(1).to(device).float()


def targets(idx, k, stats):
    t1, t2 = raw_targets(idx, k)
    mu1, sd1, mu2, sd2 = stats
    y = torch.cat([(t1 - mu1) / sd1, (t2 - mu2) / sd2], dim=1)
    return y.to(device)


class Probe(nn.Module):
    def __init__(self, pair):
        super().__init__()
        self.pair = pair
        self.proj = nn.Linear(D, args.d_proj, bias=False)
        d_in = P * args.d_proj * (2 if pair else 1)
        if args.d_hidden:
            self.mlp = nn.Sequential(
                nn.Linear(d_in, args.d_hidden),
                nn.SiLU(),
                nn.Linear(args.d_hidden, 2 * action_dim),
            )
        else:
            self.mlp = nn.Linear(d_in, 2 * action_dim)

    def forward(self, z0, z1):
        h = self.proj(z0).flatten(1)
        if self.pair:
            h = torch.cat([h, self.proj(z1 - z0).flatten(1)], dim=1)
        return self.mlp(h)


def r2_per_half(pred, y):
    out = {}
    for name, sl in [("first", slice(0, action_dim)), ("sum", slice(action_dim, 2 * action_dim))]:
        p, t = pred[:, sl], y[:, sl]
        var = ((t - t.mean(0)) ** 2).mean(0)
        mse = ((p - t) ** 2).mean(0)
        valid = var > 1e-8
        r2 = (1 - mse[valid] / var[valid]).clamp(min=-1.0)
        out[name] = r2.mean().item() if valid.any() else float("nan")
    return out


def run_probe(k, pair, train_idx, eval_sets, stats):
    torch.manual_seed(0)
    probe = Probe(pair).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    label = f"k={k} {'pair' if pair else 'state'}"
    for _ in tqdm(range(args.steps), desc=label, unit="step", leave=False):
        bi = train_idx[torch.randint(0, len(train_idx), (args.batch_size,))]
        z0 = gather_z(bi)
        z1 = gather_z(bi + k) if pair else z0
        y = targets(bi, k, stats)
        optim.zero_grad(set_to_none=True)
        loss = ((probe(z0, z1) - y) ** 2).mean()
        loss.backward()
        optim.step()

    probe.eval()
    scores = {}
    with torch.no_grad():
        for name, idx in eval_sets.items():
            if len(idx) < 8:
                scores[name] = {"first": float("nan"), "sum": float("nan")}
                continue
            preds, ys = [], []
            for i in range(0, len(idx), 256):
                bi = idx[i : i + 256]
                z0 = gather_z(bi)
                z1 = gather_z(bi + k) if pair else z0
                preds.append(probe(z0, z1).cpu())
                ys.append(targets(bi, k, stats).cpu())
            scores[name] = r2_per_half(torch.cat(preds), torch.cat(ys))
    return scores


def subsample(idx, n, seed):
    if len(idx) <= n:
        return idx
    g = torch.Generator().manual_seed(seed)
    return idx[torch.randperm(len(idx), generator=g)[:n]]


rows = []
for k in args.strides:
    train_idx, win_idx = split_pairs(train_eps, k)
    ep_idx = pairs_for(val_eps, k)
    if len(train_idx) < args.batch_size or len(ep_idx) < 8:
        print(f"stride {k}: not enough pairs, skipping")
        continue
    eval_sets = {
        "train": subsample(train_idx, args.val_pairs, 1),
        "win": subsample(win_idx, args.val_pairs, 2),
        "ep": subsample(ep_idx, args.val_pairs, 3),
    }

    t1, t2 = raw_targets(train_idx, k)
    stats = (
        t1.mean(0), t1.std(0).clamp(min=1e-6),
        t2.mean(0), t2.std(0).clamp(min=1e-6),
    )
    for pair in [False, True]:
        s = run_probe(k, pair, train_idx, eval_sets, stats)
        rows.append({
            "stride": k,
            "variant": "pair" if pair else "state",
            "r2": s,
            "n_train_pairs": len(train_idx),
            "n_win_pairs": len(win_idx),
            "n_ep_pairs": len(ep_idx),
        })
        print(
            f"stride {k:>2} | {'pair ' if pair else 'state'} | R2({half_names[1]}) "
            f"train {s['train']['sum']:+.3f} | win {s['win']['sum']:+.3f} | ep {s['ep']['sum']:+.3f}"
        )

print(f"\n=== probe target={args.target}, R2 as {half_names[0]}/{half_names[1]}, {n_val_eps} held-out episodes ===")
print(f"{'stride':>6} | {'variant':>7} | {'train':>13} | {'held win':>13} | {'held ep':>13}")
for r in rows:
    s = r["r2"]
    cells = [f"{s[n]['first']:+.2f}/{s[n]['sum']:+.2f}" for n in ["train", "win", "ep"]]
    print(f"{r['stride']:>6} | {r['variant']:>7} | {cells[0]:>13} | {cells[1]:>13} | {cells[2]:>13}")

if args.target == "state":
    print("""
how to read (pose target):
  'state' variant, 'pose t' half is the key cell: can a single frame's latents
    decode the current arm pose?
  high on held-win -> latents see the arm; per-step motion is just below the
    noise floor -> coarser strides / more data are the fix
  low on held-win too -> the camera/encoder does not resolve arm pose; no
    predictor or data scale fixes this cache""")
else:
    print("""
how to read:
  train high, held-win high, held-ep low (pair >> state) -> transitions encode
    actions within a scene but decoding does not transfer across scenes at this
    data scale; more episodes (or per-scene adaptation) is the bottleneck
  train high, held-win low -> probe memorizes frames; signal is doubtful
  train low for pair -> no extractable action signal even in-sample; check what
    the action key encodes and whether the camera/encoder resolves the arm
  pair ~ state everywhere -> transitions add nothing over the current frame""")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"target": args.target, "half_names": half_names, "d_hidden": args.d_hidden, "strides": args.strides, "val_episodes": n_val_eps, "rows": rows}, f, indent=2)
print(f"saved -> {args.out}")
