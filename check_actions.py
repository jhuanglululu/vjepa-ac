# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "lerobot",
# ]
# ///
import vjepa_common as C

import json, os

import torch
from safetensors.torch import save_file
from lerobot.datasets import LeRobotDataset

assert len(C.DATASET_IDS) == 1, "state dump assumes a single dataset in cache order"
ds = LeRobotDataset(C.DATASET_IDS[0], video_backend="pyav")
print("action feature:")
print(json.dumps(ds.meta.features.get("action"), indent=2, default=str))

hf = ds.hf_dataset
A = torch.stack([torch.as_tensor(x) for x in hf["action"]]).float()
S = torch.stack([torch.as_tensor(x) for x in hf["observation.state"]]).float()
ep = torch.tensor([int(e) for e in hf["episode_index"]])

m = ep[:-1] == ep[1:]
a = A[:-1][m]
s0 = S[:-1][m]
dS = S[1:][m] - s0
print(f"\n{len(a)} within-episode transitions")


def diagcorr(x, y):
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    denom = (xc.std(0, correction=0) * yc.std(0, correction=0)).clamp(min=1e-8)
    return (xc * yc).mean(0) / denom


def fit_r2(x, y):
    x1 = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    w = torch.linalg.lstsq(x1, y).solution
    pred = x1 @ w
    var = ((y - y.mean(0)) ** 2).mean(0)
    mse = ((pred - y) ** 2).mean(0)
    valid = var > 1e-8
    return (1 - mse[valid] / var[valid]).mean().item()


c_vel = diagcorr(a, dS)
c_tgt = diagcorr(a - s0, dS)
c_abs = diagcorr(a, s0)

print(f"\n{'dim':>3} | {'act std':>8} | {'dS std':>8} | {'corr(a,dS)':>10} | {'corr(a-s,dS)':>12} | {'corr(a,s)':>9}")
for d in range(a.shape[1]):
    print(
        f"{d:>3} | {a[:, d].std():>8.4f} | {dS[:, d].std():>8.5f} | "
        f"{c_vel[d]:>+10.3f} | {c_tgt[d]:>+12.3f} | {c_abs[d]:>+9.3f}"
    )

print(f"\nlinear fit R2 of dS: from a {fit_r2(a, dS):+.3f} | from [a, s] {fit_r2(torch.cat([a, s0], 1), dS):+.3f} | from s alone {fit_r2(s0, dS):+.3f}")

print("""
how to read:
  corr(a,dS) ~ +1 per dim -> actions are joint velocities/deltas; summing them
    across a stride (as the probe does) is correct
  corr(a-s,dS) ~ +1 and corr(a,s) ~ +1 -> actions are absolute joint targets;
    the usable motion signal is (a - s) or dS itself, not a
  per-dim corrs low but fit R2 from [a, s] >> from s alone -> actions live in a
    different frame (e.g. cartesian velocity); dS still recovers the motion
  everything low -> commands barely correlate with executed motion at 15 Hz;
    aggregate over a coarser stride
in any case: observation.state is not in the latent cache -- caching it gives a
ground-truth motion signal (dS) to use as the probe target or predictor action""")

state_path = os.path.join(C.CACHE_DIR, "state.safetensors")
save_file({"state": S}, state_path)
print(f"saved proprio states -> {state_path} (enables probe.py --target dstate)")
