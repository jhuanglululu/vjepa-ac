import argparse
import math

import torch

from vjepa_ac import data

p = argparse.ArgumentParser()
p.add_argument("--cache-dir", default=None)
args = p.parse_args()

cache = data.load_cache(args.cache_dir)
print(
    f"dataset {cache.meta.get('dataset')} [{cache.meta.get('split')}] | "
    f"camera {cache.meta.get('camera')} | trim {cache.meta.get('trim')} | "
    f"{len(cache.episodes)} episodes"
)

A, S = cache.actions, cache.states
mask = torch.zeros(len(S) - 1, dtype=torch.bool)
for a, b in cache.episodes:
    mask[a : b - 1] = True
a = A[:-1][mask]
s0 = S[:-1][mask]
dS = S[1:][mask] - s0
dS = torch.remainder(dS + math.pi, 2 * math.pi) - math.pi
print(f"{len(a)} within-episode transitions (wrap-corrected dS)")


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
c_abs = diagcorr(a, s0)

print(
    f"\n{'dim':>3} | {'act std':>8} | {'dS std':>8} | {'state std':>9} | "
    f"{'corr(a,dS)':>10} | {'corr(a,s)':>9}"
)
for d in range(a.shape[1]):
    print(
        f"{d:>3} | {a[:, d].std():>8.4f} | {dS[:, d].std():>8.5f} | {s0[:, d].std():>9.4f} | "
        f"{c_vel[d]:>+10.3f} | {c_abs[d]:>+9.3f}"
    )

print(
    f"\nlinear fit R2 of dS: from a {fit_r2(a, dS):+.3f} | "
    f"from [a, s] {fit_r2(torch.cat([a, s0], 1), dS):+.3f} | from s alone {fit_r2(s0, dS):+.3f}"
)

print("""
expected for Cosmos3-DROID: state dims 0-5 are cartesian_position [x,y,z,rx,ry,rz]
(rx/ry/rz wrap at +-pi, so dS std should be small after wrap correction), dim 6 is
gripper_position in [0,1]; action dims 0-5 are cartesian_velocity (corr(a,dS) should
be clearly positive) and dim 6 the commanded gripper position (corr(a,s) ~ +1).
Conditioning uses wrap-corrected dS on dims 0-5 and the absolute value on dim 6 --
confirm those two patterns hold before training""")
