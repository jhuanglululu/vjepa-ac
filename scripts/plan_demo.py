import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

from vjepa_ac import data
from vjepa_ac.checkpoints import load_model_weights
from vjepa_ac.cpredictor import CPredictor
from vjepa_ac.device import get_device
from vjepa_ac.predictor import Predictor
from vjepa_ac.variations import ModelConfig, TrainingConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--start", type=int, default=30)
    p.add_argument("--goal-offset", type=int, default=90)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--goal-tol", type=int, default=3)
    p.add_argument("--commit-steps", type=int, default=1)
    p.add_argument("--snap", choices=["both", "state", "latent"], default="both")
    p.add_argument("--state-topk", type=int, default=8)
    p.add_argument("--snap-range", type=int, nargs=2, default=None)
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--gain", default="auto")
    p.add_argument("--action-momentum", type=float, default=0.0)
    p.add_argument("--random-plan", action="store_true")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--elite", type=int, default=64)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--action-std", type=float, default=1.0)
    p.add_argument("--action-clip", type=float, default=2.5)
    p.add_argument("--rollout-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or get_device()
    torch.manual_seed(args.seed)

    sidecar_path = Path(args.checkpoint).with_suffix(".json")
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    mc = ModelConfig(**sidecar["config"]["model"])
    tc = TrainingConfig(**sidecar["config"]["training"])
    stride = tc.stride
    assert args.context + args.horizon <= tc.T
    assert args.commit_steps <= args.horizon

    cache = data.load_cache()
    cond = data.load_conditioner(cache.states, sidecar["conditioning"])
    _, val_eps = data.split_episodes(cache.episodes, tc.val_frac)
    a0, b0 = val_eps[args.episode % len(val_eps)]

    model_cls = CPredictor if mc.compressor else Predictor
    model = model_cls(mc, tc.T).to(device).eval()
    model.load_state_dict(load_model_weights(args.checkpoint))

    @torch.no_grad()
    def to_space(z):
        if not isinstance(model, CPredictor):
            return z
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            return model.encode(z).float()

    @torch.no_grad()
    def forward(s, a):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            return model(s, a).float()

    ep_tokens = []
    for i in range(a0, b0, 64):
        j = min(i + 64, b0)
        ep_tokens.append(to_space(cache.latents[i:j].to(device).float()))
    ep_tokens = torch.cat(ep_tokens)
    flat_ep = ep_tokens.reshape(len(ep_tokens), -1)

    s0 = a0 + args.start
    goal = max(a0, min(s0 + args.goal_offset, b0 - 1))
    goal_tok = ep_tokens[goal - a0]
    print(
        f"episode [{a0},{b0}) len {b0 - a0} | start {s0} (+{args.start}) | "
        f"goal {goal} (goal is {goal - s0:+d} frames from start) | context {args.context} real frames | "
        f"replan horizon {args.horizon} strided steps ({args.horizon * stride} frames) | "
        f"snap {args.snap} | commit {args.commit_steps} action(s)"
        + (" | RANDOM-PLAN CONTROL" if args.random_plan else "")
    )

    def executed_features(i, j):
        delta = cond.delta_cumsum[j] - cond.delta_cumsum[i]
        grip = cond.states[j][..., -1:]
        feats = torch.cat([delta[..., :-1], grip], dim=-1)
        return ((feats - cond.mean) / cond.std).to(device)

    @torch.no_grad()
    def calibrate_gain():
        idx = torch.arange(a0, b0 - stride)
        if len(idx) > 256:
            idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))[:256]]
        real_n, imag_n = 0.0, 0.0
        for i in range(0, len(idx), 64):
            bi = idx[i : i + 64]
            tok0 = ep_tokens[bi - a0]
            tok1 = ep_tokens[bi - a0 + stride]
            af = torch.stack([executed_features(int(t), int(t) + stride) for t in bi])[:, None, :]
            af = torch.cat([af, torch.zeros_like(af)], dim=1)
            s = torch.stack([tok0, tok0], dim=1)
            pred = forward(s, af)
            real_n += (tok1 - tok0).reshape(len(bi), -1).norm(dim=1).sum().item()
            imag_n += pred[:, 0].reshape(len(bi), -1).norm(dim=1).sum().item()
        return max(1.0, min(8.0, real_n / max(imag_n, 1e-9)))

    if args.snap == "latent" and args.gain == "auto":
        gain = calibrate_gain()
        print(f"actuator gain (real/imagined one-step token delta, true actions): {gain:.2f}")
    else:
        gain = 1.0 if args.gain == "auto" else float(args.gain)

    @torch.no_grad()
    def cem_plan(ctx_frames, last_action):
        C = len(ctx_frames)
        H = args.horizon
        ctx_tok = ep_tokens[[i - a0 for i in ctx_frames]]
        ctx_act = [executed_features(ctx_frames[k], ctx_frames[k + 1]) for k in range(C - 1)]
        mu = torch.zeros(H, cache.state_dim, device=device)
        if args.action_momentum > 0 and last_action is not None:
            mu = args.action_momentum * last_action.expand(H, -1).clone()
        if args.random_plan:
            return (torch.randn(H, cache.state_dim, device=device) * args.action_std).clamp(
                -args.action_clip, args.action_clip
            ), math.nan
        sigma = torch.full((H, cache.state_dim), args.action_std, device=device)
        best_e = math.inf
        for _ in range(args.iters):
            noise = torch.randn(args.samples, H, cache.state_dim, device=device)
            A = (mu + sigma * noise).clamp(-args.action_clip, args.action_clip)
            A[0] = mu
            energies = []
            for i in range(0, len(A), args.rollout_batch):
                Ab = A[i : i + args.rollout_batch]
                B = len(Ab)
                s = torch.zeros(B, C + H, *ctx_tok.shape[1:], device=device)
                s[:] = ctx_tok[-1]
                s[:, :C] = ctx_tok
                af = torch.zeros(B, C + H, cache.state_dim, device=device)
                for k in range(C - 1):
                    af[:, k] = ctx_act[k]
                af[:, C - 1 : C - 1 + H] = Ab
                for t in range(C - 1, C + H - 1):
                    pred = forward(s, af)
                    s[:, t + 1] = s[:, t] + pred[:, t]
                energies.append((s[:, -1] - goal_tok).abs().mean(dim=(1, 2)))
            e = torch.cat(energies)
            elite = A[e.topk(args.elite, largest=False).indices]
            mu = 0.5 * mu + 0.5 * elite.mean(0)
            sigma = (0.5 * sigma + 0.5 * elite.std(0)).clamp(min=0.05)
            best_e = min(best_e, e.min().item())
        return mu, best_e

    @torch.no_grad()
    def step_once(ctx_frames, actions_seq):
        C = len(ctx_frames)
        k = len(actions_seq)
        ctx_tok = ep_tokens[[i - a0 for i in ctx_frames]]
        s = torch.zeros(1, C + k, *ctx_tok.shape[1:], device=device)
        s[0] = ctx_tok[-1]
        s[0, :C] = ctx_tok
        af = torch.zeros(1, C + k, cache.state_dim, device=device)
        for i in range(C - 1):
            af[0, i] = executed_features(ctx_frames[i], ctx_frames[i + 1])
        af[0, C - 1 : C - 1 + k] = actions_seq
        for t in range(C - 1, C + k - 1):
            pred = forward(s, af)
            s[0, t + 1] = s[0, t] + pred[0, t]
        imagined = s[0, C - 1] + gain * (s[0, -1] - s[0, C - 1])
        lo, hi = pool_bounds(ctx_frames[-1])
        if lo >= hi:
            return None
        d = torch.cdist(imagined.reshape(1, -1), flat_ep[lo:hi])[0]
        return a0 + lo + int(d.argmin())

    def wrap(x):
        return torch.remainder(x + math.pi, 2 * math.pi) - math.pi

    def pool_bounds(cur):
        lo, hi = 0, b0 - a0
        if args.snap_range is not None:
            lo = max(lo, args.snap_range[0])
            hi = min(hi, args.snap_range[1] + 1)
        if args.forward_only:
            lo = max(lo, cur + 1 - a0)
        return lo, hi

    def step_state(ctx_frames, actions_seq):
        cur = ctx_frames[-1]
        feats = actions_seq.cpu() * cond.std + cond.mean
        target = cond.states[cur, :-1] + feats[:, :-1].sum(0)
        target[3:] = wrap(target[3:])
        target_grip = feats[-1, -1]
        ep_states = cond.states[a0:b0]
        diff = ep_states[:, :-1] - target
        diff[:, 3:] = wrap(diff[:, 3:])
        grip_diff = ep_states[:, -1] - target_grip
        scale = ep_states.std(0).clamp(min=1e-3)
        d = ((diff / scale[:-1]) ** 2).sum(1) + (grip_diff / scale[-1]) ** 2
        lo, hi = pool_bounds(cur)
        if lo >= hi:
            return None
        if args.snap == "state":
            return a0 + lo + int(d[lo:hi].argmin())
        d[cur - a0] = torch.inf
        k = min(args.state_topk, int(torch.isfinite(d[lo:hi]).sum()))
        if k == 0:
            return None
        cand = d[lo:hi].topk(k, largest=False).indices + lo
        cur_flat = flat_ep[cur - a0].reshape(1, -1)
        dl = torch.cdist(cur_flat, flat_ep[cand.to(device)])[0]
        return a0 + int(cand[int(dl.argmin())])

    committed = [s0]
    trace = []
    stuck = 0
    last_action = None
    while len(trace) < args.max_steps:
        cur = committed[-1]
        if abs(cur - goal) <= args.goal_tol:
            print(f"reached goal tolerance at frame {cur} ({cur - goal:+d} from goal)")
            break
        ctx = committed[-args.context :]
        mu, best_e = cem_plan(ctx, last_action)
        last_action = mu[0]
        step_fn = step_once if args.snap == "latent" else step_state
        nxt = step_fn(ctx, mu[: args.commit_steps])
        if nxt is None:
            print("episode end reached, stopping")
            break
        phys = (mu[0].cpu() * cond.std + cond.mean).tolist()
        trace.append({"from": cur, "to": nxt, "energy": best_e, "action": phys})
        print(
            f"step {len(trace):>3} | frame {cur} -> {nxt} ({nxt - cur:+d}) | "
            f"goal dist {abs(nxt - goal):>3} | energy {best_e:.4f} | "
            f"dxyz ({phys[0]:+.3f},{phys[1]:+.3f},{phys[2]:+.3f}) grip {phys[6]:.2f}"
        )
        if nxt == cur:
            stuck += 1
            if stuck >= 3:
                print("no progress for 3 consecutive steps, stopping")
                break
        else:
            stuck = 0
        committed.append(nxt)

    final = committed[-1]
    print(
        f"\ncommitted {len(committed) - 1} steps | final frame {final} ({final - s0:+d} from "
        f"start) | goal {goal} ({goal - s0:+d}) | miss {abs(final - goal)} frames"
    )

    out = args.out or os.path.join(
        os.path.dirname(args.checkpoint) or ".", f"plan_ep{args.episode}_g{goal - s0}.gif"
    )
    with open(Path(out).with_suffix(".json"), "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "episode_range": [a0, b0],
                "start": s0,
                "goal": goal,
                "stride": stride,
                "snap": args.snap,
                "commit_steps": args.commit_steps,
                "random_plan": args.random_plan,
                "committed": committed,
                "trace": trace,
            },
            f,
            indent=2,
        )

    meta = cache.meta
    if args.no_gif or "episode_ids" not in meta:
        print("skipping gif (no video provenance in cache meta)" if not args.no_gif else "")
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import av
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    import prepare_cache as pc

    ep_pos = cache.episodes.index([a0, b0])
    ep_id = meta["episode_ids"][ep_pos]
    trim = meta["trim"]
    key = meta["camera_key"]
    emeta = pq.read_table(
        pc.hub_file(f"{data.DATASET_SPLIT}/meta/episodes/chunk-000/file-000.parquet")
    ).to_pydict()
    row = emeta["episode_index"].index(ep_id)
    from_ts = emeta[f"videos/{key}/from_timestamp"][row]
    chunk = emeta[f"videos/{key}/chunk_index"][row]
    fidx = emeta[f"videos/{key}/file_index"][row]
    path = pc.hub_file(f"{data.DATASET_SPLIT}/videos/{key}/chunk-{chunk:03d}/file-{fidx:03d}.mp4")

    def grab(cache_idx):
        frame_in_video = trim + (cache_idx - a0)
        ts = from_ts + frame_in_video / pc.FPS + 1 / (2 * pc.FPS)
        with av.open(path) as container:
            st = container.streams.video[0]
            container.seek(int(ts / st.time_base), stream=st, backward=True)
            for fr in container.decode(st):
                if float(fr.pts * st.time_base) >= ts - 1 / (2 * pc.FPS):
                    return Image.fromarray(fr.to_ndarray(format="rgb24"))
        raise AssertionError(f"no frame at ts {ts}")

    goal_im = grab(goal).resize((640, 360))
    frames = []
    for t, idx in enumerate(committed):
        cur = grab(idx).resize((640, 360))
        panel = Image.new("RGB", (1280 + 8, 360 + 26), "black")
        panel.paste(cur, (0, 26))
        panel.paste(goal_im, (648, 26))
        d = ImageDraw.Draw(panel)
        d.text(
            (4, 6),
            f"mpc step {t}/{len(committed) - 1} | frame {idx} ({idx - s0:+d})",
            fill="white",
        )
        d.text((652, 6), f"goal | frame {goal} ({goal - s0:+d})", fill="white")
        frames.append(panel)
    frames += [frames[-1]] * int(args.fps)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
