import argparse
import copy
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import av
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import AutoModel, AutoVideoProcessor

from vjepa_ac import data
from vjepa_ac.device import get_device, pick_free_gpus
from vjepa_ac.variations import TRAININGS

FPS = 15
DECODE_BATCH = 32
STATE_COLS = ["observation.state.cartesian_position", "observation.state.gripper_position"]
ACTION_COLS = ["action.cartesian_velocity", "action.gripper_position"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--trim", type=int, default=15)
    p.add_argument("--cameras", nargs="+", default=list(data.CAMERAS), choices=list(data.CAMERAS))
    return p.parse_args()


def hub_file(path):
    return hf_hub_download(data.DATASET_ID, path, repo_type="dataset")


def load_plan(n_episodes, trim):
    meta = pq.read_table(hub_file(f"{data.DATASET_SPLIT}/meta/episodes/chunk-000/file-000.parquet"))
    assert meta.num_rows >= n_episodes, f"meta shard has only {meta.num_rows} episodes"
    meta = meta.slice(0, n_episodes).to_pydict()

    plan, dropped = [], 0
    for i in range(n_episodes):
        length = meta["length"][i]
        if length - trim < 2:
            dropped += 1
            continue
        ep = {"episode": meta["episode_index"][i], "length": length, "videos": {}}
        for short, key in data.CAMERAS.items():
            ep["videos"][short] = {
                "chunk": meta[f"videos/{key}/chunk_index"][i],
                "file": meta[f"videos/{key}/file_index"][i],
                "from_ts": meta[f"videos/{key}/from_timestamp"][i],
            }
        plan.append(ep)
    return plan, dropped


def load_rows(plan, trim):
    shards = sorted({(e["chunk"], e["file"]) for e in _data_locations(plan)})
    tables = [
        pq.read_table(
            hub_file(f"{data.DATASET_SPLIT}/data/chunk-{c:03d}/file-{f:03d}.parquet"),
            columns=["episode_index", "frame_index"] + STATE_COLS + ACTION_COLS,
        ).to_pydict()
        for c, f in shards
    ]
    wanted = {e["episode"] for e in plan}
    rows = {}
    for t in tables:
        for j in range(len(t["episode_index"])):
            e = _scalar(t["episode_index"][j])
            if e not in wanted:
                continue
            f = _scalar(t["frame_index"][j])
            if f < trim:
                continue
            state = _vec(t[STATE_COLS[0]][j]) + _vec(t[STATE_COLS[1]][j])
            action = _vec(t[ACTION_COLS[0]][j]) + _vec(t[ACTION_COLS[1]][j])
            rows.setdefault(e, []).append((f, state, action))
    for e in rows:
        rows[e].sort()
    return rows


def _data_locations(plan):
    meta = pq.read_table(
        hub_file(f"{data.DATASET_SPLIT}/meta/episodes/chunk-000/file-000.parquet"),
        columns=["episode_index", "data/chunk_index", "data/file_index"],
    ).to_pydict()
    wanted = {e["episode"] for e in plan}
    return [
        {"chunk": c, "file": f}
        for e, c, f in zip(meta["episode_index"], meta["data/chunk_index"], meta["data/file_index"])
        if e in wanted
    ]


def _scalar(v):
    return int(v[0]) if hasattr(v, "__len__") else int(v)


def _vec(v):
    return [float(x) for x in v] if hasattr(v, "__len__") else [float(v)]


def decode_segment(path, from_ts, n_frames, skip):
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(from_ts / stream.time_base), stream=stream, backward=True)
        got = 0
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) < from_ts - 1 / (2 * FPS):
                continue
            got += 1
            if got <= skip:
                continue
            yield torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            if got == n_frames:
                return
    raise AssertionError(f"{path}: expected {n_frames} frames from {from_ts}, got {got}")


@torch.no_grad()
def encode_batch(frames, enc, dev, mean, std):
    x = frames.to(dev).float() / 255
    if x.shape[-2:] != (data.IMG_SIZE, data.IMG_SIZE):
        x = F.interpolate(x, data.IMG_SIZE, mode="bilinear", align_corners=False, antialias=True)
    x = (x - mean) / std
    clip = x.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
        tok = enc(clip).last_hidden_state
    return tok.reshape(x.shape[0], tok.shape[1], 1024).half().cpu()


def check_health(latents, actions, episodes, T, stride):
    print("\n=== cache health check ===")
    N, P, D = latents.shape
    print(f"frames {N} | patches {P} | dim {D} | episodes {len(episodes)}")
    warns = []

    nonfinite = 0
    for i in range(0, N, 1024):
        nonfinite += (~torch.isfinite(latents[i : i + 1024])).sum().item()
    print(f"non-finite latent elements: {nonfinite}")
    if nonfinite:
        warns.append(f"{nonfinite} non-finite latent values (NaN/Inf)")

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(N, generator=g)[: min(N, 4096)]
    s = latents[idx].float()
    fn = s.norm(dim=(1, 2))
    print(
        f"latent abs-mean {s.abs().mean():.4f} | std {s.std():.4f} | "
        f"min {s.min():.3f} | max {s.max():.3f}"
    )
    print(f"per-frame L2 norm: mean {fn.mean():.2f} | std {fn.std():.2f}")
    if fn.mean() > 0 and fn.std() / fn.mean() < 1e-3:
        warns.append("per-frame latent norms nearly identical (possible encoder collapse)")

    sel = episodes[:: max(1, len(episodes) // 50)][:50]
    next_mse, sf_mse = [], []
    for a, b in sel:
        if b - a < 2:
            continue
        z = latents[a:b].float()
        next_mse.append(((z[1:] - z[:-1]) ** 2).mean().item())
        sf_mse.append(((z[0] - z[-1]) ** 2).mean().item())
    if next_mse:
        nm = sum(next_mse) / len(next_mse)
        sf = sum(sf_mse) / len(sf_mse)
        print(
            f"per-step Δ mse {nm:.4f} | start→final mse {sf:.4f} | signal ratio {sf / max(nm, 1e-9):.1f}"
        )
        if nm < 1e-4:
            warns.append("per-step latent change ~0 (static or duplicate frames)")

    a_all = actions.float()
    a_std = a_all.std(dim=0)
    a_min = a_all.min(dim=0).values
    a_max = a_all.max(dim=0).values
    print("action per-dim [min, max] std:")
    for d in range(a_all.shape[1]):
        print(f"  dim {d}: [{a_min[d]:+.3f}, {a_max[d]:+.3f}] std {a_std[d]:.4f}")
    dead = [d for d in range(a_all.shape[1]) if a_std[d] < 1e-6]
    if dead:
        warns.append(f"action dims with zero variance: {dead}")

    span = data.window_span(T, stride)
    windows = sum(max(0, (b - a) - span + 1) for a, b in episodes)
    short = sum(1 for a, b in episodes if (b - a) < span)
    print(
        f"training windows (T={T}, stride={stride}, span={span}): {windows} | "
        f"episodes shorter than span: {short}/{len(episodes)}"
    )
    if windows == 0:
        warns.append(f"no training windows: all episodes shorter than span={span}")

    print("--- verdict ---")
    if warns:
        print("UNHEALTHY:")
        for w in warns:
            print(f"  ! {w}")
    else:
        print("HEALTHY: cache passed all checks")
    return not warns


def build_camera_cache(short, plan, rows, trim, encoders, enc_devices, means, stds):
    key = data.CAMERAS[short]
    n_frames = sum(e["length"] - trim for e in plan)
    videos = {
        loc: hub_file(f"{data.DATASET_SPLIT}/videos/{key}/chunk-{loc[0]:03d}/file-{loc[1]:03d}.mp4")
        for loc in sorted({(e["videos"][short]["chunk"], e["videos"][short]["file"]) for e in plan})
    }

    latents = torch.empty(n_frames, 256, 1024, dtype=torch.float16)
    actions = torch.empty(n_frames, 7, dtype=torch.float32)
    states = torch.empty(n_frames, 7, dtype=torch.float32)
    episodes = []
    ng = len(enc_devices)
    pools = [ThreadPoolExecutor(max_workers=1) for _ in enc_devices]
    pending = deque()
    pbar = tqdm(total=n_frames, desc=f"encoding {short}", unit="frame")

    def encode_and_write(frames, offset, gi):
        z = encode_batch(frames, encoders[gi], enc_devices[gi], means[gi], stds[gi])
        latents[offset : offset + z.shape[0]] = z
        pbar.update(z.shape[0])

    off, gi = 0, 0
    for ep in plan:
        n_keep = ep["length"] - trim
        ep_rows = rows[ep["episode"]]
        assert len(ep_rows) == n_keep, (
            f"episode {ep['episode']}: {len(ep_rows)} state rows != {n_keep} kept frames"
        )
        assert ep_rows[0][0] == trim and ep_rows[-1][0] == ep["length"] - 1
        for i, (_, state, action) in enumerate(ep_rows):
            states[off + i] = torch.tensor(state)
            actions[off + i] = torch.tensor(action)
        episodes.append([off, off + n_keep])

        v = ep["videos"][short]
        batch = []
        for frame in decode_segment(
            videos[(v["chunk"], v["file"])], v["from_ts"], ep["length"], trim
        ):
            batch.append(frame)
            if len(batch) == DECODE_BATCH:
                pending.append(
                    pools[gi % ng].submit(encode_and_write, torch.stack(batch), off, gi % ng)
                )
                off += len(batch)
                batch, gi = [], gi + 1
                if len(pending) >= 2 * ng:
                    pending.popleft().result()
        if batch:
            pending.append(
                pools[gi % ng].submit(encode_and_write, torch.stack(batch), off, gi % ng)
            )
            off += len(batch)
            gi += 1
    for fut in pending:
        fut.result()
    for p in pools:
        p.shutdown()
    pbar.close()
    assert off == n_frames

    out_dir = os.path.join(data.CACHE_DIR, short)
    os.makedirs(out_dir, exist_ok=True)
    latents_path, meta_path = data.cache_paths(out_dir)
    save_file({"latents": latents, "actions": actions, "state": states}, latents_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": data.DATASET_ID,
                "split": data.DATASET_SPLIT,
                "camera": short,
                "camera_key": key,
                "trim": trim,
                "action_dim": 7,
                "state_dim": 7,
                "n_frames": n_frames,
                "episodes": episodes,
                "episode_ids": [e["episode"] for e in plan],
                "encoder": data.HF_REPO,
                "img_size": data.IMG_SIZE,
            },
            f,
        )
    print(f"saved {n_frames} frame latents ({len(episodes)} episodes) -> {latents_path}")
    tc = TRAININGS["full"]
    return check_health(latents, actions, episodes, tc.T, tc.stride)


if __name__ == "__main__":
    args = parse_args()
    free_gpus = pick_free_gpus()
    enc_devices = [f"cuda:{g}" for g in free_gpus[:4]] or [get_device()]
    print(f"encoding on: {enc_devices}")

    plan, dropped = load_plan(args.episodes, args.trim)
    print(
        f"{data.DATASET_ID} [{data.DATASET_SPLIT}]: first {args.episodes} episodes, "
        f"trim {args.trim} -> {len(plan)} kept, {dropped} dropped (too short), "
        f"{sum(e['length'] - args.trim for e in plan)} frames per camera"
    )
    rows = load_rows(plan, args.trim)

    m = AutoModel.from_pretrained(data.HF_REPO)
    processor = AutoVideoProcessor.from_pretrained(data.HF_REPO)
    base_enc = m.encoder.eval().requires_grad_(False)
    encoders = [copy.deepcopy(base_enc).to(dev) for dev in enc_devices]
    means = [torch.tensor(processor.image_mean, device=dev).view(1, 3, 1, 1) for dev in enc_devices]
    stds = [torch.tensor(processor.image_std, device=dev).view(1, 3, 1, 1) for dev in enc_devices]
    del m, base_enc

    healthy = True
    for short in args.cameras:
        healthy &= build_camera_cache(
            short, plan, rows, args.trim, encoders, enc_devices, means, stds
        )
    print(
        "now run: uv run scripts/gate_sweep.py"
        if healthy
        else "fix the cache before running the gate"
    )
