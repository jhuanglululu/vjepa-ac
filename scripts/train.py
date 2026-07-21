import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from tqdm.auto import tqdm

from safetensors.torch import load_file

from vjepa_ac import checkpoints, data
from vjepa_ac.cpredictor import CPredictor
from vjepa_ac.device import get_device
from vjepa_ac.predictor import Predictor
from vjepa_ac.records import RecordWriter
from vjepa_ac.schedule import make_scheduler
from vjepa_ac.variations import MODELS, TRAININGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--training", choices=sorted(TRAININGS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--no-rollout", action="store_true")
    p.add_argument("--compressor", default=None)
    return p.parse_args()


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main():
    args = parse_args()
    mc = MODELS[args.model]
    tc = TRAININGS[args.training]
    assert tc.batch_size % tc.grad_accum == 0
    training_name = args.training
    if args.stride is not None and args.stride != tc.stride:
        assert args.stride >= 1
        tc = tc.model_copy(update={"stride": args.stride})
        training_name += f"-s{args.stride}"
    if args.no_rollout:
        tc = tc.model_copy(update={"rollout_loss": False})
        training_name += "-noroll"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    device_type = device.split(":")[0]

    if tc.data == "synthetic":
        cache = data.synthetic_cache(mc, seed=args.seed)
    else:
        cache = data.load_cache()
    assert cache.state_dim == mc.d_action, (
        f"model d_action={mc.d_action} but cache state_dim={cache.state_dim}"
    )

    train_eps, val_eps = data.split_episodes(cache.episodes, tc.val_frac)
    train_starts = data.window_starts(train_eps, tc.T, tc.stride)
    val_starts = data.window_starts(val_eps, tc.T, tc.stride)
    assert len(train_starts) > 0 and len(val_starts) > 0, (
        f"no windows at T={tc.T} stride={tc.stride} "
        f"({len(train_eps)} train / {len(val_eps)} val episodes)"
    )
    gen = torch.Generator().manual_seed(0)
    val_starts = val_starts[torch.randperm(len(val_starts), generator=gen)[: tc.val_windows]]
    print(
        f"{len(train_eps)} train / {len(val_eps)} val episodes | "
        f"{len(train_starts)} train / {len(val_starts)} val windows | "
        f"stride {tc.stride} | rollout_loss {tc.rollout_loss} | device {device}"
    )

    cond = data.fit_conditioner(cache.states, train_eps, tc.stride)

    if mc.compressor:
        lat = cache.latents
        shape = lat.get_shape() if hasattr(lat, "get_shape") else lat.shape
        assert (shape[1], shape[2]) == (mc.comp_patches, mc.comp_d_latent), (
            f"cache latents {tuple(shape[1:])} but model expects "
            f"({mc.comp_patches}, {mc.comp_d_latent})"
        )
        model = CPredictor(mc, tc.T).to(device)
        comp_path = args.compressor or str(
            checkpoints.checkpoint_dir(args.model, f"comp-s{tc.stride}", args.seed)
            / "compressor.safetensors"
        )
        if os.path.exists(comp_path):
            weights = load_file(comp_path)
            model.load_state_dict(weights, strict=False)
            print(f"loaded phase-1 compressor {comp_path}")
        else:
            print(f"WARNING: no compressor checkpoint at {comp_path}, random init")
        with torch.no_grad():
            zs, _ = data.gather(
                cache, cond, train_starts[: min(64, len(train_starts))], tc.T, tc.stride, device
            )
            c = model.compressor(zs)
            model.set_stats(c.mean(dim=(0, 1, 2)), c.std(dim=(0, 1, 2)))
        enc_params = [p for m in (model.compressor, model.id_head) for p in m.parameters()]
        param_groups = [
            {"params": list(model.predictor.parameters())},
            {"params": enc_params, "lr": tc.compressor_lr},
        ]
    else:
        model = Predictor(mc, tc.T).to(device)
        param_groups = [{"params": list(model.parameters())}]
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.AdamW(param_groups, lr=tc.lr, betas=tc.betas, weight_decay=tc.weight_decay)
    sched = make_scheduler(optim, tc.warmup_steps, tc.total_steps)

    ckpt_dir = checkpoints.checkpoint_dir(args.model, training_name, args.seed)
    best_path = ckpt_dir / "best.json"
    run_info = {
        "model": args.model,
        "training": training_name,
        "seed": args.seed,
        "config": {"model": mc.model_dump(), "training": tc.model_dump()},
        "conditioning": cond.stats(),
    }

    start_step = 0
    best: list[tuple[float, int]] = []
    resume = checkpoints.load_checkpoint(ckpt_dir)
    if resume is not None:
        tensors, sidecar = resume
        model_sd, optim_t, rng_t = checkpoints.split_checkpoint_tensors(tensors)
        model.load_state_dict(model_sd)
        optim.load_state_dict(checkpoints.unflatten_optim_state(optim_t, sidecar["param_groups"]))
        sched.load_state_dict(sidecar["sched"])
        checkpoints.restore_rng(rng_t)
        start_step = sidecar["step"]
        if best_path.exists():
            with open(best_path) as f:
                best = [(b["val_loss"], b["step"]) for b in json.load(f)]
        print(f"resuming from step {start_step}")

    record = RecordWriter(args.model, training_name, args.seed)
    record.meta(args.model, training_name, args.seed, run_info["config"])

    def predict(z, a):
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=tc.amp):
            pred = model(z, a)
        return z + pred.float()

    def encode(z):
        assert isinstance(model, CPredictor)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=tc.amp):
            return model.encode(z).float()

    def id_loss(z, a):
        assert isinstance(model, CPredictor)
        return F.mse_loss(model.id_head(z[:, :-1], z[:, 1:]), a[:, :-1])

    @torch.no_grad()
    def validate():
        model.eval()
        tot, id_tot, std_tot, n = 0.0, 0.0, 0.0, 0
        for i in range(0, len(val_starts), tc.batch_size):
            sb = val_starts[i : i + tc.batch_size]
            z, a = data.gather(cache, cond, sb, tc.T, tc.stride, device)
            if mc.compressor:
                z = encode(z)
                id_tot += id_loss(z, a).item() * len(sb)
                std_tot += z.std(dim=(0, 1, 2)).mean().item() * len(sb)
            zhat = predict(z, a)
            tot += F.smooth_l1_loss(zhat[:, :-1], z[:, 1:]).item() * len(sb)
            n += len(sb)
        model.train()
        if mc.compressor:
            pbar.write(
                f"    collapse monitor: token std {std_tot / n:.3f} | val id {id_tot / n:.4f}"
            )
        return tot / n

    def micro_step(sb):
        z, a = data.gather(cache, cond, sb, tc.T, tc.stride, device)
        if mc.compressor:
            z = encode(z)
            target = z.detach()
        else:
            target = z
        zhat = predict(z, a)
        loss = F.smooth_l1_loss(zhat[:, :-1], target[:, 1:])
        if mc.compressor:
            loss = loss + tc.id_weight * id_loss(z, a)

        if tc.rollout_loss:
            s_in = target.clone()
            s_in[:, 1:] = zhat[:, :-1]
            frozen = {k: v.detach() for k, v in model.state_dict().items()}
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=tc.amp):
                pred2 = functional_call(model, frozen, (s_in, a))
            zhat2 = s_in + pred2.float()
            loss = loss + F.smooth_l1_loss(zhat2[:, :-1], target[:, 1:])
        return loss

    def train_step():
        sb = train_starts[torch.randint(0, len(train_starts), (tc.batch_size,))]
        optim.zero_grad(set_to_none=True)
        micro = tc.batch_size // tc.grad_accum
        loss_sum = 0.0
        for i in range(tc.grad_accum):
            loss = micro_step(sb[i * micro : (i + 1) * micro]) / tc.grad_accum
            loss.backward()
            loss_sum += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optim.step()
        sched.step()
        return loss_sum, float(grad_norm)

    model.train()
    t_start = time.monotonic()
    pbar = tqdm(
        total=tc.total_steps,
        initial=start_step,
        desc=f"{args.model}/{training_name}",
        unit="step",
    )
    for step in range(start_step + 1, tc.total_steps + 1):
        t0 = time.monotonic()
        loss, grad_norm = train_step()
        sec_per_step = time.monotonic() - t0
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        if step % tc.log_interval == 0:
            record.step(step, loss, float(sched.get_last_lr()[0]), grad_norm, sec_per_step)

        if step % tc.val_interval == 0:
            vl = validate()
            record.eval(step, vl, loss)
            checkpoints.save_checkpoint(
                ckpt_dir, step, model, optim, sched.state_dict(), vl, run_info
            )
            best.append((vl, step))
            best = checkpoints.prune_checkpoints(ckpt_dir, best, tc.keep_ckpts)
            with open(best_path, "w") as f:
                json.dump([{"val_loss": v, "step": s} for v, s in best], f, indent=2)
            pbar.write(
                f"step {step:>6}/{tc.total_steps} | {fmt_elapsed(time.monotonic() - t_start)} | "
                f"loss {loss:7.4f} | val {vl:7.4f} | diff {vl - loss:+8.4f} | "
                f"best {best[0][0]:.4f}@{best[0][1]}"
            )
    pbar.close()
    record.close()

    print("best checkpoints:")
    for v, s in best:
        print(f"  val {v:.4f} | step {s} | {ckpt_dir / f'{s}.safetensors'}")


if __name__ == "__main__":
    main()
