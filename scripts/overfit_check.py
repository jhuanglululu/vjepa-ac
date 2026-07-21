import argparse

import torch
import torch.nn.functional as F

from vjepa_ac import data
from vjepa_ac.device import get_device
from vjepa_ac.predictor import Predictor
from vjepa_ac.schedule import make_scheduler
from vjepa_ac.variations import MODELS, TRAININGS


MODEL = "base"
WINDOWS = 512
STEPS = 3000
WARMUP = 50
LR = 3e-4
EVAL_EVERY = 500
EVAL_WINDOWS = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    mc = MODELS[MODEL]
    tc = TRAININGS["full"]
    device = get_device()
    device_type = device.split(":")[0]
    amp = tc.amp and device_type == "cuda"

    cache = data.load_cache()
    assert cache.state_dim == mc.d_action
    train_eps, _ = data.split_episodes(cache.episodes, tc.val_frac)
    starts_all = data.window_starts(train_eps, tc.T, args.stride)
    g = torch.Generator().manual_seed(args.seed)
    n_sub = min(WINDOWS, len(starts_all))
    sub = starts_all[torch.randperm(len(starts_all), generator=g)[:n_sub]]
    eval_starts = sub[: min(EVAL_WINDOWS, n_sub)]
    cond = data.fit_conditioner(cache.states, train_eps, args.stride)
    micro = max(1, tc.batch_size // tc.grad_accum)
    print(
        f"{n_sub} fixed train windows | stride {args.stride} | {STEPS} steps | "
        f"batch {tc.batch_size} (micro {micro}) | eval on {len(eval_starts)} windows | {device}"
    )

    def batch(z_starts, a_starts, model):
        z, _ = data.gather(cache, cond, z_starts, tc.T, args.stride, device)
        a = cond.windows(a_starts, tc.T, args.stride).to(device)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
            pred = model(z, a)
        zhat = z + pred.float()
        return F.smooth_l1_loss(zhat[:, :-1], z[:, 1:])

    @torch.no_grad()
    def eval_loss(model, mode):
        tot, n = 0.0, 0
        for i in range(0, len(eval_starts), micro):
            zs = eval_starts[i : i + micro]
            if mode == "true":
                loss = batch(zs, zs, model)
            elif mode == "shuf":
                js = eval_starts[(torch.arange(i, i + len(zs)) + 1) % len(eval_starts)]
                loss = batch(zs, js, model)
            else:
                z, _ = data.gather(cache, cond, zs, tc.T, args.stride, device)
                a = torch.zeros(len(zs), tc.T, cache.state_dim, device=device)
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
                    pred = model(z, a)
                loss = F.smooth_l1_loss((z + pred.float())[:, :-1], z[:, 1:])
            tot += loss.item() * len(zs)
            n += len(zs)
        return tot / n

    copy_tot = 0.0
    for i in range(0, len(eval_starts), micro):
        z, _ = data.gather(cache, cond, eval_starts[i : i + micro], tc.T, args.stride, device)
        copy_tot += F.smooth_l1_loss(z[:, :-1], z[:, 1:]).item() * z.shape[0]
    copy_loss = copy_tot / len(eval_starts)
    print(f"copy baseline on eval windows: {copy_loss:.4f}\n")

    roll = torch.roll(torch.arange(n_sub), 1)

    def run_variant(name, shuffled):
        torch.manual_seed(args.seed)
        model = Predictor(mc, tc.T).to(device)
        optim = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=tc.betas, weight_decay=tc.weight_decay
        )
        sched = make_scheduler(optim, WARMUP, STEPS)
        samp = torch.Generator().manual_seed(args.seed + 1)
        model.train()
        history = []
        for step in range(1, STEPS + 1):
            idx = torch.randint(0, n_sub, (tc.batch_size,), generator=samp)
            optim.zero_grad(set_to_none=True)
            for i in range(0, tc.batch_size, micro):
                m = idx[i : i + micro]
                am = roll[m] if shuffled else m
                loss = batch(sub[m], sub[am], model) * len(m) / tc.batch_size
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optim.step()
            sched.step()
            if step % EVAL_EVERY == 0 or step == STEPS:
                model.eval()
                lt = eval_loss(model, "true")
                ls = eval_loss(model, "shuf")
                lz = eval_loss(model, "zero")
                model.train()
                history.append((step, lt, ls, lz))
                print(
                    f"[{name}] step {step:>5} | true {lt:.4f} (copy ratio {lt / copy_loss:.3f}) | "
                    f"shuf {ls:.4f} ({(ls - lt) / lt * 100:+.1f}%) | "
                    f"zero {lz:.4f} ({(lz - lt) / lt * 100:+.1f}%)"
                )
        return history

    hist_a = run_variant("true-actions", shuffled=False)
    hist_b = run_variant("shuf-actions", shuffled=True)

    a_final, b_final = hist_a[-1], hist_b[-1]
    gap = (b_final[1] - a_final[1]) / a_final[1] * 100
    sens = (a_final[2] - a_final[1]) / a_final[1] * 100
    ratio = a_final[1] / copy_loss
    print("\n=== verdict ===")
    print(
        f"true-actions model: loss {a_final[1]:.4f} (copy ratio {ratio:.3f}) | "
        f"eval-time shuffle sensitivity {sens:+.1f}%"
    )
    print(f"shuffle-trained control: loss {b_final[1]:.4f} | A/B gap {gap:+.1f}%")
    if gap >= 5 or sens >= 5:
        print(
            "actions ARE usable by this architecture under memorization pressure -> the "
            "full-run blindness is an optimization/scale problem (longer training, stronger "
            "conditioning pressure), not a structural one"
        )
    elif ratio < 0.8:
        print(
            "model fits the subset well yet gains nothing from actions -> structural: the "
            "objective/architecture does not reward routing action information; strengthen "
            "the injection (action added to every patch token) or add an auxiliary "
            "action-prediction loss before spending more GPU-hours"
        )
    else:
        print(
            "model never fit the subset (copy ratio >= 0.8) -- raise STEPS or LR at the "
            "top of this script before reading anything into the action numbers"
        )


if __name__ == "__main__":
    main()
