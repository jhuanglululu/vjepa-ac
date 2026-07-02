# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "safetensors",
#     "tqdm",
# ]
# ///
import vjepa_common as C

import os, json

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.func import functional_call
from safetensors import safe_open
from transformers import get_cosine_schedule_with_warmup

from predictor import Predictor

train_device = C.train_device
T = C.T
tc = C.training_config
batch_size = tc["batch_size"]
total_steps = tc["total_steps"]
LATENTS_PATH, CACHE_META, OUTPUT_DIR = C.LATENTS_PATH, C.CACHE_META, C.OUTPUT_DIR
atomic_save = C.atomic_save
TRAIN_STATE_PATH = os.path.join(OUTPUT_DIR, "train_state.pt")
LAST_CKPT_PATH = os.path.join(OUTPUT_DIR, "predictor_last.safetensors")


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

assert os.path.exists(LATENTS_PATH), "no latent cache -- run prepare_cache.py first"
meta = json.load(open(CACHE_META))
action_dim = meta["action_dim"]

predictor_config = dict(C.predictor_config)
predictor_config["d_action"] = action_dim

cache = safe_open(LATENTS_PATH, framework="pt", device="cpu")
latents = cache.get_slice("latents")
actions = cache.get_tensor("actions")

starts = []
for a, b in meta["episodes"]:
    starts.extend(range(a, b - T + 1))
starts = torch.tensor(starts, dtype=torch.long)
gen = torch.Generator().manual_seed(0)
perm = torch.randperm(len(starts), generator=gen)
val_n = min(tc["val_windows"], len(starts) // 10)
val_starts = starts[perm[:val_n]]
train_starts = starts[perm[val_n:]]
print(f"{len(train_starts)} train / {len(val_starts)} val windows")

_arangeT = torch.arange(T)


def gather(sb):
    z = torch.stack([latents[int(s) : int(s) + T] for s in sb]).to(train_device).float()
    flat = (sb[:, None] + _arangeT).reshape(-1)
    a = actions[flat].reshape(len(sb), T, action_dim).to(train_device).float()
    return z, a


model = Predictor(predictor_config).to(train_device)
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

optim = torch.optim.AdamW(
    model.parameters(),
    lr=tc["lr"],
    betas=tc["betas"],
    weight_decay=tc["weight_decay"],
)
sched = get_cosine_schedule_with_warmup(optim, tc["warmup_steps"], total_steps)


def state_cpu():
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}


start_step = 0
best = []
if os.path.exists(TRAIN_STATE_PATH) and os.path.exists(LAST_CKPT_PATH):
    with safe_open(LAST_CKPT_PATH, framework="pt", device="cpu") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
    ts = torch.load(TRAIN_STATE_PATH, map_location=train_device)
    optim.load_state_dict(ts["optim"])
    sched.load_state_dict(ts["sched"])
    torch.set_rng_state(ts["rng"])
    start_step = ts["step"]
    best = ts["best"]
    print(f"resuming from step {start_step}")


def predict(z, a):
    states = [z[:, t] for t in range(T)]
    acts = [a[:, t] for t in range(T)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(states, acts)
    return z + pred.float(), acts


@torch.no_grad()
def validate():
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val_starts), batch_size):
        sb = val_starts[i : i + batch_size]
        z, a = gather(sb)
        zhat, _ = predict(z, a)
        tot += F.smooth_l1_loss(zhat[:, :-1], z[:, 1:]).item() * len(sb)
        n += len(sb)
    model.train()
    return tot / n


def train_step():
    sb = train_starts[torch.randint(0, len(train_starts), (batch_size,))]
    z, a = gather(sb)
    optim.zero_grad(set_to_none=True)

    zhat, acts = predict(z, a)
    loss_cur = F.smooth_l1_loss(zhat[:, :-1], z[:, 1:])

    s_in = z.clone()
    s_in[:, 1:] = zhat[:, :-1]
    states2 = [s_in[:, t] for t in range(T)]
    frozen = {k: v.detach() for k, v in model.state_dict().items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred2 = functional_call(model, frozen, (states2, acts))
    zhat2 = s_in + pred2.float()
    loss_roll = F.smooth_l1_loss(zhat2[:, :-1], z[:, 1:])

    loss = loss_cur + loss_roll
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
    optim.step()
    sched.step()
    return loss


model.train()
pbar = tqdm(total=total_steps, initial=start_step, desc="training", unit="step")
for step in range(start_step + 1, total_steps + 1):
    loss = train_step()
    pbar.update(1)
    pbar.set_postfix(
        loss=f"{loss.item():.4f}",
        lr=f"{sched.get_last_lr()[0]:.2e}",
    )

    if step % tc["val_interval"] == 0:
        vl = validate()
        ckpt = os.path.join(OUTPUT_DIR, f"predictor_step{step}.safetensors")
        atomic_save(state_cpu(), ckpt)
        best.append((vl, step, ckpt))
        best.sort(key=lambda x: x[0])
        for _, _, p in best[tc["keep_ckpts"] :]:
            if os.path.exists(p):
                os.remove(p)
        best = best[: tc["keep_ckpts"]]
        atomic_save(state_cpu(), LAST_CKPT_PATH)
        atomic_torch_save(
            {
                "step": step,
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "rng": torch.get_rng_state(),
                "best": best,
            },
            TRAIN_STATE_PATH,
        )
        with open(os.path.join(OUTPUT_DIR, "best.json"), "w") as f:
            json.dump([{"val_loss": v, "step": s} for v, s, _ in best], f, indent=2)
        pbar.write(f"step {step:6d} | val_loss {vl:.4f} | best {best[0][0]:.4f}@{best[0][1]}")
pbar.close()

atomic_save(state_cpu(), os.path.join(OUTPUT_DIR, "predictor_final.safetensors"))
print("best checkpoints:")
for v, s, _ in best:
    print(f"  val {v:.4f} | step {s}")
