# vjepa-ac

Learning-purpose reproduction of the V-JEPA 2-AC predictor from Meta's V-JEPA 2
paper: a small block-causal transformer trained to predict the next frame's
latents from past latents and executed motion, on top of a frozen
`facebook/vjepa2-vitl-fpc64-256` encoder over the first 100 episodes of
`nvidia/Cosmos3-DROID` (success split, 640x360 @ 15 fps; the intentional bias
toward one task family is deliberate — similar scenes make cross-episode
generalization feasible at this data scale). The first `--trim` frames of each
episode are dropped (the arm often starts outside the camera), and a cache is
built per camera (`ext1`/`ext2`/`wrist` subdirs). Frames are sampled at a
temporal stride so per-step latent change clears the encoder noise floor; the
action token for each position is the wrap-corrected proprio-state delta over
that strided interval (absolute value for the gripper dim), normalized with
train-split stats that travel in the checkpoint sidecar. Training touches only
the cache, never the encoder. Torch-only: real runs happen on the remote L40S
box, while tests and the smoke variation run locally on CPU.

## Setup

```
uv sync                 # local: tests + smoke runs (CPU is fine)
uv sync --extra cache   # remote: adds transformers/pyarrow/av/pillow for cache building + gifs
```

Optional env vars (all paths, with defaults): `VJEPA_CACHE_DIR`
(`./latent_cache`), `VJEPA_CKPT_DIR` (`./checkpoints`), `VJEPA_RECORDS_DIR`
(`./records`).

Local machine can run: pytest, ruff/pyrefly, `--training smoke`.
Remote box (GPU + cache) is needed for: prepare_cache, check_actions,
gate_sweep/stride_gate, ceiling_probe/overfit_check, real training,
evaluate, plan_demo.

## Scripts, in running order

Before anything remote: `uv run pytest` (unit tests) and
`uv run scripts/train.py --model tiny --training smoke` (50-step CPU sanity
check on synthetic data) should both pass locally. Scripts whose knobs you
never touch expose only `--stride`/`--seed`; everything else is a constant at
the top of the script.

### 1. prepare_cache.py — build the per-camera latent caches

```
uv run scripts/prepare_cache.py --episodes 100 --trim 15
```

Downloads only the Cosmos3-DROID shards covering the first `--episodes`
episodes, drops the first `--trim` frames of each (arm out of view), decodes
each camera's video, and encodes every frame with the frozen V-JEPA encoder
on up to 4 free GPUs. Writes one cache per camera to `latent_cache/<cam>/`
(`--cameras ext1 ext2 wrist`) with states, actions, and episode ranges, then
prints a health check. Every later script picks its camera via `--cache-dir`
or `VJEPA_CACHE_DIR` (default `./latent_cache`, so after choosing a camera
you can move that cache to the root and drop the flag).

### 2. check_actions.py — confirm state/action semantics

```
uv run scripts/check_actions.py --cache-dir latent_cache/wrist
```

Prints per-dim correlations between commanded actions, wrap-corrected state
deltas, and absolute states from the cache. Confirm before training: dims
0-5 behave like cartesian velocity commands (corr(a,dS) clearly positive)
and the last dim is the absolute gripper (corr(a,s) ~ +1).

### 3. gate_sweep.py — pick the camera and stride

```
uv run scripts/gate_sweep.py --seeds 1     # quick pass, all cameras
uv run scripts/stride_gate.py --cache-dir latent_cache/ext1 --strides 4 6   # confirm winner, 3 seeds
```

gate_sweep launches one stride_gate per camera, each pinned to a free GPU
(at most `--max-gpus 4`), and prints a combined verdict table. stride_gate
itself trains, per stride and seed, two 23M probes to recover the exact
conditioning features — one from latent pairs (z_t, z_{t+s}), one z0-only
control — because information redundant with z_t is useless as conditioning:
the decision statistic is the pair-minus-control margin on motion dims.
Errors combine a bootstrap over held-out test episodes with seed spread, and
a stride passes only when both pair R2 − SE and margin − SE clear their
thresholds. Fails are labeled conclusive vs probe-limited (train R2 < 0.5),
and the verdict checks that training windows exist at the chosen stride.
JSON lands in `records/diagnostics/`.

### 4. ceiling_probe.py — size the prize before training

```
uv run scripts/ceiling_probe.py --stride 6
```

Ridge-regresses raw latent deltas from the conditioning features and reports
the held-out, scene-independent action-attributable share of the training
loss. On these caches it reads ~0%: actions explain none of the raw latent
delta without action-x-content interactions — the measurement that motivated
predicting in a learned compressed space instead of raw latents.

### 5. train_compressor.py — phase 1: learn the prediction space

```
uv run scripts/train_compressor.py --stride 6
```

Trains the base-c16 compressor (16 learned queries cross-attending over the
256 patches) plus an inverse-dynamics head, with a light reconstruction term
so the tokens keep forecastable context. Selects the best checkpoint on
held-out ID motion R2, saves it under
`checkpoints/base-c16/comp-s<stride>/<seed>/compressor.safetensors`, and
prints two go/no-go gates: held-out ID R2 >= 0.2 (the compressor found the
motion signal) and C-space linear ceiling >= +2% (the token space is
action-driven, unlike raw latents). Do not spend phase-2 GPU time if either
fails.

### 6. train.py — phase 2: train the predictor

```
uv run scripts/train.py --model base-c16 --training c-full --seed 0
```

For compressed models it auto-loads the phase-1 compressor (override with
`--compressor`) and fine-tunes it at `compressor_lr` with three collapse
guards: stop-grad targets, the inverse-dynamics auxiliary (`id_weight`), and
a monitor printing val token std + ID loss each val interval (falling std or
rising ID loss means the compressor is cheating — lower `compressor_lr`).
`--stride N` overrides the variation's stride (records under
`<training>-s<N>`); `--no-rollout` drops the two-pass rollout loss
(`<training>-noroll`; at stride 6 the rollout loss measurably helps, so the
default keeps it). Resumes automatically from `current.safetensors` if the
run directory exists; checkpoints bundle compressor + predictor.

### 7. evaluate.py — action sensitivity and rollout quality

```
uv run scripts/evaluate.py            # defaults to weights/model.safetensors
```

Reads config + conditioning stats from the checkpoint sidecar, rolls the
model out on held-out episodes, and prints latent L1 against copy-first /
zero-action / shuffled-action baselines per horizon plus within-episode
frame retrieval. Adoption criterion: shuffled >= +10% worse at max horizon
and model/copy <= 0.9. The shipped weights score shuffled +50-65% and
model/copy ~0.67-0.73 at h=15, with retrieval tracking the true frame
(median offset ~0 vs copy's -h*stride).

### 8. plan_demo.py — receding-horizon MPC demo

```
uv run scripts/plan_demo.py           # same default weights
```

Picks a held-out episode, takes the frame at `--start` as current state and
`--goal-offset` frames ahead (negative works) as the goal image, then loops:
CEM (re-initialized N(0,1) each step) scores imagined token rollouts by L1
to the goal tokens, only the first `--commit-steps` actions execute, and
execution is model-independent kinematics — commanded wrap-aware dstate
applied to the real proprio state, snapped to the nearest-state episode
frame. The context holds only real frames, with executed (not commanded)
motion between them. Prints required vs commanded vs executed motion per
step to separate planner error from simulator error; `--snap-range LO HI`
restricts snapping to a frame window when pose aliasing teleports across
task phases (same arm pose, different world state). Saves a side-by-side
gif of committed frames vs the goal next to the checkpoint.

### overfit_check.py — optional action-use diagnostic

```
uv run scripts/overfit_check.py --stride 6
```

Trains twin raw-space models on a fixed 512-window subset — true actions vs
permanently shuffled — and reports the A/B loss gap plus eval-time shuffle
sensitivity. Separates "optimization/scale problem" (sensitivity grows with
training) from "structural blindness" (flat at zero) when a model ignores
its actions.

## Variations

**Model**
- `tiny` — smoke runs and shape checks on a small synthetic grid, never real results
- `tiny-c` — tiny compressed-space twin for exercising the compressor path locally
- `base` — the paper-scale predictor for the vjepa2-vitl 16x16x1024 latent grid
  (kept as the documented raw-latent negative baseline)
- `base-c16` — compressor (16 learned queries over the 256 patches, trained on inverse
  dynamics + light reconstruction, then fine-tuned at `compressor_lr`) + predictor
  operating entirely in the 16x384 token space; needs a phase-1 checkpoint from
  train_compressor.py

**Training**
- `smoke` — 50-step local sanity check on synthetic linear-dynamics data, stride 2
- `full` — 3k-step recipe on the real latent cache (raw 4112-token sequences)
- `c-full` — 10k-step stride-6 recipe for compressed-space models (272-token
  sequences are ~15x cheaper per step); `compressor_lr` and `id_weight` live here

Purpose only — the numbers live in `src/vjepa_ac/variations.py`.
New variation = new entry there and a line here, in the same change.

## Layout

- `records/<model>/<training>/<seed>/record.jsonl` — meta line + per-step/eval metrics
- `checkpoints/<model>/<training>/<seed>/` — `<step>.safetensors` (+ `<step>.json`
  sidecar), `current.*` for resume, 3 best by val loss kept
- `weights/` — `model.safetensors` + `model.json`: the trained seed-0 `base-c16`/`c-full`
  model (weights + sidecar, committed to the repo); evaluate.py and plan_demo.py
  default to it, so both run without any checkpoint path
- `latent_cache/<cam>/` — `latents.safetensors` + `cache.json` per camera from prepare_cache
- `records/diagnostics/` — stride_gate/gate_sweep output

## Notes

- Train/val split is episode-level (`data.split_episodes`, seed 0), shared by
  train.py, evaluate.py, stride_gate.py, and the diagnostics; a smaller
  val_frac holds out a subset of a larger one's episodes, so probes and
  training agree on what is unseen.
- Checkpoints must use the `model.`-prefixed tensor layout and carry a JSON
  sidecar; pre-restructure checkpoints no longer load, and caches without
  states must be rebuilt with prepare_cache.py.
