# vjepa-ac

Learning-purpose reproduction of the V-JEPA 2-AC action-conditioned world
model from Meta's V-JEPA 2 paper, scaled down to what 100 robot episodes and
a handful of GPU-hours can support: a frozen V-JEPA encoder, a small
transformer predictor conditioned on executed motion, and goal-image MPC
planning with CEM. The interesting part is what the downscaling forced — the
model predicts in a learned, motion-weighted 16-token space instead of raw
patch latents, a deviation each step of which was motivated by a measurement
(see below). Data: first 100 episodes of `nvidia/Cosmos3-DROID` (success
split, one task family by intent — similar scenes make cross-episode
generalization feasible at this scale). Real runs happen on a remote GPU box;
tests and the smoke variation run locally on CPU.

## How and why this differs from Facebook's

V-JEPA 2-AC trains a ~300M block-causal predictor to output the **full patch
latents** of the next frame, on ~62 hours of DROID, over a frozen ViT-g
encoder. At that scale the model can afford to learn action-x-content
interactions from raw latents. We have ~45 minutes of robot data and a
ViT-L/256 encoder, and the direct reproduction failed in a specific,
measurable way:

- Raw-latent training was **action-blind**: shuffling actions at eval cost
  +0.2%, zero-actions matched the model, and rollouts never left the input
  frame (retrieval offset exactly -h*stride). The rollout loss did not help.
- Yet the information exists: a 23M probe decodes motion from latent pairs
  at R2 ~0.37 (stride_gate), so the failure is not the encoder.
- The reverse direction is the killer: ridge regression from actions to raw
  latent deltas explains **0.00%** of their energy (ceiling_probe). There is
  no scene-independent "moving right" direction in V-JEPA latent space —
  action use must go through content-dependent interactions, and ~99.9% of
  the training loss is content/noise the actions can never explain. An
  overfit A/B test showed the action pathway opening at a crawl (0.7 -> 5%
  sensitivity over 3000 steps on a memorized subset) — an optimization
  problem our data budget cannot buy out of.

The fix is to change the objective rather than the scale: **learn a small
token space where motion is a first-class share of the variance, and predict
there.** A frozen-then-fine-tuned compressor (16 cross-attention queries,
trained on inverse dynamics + light reconstruction) defines the space; the
predictor trains entirely in it. Same paper recipe otherwise — frozen
encoder, block-causal predictor, Δstate conditioning, CEM planning — but the
prediction target is 16x384 tokens instead of 256x1024 latents. Result:
shuffled-actions +50-65% worse at h=15 (vs +0.2% raw), model/copy 0.67-0.73,
rollouts that track the true trajectory. Secondary deviations, all
measurement-driven: frames sampled at temporal stride 6 (per-step motion at
15 Hz sits below the encoder noise floor; probed in stride_gate), and
conditioning on wrap-corrected **executed** Δstate rather than commanded
actions.

## Architecture

Frozen encoder -> compressor -> block-causal predictor, ~24M trained params:

- **Encoder** (frozen, cache-time only): `facebook/vjepa2-vitl-fpc64-256`
  maps each 256x256 frame to 256 patches x 1024 dims. Frames are encoded
  once into the latent cache; training never touches the encoder.
- **Compressor C** (~7M): linear 1024->384 + per-patch MLP block, then 16
  learned queries cross-attend over the 256 patches (8 heads), then a
  per-token MLP block -> 16 tokens x 384 dims per frame, standardized with
  train-split stats stored as model buffers. An **inverse-dynamics head**
  (train-time only) predicts the conditioning features from consecutive
  token pairs — it is what forces motion into the tokens.
- **Predictor** (~17M): block-causal transformer (d_model 512, 6 layers, 16
  heads, RoPE, SiLU MLPs). Each frame contributes its 16 tokens plus one
  action token (a linear embedding of the 7-dim conditioning), so a T=16
  window is a 272-token sequence; the block-causal mask lets frame t's
  positions see all frames <= t including t's action token. The output head
  predicts residual token deltas: z_{t+1} = z_t + f(z_<=t, a_<=t).
- **Conditioning**: per strided interval, wrap-corrected proprio Δstate on
  dims 0-5 summed over the interval, absolute gripper on dim 6, normalized
  with train-episode stats that travel in the checkpoint sidecar.

## Training process

Two phases, gated so GPU time is only spent downstream of a positive
measurement:

1. **Compressor first** (`train_compressor.py`, ~minutes): train C + the ID
   head on stride-6 latent pairs — loss = inverse dynamics MSE + 0.1 x
   reconstruction (a throwaway cross-attention decoder regressing the input
   patches, which keeps enough static context in the tokens for forecasting).
   Best checkpoint by held-out ID motion R2. Two gates must pass before
   phase 2: ID R2 >= 0.2, and a C-space ridge ceiling >= +2% (the analogue
   of the raw-space 0.00% measurement).
2. **Predictor second** (`train.py`, ~30 min for 10k steps): the phase-1
   compressor is loaded and fine-tunes with the predictor at a much smaller
   lr (`compressor_lr`, ~lr/10). Loss = teacher-forced smooth-L1 on next
   tokens + a two-pass rollout term (predictions fed back once) + the ID
   auxiliary. Unfreezing the compressor reopens the collapse pathway, so
   three guards hold it: prediction targets are stop-grad, the ID auxiliary
   keeps motion linearly readable from the tokens, and a collapse monitor
   logs val token std + ID loss every val interval. Both modules ship in one
   checkpoint.

Training is window-based: T=16 frames at stride 6 (91-frame span), episode-
level train/val split shared by every script, batch 64 with grad accumulation.

## Demo process

`plan_demo.py` runs the paper's goal-image planning loop without a robot, so
the recorded episode has to stand in for the environment. The design rule
throughout: the planner may look at the goal, but **execution may not
reference the goal or the direction of time** — otherwise the demo
manufactures its own success (a time-forward snap on a recorded trajectory
reaches any future goal by construction).

The loop, per committed step:

1. **Plan** (paper-faithful CEM): sample action sequences over an 8-step
   horizon from Gaussians re-initialized at N(0,1) each replan, roll them
   through the world model from the real context, score by L1 between the
   imagined final tokens and the goal frame's tokens, refit on the elites,
   take the mean.
2. **Execute** only the first `--commit-steps` actions, then **snap**: add
   the commanded wrap-aware Δstate to the current frame's real proprio state
   and commit the episode frame whose state is nearest (per-dim scaled,
   angle-wrapped). This is the actuator; the world model is not consulted.
3. **Re-ground**: the committed frame's real latent joins the context (last
   4 real frames), with the **executed** motion between committed frames —
   not the commanded action — as the context's action rows. Replan.

What each mechanism is for:

- **State snapping** solves the actuator problem. The model's own one-step
  imagination is ~5x too conservative (smooth-L1 regression to the mean), so
  using it as the environment stalls and, worse, tests the model against
  itself. Kinematics on ground-truth proprio is model-independent, works in
  any direction (backwards commits stay possible, so failure stays visible),
  and mirrors a position-controlled robot tracking Δstate commands.
- **`--commit-steps`** solves the granularity problem: one strided step's
  displacement can be smaller than the state gap between adjacent frames,
  which snaps back to the same frame and stalls; executing 2-3 planned
  actions before snapping clears the spacing.
- **`--snap-range LO HI`** solves pose aliasing. Proprio state does not see
  the world, and a pour episode passes through nearly the same arm pose with
  the cup on the table and again with the cup in the gripper — nearest-state
  snapping happily teleports between those task phases. Windowing the snap
  pool to a frame range excludes other phases; it mildly references the
  goal's location in time, which is why it is a flag and not the default.
- **`--action-momentum`** solves dithering. Paper-faithful CEM restarts from
  zero mean every replan, so successive plans can reverse each other;
  warm-starting the mean from the last **executed** action biases toward
  continuing real motion. It references only the past — never the goal or
  time direction — which is what keeps it honest (index-space momentum
  would not be).
- **Executed-vs-commanded bookkeeping** separates failure modes: the header
  prints the required start->goal motion, and every step prints commanded vs
  executed Δstate. Commanded agreeing with required while executed diverges
  means the simulator (no frame along the path); commanded disagreeing with
  required means the planner.

Output: a per-step trace, a JSON of committed frames and actions, and a
side-by-side gif (committed real frames vs the goal image) next to the
checkpoint.

## Setup

```
uv sync                 # local: tests + smoke runs (CPU is fine)
uv sync --extra cache   # remote: adds transformers/pyarrow/av/pillow for cache building + gifs
```

Optional env vars (all paths, with defaults): `VJEPA_CACHE_DIR`
(`./latent_cache`), `VJEPA_CKPT_DIR` (`./checkpoints`), `VJEPA_RECORDS_DIR`
(`./records`).

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
