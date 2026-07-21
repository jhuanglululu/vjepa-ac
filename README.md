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
uv sync --extra cache   # remote: adds transformers + lerobot for cache building
```

Optional env vars (all paths, with defaults): `VJEPA_CACHE_DIR`
(`./latent_cache`), `VJEPA_CKPT_DIR` (`./checkpoints`), `VJEPA_RECORDS_DIR`
(`./records`).

Local machine can run: pytest, ruff/pyrefly, `--training smoke`.
Remote box (GPU + cache) is needed for: prepare_cache, check_actions,
gate_sweep/stride_gate, ceiling_probe/overfit_check, real training,
evaluate, plan_demo.

## Usage

```
uv run pytest                                                  # 1. local: unit tests
uv run scripts/train.py --model tiny --training smoke          # 2. local: 50-step sanity check
uv run scripts/prepare_cache.py --episodes 100 --trim 15       # 3. remote: per-camera latent caches
uv run scripts/check_actions.py --cache-dir latent_cache/wrist # 4. remote: confirm action/state semantics
uv run scripts/gate_sweep.py --seeds 1                         # 5. remote: stride gate on every camera, one free GPU each
uv run scripts/stride_gate.py --cache-dir latent_cache/wrist   # 5b. single-camera gate (full 3 seeds)
uv run scripts/ceiling_probe.py --stride 6                     # 6. remote: action-attributable share of latent deltas
uv run scripts/train_compressor.py --stride 6                  # 7. remote: phase-1 compressor + gates
uv run scripts/train.py --model base-c16 --training c-full --seed 0 --no-rollout  # 8. remote: compressed-space training
uv run scripts/evaluate.py --checkpoint checkpoints/base-c16/c-full-noroll/0/current.safetensors  # 9.
uv run scripts/overfit_check.py --stride 6                     # optional: action-use A/B diagnostic
uv run scripts/plan_demo.py --checkpoint checkpoints/base-c16/c-full/0/current.safetensors  # MPC demo + gif
```

For compressed-space models (`*-c*`) train.py loads the phase-1 compressor
from `checkpoints/<model>/comp-s<stride>/<seed>/compressor.safetensors`
(override with `--compressor`), fine-tunes it at `compressor_lr` with
stop-grad targets and the inverse-dynamics auxiliary (`id_weight`), and logs
a collapse monitor (val token std + ID loss) at every val interval. The
phase-2 checkpoint bundles compressor and predictor, so evaluate.py works on
it unchanged. train_compressor.py, stride_gate.py, and overfit_check.py take
only `--stride`/`--seed`(`--seeds`) — their remaining knobs are constants at
the top of each script.

prepare_cache.py downloads only the shards covering the requested episodes
and writes one cache per camera under `latent_cache/<cam>/`; every consumer
(train, evaluate, probe, check_actions, stride_gate) picks the camera via
`--cache-dir` or the `VJEPA_CACHE_DIR` env var. gate_sweep.py launches one
stride_gate per camera, each pinned to a free GPU (at most `--max-gpus 4`),
and prints the combined verdict table when all finish.

Run stride_gate.py before training: per stride and seed it trains two 23M
probes (per-patch MLP blocks with no cross-patch mixing, then multihead
cross-attention with one learned query pooling the 256 patches into a single
vector, then MLP blocks and a projection out) — one on latent pairs
(z_t, z_{t+s}) and one z0-only control with the second frame ablated — to
recover the exact conditioning features. The decision statistic is the
pair-minus-control margin on motion dims (gripper excluded): information
redundant with z_t is useless as conditioning, so only the margin counts.
Scores are reported with errors combining a bootstrap over test episodes and
seed spread; checkpoints are selected on held-out episodes disjoint from the
reported test episodes; a stride passes when pair R2 − SE clears --threshold
and margin − SE clears --margin. Fails are split into conclusive (probe fit
its training set) vs probe-limited (train R2 < 0.5, inconclusive), and the
recommendation checks that training windows actually exist at T=16 for the
chosen stride. `--stride N` on train.py overrides the variation's stride and
records under `<training>-s<N>`; `--no-rollout` drops the two-pass rollout
loss term and records under `<training>-noroll` (suffixes combine).
evaluate.py reads the model and training config plus the conditioning stats
from the checkpoint's JSON sidecar, so it takes only the checkpoint path;
`--horizons` are counted in strided steps.

Training resumes from `current.safetensors` automatically if one exists in the
run's checkpoint directory; delete the directory to start fresh.

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
