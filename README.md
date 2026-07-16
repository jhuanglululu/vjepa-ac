# vjepa-ac

Learning-purpose reproduction of the V-JEPA 2-AC predictor from Meta's V-JEPA 2
paper: a small block-causal transformer trained to predict the next frame's
latents from past latents and executed motion, on top of a frozen
`facebook/vjepa2-vitl-fpc64-256` encoder over `lerobot/droid_100`
(wrist camera). Frames are sampled at a temporal stride so per-step latent
change clears the encoder noise floor; the action token for each position is
the wrap-corrected proprio-state delta over that strided interval (absolute
value for the gripper dim), normalized with train-split stats that travel in
the checkpoint sidecar. Frame latents are pre-encoded once into a cache;
training touches only the cache, never the encoder. Torch-only: real runs happen on
the remote L40S box, while tests and the smoke variation run locally on CPU.

## Setup

```
uv sync                 # local: tests + smoke runs (CPU is fine)
uv sync --extra cache   # remote: adds transformers + lerobot for cache building
```

Optional env vars (all paths, with defaults): `VJEPA_CACHE_DIR`
(`./latent_cache`), `VJEPA_CKPT_DIR` (`./checkpoints`), `VJEPA_RECORDS_DIR`
(`./records`).

Local machine can run: pytest, ruff/pyrefly, `--training smoke`, export.
Remote box (GPU + cache) is needed for: prepare_cache, check_actions,
stride_gate, real training, evaluate, probe.

## Usage

```
uv run pytest                                                  # 1. local: unit tests
uv run scripts/train.py --model tiny --training smoke          # 2. local: 50-step sanity check
uv run scripts/prepare_cache.py                                # 3. remote: build latent cache
uv run scripts/check_actions.py                                # 4. remote: confirm action/state semantics
uv run scripts/stride_gate.py                                  # 5. remote: pick the training stride
uv run scripts/train.py --model base --training full --stride 4 --seed 0   # 6. remote: real training
uv run scripts/evaluate.py --checkpoint checkpoints/base/full-s4/0/current.safetensors  # 7.
uv run scripts/export.py --checkpoint checkpoints/base/full-s4/0/current.safetensors    # 8. weights-only file
uv run scripts/probe.py --target dstate                        # optional: raw latent diagnostics
```

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
- `base` — the paper-scale predictor for the vjepa2-vitl 16x16x1024 latent grid

**Training**
- `smoke` — 50-step local sanity check on synthetic linear-dynamics data, stride 2
- `full` — 5k-step stride-1 recipe on the real latent cache

Purpose only — the numbers live in `src/vjepa_ac/variations.py`.
New variation = new entry there and a line here, in the same change.

## Layout

- `records/<model>/<training>/<seed>/record.jsonl` — meta line + per-step/eval metrics
- `checkpoints/<model>/<training>/<seed>/` — `<step>.safetensors` (+ `<step>.json`
  sidecar), `current.*` for resume, 3 best by val loss kept, `model.safetensors`
  from export
- `latent_cache/` — `latents.safetensors` + `cache.json` from prepare_cache
- `records/diagnostics/` — probe output

## Notes

- Train/val split is episode-level (`data.split_episodes`, seed 0), shared by
  train.py, evaluate.py, and probe.py; a smaller val_frac holds out a subset
  of a larger one's episodes, so probes and training agree on what is unseen.
- Checkpoints must use the `model.`-prefixed tensor layout and carry a JSON
  sidecar; pre-restructure checkpoints no longer load, and caches without
  states must be rebuilt with prepare_cache.py.
