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
Remote box (GPU + cache) is needed for: prepare_cache, real training,
evaluate, probe, check_actions.

## Usage

```
uv run pytest                                                  # unit tests
uv run scripts/train.py --model tiny --training smoke          # 50-step local sanity check
uv run scripts/prepare_cache.py                                # remote: build latent cache
uv run scripts/train.py --model base --training full --seed 0  # remote: real training
uv run scripts/evaluate.py --checkpoint checkpoints/base/full/0/current.safetensors
uv run scripts/export.py --checkpoint checkpoints/base/full/0/50.safetensors
uv run scripts/probe.py --target dstate                        # remote: latent probes
uv run scripts/check_actions.py                                # remote: action/state semantics
```

`--no-rollout` on train.py drops the two-pass rollout loss term; such runs
record and checkpoint under `<training>-noroll`. evaluate.py reads the model
and training config plus the conditioning stats from the checkpoint's JSON
sidecar, so it takes only the checkpoint path; `--horizons` are counted in
strided steps.

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
