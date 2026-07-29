# vjepa-ac

[繁體中文](README-TW.md)

A small-scale reproduction of Meta's **V-JEPA 2-AC** for learning and
experimentation. It predicts how a robot's camera view will change after a
motion, then uses those predictions to plan toward a goal image.

The original model uses about 62 hours of robot data and a 300M-parameter
predictor. This project uses the first 100 successful episodes from
`nvidia/Cosmos3-DROID` (about 45 minutes) and trains roughly 24M parameters in
a few GPU-hours. Tests and smoke runs work locally on CPU; full runs use a
remote GPU machine.

## Outcome

Directly predicting V-JEPA's full image features did not work at this scale:
the model mostly ignored robot motion. The final design first compresses each
frame into 16 motion-aware tokens and predicts in that smaller space.

| Held-out result | Raw features | Final model |
| --- | ---: | ---: |
| Error increase after shuffling actions | +0.2% | +50-65% at horizon 15 |
| Model error / repeat-input-frame error | about 1.0 | 0.67-0.73 |
| Retrieved-frame timing | stayed at input | tracks the true frame |

Shuffled actions should hurt because the future depends on the action. The
large increase therefore shows that the final model uses motion information.
A model/copy ratio below 1 means it predicts better than simply repeating the
input frame.

Goal-image planning on held-out episodes (left: selected frames; right: goal):

![Goal reached in 11 steps](gif/plan-roll-1.gif)

![Goal reached in 4 steps](gif/plan-roll-2.gif)

![71 of 90 frames covered before the step limit](gif/plan-roll-3.gif)

## Why the design changed

Three measurements explain the change from the paper's full-feature target:

1. **The raw-feature model ignored actions.** Shuffling actions changed error
   by only 0.2%, zero actions behaved similarly, and predicted rollouts stayed
   near the input frame.
2. **The encoder still contained motion information.** A 23M-parameter probe
   decoded motion from pairs of encoded frames with R2 around 0.37.
3. **Actions alone could not explain raw feature changes.** Ridge regression
   explained 0.00% of raw latent-delta energy across scenes. Learning the
   required action-and-scene interaction would need more data and training.

The practical fix is a learned compressor trained with inverse dynamics: from
two consecutive token sets, an auxiliary head must recover the robot motion.
This makes motion visible in the prediction target instead of letting image
content dominate the loss.

Two other choices came from measurements:

- Frames use stride 6 because motion at the original 15 Hz was below encoder
  noise.
- Conditioning uses the robot motion that actually occurred, with angle wrap
  correction, rather than the requested command.

## System design

`frozen V-JEPA encoder -> motion-aware compressor -> causal predictor -> CEM planner`

- **Encoder:** `facebook/vjepa2-vitl-fpc64-256` converts a 256x256 frame into
  256 x 1024 patch features. Frames are encoded once and cached; the encoder
  is not trained.
- **Compressor (~7M parameters):** 16 learned queries attend to the 256 patches
  and produce 16 x 384 tokens. An inverse-dynamics head makes the tokens retain
  motion; a light reconstruction loss preserves scene information.
- **Predictor (~17M parameters):** a 6-layer block-causal transformer predicts
  the next token change from up to 16 frames and their actions:
  `z[t+1] = z[t] + f(z[<=t], a[<=t])`.
- **Action input (7 values):** summed, angle-corrected state change for robot
  dimensions 0-5 and absolute gripper state for dimension 6. Training-set
  normalization values are stored with the checkpoint.

### Training

Training has two stages so expensive predictor training starts only after the
compressed space passes basic checks.

1. `train_compressor.py` trains the compressor, inverse-dynamics head, and
   temporary reconstruction decoder. Continue only if held-out motion R2 is at
   least 0.2 and the compressed-space linear ceiling is at least +2%.
2. `train.py` fine-tunes the compressor at a low learning rate while training
   the predictor. Its loss combines next-token prediction, a two-pass rollout,
   and inverse dynamics. Stop-gradient targets and motion monitoring guard
   against token collapse.

The main recipe uses 16 frames at stride 6, spanning 91 source frames, with an
episode-level train/validation split and batch size 64 via gradient
accumulation.

### Planning demo

`plan_demo.py` demonstrates model-predictive control (MPC) without a live
robot. At every step it:

1. samples 8-step action sequences with CEM;
2. predicts their outcomes and chooses the sequence ending closest to the goal
   tokens;
3. executes only the first few actions, adds them to the recorded robot state,
   and selects the recorded frame with the nearest state;
4. adds that real frame and its actual motion to the context, then plans again.

![State-based frame selection](assets/snapping.svg)

The recorded episode acts as a simple environment, not as proof of real-robot
control. Execution never uses the goal frame's time index or assumes that the
next frame is later in the recording. This prevents a future goal from being
reached automatically.

The demo runs on a random held-out episode by default; `--episode N` picks a
specific one. Two fixed settings (constants at the top of the script) matter:

- 6 planned actions are combined per commit, because one action is often too
  small to reach a different recorded state.
- Frame selection is limited to frames 30-150 of the episode, because the same
  arm pose appears in different task stages. This uses time-range knowledge, so
  the demo is a controlled illustration rather than a fully blind rollout.

The trace compares required, commanded, and executed motion. A good command
but bad execution points to missing states in the recording; a bad command
points to the planner.

## Quick start

```bash
uv sync                 # tests and CPU smoke runs
uv sync --extra cache   # cache building, evaluation GIFs, and GPU runs

uv run pytest
uv run scripts/train.py --smoke   # CPU check on synthetic data, saves nothing
```

After preparing a latent cache and training with the full workflow below:

```bash
uv run scripts/evaluate.py       # uses checkpoints/current.safetensors
uv run scripts/plan_demo.py      # uses the same checkpoint and cache
```

Scripts take no arguments and read no environment variables. Every setting is
a named constant: the model and training recipe in `src/vjepa_ac/variations.py`
and per-script knobs at the top of each script. The only flags are
`train.py --smoke`, `plan_demo.py --episode N`, and optional
`--cameras/--strides/--seeds` on the rarely rerun `gate_sweep.py`.

## Full workflow

Run the scripts in this order:

```bash
# 1. Download the first 100 episodes and cache V-JEPA features for the chosen
#    camera (the CAMERA constant at the top; ext1 lands in latent_cache/)
uv run scripts/prepare_cache.py

# 2. Confirm action/state meanings
uv run scripts/check_actions.py

# 3. Screen cameras and strides (1 seed each), then confirm the chosen camera
#    at strides 4 and 6 with 3 seeds. Sweeping ext2/wrist first needs their
#    caches: rerun prepare_cache.py with CAMERA changed
uv run scripts/gate_sweep.py --cameras ext1
uv run scripts/stride_gate.py

# 4. Confirm that raw features have little action-predictable change
uv run scripts/ceiling_probe.py

# 5. Train and validate the compressed token space
uv run scripts/train_compressor.py

# 6. Train the action-conditioned predictor
uv run scripts/train.py

# 7. Measure action use and rollout quality
uv run scripts/evaluate.py

# 8. Run goal-image planning
uv run scripts/plan_demo.py
```

`evaluate.py` reports prediction error against copy, zero-action, and
shuffled-action baselines, plus frame retrieval over time. Acceptance targets
are shuffled-action error at least 10% worse at the longest horizon and a
model/copy ratio no higher than 0.9.

For deeper diagnosis, `overfit_check.py` compares two raw-feature models
trained on a fixed 512-window sample: one receives correct actions and
one receives permanently shuffled actions. It distinguishes slow learning
from a model that is structurally unable to use actions.

## Configuration and outputs

`src/vjepa_ac/variations.py` holds the single training recipe: `MODEL` and
`TRAINING` are the final 16 x 384 compressor-plus-predictor trained for 10,000
steps at stride 6, and `SMOKE_MODEL`/`SMOKE_TRAINING` are the tiny synthetic
configuration behind `train.py --smoke`. The raw-feature negative baseline
survives only inside `overfit_check.py`.

Main outputs:

- `latent_cache/`: cached features, state, actions, and episode ranges for the
  chosen camera (ext1); rebuilding with a different `CAMERA` writes to
  `latent_cache/ext2/` or `latent_cache/wrist/` instead.
- `checkpoints/`: `compressor.safetensors` from stage 1 and the predictor's
  `current.safetensors` with its `current.json` sidecar (config, conditioning
  stats, latest validation loss), overwritten at every validation. Evaluation
  and planning load these. Smoke runs save nothing.
- `records/compressor.jsonl` and `records/train.jsonl`: training metrics.
- `records/eval_results.json` and `records/plan_ep*_g*.gif`: evaluation and
  planning outputs.
- `records/diagnostics/`: camera, stride, and ceiling measurements.
