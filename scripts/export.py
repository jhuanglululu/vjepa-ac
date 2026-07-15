import argparse
import json
from pathlib import Path

from vjepa_ac.checkpoints import atomic_save_file, load_model_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    src = Path(args.checkpoint)
    out = Path(args.out) if args.out else src.with_name("model.safetensors")
    weights = load_model_weights(src)
    atomic_save_file({f"model.{k}": v for k, v in weights.items()}, out)

    src_sidecar = src.with_suffix(".json")
    assert src_sidecar.exists(), f"no sidecar at {src_sidecar}"
    with open(src_sidecar) as f:
        sidecar = json.load(f)
    keep = ("model", "training", "seed", "step", "val_loss", "config", "conditioning")
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({k: sidecar[k] for k in keep if k in sidecar}, f, indent=2)

    n_params = sum(v.numel() for v in weights.values())
    print(f"exported {n_params:,} params ({len(weights)} tensors) -> {out}")


if __name__ == "__main__":
    main()
