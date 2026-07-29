import datetime
import json
import subprocess
from pathlib import Path

RECORDS_ROOT = Path("records")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


class RecordWriter:
    def __init__(self, name: str):
        self.path = RECORDS_ROOT / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj) + "\n")
        self._f.flush()

    def meta(self, name: str, config: dict) -> None:
        self.write(
            {
                "type": "meta",
                "name": name,
                "config": config,
                "git_commit": git_commit(),
                "started": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        )

    def step(
        self, step: int, loss: float, lr: float, grad_norm: float, sec_per_step: float
    ) -> None:
        self.write(
            {
                "type": "step",
                "step": step,
                "loss": round(loss, 6),
                "lr": lr,
                "grad_norm": round(grad_norm, 6),
                "sec_per_step": round(sec_per_step, 4),
            }
        )

    def eval(self, step: int, val_loss: float, train_loss: float) -> None:
        self.write(
            {
                "type": "eval",
                "step": step,
                "val_loss": round(val_loss, 6),
                "train_loss": round(train_loss, 6),
            }
        )

    def close(self) -> None:
        self._f.close()
