import subprocess

import torch


def pick_free_gpus(threshold_mb: int = 1000) -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except Exception:
        return []
    free = []
    for line in out.strip().splitlines():
        idx, used = line.split(",")
        if int(used.strip()) < threshold_mb:
            free.append(int(idx.strip()))
    return free


def get_device() -> str:
    free = pick_free_gpus()
    if free:
        return f"cuda:{free[0]}"
    if torch.cuda.device_count():
        return "cuda:0"
    return "cpu"
