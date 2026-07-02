import os, subprocess, warnings

warnings.filterwarnings(
    "ignore", message="The video decoding and encoding capabilities of torchvision"
)

import torch
from safetensors.torch import save_file


def pick_free_gpus(threshold_mb=1000):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
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


free_gpus = pick_free_gpus()
if free_gpus:
    train_device = f"cuda:{free_gpus[0]}"
elif torch.cuda.device_count():
    train_device = "cuda:0"
else:
    train_device = "cpu"

hf_repo = "facebook/vjepa2-vitl-fpc64-256"

DATASET_IDS = [
    "lerobot/droid_100",
]
CAMERA_KEY = "observation.images.wrist_image_left"
IMG_SIZE = 256

training_config = {
    "lr": 1e-4,
    "weight_decay": 0.001,
    "betas": (0.9, 0.95),
    "grad_clip": 1.0,
    "batch_size": 64,
    "T": 16,
    "warmup_steps": 1000,
    "total_steps": 20000,
    "val_interval": 2000,
    "val_windows": 1024,
    "keep_ckpts": 3,
}

T = training_config["T"]

PATCH_DIM = 1024
PATCH_GRID = 16

predictor_config = {
    "latent_size": (PATCH_GRID, PATCH_GRID),
    "d_state": PATCH_DIM,
    "d_action": 7,
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 16,
    "n_layers": 6,
    "eps": 1e-6,
    "max_seq_len": T * (PATCH_GRID ** 2 + 1),
}

CACHE_DIR = "./latent_cache/"
LATENTS_PATH = os.path.join(CACHE_DIR, "latents.safetensors")
CACHE_META = os.path.join(CACHE_DIR, "cache.json")
OUTPUT_DIR = "./outputs"


def atomic_save(state, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    save_file(state, tmp)
    os.replace(tmp, path)
