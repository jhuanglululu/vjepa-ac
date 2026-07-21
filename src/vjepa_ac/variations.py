from typing import Literal

from pydantic import BaseModel


class ModelConfig(BaseModel):
    d_state: int = 1024
    patch_grid: int = 16
    d_action: int = 7
    d_model: int
    d_ff: int
    n_heads: int
    n_layers: int
    per_patch_action: bool = False
    compressor: bool = False
    comp_patches: int = 256
    comp_d_latent: int = 1024
    comp_heads: int = 8
    eps: float = 1e-6

    @property
    def n_patches(self) -> int:
        return self.patch_grid**2


class TrainingConfig(BaseModel):
    lr: float
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    batch_size: int
    grad_accum: int = 1
    T: int
    stride: int = 1
    compressor_lr: float = 1e-5
    id_weight: float = 1.0
    warmup_steps: int
    total_steps: int
    val_interval: int
    val_frac: float = 0.1
    val_windows: int
    rollout_loss: bool = True
    keep_ckpts: int = 3
    log_interval: int
    amp: bool
    data: Literal["cache", "synthetic"] = "cache"


MODELS: dict[str, ModelConfig] = {
    "base": ModelConfig(d_model=512, d_ff=2048, n_heads=16, n_layers=6),
    "base-pp": ModelConfig(d_model=512, d_ff=2048, n_heads=16, n_layers=6, per_patch_action=True),
    "base-c16": ModelConfig(
        d_state=384,
        patch_grid=4,
        d_model=512,
        d_ff=2048,
        n_heads=16,
        n_layers=6,
        compressor=True,
    ),
    "tiny": ModelConfig(d_state=32, patch_grid=4, d_model=64, d_ff=256, n_heads=4, n_layers=2),
    "tiny-c": ModelConfig(
        d_state=16,
        patch_grid=2,
        d_model=64,
        d_ff=256,
        n_heads=4,
        n_layers=2,
        compressor=True,
        comp_patches=16,
        comp_d_latent=32,
        comp_heads=4,
    ),
}

TRAININGS: dict[str, TrainingConfig] = {
    "c-full": TrainingConfig(
        lr=1e-4,
        batch_size=64,
        grad_accum=2,
        T=16,
        stride=6,
        warmup_steps=300,
        total_steps=10000,
        val_interval=500,
        val_windows=1024,
        log_interval=100,
        amp=True,
    ),
    "full": TrainingConfig(
        lr=1e-4,
        batch_size=64,
        grad_accum=8,
        T=16,
        warmup_steps=300,
        total_steps=3000,
        val_interval=500,
        val_windows=1024,
        log_interval=100,
        amp=True,
    ),
    "smoke": TrainingConfig(
        lr=1e-3,
        batch_size=8,
        grad_accum=2,
        T=4,
        stride=2,
        warmup_steps=10,
        total_steps=50,
        val_interval=25,
        val_frac=0.2,
        val_windows=16,
        log_interval=1,
        amp=False,
        data="synthetic",
    ),
}
