import torch
from torch import Tensor, nn

from .compressor import Compressor, IDHead
from .predictor import Predictor
from .variations import ModelConfig


class CPredictor(nn.Module):
    def __init__(self, config: ModelConfig, max_T: int):
        super().__init__()
        self.config = config
        self.compressor = Compressor(
            config.comp_d_latent, config.n_patches, config.d_state, config.comp_heads
        )
        self.id_head = IDHead(config.d_state, config.d_action)
        self.predictor = Predictor(config, max_T)
        self.c_mean: Tensor
        self.c_std: Tensor
        self.register_buffer("c_mean", torch.zeros(config.d_state))
        self.register_buffer("c_std", torch.ones(config.d_state))

    def encode(self, z: Tensor) -> Tensor:
        return (self.compressor(z) - self.c_mean) / self.c_std

    def set_stats(self, mean: Tensor, std: Tensor) -> None:
        self.c_mean.copy_(mean)
        self.c_std.copy_(std.clamp(min=1e-6))

    def forward(self, tokens: Tensor, actions: Tensor) -> Tensor:
        return self.predictor(tokens, actions)
