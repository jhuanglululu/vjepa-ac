import torch
from torch import Tensor, nn


class MlpBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(nn.functional.silu(self.up(self.norm(x))))


class Compressor(nn.Module):
    def __init__(self, d_latent: int, n_tokens: int, d_c: int, n_heads: int = 8):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_c = d_c
        self.inp = nn.Linear(d_latent, d_c)
        self.patch_mlp = MlpBlock(d_c)
        self.query = nn.Parameter(torch.randn(n_tokens, d_c) * 0.02)
        self.attn = nn.MultiheadAttention(d_c, n_heads, batch_first=True)
        self.token_mlp = MlpBlock(d_c)

    def forward(self, z: Tensor) -> Tensor:
        lead, P, D = z.shape[:-2], z.shape[-2], z.shape[-1]
        h = self.patch_mlp(self.inp(z.reshape(-1, P, D)))
        q = self.query.expand(h.shape[0], -1, -1)
        v, _ = self.attn(q, h, h, need_weights=False)
        return self.token_mlp(v).reshape(*lead, self.n_tokens, self.d_c)


class IDHead(nn.Module):
    def __init__(self, d_c: int, d_out: int):
        super().__init__()
        self.inp = nn.Linear(2 * d_c, d_c)
        self.mlp = MlpBlock(d_c)
        self.out = nn.Linear(d_c, d_out)

    def forward(self, c0: Tensor, c1: Tensor) -> Tensor:
        x = self.inp(torch.cat([c0, c1 - c0], dim=-1)).mean(dim=-2)
        return self.out(self.mlp(x))


class ReconHead(nn.Module):
    def __init__(self, d_latent: int, n_patches: int, d_c: int, n_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_patches, d_c) * 0.02)
        self.attn = nn.MultiheadAttention(d_c, n_heads, batch_first=True)
        self.out = nn.Linear(d_c, d_latent)

    def forward(self, c: Tensor) -> Tensor:
        q = self.query.expand(c.shape[0], -1, -1)
        v, _ = self.attn(q, c, c, need_weights=False)
        return self.out(v)
