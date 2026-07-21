import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .variations import ModelConfig


def block_causal_mask(T: int, P: int, device: torch.device | str = "cpu") -> Tensor:
    ts = torch.arange(T, device=device).repeat_interleave(P + 1)
    return ts[None, :] <= ts[:, None]


class SiLUMlp(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        nn.init.kaiming_uniform_(self.up_proj.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.down_proj.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class RoPE(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self._build_cache(base, d_head, max_seq_len)

    def _build_cache(self, base, d_head, max_seq_len):
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        seq_idx = torch.arange(max_seq_len).float()

        freqs = torch.outer(seq_idx, inv_freq)
        emb = torch.cat([freqs, freqs], dim=1)

        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        neg_half_x = torch.cat([-x2, x1], dim=-1)

        seq_len = x.shape[2]

        cos = self.cos_cached[:seq_len].to(x.dtype)[None, None]
        sin = self.sin_cached[:seq_len].to(x.dtype)[None, None]
        return x * cos + neg_half_x * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RoPE):
        super().__init__()

        self.n_heads = n_heads
        self.d_heads = d_model // n_heads

        self.rope = rope

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_uniform_(self.w_qkv.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.w_qkv(x)

        qkv = qkv.reshape(batch_size, seq_len, self.n_heads, 3 * self.d_heads)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.rope(q)
        k = self.rope(k)

        value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        value = value.permute(0, 2, 1, 3).reshape(x.shape)

        return self.w_o(value)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, rope: RoPE):
        super().__init__()

        self.norm1 = nn.RMSNorm(config.d_model, config.eps)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, rope)
        self.norm2 = nn.RMSNorm(config.d_model, config.eps)
        self.mlp = SiLUMlp(config.d_model, config.d_ff)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Predictor(nn.Module):
    def __init__(self, config: ModelConfig, max_T: int):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.n_patches = config.n_patches
        max_seq_len = max_T * (config.n_patches + 1)

        self.state_embed = nn.Linear(config.d_state, config.d_model)
        self.action_embed = nn.Linear(config.d_action, config.d_model)

        rope = RoPE(config.d_model // config.n_heads, max_seq_len)

        self.decoder_stack = nn.ModuleList(
            [DecoderBlock(config, rope) for _ in range(config.n_layers)]
        )

        self.out_norm = nn.RMSNorm(config.d_model, config.eps)
        self.out_proj = nn.Linear(config.d_model, config.d_state)

        nn.init.normal_(self.state_embed.weight, std=0.02)
        nn.init.normal_(self.action_embed.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        B, T, P, D = states.shape
        assert P == self.n_patches and D == self.config.d_state
        assert actions.shape[:2] == (B, T)

        s = self.state_embed(states)
        a = self.action_embed(actions).unsqueeze(2)
        if self.config.per_patch_action:
            s = s + a
        x = torch.cat([s, a], dim=-2).reshape(B, -1, self.d_model)

        mask = block_causal_mask(T, P, device=s.device)

        for block in self.decoder_stack:
            x = block(x, mask)

        x = x.reshape(B, T, P + 1, self.d_model)[:, :, :-1, :]
        x = self.out_norm(x)
        x = self.out_proj(x)

        return x
