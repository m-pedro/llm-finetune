from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    vocab_size: int = 264
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 128
    max_seq_len: int = 256
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


class CausalAttention(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.dim = cfg.dim

        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if seq_len > 1:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + mask

        probs = F.softmax(scores.float(), dim=-1).type_as(x)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.wo(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w_down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = CausalAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        return h + self.mlp(self.mlp_norm(h))


class CausalLMBackbone(nn.Module):
    """Self-contained Llama-style causal Transformer with standard target module names."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.dim)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        h = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)

    @torch.no_grad()
    def greedy_generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        tokens = prompt_ids.clone()
        for _ in range(max_new_tokens):
            ctx = tokens[:, -self.config.max_seq_len :]
            logits = self(ctx)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens
