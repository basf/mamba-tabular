import torch.nn as nn
import torch
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
import torch.nn.functional as F
import numpy as np


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rotary=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary = rotary
        dim = np.int64(dim / 2)
        self.rotary_embedding = RotaryEmbedding(dim=dim)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if self.rotary:
            q = self.rotary_embedding.rotate_queries_or_keys(q)
            k = self.rotary_embedding.rotate_queries_or_keys(k)
        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", dropped_attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, rotary=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                            rotary=rotary,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x, return_attn=False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)
