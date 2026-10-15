"""DINO backbone + projection head.

Student and teacher are both instances of `DINOBackbone`; the teacher
is built by `DINOTask.attach` (deep-copy of the student + freeze). This
file only defines what one network looks like, not the training
scheme — that lives in `tasks/dino.py`.

Architecture:
    image  → patch embedding → TransformerEncoder → [CLS] → head → (B, K)
    head:  Linear(d, h) → GELU → Linear(h, h) → GELU → Linear(h, K)
           then L2-normalise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.interfaces import Model


class DINOHead(nn.Module):
    """Projection head: MLP with L2-normalised output."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 1024) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1, p=2)


class DINOBackbone(Model):
    """ViT backbone + projection head. Returns (B, head_out)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        head_hidden: int = 2048,
        head_out: int = 1024,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = DINOHead(embed_dim, hidden_dim=head_hidden, out_dim=head_out)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        p = torch.cat([cls, p], dim=1) + self.pos_embed
        p = self.encoder(p)
        p = self.norm(p)
        return self.head(p[:, 0])  # (B, head_out)
