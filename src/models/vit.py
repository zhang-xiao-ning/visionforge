import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """把 (B, 3, 32, 32) 切成 patch 并线性投影到 embed_dim。"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size 必须能被 patch_size 整除"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        # x: (B, 3, 32, 32) -> (B, embed_dim, 8, 8) -> (B, 64, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                       # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)                           # (B, N+1, D)
        x = self.norm(x)
        cls_out = x[:, 0]                             # 取 [CLS]
        return self.head(cls_out)                     # (B, num_classes)