"""A minimal Mamba implementation for learning purposes.

Maps the theoretical formulas to code:
    Continuous SSM:  h'(t) = A·h(t) + B·x(t),  y(t) = C·h(t)
    Discretization:  A_bar = I + Δ·A,  B_bar = Δ·B
    Selectivity:     Δ, B, C are computed from the current input x_k
    Recurrence:      h_k = A_bar_k · h_{k-1} + B_bar_k · x_k
                     y_k = C_k · h_k

Note: this is a teaching implementation. It uses a Python for-loop
instead of the parallel scan. Slow but transparent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.interfaces import Model


class MambaBlock(nn.Module):
    """Single Mamba block (selective SSM + gating)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # (1) Input projection: split x into two paths.
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # (2) Depthwise conv: local receptive field for the SSM.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # (3) Selective projection: compute Δ, B, C from x.
        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # (4) A matrix: S4D-Real init.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # (5) Output projection.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        B, L, _ = x.shape

        # ---- (1) Input projection + split into SSM path and gate path ----
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # ---- (2) Causal depthwise conv (must not see the future) ----
        x_conv = x_in.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]  # trim right padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # ---- (3) Selective: compute Δ, B, C from the current input ----
        params = self.x_proj(x_conv)  # (B, L, 1 + 2*d_state)
        dt_raw, Bk, Ck = torch.split(params, [1, self.d_state, self.d_state], dim=-1)
        # Δ = softplus(Linear(x_k)) -> always positive
        delta = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        # ---- (4) Discretization: A_bar = exp(Δ·A), B_bar = Δ·B ----
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative
        A_bar = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, d_inner, d_state)
        B_bar = delta.unsqueeze(-1) * Bk.unsqueeze(2)  # (B, L, d_inner, d_state)

        # ---- (5) Recurrence: h_k = A_bar_k · h_{k-1} + B_bar_k · x_k ----
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for k in range(L):
            h = A_bar[:, k] * h + B_bar[:, k] * x_conv[:, k].unsqueeze(-1)
            y_k = (h * Ck[:, k].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_k)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Skip connection (the D term in the paper)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # ---- (6) Gate + output projection ----
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaClassifier(Model):
    """CIFAR-10 classifier: patch embedding + Mamba blocks + CLS head."""

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch_size: int = 8,  # was 4 -> seq_len 65 becomes 17
        in_channels: int = 3,
        d_model: int = 64,  # was 128
        depth: int = 2,  # was 4
        d_state: int = 8,  # was 16
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding (same idea as ViT)
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # [CLS] token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

        # Stack of Mamba blocks
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # (1) Image -> sequence of patches
        p = self.patch_embed(x)  # (B, d_model, h, w)
        p = p.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        p = torch.cat([cls, p], dim=1)  # (B, num_patches+1, d_model)
        p = p + self.pos_embed

        # (2) Through Mamba blocks with residual connections
        for blk in self.blocks:
            p = p + blk(p)

        # (3) Classify via the [CLS] token
        cls_out = self.norm(p[:, 0])
        return self.head(cls_out)
