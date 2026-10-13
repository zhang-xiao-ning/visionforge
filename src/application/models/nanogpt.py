"""A minimal decoder-only Transformer language model.

Architecture (GPT-style, pre-norm):
    tokens → token_embed + pos_embed → TransformerEncoder(causal) → LayerNorm → lm_head

Weight tying: lm_head shares `token_embed.weight` (standard for LMs).
"""

import torch
import torch.nn as nn

from framework.inference import SamplingConfig, sample
from framework.interfaces import Model


class NanoGPT(Model):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        d_model: int = 192,
        num_heads: int = 6,
        num_layers: int = 6,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size must be >= 1")

        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights before init so we only init the shared tensor once.
        self.lm_head.weight = self.token_embed.weight

        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask; True = don't attend."""
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) long
        Returns:
            logits:    (B, L, vocab_size)
        """
        B, L = input_ids.shape
        if L > self.block_size:
            raise ValueError(f"input length {L} exceeds block_size {self.block_size}")

        pos = torch.arange(L, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.drop(x)

        mask = self._causal_mask(L, input_ids.device)
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_id: int | None = None,
        config: SamplingConfig | None = None,
    ) -> torch.Tensor:
        """Thin wrapper over `framework.inference.sample`."""
        if config is None:
            config = SamplingConfig()
        return sample(
            self.forward,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
            config=config,
        )
