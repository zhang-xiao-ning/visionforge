"""Image captioning model: ViT-style encoder + Transformer decoder.

Architecture:
    image  → patch embedding → TransformerEncoder → memory
    tokens → embedding       → TransformerDecoder(×L) using memory as cross-attn
    → linear → logits over vocabulary

This is the simplest thing that works for captioning. Training is teacher-forced:
    input_ids  = [BOS, c1, c2, ..., c_{N-1}]
    target_ids = [c1, c2, ..., c_N, EOS]
"""

import torch
import torch.nn as nn

from models.base import ExperimentBundle, SetupContext


class ImageEncoder(nn.Module):
    """Patchify + TransformerEncoder. Output: (B, num_patches, d_model)."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

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

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W)
        x = self.patch_embed(images)  # (B, d_model, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        return self.norm(x)


class CaptioningModel(nn.Module):
    """Full captioning model: image encoder + text decoder."""

    # Tokenizer backend name. Change to "sp" after training SentencePiece.
    tokenizer_name: str = "tiktoken"

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int = 64,
        d_model: int = 256,
        num_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        image_size: int = 224,
        patch_size: int = 16,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        if max_len < 1:
            raise ValueError("max_len must be >= 1")

        self.pad_id = pad_id
        self.max_len = max_len

        self.encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, max_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # init
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask; True = don't attend."""
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images:    (B, 3, H, W)
            input_ids: (B, L)
        Returns:
            logits:    (B, L, vocab_size)
        """
        B, L = input_ids.shape
        if L > self.max_len:
            raise ValueError(f"input length {L} exceeds max_len {self.max_len}")

        memory = self.encoder(images)  # (B, num_patches, d_model)

        tgt = self.token_embed(input_ids)  # (B, L, d_model)
        tgt = tgt + self.pos_embed_dec[:, :L]

        tgt_mask = self._causal_mask(L, images.device)
        tgt_key_padding_mask = input_ids == self.pad_id  # True = ignore

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, L, d_model)
        return self.lm_head(out)  # (B, L, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decode.

        Args:
            images: (B, 3, H, W)
        Returns:
            generated ids (B, T) including BOS, excluding trailing padding.
        """
        B = images.size(0)
        device = images.device
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        # BOS 已占 1 位，剩下最多生成 max_len - 1 个
        limit = min(max_new_tokens, self.max_len - 1)

        for _ in range(limit):
            logits = self.forward(images, generated)  # (B, L, V)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return generated

    @classmethod
    def setup(cls, ctx: SetupContext) -> ExperimentBundle:
        """Build (model, task, loaders) for captioning.

        Deferred imports avoid top-level circular deps:
        models → data → tasks → models
        """
        from data.flickr8k import build_captioning_loaders
        from data.tokenizers import build_tokenizer
        from tasks.captioning import CaptioningTask
        from utils.path import DATASETS_PATH

        tokenizer = build_tokenizer(cls.tokenizer_name)
        loader_train, loader_val, loader_test = build_captioning_loaders(
            root=DATASETS_PATH,
            tokenizer=tokenizer,
            batch_size=ctx.batch_size,
            num_train=ctx.num_train,
        )
        model = cls(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
        )
        task = CaptioningTask()

        return ExperimentBundle(
            model=model,
            task=task,
            loader_train=loader_train,
            loader_val=loader_val,
            loader_test=loader_test,
        )
