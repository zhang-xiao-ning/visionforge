"""Tests for the DINO backbone."""

import pytest
import torch

from application.models.dino import DINOBackbone, DINOHead


def _make_model(head_out: int = 32) -> DINOBackbone:
    return DINOBackbone(
        img_size=32,
        patch_size=4,
        embed_dim=32,
        depth=1,
        num_heads=2,
        head_hidden=64,
        head_out=head_out,
        dropout=0.0,
    )


def test_forward_output_shape() -> None:
    model = _make_model(head_out=32)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 32)


def test_output_is_l2_normalized() -> None:
    model = _make_model(head_out=16)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_head_exposes_out_dim() -> None:
    head = DINOHead(in_dim=8, hidden_dim=16, out_dim=42)
    assert head.out_dim == 42


def test_invalid_image_patch_size_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DINOBackbone(img_size=30, patch_size=4, embed_dim=16, depth=1, num_heads=2)


def test_param_groups_default_single_group() -> None:
    model = _make_model()
    groups = model.param_groups()
    assert set(groups.keys()) == {"all"}
