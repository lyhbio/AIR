"""Unified backbone wrappers for Dream-7B-Instruct and LLaDA-8B-Instruct.

Crucial architectural difference handled transparently here:
- LLaDA-8B-Instruct outputs position-aligned logits (logit[i] predicts token[i]).
- Dream-7B-Instruct is adapted from autoregressive weights and outputs causally
  shifted logits, requiring: logits = cat([logits[:, :1], logits[:, :-1]], dim=1).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_LLADA_MASK_ID = 126336


def infer_backbone_type(model_or_name: Any) -> str:
    """Infers 'dream' vs 'llada' from a model object or repository identifier."""
    if isinstance(model_or_name, str):
        return "dream" if "dream" in model_or_name.lower() else "llada"
    cls_name = type(model_or_name).__name__.lower()
    cfg = getattr(model_or_name, "config", None)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    name_or_path = str(getattr(cfg, "_name_or_path", "")).lower()
    if "dream" in cls_name or "dream" in model_type or "dream" in name_or_path:
        return "dream"
    return "llada"


def align_logits_for_backbone(logits: torch.Tensor, backbone: str) -> torch.Tensor:
    """Applies Dream's 1-token causal logit shift when `backbone == 'dream'`."""
    if backbone == "dream":
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


class BackboneAdapter:
    """Callable forward wrapper normalizing output logits to aligned [1, L, V]."""

    def __init__(self, model: Any, backbone: str = "auto") -> None:
        self.model = model
        self.backbone = infer_backbone_type(model) if backbone == "auto" else backbone
        if self.backbone not in {"dream", "llada"}:
            raise ValueError(f"Unsupported backbone type: {self.backbone!r}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out.logits if hasattr(out, "logits") else out
        return align_logits_for_backbone(logits, self.backbone)


DreamForwardWrapper = BackboneAdapter
LLaDAForwardWrapper = BackboneAdapter


def resolve_mask_token_id(tokenizer: Any, override: Optional[int] = None) -> int:
    """Resolves the [MASK] token ID with LLaDA fallback (126336)."""
    if override is not None:
        return int(override)
    mask_id = getattr(tokenizer, "mask_token_id", None) if tokenizer is not None else None
    return int(mask_id) if mask_id is not None else DEFAULT_LLADA_MASK_ID


def resolve_forward_fn(model_or_forward: Any, backbone: str = "auto") -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolves a HuggingFace model, BackboneAdapter, or raw callable to aligned [B, L, V] logits."""
    if isinstance(model_or_forward, BackboneAdapter):
        return model_or_forward
    if hasattr(model_or_forward, "config") and hasattr(model_or_forward, "forward"):
        return BackboneAdapter(model_or_forward, backbone=backbone)
    if callable(model_or_forward):
        def _wrapped(x: torch.Tensor) -> torch.Tensor:
            out = model_or_forward(x)
            logits = out.logits if hasattr(out, "logits") else out
            target_bb = "llada" if backbone == "auto" else backbone
            return align_logits_for_backbone(logits, target_bb)
        return _wrapped
    raise TypeError("Expected an AutoModel, BackboneAdapter, or callable(x) -> logits")


def load_backbone(
    model_name_or_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    revision: Optional[str] = None,
) -> Tuple[Any, Any, str]:
    """Loads a pretrained masked diffusion model and tokenizer."""
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, revision=revision, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    backbone = infer_backbone_type(model) if "dream" in model_name_or_path.lower() else infer_backbone_type(model)
    return model, tokenizer, backbone
