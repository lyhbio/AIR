"""Fixed-budget baseline samplers (Vanilla Entropy & ReMDM) sharing AIR's forward protocol."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .backbones import resolve_forward_fn, resolve_mask_token_id
from .sampler import sample_tokens


@dataclasses.dataclass(frozen=True)
class VanillaConfig:
    """Configuration for the fixed-budget Vanilla confidence sampler."""

    budget: int = 64
    gen_length: int = 256
    sampling_alg: str = "entropy"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    eps: float = 1e-3
    backbone: str = "auto"


@dataclasses.dataclass(frozen=True)
class ReMDMConfig:
    """Configuration for the fixed-budget ReMDM stochastic remasking baseline."""

    budget: int = 64
    gen_length: int = 256
    remask_prob: float = 0.25         # Maximum per-step stochastic remask probability eta
    t_on: float = 0.55                # Active window upper normalized noise threshold
    t_off: float = 0.05               # Active window lower normalized noise threshold
    sampling_alg: str = "entropy"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    eps: float = 1e-3
    backbone: str = "auto"


def _fill_remaining_masks(
    gen_slice: torch.Tensor,
    logits_gen: torch.Tensor,
    mask_id: int,
) -> None:
    remaining = (gen_slice == mask_id).nonzero(as_tuple=True)[0]
    if remaining.numel() > 0:
        final_logits = logits_gen[remaining].clone()
        if 0 <= mask_id < final_logits.shape[-1]:
            final_logits[:, mask_id] = -torch.inf
        gen_slice[remaining] = final_logits.argmax(dim=-1)


@torch.inference_mode()
def vanilla_generate(
    model_or_forward: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    cfg: VanillaConfig,
    device: Optional[str] = None,
    mask_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fixed-budget confidence-based MDLM sampler without remasking (Vanilla)."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    target_device = torch.device(device) if device is not None else prompt_ids.device
    prompt_ids = prompt_ids.to(target_device)
    prompt_len = prompt_ids.shape[1]

    resolved_mask_id = resolve_mask_token_id(tokenizer, override=mask_id)
    forward_fn = resolve_forward_fn(model_or_forward, backbone=cfg.backbone)

    x = torch.full((1, prompt_len + cfg.gen_length), resolved_mask_id, device=target_device, dtype=torch.long)
    x[:, :prompt_len] = prompt_ids
    times = torch.linspace(1.0, cfg.eps, cfg.budget + 1, device=target_device)

    nfe = 0
    for step in range(cfg.budget):
        last_step = step == (cfg.budget - 1)
        logits = forward_fn(x)
        nfe += 1
        logits_gen = logits[0, prompt_len:].float()
        del logits

        gen_slice = x[0, prompt_len:]
        masked = (gen_slice == resolved_mask_id).nonzero(as_tuple=True)[0]
        if masked.numel() > 0:
            conf, cands = sample_tokens(
                logits_gen[masked],
                mask_id=resolved_mask_id,
                alg=cfg.sampling_alg,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                generator=generator,
            )
            ratio = 1.0 if last_step else float(1.0 - times[step + 1] / times[step])
            n_transfer = min(masked.numel(), max(1, int(masked.numel() * ratio)))
            chosen = torch.argsort(conf, descending=True, stable=True)[:n_transfer]
            gen_slice[masked[chosen]] = cands[chosen]

        if last_step:
            _fill_remaining_masks(gen_slice, logits_gen, resolved_mask_id)
        del logits_gen

    return x[0, prompt_len:], {
        "config": asdict(cfg),
        "nfe": nfe,
        "remaining_masks": int((x[0, prompt_len:] == resolved_mask_id).sum().item()),
    }


@torch.inference_mode()
def remdm_generate(
    model_or_forward: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    cfg: ReMDMConfig,
    device: Optional[str] = None,
    mask_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fixed-budget stochastic remasking baseline (ReMDM)."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    target_device = torch.device(device) if device is not None else prompt_ids.device
    prompt_ids = prompt_ids.to(target_device)
    prompt_len = prompt_ids.shape[1]

    resolved_mask_id = resolve_mask_token_id(tokenizer, override=mask_id)
    forward_fn = resolve_forward_fn(model_or_forward, backbone=cfg.backbone)

    x = torch.full((1, prompt_len + cfg.gen_length), resolved_mask_id, device=target_device, dtype=torch.long)
    x[:, :prompt_len] = prompt_ids
    times = torch.linspace(1.0, cfg.eps, cfg.budget + 1, device=target_device)

    nfe = 0
    remasked_total = 0
    for step in range(cfg.budget):
        last_step = step == (cfg.budget - 1)
        t_cur = float(times[step])
        t_next = float(times[step + 1])

        logits = forward_fn(x)
        nfe += 1
        logits_gen = logits[0, prompt_len:].float()
        del logits

        gen_slice = x[0, prompt_len:]
        unmasked_before = (gen_slice != resolved_mask_id).nonzero(as_tuple=True)[0]

        # Stochastic remasking on already unmasked positions within [t_off, t_on]
        if not last_step and unmasked_before.numel() > 0 and (cfg.t_off <= t_cur <= cfg.t_on):
            sigma = min(cfg.remask_prob, (1.0 - t_next) / max(t_cur, 1e-6)) * 0.5 * (
                1.0 + math.cos(math.pi * (1.0 - t_cur))
            )
            if sigma > 0.0:
                rand_draws = torch.rand(unmasked_before.numel(), device=target_device, generator=generator)
                drop_mask = rand_draws < sigma
                if drop_mask.any():
                    gen_slice[unmasked_before[drop_mask]] = resolved_mask_id
                    remasked_total += int(drop_mask.sum().item())

        # Unmask top-N_transfer masked positions
        masked = (gen_slice == resolved_mask_id).nonzero(as_tuple=True)[0]
        if masked.numel() > 0:
            conf, cands = sample_tokens(
                logits_gen[masked],
                mask_id=resolved_mask_id,
                alg=cfg.sampling_alg,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                generator=generator,
            )
            ratio = 1.0 if last_step else float(1.0 - t_next / max(t_cur, 1e-6))
            n_transfer = min(masked.numel(), max(1, int(masked.numel() * ratio)))
            chosen = torch.argsort(conf, descending=True, stable=True)[:n_transfer]
            gen_slice[masked[chosen]] = cands[chosen]

        if last_step:
            _fill_remaining_masks(gen_slice, logits_gen, resolved_mask_id)
        del logits_gen

    return x[0, prompt_len:], {
        "config": asdict(cfg),
        "nfe": nfe,
        "remasked_tokens": remasked_total,
        "remaining_masks": int((x[0, prompt_len:] == resolved_mask_id).sum().item()),
    }
