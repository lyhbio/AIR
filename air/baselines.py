"""Baseline samplers sharing AIR's forward protocol.

Includes:
- Fixed-budget baselines: Vanilla Entropy & ReMDM (Tables 1 & 2)
- Dynamic-stopping baselines: EB-Sampler & SlowFast Sampling (Table 1 & Figure 3)
"""

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


@dataclasses.dataclass(frozen=True)
class EBSamplerConfig:
    """Configuration for the dynamic-stopping Entropy-Bounded (EB) baseline sampler."""

    max_nfe: int = 256                # Safety NFE cap equal to generation length L
    gen_length: int = 256
    eb_gamma: float = 0.5             # Cumulative entropy error threshold gamma
    sampling_alg: str = "entropy"
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: Optional[int] = None
    backbone: str = "auto"


@dataclasses.dataclass(frozen=True)
class SlowFastConfig:
    """Configuration for the dynamic-stopping SlowFast two-phase baseline sampler."""

    max_nfe: int = 256                # Hard forward-pass cap
    gen_length: int = 256
    exploration_steps: int = 8        # Slow exploration window length
    history_window: int = 2           # Endpoint stability lookback window
    stability_std_threshold: float = 1.0
    min_confidence: float = 0.1
    high_confidence: float = 0.85
    slow_top_k: int = 1
    fast_top_k: int = 1
    max_cycles: int = 256
    temperature: float = 0.0
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


@torch.inference_mode()
def eb_sampler_generate(
    model_or_forward: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    cfg: EBSamplerConfig,
    device: Optional[str] = None,
    mask_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Dynamic-stopping Entropy-Bounded sampler (EB-Sampler)."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    target_device = torch.device(device) if device is not None else prompt_ids.device
    prompt_ids = prompt_ids.to(target_device)
    prompt_len = prompt_ids.shape[1]

    resolved_mask_id = resolve_mask_token_id(tokenizer, override=mask_id)
    forward_fn = resolve_forward_fn(model_or_forward, backbone=cfg.backbone)

    x = torch.full((1, prompt_len + cfg.gen_length), resolved_mask_id, device=target_device, dtype=torch.long)
    x[:, :prompt_len] = prompt_ids
    gen_slice = x[0, prompt_len:]

    nfe = 0
    while (gen_slice == resolved_mask_id).any() and nfe < cfg.max_nfe:
        logits = forward_fn(x)
        nfe += 1
        logits_gen = logits[0, prompt_len:].float()
        del logits

        masked = (gen_slice == resolved_mask_id).nonzero(as_tuple=True)[0]
        if nfe == cfg.max_nfe:
            _fill_remaining_masks(gen_slice, logits_gen, resolved_mask_id)
            del logits_gen
            break

        conf, cands = sample_tokens(
            logits_gen[masked],
            mask_id=resolved_mask_id,
            alg=cfg.sampling_alg,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            generator=generator,
        )
        rows = logits_gen[masked].clone()
        if 0 <= resolved_mask_id < rows.shape[-1]:
            rows[:, resolved_mask_id] = -torch.inf
        log_probs = F.log_softmax(rows, dim=-1)
        entropy = -(log_probs.exp() * log_probs.clamp_min(-1e30)).sum(dim=-1)

        order = torch.argsort(conf, descending=True, stable=True)
        ordered_entropy = entropy[order]
        bound = ordered_entropy.cumsum(0) - ordered_entropy.cummax(0).values
        eb_count = max(1, int((bound <= cfg.eb_gamma).sum().item()))
        chosen = order[: min(eb_count, masked.numel())]
        gen_slice[masked[chosen]] = cands[chosen]
        del logits_gen

    return x[0, prompt_len:], {
        "config": asdict(cfg),
        "nfe": nfe,
        "remaining_masks": int((gen_slice == resolved_mask_id).sum().item()),
    }


def _slowfast_select(
    masked: torch.Tensor,
    confidence: torch.Tensor,
    start: int,
    end: int,
    top_k: int,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    in_region = masked[(masked >= start) & (masked < end)]
    if not in_region.numel():
        return in_region
    vals = confidence[in_region]
    if threshold is not None:
        high = in_region[vals > threshold]
        if high.numel():
            return high
    order = torch.argsort(vals, descending=True, stable=True)
    return in_region[order[: min(top_k, in_region.numel())]]


@torch.inference_mode()
def slowfast_generate(
    model_or_forward: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    cfg: SlowFastConfig,
    device: Optional[str] = None,
    mask_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Dynamic-stopping SlowFast two-phase adaptive sampler."""
    del generator  # SlowFast uses greedy confidence predictions
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    target_device = torch.device(device) if device is not None else prompt_ids.device
    prompt_ids = prompt_ids.to(target_device)
    prompt_len = prompt_ids.shape[1]

    resolved_mask_id = resolve_mask_token_id(tokenizer, override=mask_id)
    forward_fn = resolve_forward_fn(model_or_forward, backbone=cfg.backbone)

    x = torch.full((1, prompt_len + cfg.gen_length), resolved_mask_id, device=target_device, dtype=torch.long)
    x[:, :prompt_len] = prompt_ids
    gen_slice = x[0, prompt_len:]

    nfe = slow_nfe = fast_nfe = cycles = 0
    endpoint_history = []
    start = 0

    def _run_phase(phase: str, region_start: int, region_end: int) -> Tuple[Optional[torch.Tensor], bool]:
        nonlocal nfe, slow_nfe, fast_nfe
        masked = (gen_slice == resolved_mask_id).nonzero(as_tuple=True)[0]
        if not masked.numel():
            return None, False
        logits = forward_fn(x)
        nfe += 1
        if phase == "slow":
            slow_nfe += 1
        else:
            fast_nfe += 1
        rows = logits[0, prompt_len:].float().clone()
        del logits
        if 0 <= resolved_mask_id < rows.shape[-1]:
            rows[:, resolved_mask_id] = -torch.inf
        probs = F.softmax(rows, dim=-1)
        confidence, preds = probs.max(dim=-1)

        if nfe == cfg.max_nfe:
            gen_slice[masked] = preds[masked]
            return confidence, True

        if phase == "slow":
            selected = _slowfast_select(masked, confidence, region_start, region_end, cfg.slow_top_k)
        else:
            selected = _slowfast_select(
                masked, confidence, region_start, region_end, cfg.fast_top_k, threshold=cfg.high_confidence
            )
        if selected.numel():
            gen_slice[selected] = preds[selected]
        return confidence, False

    while (gen_slice == resolved_mask_id).any() and nfe < cfg.max_nfe and cycles < cfg.max_cycles:
        cycles += 1
        start_before = start
        endpoints = []
        stable = False

        for _ in range(cfg.exploration_steps):
            if not (gen_slice == resolved_mask_id).any() or nfe >= cfg.max_nfe:
                break
            masked = (gen_slice == resolved_mask_id).nonzero(as_tuple=True)[0]
            confidence, forced = _run_phase("slow", start, cfg.gen_length)
            if confidence is None:
                break
            cands = masked[(masked >= start) & (confidence[masked] > cfg.min_confidence)]
            endpoint = int(cands[-1].item()) + 1 if cands.numel() else min(start + 1, cfg.gen_length)
            endpoint = min(cfg.gen_length, max(start, endpoint))
            endpoints.append(endpoint)
            endpoint_history.append(endpoint)
            if len(endpoint_history) > cfg.history_window:
                endpoint_history.pop(0)
            if len(endpoint_history) >= cfg.history_window:
                mean_ep = sum(endpoint_history) / len(endpoint_history)
                var_ep = sum((v - mean_ep) ** 2 for v in endpoint_history) / len(endpoint_history)
                stable = var_ep < (cfg.stability_std_threshold ** 2)
            if forced:
                break
            if stable:
                start = min(cfg.gen_length, max(start_before + 1, int(mean_ep)))
                break

        if not (gen_slice == resolved_mask_id).any() or nfe >= cfg.max_nfe:
            break
        if not stable:
            fallback = endpoints[-1] if endpoints else min(start_before + 1, cfg.gen_length)
            start = min(cfg.gen_length, max(start_before + 1, fallback))

        if (gen_slice[start_before:start] == resolved_mask_id).any() and nfe < cfg.max_nfe:
            _run_phase("fast", start_before, start)

    return x[0, prompt_len:], {
        "config": asdict(cfg),
        "nfe": nfe,
        "slow_nfe": slow_nfe,
        "fast_nfe": fast_nfe,
        "cycles": cycles,
        "remaining_masks": int((gen_slice == resolved_mask_id).sum().item()),
    }
