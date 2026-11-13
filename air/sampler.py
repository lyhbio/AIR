"""Core AIR (Accelerating Inference via Retrospective Remasking) sampler.

Strictly follows Algorithm 1 of:
  "AIR: Retrospective Remasking for Step-Efficient Diffusion Language Model Inference"

Single-forward execution model (1 NFE per diffusion step):
  1. Forward pass:   Z^(t) = f_theta(x_{t-1})
  2. Phase 1 (Sample): Select top-N_transfer masked positions and commit candidates
                       (skipped when is_heavy is True, setting Delta_t = empty).
  3. Phase 2 (Verify): At checkpoints (t mod K == 0 and t < T), rank tokens in the
                       PRECEDING interval accumulator A_{t-1} using the same Z^(t)
                       and remask the Bottom-N_drop tokens under schedule r(p).
  4. Phase 3 (Accumulate & Reset): At checkpoints, reset accumulator to Delta_t \\ M_t;
                       otherwise accumulate A <- A U Delta_t.
  5. Terminal guarantee: At step T-1, fill any remaining [MASK] positions via argmax.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .backbones import resolve_forward_fn, resolve_mask_token_id


@dataclasses.dataclass(frozen=True)
class AIRConfig:
    """Configuration for AIR sampling under a fixed forward-pass budget."""

    budget: int = 64                  # Total forward-pass (NFE) budget T
    gen_length: int = 256             # Target generation length L
    interval: int = 2                 # Verification checkpoint spacing K
    heavy_ratio: int = 2              # Regular checkpoints between Heavy checkpoints (-1 disables Heavy)
    r_start: float = 0.85             # Initial remasking rate
    r_end: float = 0.30               # Terminal remasking rate
    schedule: str = "cosine"          # "cosine" (default), "linear", or "fixed"
    fixed_rate: Optional[float] = None  # Constant remask fraction used when schedule == "fixed"
    score_mode: str = "prob"          # Verification: "prob", "margin"/"committed_margin", "top2_margin", or "entropy"
    sampling_alg: str = "entropy"     # Phase 1: "entropy", "argmax", "maskgit_plus", or "topk_margin"
    temperature: float = 0.0          # Sampling temperature (0.0 for greedy argmax)
    top_p: float = 1.0                # Nucleus sampling threshold (1.0 disables)
    top_k: Optional[int] = None       # Top-k truncation (None disables)
    eps: float = 1e-3                 # Terminal noise level epsilon in Eq. (3)
    backbone: str = "auto"            # "auto", "dream", or "llada"

    def __post_init__(self) -> None:
        if self.budget < 1 or self.gen_length < 1 or self.interval < 1:
            raise ValueError("budget, gen_length, and interval must be positive integers")
        if self.heavy_ratio < -1:
            raise ValueError("heavy_ratio must be >= 0, or -1 to disable Heavy checkpoints")
        if self.schedule not in {"cosine", "linear", "fixed"}:
            raise ValueError(f"Unsupported schedule: {self.schedule!r}")
        if not (0.0 <= self.r_start <= 1.0 and 0.0 <= self.r_end <= 1.0):
            raise ValueError("r_start and r_end must lie in [0, 1]")
        if self.fixed_rate is not None and not (0.0 <= self.fixed_rate <= 1.0):
            raise ValueError("fixed_rate must lie in [0, 1]")
        if self.score_mode not in {"prob", "margin", "committed_margin", "top2_margin", "entropy"}:
            raise ValueError(f"Unsupported score_mode: {self.score_mode!r}")
        if self.sampling_alg not in {"entropy", "argmax", "maskgit_plus", "topk_margin"}:
            raise ValueError(f"Unsupported sampling_alg: {self.sampling_alg!r}")
        if self.temperature < 0.0 or not (0.0 < self.top_p <= 1.0):
            raise ValueError("temperature must be >= 0 and top_p must lie in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be a positive integer when provided")
        if not (0.0 < self.eps < 1.0):
            raise ValueError("eps must lie in (0, 1)")


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Applies top-k and nucleus (top-p) filtering to a 2D [N, V] logits tensor."""
    filtered = logits
    if top_k is not None and top_k < filtered.shape[-1]:
        kth_vals = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth_vals, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_mask = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
        filtered = filtered.masked_fill(remove_mask, -torch.inf)
    return filtered


def sample_tokens(
    logits: torch.Tensor,
    mask_id: int,
    alg: str = "entropy",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples candidate tokens and returns (confidence_scores, candidate_ids).

    Never proposes `mask_id` as a candidate token.
    """
    work = logits.float().clone()
    if 0 <= mask_id < work.shape[-1]:
        work[:, mask_id] = -torch.inf
    if temperature > 0.0:
        work = work / temperature
    work = apply_top_k_top_p(work, top_k=top_k, top_p=top_p)

    log_probs = F.log_softmax(work, dim=-1)
    probs = log_probs.exp()

    if alg == "argmax" or temperature <= 0.0:
        candidates = work.argmax(dim=-1)
    else:
        candidates = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    if alg == "entropy":
        # Negative predictive entropy: higher score indicates sharper confidence
        confidence = (probs * log_probs.clamp_min(-1e30)).sum(dim=-1)
    elif alg == "topk_margin":
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        confidence = top2[:, 0] - top2[:, 1] if top2.shape[-1] == 2 else top2[:, 0]
    elif alg in {"argmax", "maskgit_plus"}:
        confidence = probs.gather(1, candidates.unsqueeze(1)).squeeze(1)
    else:
        raise ValueError(f"Unsupported sampling_alg: {alg!r}")

    return confidence, candidates


def compute_remask_rate(cfg: AIRConfig, progress: float) -> float:
    """Computes scheduled remasking fraction r(p) at normalized progress p in [0, 1)."""
    if cfg.schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.r_end + (cfg.r_start - cfg.r_end) * decay
    if cfg.schedule == "linear":
        return cfg.r_start * (1.0 - progress) + cfg.r_end * progress
    if cfg.schedule == "fixed":
        return cfg.r_start if cfg.fixed_rate is None else cfg.fixed_rate
    raise ValueError(f"Unsupported schedule: {cfg.schedule!r}")


def compute_verification_scores(
    logits: torch.Tensor,
    committed_ids: torch.Tensor,
    score_mode: str = "prob",
) -> torch.Tensor:
    """Computes Phase 2 retrospective verification scores for committed tokens."""
    probs = F.softmax(logits.float(), dim=-1)
    if score_mode == "prob":
        return probs.gather(1, committed_ids.unsqueeze(1)).squeeze(1)
    if score_mode in {"margin", "committed_margin"}:
        p_committed = probs.gather(1, committed_ids.unsqueeze(1)).squeeze(1)
        probs_alt = probs.clone()
        probs_alt.scatter_(1, committed_ids.unsqueeze(1), -1.0)
        p_competitor = probs_alt.max(dim=-1).values
        return p_committed - p_competitor
    if score_mode == "top2_margin":
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        return top2[:, 0] - top2[:, 1] if top2.shape[-1] == 2 else top2[:, 0]
    if score_mode == "entropy":
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return (probs * log_probs.clamp_min(-1e30)).sum(dim=-1)
    raise ValueError(f"Unsupported score_mode: {score_mode!r}")


@torch.inference_mode()
def air_generate(
    model_or_forward: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    cfg: AIRConfig,
    device: Optional[str] = None,
    mask_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    record_trace: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Runs AIR generation for batch size 1 under a strict NFE budget `cfg.budget`.

    Args:
        model_or_forward: HuggingFace diffusion model, BackboneAdapter, or callable(x) -> logits.
        tokenizer: HuggingFace tokenizer (or None when `mask_id` is passed explicitly).
        prompt_ids: 1D [P] or 2D [1, P] tensor of prompt token IDs.
        cfg: AIR sampling configuration.
        device: Target device string; inferred from `prompt_ids` if omitted.
        mask_id: Explicit [MASK] token ID override.
        generator: Optional torch.Generator for reproducible stochastic sampling.
        record_trace: If True, returns per-checkpoint audit metadata under `stats["checkpoints"]`.

    Returns:
        (generated_tokens_1d, stats_dict) where `generated_tokens_1d` has length `cfg.gen_length`.
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("air_generate expects batch_size=1 prompt_ids of shape [P] or [1, P]")

    target_device = torch.device(device) if device is not None else prompt_ids.device
    prompt_ids = prompt_ids.to(target_device)
    prompt_len = prompt_ids.shape[1]

    resolved_mask_id = resolve_mask_token_id(tokenizer, override=mask_id)
    forward_fn = resolve_forward_fn(model_or_forward, backbone=cfg.backbone)

    x = torch.full(
        (1, prompt_len + cfg.gen_length),
        resolved_mask_id,
        device=target_device,
        dtype=torch.long,
    )
    x[:, :prompt_len] = prompt_ids

    accumulator = torch.zeros(cfg.gen_length, dtype=torch.bool, device=target_device)
    times = torch.linspace(1.0, cfg.eps, cfg.budget + 1, device=target_device)

    nfe = 0
    remasked_total = 0
    heavy_count = 0
    checkpoint_counter = 0
    traces = []

    for step in range(cfg.budget):
        progress = step / cfg.budget
        last_step = step == (cfg.budget - 1)
        is_chk = step > 0 and (step % cfg.interval == 0) and not last_step

        is_heavy = False
        if is_chk:
            checkpoint_counter += 1
            if cfg.heavy_ratio >= 0 and (checkpoint_counter % (cfg.heavy_ratio + 1) == 0):
                is_heavy = True
                heavy_count += 1

        # Single forward pass per step (1 NFE)
        logits = forward_fn(x)
        nfe += 1
        if logits.ndim != 3 or logits.shape[:2] != x.shape:
            raise ValueError(f"Expected logits shape {tuple(x.shape) + ('V',)}, got {tuple(logits.shape)}")
        logits_gen = logits[0, prompt_len:].float()
        del logits

        gen_slice = x[0, prompt_len:]
        prior_tokens = gen_slice.clone()
        prior_accumulator = accumulator.clone()

        # Phase 1: Forward Sampling (bypassed at Heavy Interval checkpoints)
        newly_committed = torch.zeros_like(accumulator)
        if not is_heavy:
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
                pos = masked[chosen]
                gen_slice[pos] = cands[chosen]
                newly_committed[pos] = True

        # Phase 2: Interval-Scoped Retrospective Verification using Z^(t)
        dropped_positions = torch.empty(0, dtype=torch.long, device=target_device)
        if is_chk and prior_accumulator.any():
            active_pos = prior_accumulator.nonzero(as_tuple=True)[0]
            committed_ids = prior_tokens[active_pos]
            scores = compute_verification_scores(
                logits_gen[active_pos],
                committed_ids=committed_ids,
                score_mode=cfg.score_mode,
            )
            rate = compute_remask_rate(cfg, progress)
            n_drop = min(active_pos.numel(), max(1, int(active_pos.numel() * rate)))
            drop_order = torch.argsort(scores, stable=True)[:n_drop]
            dropped_positions = active_pos[drop_order]
            gen_slice[dropped_positions] = resolved_mask_id
            remasked_total += n_drop

            if record_trace:
                traces.append({
                    "step": step,
                    "heavy": is_heavy,
                    "rate": rate,
                    "n_drop": n_drop,
                    "eligible_positions": active_pos.tolist(),
                    "remasked_positions": dropped_positions.tolist(),
                    "new_positions": newly_committed.nonzero(as_tuple=True)[0].tolist(),
                })

        # Phase 3: Token Settlement & Accumulator Reset / Update
        if is_chk:
            accumulator = newly_committed.clone()
            if dropped_positions.numel() > 0:
                accumulator[dropped_positions] = False
        else:
            accumulator = accumulator | newly_committed

        # Terminal step guarantee: zero remaining [MASK] tokens without extra NFEs
        if last_step:
            remaining = (gen_slice == resolved_mask_id).nonzero(as_tuple=True)[0]
            if remaining.numel() > 0:
                final_logits = logits_gen[remaining].clone()
                if 0 <= resolved_mask_id < final_logits.shape[-1]:
                    final_logits[:, resolved_mask_id] = -torch.inf
                gen_slice[remaining] = final_logits.argmax(dim=-1)

        del logits_gen

    stats: Dict[str, Any] = {
        "config": asdict(cfg),
        "nfe": nfe,
        "remasked_tokens": remasked_total,
        "heavy_checkpoints": heavy_count,
        "remaining_masks": int((x[0, prompt_len:] == resolved_mask_id).sum().item()),
    }
    if record_trace:
        stats["checkpoints"] = traces
    return x[0, prompt_len:], stats
