"""CPU unit tests verifying Algorithm 1 invariants, baselines, and paper statistical helpers."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air import (
    AIRConfig,
    EBSamplerConfig,
    ReMDMConfig,
    SlowFastConfig,
    VanillaConfig,
    air_generate,
    compute_remask_rate,
    eb_sampler_generate,
    remdm_generate,
    slowfast_generate,
    vanilla_generate,
)
from air.backbones import align_logits_for_backbone
from benchmarks import extract_gsm8k_answer, extract_pubmedqa_answer, format_pubmedqa_prompt
from scripts.run_ablations import build_ablation_plan
from scripts.run_eval import build_paper_config
from scripts.stats_analysis import exact_mcnemar_test, wilson_confidence_interval


class ToyDiffusionForward:
    """Synthetic deterministic forward model favoring MASK pre-filter to test safety."""

    def __init__(self, vocab_size: int = 12, mask_id: int = 11) -> None:
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        pos = torch.arange(x.shape[1], dtype=torch.float32).view(1, -1, 1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        logits = torch.sin(pos * 0.7 + vocab + self.calls * 0.8) * 3.0
        logits[:, :, self.mask_id] = 6.0  # Adversarial high logit on [MASK]
        return logits


def test_algorithm1_nfe_and_interval_invariants():
    model = ToyDiffusionForward()
    prompt = torch.tensor([[1, 2, 3, 4]])
    cfg = AIRConfig(budget=10, gen_length=24, interval=2, heavy_ratio=2, backbone="llada")
    tokens, stats = air_generate(model, None, prompt, cfg, mask_id=11, record_trace=True)

    assert tokens.shape == (24,)
    assert stats["nfe"] == model.calls == cfg.budget == 10
    assert stats["remaining_masks"] == 0
    assert not (tokens == 11).any()

    checkpoints = stats["checkpoints"]
    assert [c["step"] for c in checkpoints] == [2, 4, 6, 8]
    assert [c["heavy"] for c in checkpoints] == [False, False, True, False]
    assert stats["heavy_checkpoints"] == 1

    for chk in checkpoints:
        assert set(chk["remasked_positions"]) <= set(chk["eligible_positions"])
        # Retrospective rule: eligible_positions never overlap current-step commitments
        assert not (set(chk["eligible_positions"]) & set(chk["new_positions"]))
        if chk["heavy"]:
            assert chk["new_positions"] == []


@pytest.mark.parametrize("score_mode", ["prob", "margin", "entropy"])
@pytest.mark.parametrize("schedule", ["cosine", "linear", "fixed"])
def test_score_and_schedule_modes(score_mode: str, schedule: str):
    model = ToyDiffusionForward()
    prompt = torch.tensor([[1, 2, 3]])
    cfg = AIRConfig(
        budget=8,
        gen_length=16,
        interval=2,
        heavy_ratio=-1,
        score_mode=score_mode,
        schedule=schedule,
        fixed_rate=0.4,
    )
    tokens, stats = air_generate(model, None, prompt, cfg, mask_id=11)
    assert stats["nfe"] == 8
    assert stats["heavy_checkpoints"] == 0
    assert stats["remaining_masks"] == 0
    assert tokens.shape == (16,)


def test_dream_causal_logit_shift():
    logits = torch.arange(20, dtype=torch.float32).view(1, 5, 4)
    shifted = align_logits_for_backbone(logits, "dream")
    assert torch.equal(shifted[:, 0], logits[:, 0])
    assert torch.equal(shifted[:, 1:], logits[:, :-1])
    unshifted = align_logits_for_backbone(logits, "llada")
    assert torch.equal(unshifted, logits)


def test_cosine_schedule_bounds():
    cfg = AIRConfig(r_start=0.85, r_end=0.30, schedule="cosine")
    assert compute_remask_rate(cfg, 0.0) == pytest.approx(0.85)
    assert compute_remask_rate(cfg, 0.5) == pytest.approx(0.5 * (0.85 + 0.30))
    assert compute_remask_rate(cfg, 1.0) == pytest.approx(0.30)


def test_baselines_exact_nfe():
    prompt = torch.tensor([[1, 2, 3]])
    v_model = ToyDiffusionForward()
    v_tok, v_stats = vanilla_generate(v_model, None, prompt, VanillaConfig(budget=8, gen_length=16), mask_id=11)
    assert v_tok.shape == (16,)
    assert v_stats["nfe"] == v_model.calls == 8
    assert v_stats["remaining_masks"] == 0

    r_model = ToyDiffusionForward()
    r_tok, r_stats = remdm_generate(
        r_model,
        None,
        prompt,
        ReMDMConfig(budget=8, gen_length=16),
        mask_id=11,
        generator=torch.Generator().manual_seed(0),
    )
    assert r_tok.shape == (16,)
    assert r_stats["nfe"] == r_model.calls == 8
    assert r_stats["remaining_masks"] == 0


def test_paper_preset_configs():
    model_id, cfg = build_paper_config("humaneval", "dream", "air", 64)
    assert "Dream" in model_id
    assert (cfg.budget, cfg.interval, cfg.heavy_ratio, cfg.r_start, cfg.r_end) == (64, 2, 2, 0.85, 0.60)

    model_id, cfg = build_paper_config("gsm8k", "llada", "air", 128)
    assert "LLaDA" in model_id
    assert (cfg.budget, cfg.interval, cfg.heavy_ratio, cfg.r_start, cfg.r_end) == (128, 24, 2, 0.55, 0.30)


def test_stats_helpers_match_manuscript():
    # Appendix A Table A1: HumanEval N=164, 32 steps r_end=0.7 -> 40/164 (24.4%, [18.5%, 31.5%])
    p, lo, hi = wilson_confidence_interval(40, 164)
    assert round(p * 100, 1) == 24.4
    assert round(lo * 100, 1) == 18.5
    assert round(hi * 100, 1) == 31.5

    # Table 6: two-sided exact McNemar p-values
    assert exact_mcnemar_test(5, 7) == pytest.approx(0.7744140625)
    assert exact_mcnemar_test(8, 6) == pytest.approx(0.79052734375)


def test_benchmark_extractors():
    doc = {"context": {"contexts": ["Abstract text."]}, "question": "Does it help?", "final_decision": "yes"}
    prompt = format_pubmedqa_prompt(doc)
    assert "Answer: yes" in prompt
    assert extract_pubmedqa_answer("Reasoning: ... Therefore Answer: YES.") == "yes"
    assert extract_gsm8k_answer("Let's compute 12 + 30 = 42.\n#### 42") == "42"


@pytest.mark.parametrize("score_mode", ["committed_margin", "top2_margin"])
def test_argmax_and_margin_variants(score_mode: str):
    prompt = torch.tensor([[1, 2, 3]])
    model = ToyDiffusionForward()
    cfg = AIRConfig(
        budget=8,
        gen_length=16,
        interval=2,
        heavy_ratio=1,
        score_mode=score_mode,
        sampling_alg="argmax",
        temperature=0.0,
    )
    tok, stats = air_generate(model, None, prompt, cfg, mask_id=11, record_trace=True)
    assert tok.shape == (16,)
    assert stats["nfe"] == model.calls == 8
    assert stats["remaining_masks"] == 0
    assert (tok != 11).all()


def test_dynamic_stopping_baselines():
    prompt = torch.tensor([[1, 2, 3]])

    eb_model = ToyDiffusionForward()
    eb_tok, eb_stats = eb_sampler_generate(
        eb_model, None, prompt, EBSamplerConfig(max_nfe=16, gen_length=12, eb_gamma=0.5), mask_id=11
    )
    assert eb_tok.shape == (12,)
    assert 1 <= eb_stats["nfe"] == eb_model.calls <= 16
    assert eb_stats["remaining_masks"] == 0

    sf_model = ToyDiffusionForward()
    sf_tok, sf_stats = slowfast_generate(
        sf_model, None, prompt, SlowFastConfig(max_nfe=16, gen_length=12), mask_id=11
    )
    assert sf_tok.shape == (12,)
    assert 1 <= sf_stats["nfe"] == sf_model.calls <= 16
    assert sf_stats["remaining_masks"] == 0


def test_ablation_plan_generation(tmp_path: Path):
    _, eb_cfg = build_paper_config("humaneval", "dream", "eb", 64)
    _, sf_cfg = build_paper_config("humaneval", "dream", "slowfast", 64)
    assert eb_cfg.max_nfe == 256
    assert sf_cfg.max_nfe == 256

    _, argmax_cfg = build_paper_config(
        "humaneval", "dream", "air", 64, score_mode="margin", sampling_alg_override="argmax", temperature_override=0.0
    )
    assert argmax_cfg.sampling_alg == "argmax"
    assert argmax_cfg.temperature == 0.0
    assert argmax_cfg.score_mode == "margin"

    jobs = build_ablation_plan("all", tmp_path)
    # 28 (Table 3) + 28 (Table 4) + 32 (Table 5) + 12 (Table 6 prob/margin/top2_margin/entropy + argmax)
    assert len(jobs) == 100
