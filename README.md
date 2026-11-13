# AIR: Retrospective Remasking for Step-Efficient Diffusion Language Model Inference

Official reference implementation and reproduction suite for:

> **AIR: Retrospective Remasking for Step-Efficient Diffusion Language Model Inference**  
> Zhongcheng Yang and Yinghong Li  
> *Chongqing Key Laboratory of Big Data for Bio Intelligence, Chongqing University of Posts and Telecommunications*

---

## Overview

Masked diffusion language models (MDLMs) enable parallel token generation, but under constrained forward-pass budgets ($T \in \{32, 64, 128\}$) early parallel commitments made from incomplete context can propagate errors across later rounds. **AIR** addresses this without extra forward passes or auxiliary training via a three-phase single-forward execution loop (Algorithm 1 in the paper):

1. **Phase 1 — Sample:** Commit top-$N_{\text{transfer}}$ masked positions using the backbone sampler (skipped at *Heavy Interval* checkpoints to let corrected context stabilize before new commitments).
2. **Phase 2 — Verify:** At periodic checkpoints ($t \bmod K = 0,\ t < T$), reuse the **same forward-pass logits** $\mathbf{Z}^{(t)}$ to rank tokens committed during the preceding interval $\mathcal{A}_t$ and remask the lowest-scoring $\operatorname{Bottom}\text{-}N_{\text{drop}}$ tokens using a progress-adaptive cosine decay rate $r(p)$.
3. **Phase 3 — Accumulate & Reset:** Reset the interval accumulator at checkpoints so each checkpoint verifies only the immediately preceding interval's commitments.

---

## Repository Structure

```text
.
├── air/
│   ├── __init__.py          # Public API exports
│   ├── sampler.py           # Core AIR implementation (Algorithm 1, Heavy Interval, cosine/linear/fixed, prob/margin/entropy)
│   ├── backbones.py         # Unified Dream-7B-Instruct (causal logit shift) and LLaDA-8B-Instruct adapters
│   ├── baselines.py         # Fixed-budget Vanilla (entropy) and ReMDM baseline samplers
│   └── lm_eval_adapter.py   # Drop-in lm-evaluation-harness wrapper for both backbones
├── benchmarks/
│   ├── gsm8k_cot.py         # GSM8K zero-shot CoT prompt & numeric extractor
│   └── pubmedqa_cot.py      # PubMedQA zero-shot CoT prompt & yes/no/maybe extractor
├── configs/
│   ├── code_generation.json # Hyperparameters for HumanEval & MBPP (Table 1)
│   ├── reasoning_qa.json    # Hyperparameters for GSM8K & PubMedQA (Table 2)
│   └── ablations.json       # Sweep and ablation specifications (Tables 3-6, Appendix A)
├── scripts/
│   ├── quick_start.py       # Single-prompt generation demo (Dream-7B / LLaDA-8B)
│   ├── run_eval.py          # Benchmark CLI wrapper via lm-evaluation-harness
│   ├── stats_analysis.py    # Exact two-sided McNemar test & 95% Wilson score CI
│   └── plot_pareto_fig3.py  # Reproduces Figure 3 (Accuracy vs. NFE Pareto frontier)
├── tests/
│   └── test_air_sampler.py  # CPU unit tests for Algorithm 1 invariants and statistics
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/lyhbio/AIR.git
cd AIR
conda create -n air python=3.10 -y
conda activate air
pip install -r requirements.txt
pip install -e .
```

Verify the installation with CPU unit tests (no GPU or model weights required):

```bash
pytest -v
```

---

## Quick Start

### Python API

```python
import torch
from air import AIRConfig, BackboneAdapter, air_generate, load_backbone

model, tokenizer, backbone = load_backbone(
    "Dream-org/Dream-v0-Instruct-7B",  # or "GSAI-ML/LLaDA-8B-Instruct"
    device="cuda",
    dtype="bfloat16",
)
adapter = BackboneAdapter(model, backbone=backbone)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Python function to check if a number is prime."}],
    tokenize=False,
    add_generation_prompt=True,
)
prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]

cfg = AIRConfig(
    budget=64,
    gen_length=256,
    interval=2,
    heavy_ratio=2,
    r_start=0.85,
    r_end=0.60,
    score_mode="prob",
    backbone=backbone,
)

tokens, stats = air_generate(adapter, tokenizer, prompt_ids, cfg, device="cuda")
print(tokenizer.decode(tokens.tolist(), skip_special_tokens=True))
print(stats)
```

### Command-Line Demo

```bash
python scripts/quick_start.py --backbone dream --budget 64 --r-end 0.60
```

---

## Reproducing Paper Configurations

All paper hyperparameter presets are stored under `configs/`:

- **Code Generation (`HumanEval`, `MBPP` — Table 1):** `r_start = 0.85`, `K = 2`, `rho_heavy = 2`, budget-specific `r_end in {0.70, 0.60, 0.30, 0.40}` for `T in {32, 64, 128, 256}`.
- **Reasoning & Medical QA (`GSM8K`, `PubMedQA` — Table 2):** `r_start = 0.55`, `r_end = 0.30`, `K = 0.1875 * T` (`K in {6, 12, 24, 48}`), `rho_heavy = 2`.

Inspect or run any configuration via `scripts/run_eval.py`:

```bash
# Print exact preset configuration without loading weights
python scripts/run_eval.py --task humaneval --backbone dream --method air --steps 64 --print-config-only

# Run evaluation with lm-evaluation-harness
python scripts/run_eval.py --task humaneval --backbone dream --method air --steps 64 --output outputs/he_64.json
```

Reproduce Figure 3 (Accuracy–NFE Pareto Frontier):

```bash
python scripts/plot_pareto_fig3.py --out-dir outputs
```

---

## Statistical Verification

`scripts/stats_analysis.py` implements the exact two-sided McNemar test (Section 4.7, Table 6) and two-sided 95% Wilson score confidence intervals (Appendix A, Table A1).

---

## License

This repository is released under the [MIT License](LICENSE).
