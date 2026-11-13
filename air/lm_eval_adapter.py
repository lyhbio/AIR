"""lm-evaluation-harness adapter for AIR, Vanilla, and ReMDM on Dream-7B and LLaDA-8B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, List, Optional

import torch
from tqdm import tqdm
from lm_eval.api.model import LM

from .backbones import BackboneAdapter, infer_backbone_type, load_backbone, resolve_mask_token_id
from .baselines import ReMDMConfig, VanillaConfig, remdm_generate, vanilla_generate
from .sampler import AIRConfig, air_generate


class DiffusionAIRHarnessLM(LM):
    """Generation-only lm-eval wrapper supporting Dream-7B-Instruct and LLaDA-8B-Instruct."""

    def __init__(
        self,
        pretrained: str,
        method: str = "air",
        backbone: str = "auto",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        revision: Optional[str] = None,
        seed: int = 0,
        metrics_path: Optional[str] = None,
        record_trace: bool = False,
        **sampler_kwargs: Any,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; pass device='cpu' or install CUDA PyTorch")
        if method not in {"air", "vanilla", "remdm"}:
            raise ValueError(f"Unsupported method: {method!r}")

        self.method = method
        self.model, self.tokenizer, detected_backbone = load_backbone(
            pretrained, device=str(self.device), dtype=dtype, revision=revision
        )
        self.backbone = detected_backbone if backbone == "auto" else backbone
        self.adapter = BackboneAdapter(self.model, backbone=self.backbone)
        self.mask_id = resolve_mask_token_id(self.tokenizer)
        self.seed = seed
        self.batch_size = 1
        self.record_trace = record_trace
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if method == "air":
            self.cfg = AIRConfig(backbone=self.backbone, **sampler_kwargs)
        elif method == "vanilla":
            self.cfg = VanillaConfig(backbone=self.backbone, **sampler_kwargs)
        else:
            self.cfg = ReMDMConfig(backbone=self.backbone, **sampler_kwargs)
        self.records: List[dict] = []

    def loglikelihood(self, requests):
        raise NotImplementedError("DiffusionAIRHarnessLM supports generate_until tasks only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("DiffusionAIRHarnessLM supports generate_until tasks only")

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def chat_template(self, chat_template=False):
        return self.tokenizer.get_chat_template()

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def generate_until(self, requests, disable_tqdm: bool = False):
        answers = []
        for request in tqdm(requests, desc=f"{self.backbone}-{self.method}", disable=disable_tqdm):
            context, generation_kwargs = request.args
            ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            context_hash = hashlib.sha256(context.encode()).hexdigest()
            per_prompt_seed = (int(context_hash[:16], 16) + self.seed) % (2**63 - 1)
            generator = torch.Generator(device=self.device).manual_seed(per_prompt_seed)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()

            if self.method == "air":
                gen_tokens, stats = air_generate(
                    self.adapter,
                    self.tokenizer,
                    ids,
                    self.cfg,
                    device=str(self.device),
                    mask_id=self.mask_id,
                    generator=generator,
                    record_trace=self.record_trace,
                )
            elif self.method == "vanilla":
                gen_tokens, stats = vanilla_generate(
                    self.adapter,
                    self.tokenizer,
                    ids,
                    self.cfg,
                    device=str(self.device),
                    mask_id=self.mask_id,
                    generator=generator,
                )
            else:
                gen_tokens, stats = remdm_generate(
                    self.adapter,
                    self.tokenizer,
                    ids,
                    self.cfg,
                    device=str(self.device),
                    mask_id=self.mask_id,
                    generator=generator,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - start

            generated_list = gen_tokens.tolist()
            eos_id = self.tokenizer.eos_token_id
            eos_ids = set(eos_id if isinstance(eos_id, list) else ([eos_id] if eos_id is not None else []))
            eos_pos = next((idx for idx, tok in enumerate(generated_list) if tok in eos_ids), None)
            text = self.tokenizer.decode(generated_list[:eos_pos], skip_special_tokens=True)

            stops = generation_kwargs.get("until", [])
            if isinstance(stops, str):
                stops = [stops]
            for stop in stops:
                if stop:
                    text = text.split(stop)[0]

            record = {
                "request_index": len(self.records),
                "doc_id": getattr(request, "doc_id", None),
                "prompt_sha256": context_hash,
                "generation_seconds": elapsed,
                "response": text,
                **stats,
            }
            self.records.append(record)
            if self.metrics_path is not None:
                with self.metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            answers.append(text)
            self.cache_hook.add_partial("generate_until", request.args, text)
        return answers
