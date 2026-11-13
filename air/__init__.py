"""AIR: Accelerating Inference via Retrospective Remasking for Masked Diffusion Language Models."""

from .backbones import BackboneAdapter, DreamForwardWrapper, LLaDAForwardWrapper, load_backbone
from .baselines import ReMDMConfig, VanillaConfig, remdm_generate, vanilla_generate
from .sampler import AIRConfig, air_generate, compute_remask_rate, compute_verification_scores, sample_tokens

__all__ = [
    "AIRConfig",
    "BackboneAdapter",
    "DreamForwardWrapper",
    "LLaDAForwardWrapper",
    "ReMDMConfig",
    "VanillaConfig",
    "air_generate",
    "compute_remask_rate",
    "compute_verification_scores",
    "load_backbone",
    "remdm_generate",
    "sample_tokens",
    "vanilla_generate",
]
