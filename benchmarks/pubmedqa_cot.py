"""Zero-shot Chain-of-Thought prompt formatting and answer extraction for PubMedQA."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_ANSWER_REGEX = re.compile(r"Answer\s*:\s*(yes|no|maybe)\b", flags=re.IGNORECASE)
_FALLBACK_REGEX = re.compile(r"\b(yes|no|maybe)\b", flags=re.IGNORECASE)


def format_pubmedqa_prompt(doc: Dict[str, Any]) -> str:
    """Formats a PubMedQA example with zero-shot CoT instructions."""
    contexts = doc["context"]["contexts"] if isinstance(doc["context"], dict) else doc["context"]
    ctx_block = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
    return (
        "Read the medical abstract below and answer the question.\n"
        "1. First, think step-by-step and write down your reasoning.\n"
        "2. Finally, you MUST end your response with the exact format "
        "'Answer: yes', 'Answer: no', or 'Answer: maybe'.\n\n"
        f"Abstract:\n{ctx_block}\n\n"
        f"Question: {doc['question']}\n\n"
        "### Response:\n"
        "Reasoning:"
    )


def extract_pubmedqa_answer(response_text: str) -> str:
    """Extracts the yes/no/maybe decision from model output."""
    matches = _ANSWER_REGEX.findall(response_text)
    if matches:
        return matches[-1].lower().strip()
    fallback = _FALLBACK_REGEX.findall(response_text)
    return fallback[-1].lower().strip() if fallback else ""


def score_pubmedqa_prediction(doc: Dict[str, Any], response_text: str) -> Dict[str, float]:
    """Computes exact-match accuracy for a single PubMedQA sample."""
    pred = extract_pubmedqa_answer(response_text)
    gold = str(doc["final_decision"]).lower().strip()
    return {"exact_match": float(pred == gold)}
