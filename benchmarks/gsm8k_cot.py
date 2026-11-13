"""Zero-shot Chain-of-Thought prompt formatting and numeric extraction for GSM8K."""

from __future__ import annotations

import re
from typing import Any, Dict


_NUM_REGEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_HASH_REGEX = re.compile(r"####\s*(-?[\d,.$]+)")


def format_gsm8k_prompt(doc: Dict[str, Any]) -> str:
    """Formats a GSM8K test problem using zero-shot Chain-of-Thought prompting."""
    return (
        "Solve the following math word problem step by step. "
        "End your final line with '#### <final_number>'.\n\n"
        f"Question: {doc['question']}\n\n"
        "Answer:"
    )


def _normalize_number(raw: str) -> str:
    cleaned = raw.replace("$", "").replace(",", "").strip().rstrip(".")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned


def extract_gsm8k_answer(text: str) -> str:
    """Extracts the final numeric answer using flexible GSM8K matching."""
    hash_matches = _HASH_REGEX.findall(text)
    if hash_matches:
        return _normalize_number(hash_matches[-1])
    num_matches = _NUM_REGEX.findall(text)
    if not num_matches:
        return ""
    last_tuple = num_matches[-1]
    token = last_tuple[0] or last_tuple[1]
    return _normalize_number(token)


def score_pubmed_or_gsm_gold(raw_answer: str) -> str:
    """Extracts ground-truth number after '####' from GSM8K `answer` field."""
    return extract_gsm8k_answer(raw_answer)


def score_gsm8k_prediction(doc: Dict[str, Any], response_text: str) -> Dict[str, float]:
    """Computes flexible exact-match accuracy for a GSM8K item."""
    pred = extract_gsm8k_answer(response_text)
    gold = score_pubmed_or_gsm_gold(str(doc["answer"]))
    return {"exact_match": float(bool(pred) and pred == gold)}
