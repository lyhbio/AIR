"""Statistical analysis utilities: exact 95% Wilson score CIs and two-sided McNemar test.

Used for Appendix A (Table A1) and Section 4.7 verification-score ablation (Table 6).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from scipy.stats import binomtest


def wilson_confidence_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float, float]:
    """Computes Pass@1 point estimate and two-sided 95% Wilson score interval.

    Returns:
        (proportion, lower_bound, upper_bound) in [0, 1].
    """
    if n <= 0 or not (0 <= k <= n):
        raise ValueError(f"Invalid counts k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return p, max(0.0, center - half_width), min(1.0, center + half_width)


def exact_mcnemar_test(wins_b_over_a: int, wins_a_over_b: int) -> float:
    """Computes the exact two-sided McNemar p-value from discordant item counts."""
    total_discordant = wins_b_over_a + wins_a_over_b
    if total_discordant == 0:
        return 1.0
    return float(binomtest(wins_b_over_a, n=total_discordant, p=0.5, alternative="two-sided").pvalue)


def paired_outcomes_mcnemar(outcomes_a: Sequence[bool], outcomes_b: Sequence[bool]) -> Tuple[int, int, float]:
    """Computes (b_only_wins, a_only_wins, exact_two_sided_p) for paired binary results."""
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("Paired outcome sequences must have equal length")
    b_wins = sum((not a) and b for a, b in zip(outcomes_a, outcomes_b))
    a_wins = sum(a and (not b) for a, b in zip(outcomes_a, outcomes_b))
    return b_wins, a_wins, exact_mcnemar_test(b_wins, a_wins)
