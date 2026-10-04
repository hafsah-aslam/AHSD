"""S3 two-sided score and post-hoc OOD scores. Higher = more attack, always."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def two_sided_stress(stress: np.ndarray, median_benign_val: float) -> np.ndarray:
    """|log(stress / median benign val stress)|: both too-high and too-low stress are anomalous."""
    return np.abs(np.log(np.maximum(stress, 1e-12) / max(median_benign_val, 1e-12)))


def regularity(delta_norm: np.ndarray) -> np.ndarray:
    """var_t ‖Δ_t‖ per window; its LOW tail is anomalous, so the score is the negative."""
    return -np.var(delta_norm, axis=1)


def rank_normalise(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Empirical CDF of x under the reference (benign validation) distribution."""
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def s3_score(stress, delta_norm, val_benign_stress, val_benign_delta_norm) -> np.ndarray:
    """Mean of rank-normalised two-sided stress and regularity; no learned weights."""
    med = float(np.median(val_benign_stress))
    a = rank_normalise(two_sided_stress(stress, med), two_sided_stress(val_benign_stress, med))
    b = rank_normalise(regularity(delta_norm), regularity(val_benign_delta_norm))
    return (a + b) / 2


def ranks01(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 1) / max(len(x) - 1, 1)
