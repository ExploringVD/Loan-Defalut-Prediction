"""
APPROVE / REVIEW / REJECT decision bands - one small module shared by business.py, explain.py and the API
(kept free of heavy imports so the API starts fast).

The cut-offs are chosen in src/business.py and stored in models/model_meta.json -> decision_cutoffs.
"""

import numpy as np

BANDS = ["APPROVE", "REVIEW", "REJECT"]


def assign_bands(p, review_cutoff: float, reject_cutoff: float) -> np.ndarray:
    """APPROVE if p < review_cutoff, REJECT if p >= reject_cutoff, REVIEW in between."""
    p = np.asarray(p)
    return np.where(p >= reject_cutoff, "REJECT", np.where(p >= review_cutoff, "REVIEW", "APPROVE"))


def decision_band(p: float, cutoffs: dict) -> str:
    """Band of one probability, with the cut-offs dict from model_meta.json."""
    return str(assign_bands([p], cutoffs["review"], cutoffs["reject"])[0])
