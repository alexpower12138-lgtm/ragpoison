"""Fusion and the filter/refill policy (WP6).

Planned modules:
    threshold.py     mean of active signals against a calibrated threshold
    weighted.py      calibrated weights multiplied by the retrieval prior
    logistic.py      logistic regression fitted on the calibration split only
    filter_policy.py drop, refill to k, and the insufficient-context fallback
"""

from __future__ import annotations
