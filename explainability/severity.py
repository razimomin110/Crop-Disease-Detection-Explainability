"""
Disease Severity Estimation.

Uses the Grad-CAM activation heatmap as a proxy for the spatial extent of
disease symptoms.  The heatmap is thresholded and the proportion of highly-
activated pixels is reported as an approximate severity percentage.

NOTE: This is a rough heuristic — not a ground-truth segmentation.  It is
intended to demonstrate the concept of severity estimation in the absence of
pixel-level annotations.
"""

import numpy as np


def estimate_severity(
    heatmap: np.ndarray,
    threshold: float = 0.45,
) -> dict:
    """Estimate disease severity from a Grad-CAM heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        2-D array with values in [0, 1] produced by ``GradCAM.generate_heatmap``.
    threshold : float
        Pixels with activation ≥ this value are considered "affected".

    Returns
    -------
    dict with keys:
        severity_pct : float
            Percentage of the image area estimated to be diseased.
        affected_pixels : int
            Number of pixels above the threshold.
        total_pixels : int
            Total pixels in the heatmap.
        severity_label : str
            Human-readable label: Healthy / Mild / Moderate / Severe.
    """
    total_pixels = heatmap.size
    affected_pixels = int(np.sum(heatmap >= threshold))
    severity_pct = (affected_pixels / total_pixels) * 100.0

    # Map percentage to a qualitative label
    if severity_pct < 5:
        label = "Healthy"
    elif severity_pct < 25:
        label = "Mild"
    elif severity_pct < 50:
        label = "Moderate"
    else:
        label = "Severe"

    return {
        "severity_pct": round(severity_pct, 2),
        "affected_pixels": affected_pixels,
        "total_pixels": total_pixels,
        "severity_label": label,
    }
