"""Convert 40-class Coralscapes label masks into SeaQuery reef measurements."""

import numpy as np
from scipy import ndimage



CATEGORIES = {
    "live": [6, 17, 21, 22, 25, 27, 28, 31, 34, 36],
    "bleached": [4, 16, 19, 33],
    "dead": [3, 20, 23, 32, 37],
    "algae": [10],
}


def coverage_from_counts(counts):
    """Percentage of foreground pixels, excluding void (0) and background (13)."""
    counts = np.asarray(counts)
    if counts.shape != (40,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("Expected 40 nonnegative integer class counts.")
    foreground = int(counts.sum()) - int(counts[0]) - int(counts[13])
    return {
        name: 100.0 * int(counts[ids].sum()) / foreground if foreground else 0.0
        for name, ids in CATEGORIES.items()
    }


def measure_mask(mask):
    """Return four area percentages and a count of crown-of-thorns starfish."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or mask.dtype.kind not in "iu" or mask.min() < 0 or mask.max() > 39:
        raise ValueError("Expected a nonempty 2-D integer label mask with IDs 0 through 39.")
    metrics = coverage_from_counts(np.bincount(mask.ravel().astype(np.intp), minlength=40))
    labels, _ = ndimage.label(mask == 38, structure=ndimage.generate_binary_structure(2, 1))
    metrics["cots"] = int(np.count_nonzero(np.bincount(labels.ravel())[1:] > 100))
    return metrics

