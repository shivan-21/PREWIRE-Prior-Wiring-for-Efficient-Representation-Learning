"""Data-equivalence: how many labels a pre-trained prior is worth.

For a method that reaches accuracy ``a`` with ``n`` labels, the equivalent label count is the
number of labels ``n_eq`` the baseline (random init) needs to reach the same accuracy, found by
linear interpolation of the baseline's accuracy against ``log(n)``. The prior is then worth
``n_eq - n`` extra labels, or a ``n_eq / n`` data-efficiency ratio.

If the method's accuracy is below the baseline's at its smallest count, the prior is worth no
labels (or fewer than that count); if above the baseline's best accuracy, the equivalent count
lies beyond the measured range and is reported as ``inf``.
"""

import math

import numpy as np


def equivalent_labels(
    accuracy: float, baseline_counts: list[int], baseline_accuracies: list[float]
) -> float:
    """Baseline label count needed to reach ``accuracy`` (log-linear interpolation).

    The baseline curve is made monotone (running maximum) first, so interpolation is well defined.

    Returns:
        A count within ``[min(counts), max(counts)]``; ``nan`` if ``accuracy`` is below the
        baseline's smallest-count accuracy, ``inf`` if above its best accuracy.
    """
    order = np.argsort(baseline_counts)
    counts = np.asarray(baseline_counts, dtype=float)[order]
    accs = np.maximum.accumulate(np.asarray(baseline_accuracies, dtype=float)[order])
    if accuracy < accs[0]:
        return math.nan
    if accuracy > accs[-1]:
        return math.inf
    # First segment whose upper end reaches the accuracy.
    i = int(np.searchsorted(accs, accuracy))
    if i == 0 or accs[i] == accs[i - 1]:
        return float(counts[i])
    t = (accuracy - accs[i - 1]) / (accs[i] - accs[i - 1])
    return float(np.exp(np.log(counts[i - 1]) + t * (np.log(counts[i]) - np.log(counts[i - 1]))))


def data_equivalence_table(
    method_counts: list[int],
    method_accuracies: list[float],
    baseline_counts: list[int],
    baseline_accuracies: list[float],
) -> list[dict]:
    """For each of the method's label counts: accuracy, equivalent baseline count and ratio."""
    rows = []
    for n, acc in zip(method_counts, method_accuracies, strict=True):
        n_eq = equivalent_labels(acc, baseline_counts, baseline_accuracies)
        rows.append(
            {
                "labeled_n": n,
                "accuracy": acc,
                "equivalent_n": n_eq,
                "labels_saved": n_eq - n,
                "ratio": n_eq / n,
            }
        )
    return rows
