import numpy as np


def binned_distribution(values, prefix: str, n_bins: int = 10) -> dict[str, float]:
    """
    Compute ARFF-style binned distribution features for one axis.

    Splits [min, max] into equal-width bins and returns the fraction of values
    falling in each bin. Keys are ``{prefix}0`` .. ``{prefix}{n_bins - 1}``.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    keys = [f"{prefix}{i}" for i in range(n_bins)]

    if n == 0:
        return dict.fromkeys(keys, 0.0)

    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmin == vmax:
        result = dict.fromkeys(keys, 0.0)
        result[keys[0]] = 1.0
        return result

    counts, _ = np.histogram(arr, bins=n_bins, range=(vmin, vmax))
    fractions = counts / n
    return {keys[i]: float(fractions[i]) for i in range(n_bins)}
