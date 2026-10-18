"""Monte Carlo tail probabilities for explicitly generated null statistics."""

import numpy as np


def monte_carlo_pvalue(observed, surrogates, alternative="greater"):
    """Return (1 + equally-or-more-extreme draws) / (B + 1).

    Ties count against rejection. Every statistic must be finite; silently
    discarding failed null draws can bias inference. This finite-simulation
    correction does not establish that the chosen surrogate scheme is a valid
    null model (in particular for dependent time series or selected edges).

    Reference: Phipson & Smyth (2010), doi:10.2202/1544-6115.1585.
    """
    values = np.asarray(surrogates, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("At least one surrogate statistic is required")
    if not np.isfinite(observed) or not np.isfinite(values).all():
        raise ValueError("Observed and surrogate statistics must all be finite")
    tolerance = 100 * np.finfo(float).eps * abs(float(observed))
    if alternative == "greater":
        extreme = np.count_nonzero(values >= observed - tolerance)
    elif alternative == "less":
        extreme = np.count_nonzero(values <= observed + tolerance)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return float((extreme + 1) / (len(values) + 1))
