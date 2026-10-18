"""Methods package with lazy imports to avoid dependency issues."""

from . import granger, correlation, consensus


# Lazy imports for optional dependencies
def _get_tigramite_pcmci():
    """Lazy import of tigramite_pcmci to handle optional dependencies."""
    try:
        from . import tigramite_pcmci

        return tigramite_pcmci
    except ImportError as e:
        raise ImportError(f"PCMCI+ requires tigramite and its dependencies: {e}")


def _get_transfer_entropy():
    """Lazy import of transfer_entropy to handle optional dependencies."""
    try:
        from . import transfer_entropy

        return transfer_entropy
    except ImportError as e:
        raise ImportError(f"Transfer Entropy requires pyinform: {e}")


def _get_varlingam():
    """Lazy import of varlingam to handle optional lingam dependency."""
    try:
        from . import varlingam

        return varlingam
    except ImportError as e:
        raise ImportError(f"VAR-LiNGAM requires lingam: {e}")


def _get_lpcmci():
    """Lazy import of lpcmci to handle optional tigramite dependency."""
    try:
        from . import lpcmci

        return lpcmci
    except ImportError as e:
        raise ImportError(f"LPCMCI requires tigramite ≥5.2: {e}")


def _get_predictive_baseline():
    """Lazy import of predictive baseline (RF/XGBoost)."""
    try:
        from . import predictive_baseline

        return predictive_baseline
    except ImportError as e:
        raise ImportError(f"Predictive baseline requires scikit-learn: {e}")


__all__ = [
    "granger",
    "correlation",
    "consensus",
    "_get_tigramite_pcmci",
    "_get_transfer_entropy",
    "_get_varlingam",
    "_get_lpcmci",
    "_get_predictive_baseline",
]
