"""Adapter connecting causal-audit recommendations to autocause configuration.

Maps the output of causal-audit's RiskAwareGatekeeper.analyze() to autocause's
run_causal_discovery_workflow() parameters. No hard import dependency on
causal-audit; the package is optional.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def try_import_causal_audit():
    """Attempt to import causal-audit.

    Returns
    -------
    tuple
        (module, True) if available, (None, False) otherwise.
    """
    try:
        import causal_audit

        return causal_audit, True
    except ImportError:
        return None, False


def map_audit_to_config(
    audit_output: dict[str, Any],
    user_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map causal-audit recommendation to autocause workflow parameters.

    Parameters
    ----------
    audit_output : dict
        Output from causal-audit's RiskAwareGatekeeper.analyze().
        Expected structure:
        - policy: object with .decision, .recommended_method, .confidence
        - risk_profile: object with .risks dict
        - audit_evidence: object with .safe_tau_max dict, .t_eff dict
    user_overrides : dict | None
        User-specified parameters that take precedence over audit recommendations.

    Returns
    -------
    dict
        Configuration dict compatible with run_causal_discovery_workflow() kwargs.
        Keys may include: method_config, alpha, tau_max, enable_robustness,
        enable_surrogates, fdr_method.

    Notes
    -----
    For each parameter, logs whether it was set from audit or overridden by user.
    """
    if user_overrides is None:
        user_overrides = {}

    config: dict[str, Any] = {}

    # Extract policy
    policy = audit_output.get("policy")
    if policy is None:
        logger.warning("No policy found in audit output; returning empty config.")
        return config

    # Extract attributes (handle both object and dict forms)
    decision = _get_attr(policy, "decision", "recommend")
    recommended_method = _get_attr(policy, "recommended_method", None)
    confidence = _get_attr(policy, "confidence", 0.5)

    # Extract risk profile
    risk_profile = audit_output.get("risk_profile")
    confounding_risk = 0.0
    if risk_profile is not None:
        risks = _get_attr(risk_profile, "risks", {})
        if isinstance(risks, dict):
            confounding_entry = risks.get(
                "ConfoundingRisk", risks.get("confounding", {})
            )
            if isinstance(confounding_entry, dict):
                confounding_risk = confounding_entry.get("mean", 0.0)
            elif isinstance(confounding_entry, (int, float)):
                confounding_risk = float(confounding_entry)

    # Extract audit evidence for tau_max
    audit_evidence = audit_output.get("audit_evidence")
    safe_tau_max = None
    if audit_evidence is not None:
        tau_max_info = _get_attr(audit_evidence, "safe_tau_max", {})
        if isinstance(tau_max_info, dict):
            safe_tau_max = tau_max_info.get("data_driven")

    # Map recommended method to method_config
    if "method_config" in user_overrides:
        config["method_config"] = user_overrides["method_config"]
        logger.info("method_config: set from user override")
    elif recommended_method:
        method_map = {
            "PCMCI+": {
                "granger": {"enabled": False},
                "transfer_entropy": {"enabled": False},
                "pcmci": {"enabled": True},
            },
            "Granger": {
                "granger": {"enabled": True},
                "transfer_entropy": {"enabled": False},
                "pcmci": {"enabled": False},
            },
            "LPCMCI": {
                "granger": {"enabled": False},
                "transfer_entropy": {"enabled": False},
                "pcmci": {"enabled": True},
                "lpcmci": {"enabled": True},
            },
        }
        config["method_config"] = method_map.get(recommended_method, {})
        logger.info(
            f"method_config: set from audit recommendation ({recommended_method})"
        )

    # Map tau_max
    if "tau_max" in user_overrides:
        config["tau_max"] = user_overrides["tau_max"]
        logger.info("tau_max: set from user override")
    elif safe_tau_max is not None:
        config["tau_max"] = int(safe_tau_max)
        logger.info(f"tau_max: set from audit evidence ({safe_tau_max})")

    # Map confidence to robustness/surrogate flags
    if confidence < 0.5:
        if "enable_robustness" not in user_overrides:
            config["enable_robustness"] = True
            logger.info("enable_robustness: enabled due to low audit confidence")
        if "enable_surrogates" not in user_overrides:
            config["enable_surrogates"] = True
            logger.info("enable_surrogates: enabled due to low audit confidence")

    # Map confounding risk to LPCMCI
    if confounding_risk > 0.7:
        if "method_config" not in config:
            config["method_config"] = {}
        if "lpcmci" not in config.get("method_config", {}):
            config.setdefault("method_config", {})["lpcmci"] = {"enabled": True}
            logger.info("LPCMCI: enabled due to high confounding risk")

    # Handle abstention
    if decision == "abstain":
        logger.warning(
            "causal-audit recommends ABSTAINING from causal discovery on this dataset. "
            "Proceeding with caution; enable robustness and surrogate validation."
        )
        config.setdefault("enable_robustness", True)
        config.setdefault("enable_surrogates", True)
        config.setdefault("enable_stability", True)

    # Apply remaining user overrides
    for key, value in user_overrides.items():
        if key not in config:
            config[key] = value
            logger.info(f"{key}: set from user override")

    return config


def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
