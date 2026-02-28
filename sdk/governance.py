"""
NeuroVitals — Governance, Drift & Bias Monitoring
==================================================
Production governance layer for model risk management:

- Kolmogorov–Smirnov drift detection
- Bias / fairness reporting (demographic parity & equalized odds)
- Model calibration checks (Brier score)
- Aggregated governance report
"""

import numpy as np
from scipy import stats
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Drift Detection (KS Test)
# ---------------------------------------------------------------------------

def ks_drift_test(reference: np.ndarray,
                  current: np.ndarray,
                  alpha: float = 0.05) -> Dict[str, Any]:
    """Two-sample Kolmogorov–Smirnov test for feature distribution drift.

    Parameters
    ----------
    reference : 1-D array  Baseline / training distribution samples.
    current   : 1-D array  Recent production distribution samples.
    alpha     : float       Significance level.

    Returns
    -------
    dict with keys: ``statistic``, ``p_value``, ``drift_detected``.
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
    }


def multi_feature_drift(reference_dict: Dict[str, np.ndarray],
                        current_dict: Dict[str, np.ndarray],
                        alpha: float = 0.05) -> Dict[str, Dict]:
    """Run KS drift test per feature and return a combined report."""
    report: Dict[str, Dict] = {}
    for feat in reference_dict:
        if feat in current_dict:
            report[feat] = ks_drift_test(reference_dict[feat],
                                         current_dict[feat], alpha)
    return report


# ---------------------------------------------------------------------------
# Bias / Fairness
# ---------------------------------------------------------------------------

def demographic_parity(predictions: np.ndarray,
                       demographics: np.ndarray) -> Dict[str, float]:
    """Compute positive-prediction rate per demographic group.

    Parameters
    ----------
    predictions  : 1-D binary array (0/1 or bool).
    demographics : 1-D array of group labels (e.g. ``["A", "B", ...]``).

    Returns
    -------
    dict  {group_label: positive_rate}
    """
    groups = np.unique(demographics)
    rates: Dict[str, float] = {}
    for g in groups:
        mask = demographics == g
        if mask.sum() == 0:
            continue
        rates[str(g)] = float(predictions[mask].mean())
    return rates


def equalized_odds(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   demographics: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute TPR and FPR per demographic group for equalized-odds check.

    Returns
    -------
    dict  {group: {"tpr": …, "fpr": …}}
    """
    groups = np.unique(demographics)
    result: Dict[str, Dict[str, float]] = {}
    for g in groups:
        mask = demographics == g
        yt = y_true[mask]
        yp = y_pred[mask]

        pos = yt == 1
        neg = yt == 0
        tpr = float(yp[pos].mean()) if pos.sum() > 0 else float("nan")
        fpr = float(yp[neg].mean()) if neg.sum() > 0 else float("nan")
        result[str(g)] = {"tpr": tpr, "fpr": fpr}
    return result


def bias_fairness_report(predictions: np.ndarray,
                         demographics: np.ndarray,
                         y_true: Optional[np.ndarray] = None) -> Dict:
    """Combined fairness report including demographic parity and
    (optionally) equalized odds."""
    report: Dict[str, Any] = {
        "demographic_parity": demographic_parity(predictions, demographics),
    }
    if y_true is not None:
        report["equalized_odds"] = equalized_odds(y_true, predictions,
                                                  demographics)
    # Disparity metric: max − min positive rate
    rates = list(report["demographic_parity"].values())
    if rates:
        report["max_disparity"] = float(max(rates) - min(rates))
    return report


# ---------------------------------------------------------------------------
# Model Calibration
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score — lower is better.  Target < 0.15 for clinical use."""
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_curve_data(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           n_bins: int = 10) -> Dict[str, List[float]]:
    """Compute reliability diagram data (mean predicted vs actual per bin)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted: List[float] = []
    fraction_positive: List[float] = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        fraction_positive.append(float(y_true[mask].mean()))

    return {
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
    }


def model_calibration_check(y_true: np.ndarray,
                            y_prob: np.ndarray,
                            brier_threshold: float = 0.15) -> Dict:
    """Full calibration check including Brier score and reliability data."""
    bs = brier_score(y_true, y_prob)
    return {
        "brier_score": bs,
        "calibration_pass": bool(bs < brier_threshold),
        "reliability_curve": calibration_curve_data(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Governance Monitor
# ---------------------------------------------------------------------------

class GovernanceMonitor:
    """Aggregates drift, bias, and calibration checks into a single report.

    Usage::

        monitor = GovernanceMonitor()
        report = monitor.generate_report(
            reference_features={"hr": ref_hr, ...},
            current_features={"hr": cur_hr, ...},
            predictions=preds,
            demographics=demos,
            y_true=labels,
            y_prob=probs,
        )
    """

    def __init__(self, drift_alpha: float = 0.05,
                 brier_threshold: float = 0.15):
        self.drift_alpha = drift_alpha
        self.brier_threshold = brier_threshold

    def generate_report(self,
                        reference_features: Optional[Dict[str, np.ndarray]] = None,
                        current_features: Optional[Dict[str, np.ndarray]] = None,
                        predictions: Optional[np.ndarray] = None,
                        demographics: Optional[np.ndarray] = None,
                        y_true: Optional[np.ndarray] = None,
                        y_prob: Optional[np.ndarray] = None) -> Dict:
        """Return a comprehensive governance report dict."""
        report: Dict[str, Any] = {"status": "generated"}

        # Drift
        if reference_features is not None and current_features is not None:
            report["drift"] = multi_feature_drift(
                reference_features, current_features, self.drift_alpha)
            # Overall drift flag
            report["any_drift_detected"] = any(
                v.get("drift_detected", False)
                for v in report["drift"].values()
            )
        else:
            report["drift"] = None
            report["any_drift_detected"] = False

        # Bias
        if predictions is not None and demographics is not None:
            report["fairness"] = bias_fairness_report(
                predictions, demographics, y_true)
        else:
            report["fairness"] = None

        # Calibration
        if y_true is not None and y_prob is not None:
            report["calibration"] = model_calibration_check(
                y_true, y_prob, self.brier_threshold)
        else:
            report["calibration"] = None

        return report
