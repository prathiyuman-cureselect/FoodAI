"""Tests for NeuroVitals SDK — Governance Layer."""
import numpy as np
import pytest

from sdk.governance import (
    ks_drift_test,
    multi_feature_drift,
    demographic_parity,
    equalized_odds,
    bias_fairness_report,
    brier_score,
    calibration_curve_data,
    model_calibration_check,
    GovernanceMonitor,
)


class TestDrift:
    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = ks_drift_test(ref, cur)
        assert result["drift_detected"] is False
        assert result["p_value"] > 0.05

    def test_drift_shifted_distribution(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(3, 1, 500)  # shifted mean
        result = ks_drift_test(ref, cur)
        assert result["drift_detected"] is True
        assert result["p_value"] < 0.05

    def test_multi_feature_drift(self):
        rng = np.random.default_rng(42)
        ref = {"hr": rng.normal(70, 5, 100), "rmssd": rng.normal(0.04, 0.01, 100)}
        cur = {"hr": rng.normal(70, 5, 100), "rmssd": rng.normal(0.10, 0.01, 100)}
        report = multi_feature_drift(ref, cur)
        assert "hr" in report
        assert "rmssd" in report
        assert report["rmssd"]["drift_detected"] is True


class TestBias:
    def test_demographic_parity(self):
        preds = np.array([1, 1, 0, 0, 1, 0])
        demos = np.array(["A", "A", "A", "B", "B", "B"])
        rates = demographic_parity(preds, demos)
        assert "A" in rates
        assert "B" in rates

    def test_equalized_odds(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0])
        demos = np.array(["A", "A", "A", "B", "B", "B"])
        result = equalized_odds(y_true, y_pred, demos)
        assert "A" in result
        assert "tpr" in result["A"]
        assert "fpr" in result["A"]

    def test_bias_report_has_disparity(self):
        preds = np.array([1, 1, 1, 0, 0, 0])
        demos = np.array(["A", "A", "A", "B", "B", "B"])
        report = bias_fairness_report(preds, demos)
        assert "max_disparity" in report
        assert report["max_disparity"] == 1.0  # A=100%, B=0%


class TestCalibration:
    def test_brier_score_perfect(self):
        y_true = np.array([1.0, 0.0, 1.0])
        y_prob = np.array([1.0, 0.0, 1.0])
        assert brier_score(y_true, y_prob) == 0.0

    def test_brier_score_worst(self):
        y_true = np.array([1.0, 0.0])
        y_prob = np.array([0.0, 1.0])
        assert brier_score(y_true, y_prob) == 1.0

    def test_calibration_check_pass(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        result = model_calibration_check(y_true, y_prob)
        assert result["calibration_pass"] is True

    def test_calibration_curve_data(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        data = calibration_curve_data(y_true, y_prob, n_bins=5)
        assert "mean_predicted" in data
        assert "fraction_positive" in data


class TestGovernanceMonitor:
    def test_empty_report(self):
        monitor = GovernanceMonitor()
        report = monitor.generate_report()
        assert report["status"] == "generated"
        assert report["any_drift_detected"] is False

    def test_full_report(self):
        rng = np.random.default_rng(42)
        monitor = GovernanceMonitor()
        report = monitor.generate_report(
            reference_features={"hr": rng.normal(70, 5, 100)},
            current_features={"hr": rng.normal(70, 5, 100)},
            predictions=np.array([1, 0, 1, 0]),
            demographics=np.array(["A", "A", "B", "B"]),
            y_true=np.array([1, 0, 1, 0]),
            y_prob=np.array([0.9, 0.1, 0.8, 0.2]),
        )
        assert "drift" in report
        assert "fairness" in report
        assert "calibration" in report
