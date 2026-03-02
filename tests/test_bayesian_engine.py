"""Tests for NeuroVitals SDK — Bayesian Mental Health Engine."""
import math
import pytest

from sdk.bayesian_engine import (
    bayesian_update,
    BayesianMentalHealthEngine,
)


class TestBayesianUpdate:
    def test_high_likelihood_increases_posterior(self):
        post = bayesian_update(0.2, 0.9)
        assert post > 0.2  # posterior > prior when likelihood is high

    def test_low_likelihood_decreases_posterior(self):
        post = bayesian_update(0.5, 0.1)
        assert post < 0.5

    def test_neutral_likelihood(self):
        post = bayesian_update(0.5, 0.5)
        assert abs(post - 0.5) < 0.01

    def test_clips_extreme_prior(self):
        post = bayesian_update(0.0, 0.9)
        assert 0.0 < post < 1.0

    def test_clips_extreme_likelihood(self):
        post = bayesian_update(0.5, 1.0)
        assert 0.0 < post < 1.0


class TestBayesianEngine:
    @pytest.fixture
    def engine(self):
        return BayesianMentalHealthEngine()

    def test_update_returns_all_conditions(self, engine):
        features = {"rmssd": 0.02, "lf_hf_ratio": 3.0,
                     "spectral_entropy": 4.0, "stress_reactivity": 0.7,
                     "rsa": 0.0005, "ptv": 0.2, "sdnn": 0.02,
                     "sympathetic_index": 0.7}
        posteriors = engine.update(features)
        assert "depression" in posteriors
        assert "anxiety" in posteriors
        assert "burnout" in posteriors
        assert "ptsd" in posteriors

    def test_high_risk_features_increase_posteriors(self, engine):
        # Features indicating high risk
        features = {"rmssd": 0.01, "lf_hf_ratio": 4.0,
                     "spectral_entropy": 5.0, "stress_reactivity": 0.9,
                     "rsa": 0.0001, "ptv": 0.3, "sdnn": 0.01,
                     "sympathetic_index": 0.8}
        posteriors = engine.update(features)
        # All posteriors should be above their priors
        for cond, post in posteriors.items():
            assert post > engine.priors[cond]

    def test_classify_low(self, engine):
        assert engine.classify({"depression": 0.1, "anxiety": 0.1}) == "Low"

    def test_classify_moderate(self, engine):
        assert engine.classify({"depression": 0.3, "anxiety": 0.1}) == "Moderate"

    def test_classify_high(self, engine):
        assert engine.classify({"depression": 0.55}) == "High"

    def test_classify_critical(self, engine):
        assert engine.classify({"depression": 0.8}) == "Critical"

    def test_full_inference(self, engine):
        features = {"rmssd": 0.02, "lf_hf_ratio": 2.0,
                     "spectral_entropy": 3.0}
        result = engine.full_inference(features)
        assert "posteriors" in result
        assert "risk_class" in result
        assert "dominant_condition" in result
        assert result["risk_class"] in ("Low", "Moderate", "High", "Critical")
