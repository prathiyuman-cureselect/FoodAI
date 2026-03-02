"""Tests for NeuroVitals SDK — Feature Engineering."""
import math
import numpy as np
import pytest

from sdk.feature_engineer import (
    detect_peaks,
    compute_heart_rate,
    compute_rr_intervals,
    compute_rmssd_from_peaks,
    compute_sdnn,
    compute_lf_hf_ratio,
    compute_rsa,
    compute_ptv,
    spectral_entropy,
    compute_sympathetic_index,
    compute_sqi,
    compute_stress_reactivity,
    extract_all_features,
)


@pytest.fixture
def sine_signal():
    """1 Hz sine wave at 30 FPS → 60 BPM heart rate."""
    fs = 30.0
    t = np.arange(0, 10, 1.0 / fs)  # 10 seconds
    signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz
    return signal, fs


class TestPeakDetection:
    def test_detects_peaks(self, sine_signal):
        signal, fs = sine_signal
        peaks = detect_peaks(signal, fs)
        # 10 seconds of 1 Hz → ~10 peaks
        assert len(peaks) >= 8
        assert len(peaks) <= 12

    def test_empty_signal(self):
        peaks = detect_peaks(np.array([]), fs=30.0)
        assert len(peaks) == 0


class TestTimeDomainHRV:
    def test_heart_rate_1hz(self, sine_signal):
        signal, fs = sine_signal
        peaks = detect_peaks(signal, fs)
        hr = compute_heart_rate(peaks, fs)
        # 1 Hz → ~60 BPM
        assert 50 < hr < 70

    def test_rmssd_finite(self, sine_signal):
        signal, fs = sine_signal
        peaks = detect_peaks(signal, fs)
        rmssd = compute_rmssd_from_peaks(peaks, fs)
        assert not math.isnan(rmssd)
        assert rmssd >= 0

    def test_sdnn_finite(self, sine_signal):
        signal, fs = sine_signal
        peaks = detect_peaks(signal, fs)
        sdnn = compute_sdnn(peaks, fs)
        assert not math.isnan(sdnn)
        assert sdnn >= 0

    def test_rr_intervals(self, sine_signal):
        signal, fs = sine_signal
        peaks = detect_peaks(signal, fs)
        rr = compute_rr_intervals(peaks, fs)
        assert len(rr) > 0
        # Each RR interval should be ~1 second for 1 Hz
        assert np.all(rr > 0.5)
        assert np.all(rr < 1.5)


class TestFrequencyDomain:
    def test_lf_hf_finite(self, sine_signal):
        signal, fs = sine_signal
        ratio = compute_lf_hf_ratio(signal, fs)
        # May be nan or finite depending on signal — just check no crash
        assert isinstance(ratio, float)

    def test_rsa_non_negative(self, sine_signal):
        signal, fs = sine_signal
        rsa = compute_rsa(signal, fs)
        assert rsa >= 0

    def test_sympathetic_index_range(self, sine_signal):
        signal, fs = sine_signal
        idx = compute_sympathetic_index(signal, fs)
        if not math.isnan(idx):
            assert 0.0 <= idx <= 1.0


class TestOther:
    def test_spectral_entropy_positive(self, sine_signal):
        signal, fs = sine_signal
        ent = spectral_entropy(signal, fs)
        assert ent > 0

    def test_sqi_range(self, sine_signal):
        signal, _ = sine_signal
        sqi = compute_sqi(signal)
        assert 0.0 <= sqi <= 1.0

    def test_ptv_finite(self, sine_signal):
        signal, fs = sine_signal
        peaks = detect_peaks(signal, fs)
        ptv = compute_ptv(peaks, fs)
        assert isinstance(ptv, float)

    def test_stress_reactivity(self):
        score = compute_stress_reactivity(0.02, 0.03, 3.0)
        assert 0.0 <= score <= 1.0

    def test_stress_reactivity_nan(self):
        score = compute_stress_reactivity(float("nan"), 0.03, 3.0)
        assert math.isnan(score)


class TestExtractAll:
    def test_returns_all_keys(self, sine_signal):
        signal, fs = sine_signal
        features = extract_all_features(signal, fs)
        expected_keys = {
            "heart_rate_bpm", "rmssd", "sdnn", "lf_hf_ratio",
            "rsa", "ptv", "spectral_entropy", "sympathetic_index",
            "signal_quality_index", "stress_reactivity",
        }
        assert expected_keys == set(features.keys())

    def test_all_values_are_float(self, sine_signal):
        signal, fs = sine_signal
        features = extract_all_features(signal, fs)
        for k, v in features.items():
            assert isinstance(v, float), f"{k} is not float"
