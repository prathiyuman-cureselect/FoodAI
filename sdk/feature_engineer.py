"""
NeuroVitals — Clinical Feature Engineering Layer
=================================================
Extracts neurocardiac biomarkers from rPPG-derived pulse signals for
mental health phenotyping.

Feature set:
  HR, RMSSD, SDNN, LF/HF ratio, RSA (HF power), PTV, Spectral Entropy,
  Sympathetic Dominance Index, Stress Reactivity Score, SQI.
"""

import math
import numpy as np
from scipy.signal import find_peaks, welch
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def detect_peaks(signal: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Detect pulse peaks with minimum distance ~0.5 s."""
    # Adaptive threshold: distance corresponds to max HR ~120 BPM
    distance = int(0.5 * fs)
    # Use higher sensitivity: look for peaks above mean + 0.5 * std
    height = float(np.mean(signal) + 0.5 * np.std(signal))
    peaks, _ = find_peaks(signal, distance=distance, height=height)
    return peaks


# ---------------------------------------------------------------------------
# Time-domain HRV
# ---------------------------------------------------------------------------

def compute_rr_intervals(peaks: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Convert peak indices to RR-interval series (seconds)."""
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs


def compute_heart_rate(peaks: np.ndarray, fs: float = 30.0) -> float:
    """Mean heart rate in BPM from peak indices."""
    rr = compute_rr_intervals(peaks, fs)
    if len(rr) == 0:
        return float("nan")
    mean_rr = float(np.mean(rr))
    return 60.0 / mean_rr if mean_rr > 0 else float("nan")


def compute_rmssd_from_peaks(peaks: np.ndarray, fs: float = 30.0) -> float:
    """Root Mean Square of Successive RR-interval Differences (ms-scale)."""
    rr = compute_rr_intervals(peaks, fs)
    if len(rr) < 2:
        return float("nan")
    diff_rr = np.diff(rr)
    return float(np.sqrt(np.mean(diff_rr ** 2)))


def compute_sdnn(peaks: np.ndarray, fs: float = 30.0) -> float:
    """Standard Deviation of NN (RR) intervals."""
    rr = compute_rr_intervals(peaks, fs)
    if len(rr) < 2:
        return float("nan")
    return float(np.std(rr, ddof=1))


# ---------------------------------------------------------------------------
# Frequency-domain HRV
# ---------------------------------------------------------------------------

def _band_power(freqs: np.ndarray, psd: np.ndarray,
                low: float, high: float) -> float:
    """Integrate PSD within [low, high] Hz band."""
    idx = np.where((freqs >= low) & (freqs <= high))
    # Use np.trapezoid if available (Numpy 2.0+), else np.trapz
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_func is None:
         # Fallback to a simple manual trapezoidal rule if both are missing
         y = psd[idx]
         x = freqs[idx]
         if len(x) < 2: return 0.0
         return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2.0))
    return float(trapz_func(psd[idx], freqs[idx]))


def compute_lf_hf_ratio(signal: np.ndarray, fs: float = 30.0) -> float:
    """LF / HF spectral power ratio."""
    n = len(signal)
    nperseg = min(256, n) if n >= 64 else n
    if n < 32: return float("nan")
    
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    lf = _band_power(freqs, psd, 0.04, 0.15)
    hf = _band_power(freqs, psd, 0.15, 0.40)
    if hf < 1e-12:
        return float("nan")
    return float(lf / hf)


def compute_rsa(signal: np.ndarray, fs: float = 30.0) -> float:
    """Respiratory Sinus Arrhythmia — approximated as HF band power."""
    n = len(signal)
    nperseg = min(256, n) if n >= 64 else n
    if n < 32: return float("nan")
    
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return _band_power(freqs, psd, 0.15, 0.40)


def compute_sympathetic_index(signal: np.ndarray, fs: float = 30.0) -> float:
    """Sympathetic Dominance Index = LF / (LF + HF)."""
    n = len(signal)
    nperseg = min(256, n) if n >= 64 else n
    if n < 32: return float("nan")
    
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    lf = _band_power(freqs, psd, 0.04, 0.15)
    hf = _band_power(freqs, psd, 0.15, 0.40)
    total = lf + hf
    if total < 1e-12:
        return float("nan")
    return float(lf / total)


# ---------------------------------------------------------------------------
# Pulse Transit Variability
# ---------------------------------------------------------------------------

def compute_ptv(peaks: np.ndarray, fs: float = 30.0) -> float:
    """Pulse Transit Variability."""
    rr = compute_rr_intervals(peaks, fs)
    if len(rr) < 2:
        return float("nan")
    mean_rr = float(np.mean(rr))
    if mean_rr < 1e-12:
        return float("nan")
    return float(np.std(rr, ddof=1) / mean_rr)


# ---------------------------------------------------------------------------
# Spectral Entropy
# ---------------------------------------------------------------------------

def spectral_entropy(signal: np.ndarray, fs: float = 30.0) -> float:
    """Shannon spectral entropy of the PSD."""
    n = len(signal)
    nperseg = min(256, n) if n >= 64 else n
    if n < 32: return 0.0
    
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))


# ---------------------------------------------------------------------------
# Signal Quality Index
# ---------------------------------------------------------------------------

def compute_sqi(signal: np.ndarray) -> float:
    """Signal Quality Index ∈ [0, 1]."""
    power = float(np.var(signal))
    ps = np.abs(signal)
    ps_sum = ps.sum() + 1e-12
    entropy = -np.sum((ps / ps_sum) * np.log(ps / ps_sum + 1e-12))

    if power < 1e-6:
        return 0.2
    if entropy < 1e-3:
        return 0.3
    return min(1.0, power / (np.max(np.abs(signal)) + 1e-12))


# ---------------------------------------------------------------------------
# Composite Risk Scores
# ---------------------------------------------------------------------------

def compute_stress_reactivity(rmssd: float, sdnn: float,
                               lf_hf: float) -> float:
    """Composite Stress Reactivity Score ∈ [0, 1]."""
    if any(math.isnan(v) for v in (rmssd, sdnn, lf_hf)):
        return float("nan")
    hrv_component = 1.0 - min(rmssd / 0.1, 1.0)
    sdnn_component = 1.0 - min(sdnn / 0.1, 1.0)
    lf_hf_component = min(lf_hf / 5.0, 1.0)
    score = 0.4 * hrv_component + 0.3 * sdnn_component + 0.3 * lf_hf_component
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Aggregate Feature Dictionary
# ---------------------------------------------------------------------------

def extract_all_features(signal: np.ndarray, fs: float = 30.0) -> Dict[str, float]:
    """Run the full feature extraction pipeline."""
    peaks = detect_peaks(signal, fs)

    hr = compute_heart_rate(peaks, fs)
    rmssd = compute_rmssd_from_peaks(peaks, fs)
    sdnn = compute_sdnn(peaks, fs)
    lf_hf = compute_lf_hf_ratio(signal, fs)
    rsa = compute_rsa(signal, fs)
    ptv = compute_ptv(peaks, fs)
    entropy = spectral_entropy(signal, fs)
    sympathetic = compute_sympathetic_index(signal, fs)
    sqi = compute_sqi(signal)
    stress = compute_stress_reactivity(rmssd, sdnn, lf_hf)

    return {
        "heart_rate_bpm": hr,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "lf_hf_ratio": lf_hf,
        "rsa": rsa,
        "ptv": ptv,
        "spectral_entropy": entropy,
        "sympathetic_index": sympathetic,
        "signal_quality_index": sqi,
        "stress_reactivity": stress,
    }
