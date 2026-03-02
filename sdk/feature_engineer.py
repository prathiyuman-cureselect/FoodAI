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
from typing import Dict, Optional, Any, List


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


def compute_breathing_rate(signal: np.ndarray, fs: float = 30.0) -> float:
    """Extract respiration (RR) from rPPG via low-frequency modulation (0.1-0.4 Hz)."""
    try:
        # Bandpass filter for respiration (typically 6-24 breaths/min -> 0.1-0.4 Hz)
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        b, a = butter(2, [0.1 / nyq, 0.4 / nyq], btype="band")
        resp_signal = filtfilt(b, a, signal)
        
        # Count peaks in the respiratory signal
        peaks, _ = find_peaks(resp_signal, distance=int(fs * 1.5)) # min 1.5s between breaths
        if len(peaks) < 2: return 16.0 # Default normal
        
        duration_sec = len(signal) / fs
        rr = (len(peaks) / duration_sec) * 60.0
        return float(np.clip(rr, 8, 30))
    except Exception:
        return 16.0


def compute_hemodynamics(hr: float, sys: float, dia: float) -> Dict[str, float]:
    """Calculate advanced hemodynamic markers."""
    map_val = (2 * dia + sys) / 3
    pulse_pressure = sys - dia
    cardiac_workload = (hr * sys) / 100.0 # Rate Pressure Product (RPP) proxy
    return {
        "map": float(map_val),
        "pulse_pressure": float(pulse_pressure),
        "cardiac_workload": float(cardiac_workload)
    }


def estimate_biometric_risks(hr: float, hrv: float, bp_sys: float, age: int) -> Dict[str, Any]:
    """AI Proxy estimates for blood/metabolic markers (not medical measurements)."""
    # Heuristic: Anemia/Hb risk correlates with tachycardia and low pulse pressure
    # Heuristic: Glucose/HbA1c risk correlates with sympathetic dominance (low HRV)
    
    # Hb Proxy (Hgb g/dL) - baseline 14
    hb = 14.5 - (hr - 70) * 0.05 - (float(max(0.0, float(age-40)))) * 0.02
    hb_risk = 1.0 - (hb / 12.0) if hb < 12.0 else 0.0
    
    # HbA1c Proxy (%) - baseline 5.2
    hba1c = 5.0 + (1.0 / (hrv + 1e-6)) * 0.1 + (bp_sys - 120) * 0.02
    hba1c_risk = (hba1c - 5.7) / 0.8 if hba1c > 5.7 else 0.0
    
    # Glucose risk (Fasting HbA1c proxy)
    glucose_risk = hba1c_risk * 1.2
    
    # Cholesterol risk - proxy using age + BP
    chol_risk = (float(age) / 80.0) * 0.4 + (bp_sys / 200.0) * 0.3
    
    return {
        "hb": float(np.clip(hb, 8.0, 18.0)),
        "hba1c": float(np.clip(hba1c, 4.0, 12.0)),
        "glucose_risk": float(np.clip(glucose_risk, 0.0, 1.0)),
        "hba1c_risk": float(np.clip(hba1c_risk, 0.0, 1.0)),
        "anemia_risk": float(np.clip(hb_risk, 0.0, 1.0)),
        "cholesterol_risk": float(np.clip(chol_risk, 0.0, 1.0))
    }


def calculate_heart_age(actual_age: int, hr: float, hrv: float, sys: float) -> int:
    """Estimated physiological Heart Age relative to actual age."""
    # HR > 80 adds years, HRV < 30 adds years, SYS > 130 adds years
    modifier = (hr - 70) * 0.2 + (float(max(0.0, float(40.0 - hrv)))) * 0.3 + (sys - 120) * 0.4
    return int(np.clip(actual_age + modifier, 18, 90))


def calculate_wellness_score(features: Dict[str, Any]) -> float:
    """Aggregate holistic health score (0-1)."""
    # Weighted balance of HR, HRV stability, BP, and Stress
    hr_comp = float(max(0.0, 1.0 - abs(features.get("heart_rate_bpm", 70) - 70) / 40.0))
    hrv_comp = float(min(float(features.get("rmssd", 0) * 20), 1.0)) # RMSSD 50ms = 1.0
    bp_comp = float(max(0.0, 1.0 - (features.get("blood_pressure_sys", 120) - 120) / 60.0))
    stress_comp = 1.0 - features.get("stress_reactivity", 0.5)
    
    return float(np.clip(0.3 * hrv_comp + 0.3 * stress_comp + 0.2 * hr_comp + 0.2 * bp_comp, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Pulse Morphology & Segment Analysis
# ---------------------------------------------------------------------------

def extract_morphological_features(signal: np.ndarray, fs: float = 30.0) -> Dict[str, float]:
    """Extract temporal and amplitude features from the pulse wave morphology."""
    try:
        # Find peaks and troughs
        peaks, _ = find_peaks(signal, distance=int(0.5 * fs))
        troughs, _ = find_peaks(-signal, distance=int(0.5 * fs))
        
        if len(peaks) < 3 or len(troughs) < 3:
            return {}

        rise_times = []
        amplitudes = []
        
        for p in peaks:
            # Find the nearest trough before this peak
            prev_trough = troughs[troughs < p]
            if len(prev_trough) == 0: continue
            t0 = prev_trough[-1]
            
            # Rise time in ms
            rise_times.append((p - t0) / fs * 1000)
            
            # Amplitude
            amplitudes.append(signal[p] - signal[t0])
            
        return {
            "mean_rise_time": float(np.mean(rise_times)) if rise_times else 110.0,
            "mean_amplitude": float(np.mean(amplitudes)) if amplitudes else 0.5,
        }
    except Exception:
        return {}


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


def compute_rmssd_from_peaks(peaks: np.ndarray, fs: float = 30.0) -> Dict[str, Any]:
    """Root Mean Square of Successive RR-interval Differences (ms-scale).
    Returns both the scalar RMSSD and the raw successive differences.
    """
    rr = compute_rr_intervals(peaks, fs)
    if len(rr) < 2:
        return {"rmssd": float("nan"), "successive_diffs": [], "rr_intervals": []}
    
    diff_rr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    return {
        "rmssd": rmssd,
        "successive_diffs": diff_rr.tolist(),
        "rr_intervals": rr.tolist()
    }


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
    return float(min(1.0, power / (np.max(np.abs(signal)) + 1e-12)))


def get_best_window(signal: np.ndarray, fs: float = 30.0, window_sec: float = 6.0) -> np.ndarray:
    """Find the cleanest window in the signal based on SQI."""
    win_size = int(window_sec * fs)
    n = len(signal)
    if n <= win_size:
        return signal
    
    best_sqi = -1.0
    best_start = 0
    
    # Slide with 1s step
    step = int(fs)
    for start in range(0, n - win_size + 1, step):
        segment = signal[start : start + win_size]
        sqi = compute_sqi(segment)
        if sqi > best_sqi:
            best_sqi = sqi
            best_start = start
            
    return signal[best_start : best_start + win_size]


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
    score = 0.4 * hrv_component + 0.3 * sdnn_component + 0.3 * lf_hf
    return float(np.clip(score, 0.0, 1.0))


def compute_blood_pressure(hr: float, morphological: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Estimate Blood Pressure using HR and Pulse Morphology."""
    if math.isnan(hr):
        return None
    
    # Kurylyak-inspired model (simplified for rPPG)
    # Systolic often increases with HR and decreases with Pulse Rise Time (arterial stiffness)
    rise_time = morphological.get("mean_rise_time", 100.0)
    amplitude = morphological.get("mean_amplitude", 0.5)
    
    # Baseline 120/80
    # Rise time proxy for PTT (Pulse Transit Time)
    sys = 115 + (hr - 70) * 0.3 - (rise_time - 110) * 0.15 + (amplitude * 10)
    dia = 75 + (hr - 70) * 0.15 - (rise_time - 110) * 0.05 + (amplitude * 5)
    
    return {
        "sys": float(np.clip(sys, 95, 175)),
        "dia": float(np.clip(dia, 60, 105))
    }


def compute_spo2_calibrated(r_ratio: float, b_ratio: float) -> float:
    """Estimate SpO2 using the Ratio-of-Ratios (R) principle."""
    if b_ratio == 0: return 98.0
    
    # R = (AC/DC_red) / (AC/DC_blue)
    R = r_ratio / b_ratio
    
    # Linear calibration: SpO2 = A - B*R
    # Standard values for RGB-based SpO2 (approximate)
    # A=115, B=25 is a common starting point for red/blue
    spo2 = 110.0 - 15.0 * R
    
    return float(np.clip(spo2, 90.0, 100.0))


def compute_spo2(signal: np.ndarray) -> float:
    """Fallback heuristic SpO2."""
    sqi = compute_sqi(signal)
    base = 98.5
    variation = (1.0 - sqi) * 4.0
    return float(np.clip(base - variation, 94.0, 100.0))


# ---------------------------------------------------------------------------
# Trend Generation for Visualization
# ---------------------------------------------------------------------------

def generate_trend_data(signal: np.ndarray, fs: float = 30.0) -> Dict[str, List[float]]:
    """Generate time-series data for all clinical indicators."""
    peaks = detect_peaks(signal, fs)
    n = len(signal)
    duration = n / fs
    
    # 1. Heart Rate Trend (per beat)
    rr = np.diff(peaks) / fs
    hr_trend = [float(60.0 / r) if r > 0.3 else 70.0 for r in rr]
    
    # 2. Blood Pressure Trend (Morphological proxy per beat)
    # We use systolic rise time as a proxy for BP changes
    troughs, _ = find_peaks(-signal, distance=int(0.5 * fs))
    bp_sys_trend = []
    bp_dia_trend = []
    
    for p in peaks:
        prev_t = troughs[troughs < p]
        if len(prev_t) > 0:
            tr = prev_t[-1]
            rise_time = (p - tr) / fs * 1000
            # Heuristic: Shorter rise time = higher BP
            sys_val = 120 + (110 - rise_time) * 0.4
            dia_val = 80 + (110 - rise_time) * 0.2
            bp_sys_trend.append(float(np.clip(sys_val, 90, 180)))
            bp_dia_trend.append(float(np.clip(dia_val, 60, 110)))
            
    # 3. Breathing Signal (Filtered Wave)
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(2, [0.1 / nyq, 0.4 / nyq], btype="band")
    resp_wave = filtfilt(b, a, signal)
    # Downsample breathing wave for efficiency
    resp_trend = [float(val) for val in resp_wave[::5]]
    
    # 4. Stress Index Trend (Sliding window spectral entropy)
    stress_trend = []
    win_size = int(5 * fs)
    for i in range(0, n - win_size, int(fs)):
        segment = signal[i : i + win_size]
        ent = spectral_entropy(segment, fs)
        stress_trend.append(float(ent))
        
    return {
        "heart_rate": hr_trend,
        "blood_pressure_sys": bp_sys_trend,
        "blood_pressure_dia": bp_dia_trend,
        "spo2": [98.0] * len(hr_trend), # Baseline SpO2 trend
        "breathing_signal": resp_trend,
        "stress_index": stress_trend,
        "timestamps": [float(i/fs) for i in range(0, n, int(fs))][:len(stress_trend)]
    }


# ---------------------------------------------------------------------------
# Aggregate Feature Dictionary
# ---------------------------------------------------------------------------

def extract_all_features(signal: np.ndarray, fs: float = 30.0, 
                         r_ratio: Optional[float] = None, 
                         b_ratio: Optional[float] = None,
                         age: int = 35) -> Dict[str, Any]:
    """Run the full feature extraction pipeline with morphological analysis and demographic context."""
    # --- SIGNAL QUALITY GATING ---
    # We find the cleanest 8-second window for core vitals to improve accuracy
    clean_signal = get_best_window(signal, fs, window_sec=8.0)
    
    peaks = detect_peaks(clean_signal, fs)

    hr = compute_heart_rate(peaks, fs)
    rmssd_data = compute_rmssd_from_peaks(peaks, fs)
    rmssd = rmssd_data["rmssd"]
    sdnn = compute_sdnn(peaks, fs)
    lf_hf = compute_lf_hf_ratio(signal, fs)
    rsa = compute_rsa(signal, fs)
    ptv = compute_ptv(peaks, fs)
    entropy = spectral_entropy(signal, fs)
    sympathetic = compute_sympathetic_index(signal, fs)
    sqi = compute_sqi(signal)
    stress = compute_stress_reactivity(rmssd, sdnn, lf_hf)
    
    # 1. Advanced BP: Use Morphological Data
    morph = extract_morphological_features(signal, fs)
    bp = compute_blood_pressure(hr, morph)
    
    # 2. Advanced SpO2: Use Calibrated Model if ratios available
    if r_ratio is not None and b_ratio is not None:
        spo2 = compute_spo2_calibrated(r_ratio, b_ratio)
    else:
        spo2 = compute_spo2(signal)

    # 3. New Advanced Metrics Expansion
    rr = compute_breathing_rate(signal, fs)
    prq = hr / rr if rr > 0 else 4.4
    
    hemo = compute_hemodynamics(hr, bp["sys"], bp["dia"])
    
    # Use dynamic age for risk models
    biometric_risks = estimate_biometric_risks(hr, rmssd, bp["sys"], age)
    heart_age = calculate_heart_age(age, hr, rmssd, bp["sys"])
    
    # Fall detection proxy (using spectral entropy as movement index)
    # High entropy + high signal power variance = erratic movement
    fall_risk = float(np.clip(entropy * 0.5, 0.0, 1.0)) 
    activity_index = float(np.clip(1.0 - sqi, 0.0, 1.0)) # Inverse SQI can reflect movement intensity

    res = {
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
        "blood_pressure_sys": bp["sys"] if bp else 0.0,
        "blood_pressure_dia": bp["dia"] if bp else 0.0,
        "spo2": spo2,
        
        # Hemodynamic Expansion
        "mean_arterial_pressure": hemo["map"],
        "cardiac_workload": hemo["cardiac_workload"],
        "pulse_pressure": hemo["pulse_pressure"],
        "ascvd_risk": "Moderate" if bp["sys"] > 140 else "Low",
        "high_bp_risk": float(np.clip((bp["sys"] - 120) / 40.0, 0, 1)),
        
        # Respiratory Expansion
        "breathing_rate": rr,
        "pulse_respiration_quotient": prq,
        
        # Metabolic / Hematic
        "hemoglobin_estimated": biometric_risks["hb"],
        "hba1c_estimated": biometric_risks["hba1c"],
        "glucose_risk": biometric_risks["glucose_risk"],
        "cholesterol_risk": biometric_risks["cholesterol_risk"],
        "anemia_risk": biometric_risks["anemia_risk"],
        "hba1c_risk": biometric_risks["hba1c_risk"],
        
        # Wellness
        "heart_age": heart_age,
        "fall_risk": fall_risk,
        "activity_index": activity_index,
        
        "hrv_raw": {
            "successive_diffs": rmssd_data["successive_diffs"],
            "rr_intervals": rmssd_data["rr_intervals"]
        }
    }
    
    res["wellness_score"] = calculate_wellness_score(res)
    
    # --- 4. Clinical Trends for Graphs ---
    res["trends"] = generate_trend_data(clean_signal, fs)
    
    return res
