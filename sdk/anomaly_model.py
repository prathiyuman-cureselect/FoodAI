"""
NeuroVitals — Waveform Anomaly Detection (Lightweight / NumPy-only)
===================================================================
Statistical signal-quality validator.  Replaces the previous LSTM stub
with a zero-dependency heuristic so that PyTorch is NOT required at
runtime, cutting ~800 MB of RAM on headless deploy targets.
"""

import numpy as np


class WaveformValidator:
    """Score how 'biological' a pulse waveform looks using signal statistics.

    Returns an authenticity score in [0, 1].
    """

    def validate_signal(self, signal: np.ndarray) -> float:
        """
        Returns an 'authenticity' score [0-1].
        1.0 = highly biological; 0.0 = anomalous (synthetic/noisy).
        """
        if len(signal) < 30:
            return 0.0

        s = np.array(signal, dtype=np.float64)
        s_norm = (s - np.mean(s)) / (np.std(s) + 1e-12)

        # 1. Autocorrelation at ~1-second lag → periodic biological pulse
        lag = min(30, len(s_norm) // 2)
        autocorr = np.corrcoef(s_norm[:-lag], s_norm[lag:])[0, 1]
        autocorr = max(0.0, autocorr)  # negative = non-biological

        # 2. Signal-to-noise proxy (peak spectral power / total power)
        fft_mag = np.abs(np.fft.rfft(s_norm))
        snr = float(np.max(fft_mag) / (np.mean(fft_mag) + 1e-12))
        snr_score = min(snr / 10.0, 1.0)

        # 3. Zero-crossing rate (too high = noise, too low = flat)
        zc = np.sum(np.diff(np.sign(s_norm)) != 0) / len(s_norm)
        zc_score = 1.0 - abs(zc - 0.15) / 0.15  # peak around 15%
        zc_score = max(0.0, min(zc_score, 1.0))

        # Weighted blend
        score = 0.4 * autocorr + 0.35 * snr_score + 0.25 * zc_score
        return float(np.clip(score, 0.0, 1.0))
