"""
NeuroVitals — Bayesian Mental Health Inference Engine
=====================================================
Computes posterior probabilities for mental health conditions using
rPPG-derived autonomic features and clinically-grounded likelihood
mappings.

Conditions modelled:
  Depression, Anxiety, Burnout, PTSD

Each condition is associated with a set of feature→likelihood mappings
derived from published neurocardiac correlates (Section 3 of the
NeuroVitals architecture blueprint).
"""

import math
import numpy as np
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Core Bayesian Update
# ---------------------------------------------------------------------------

def bayesian_update(prior: float, likelihood: float) -> float:
    """Single-step Bayesian posterior update.

    P(H|E) = P(E|H)·P(H) / [ P(E|H)·P(H) + P(E|¬H)·P(¬H) ]

    Parameters
    ----------
    prior : float       Prior probability P(H) ∈ (0,1)
    likelihood : float  Likelihood P(E|H) ∈ (0,1)

    Returns
    -------
    float  Posterior probability P(H|E)
    """
    if prior <= 0.0 or prior >= 1.0:
        prior = np.clip(prior, 1e-6, 1.0 - 1e-6)
    if likelihood <= 0.0 or likelihood >= 1.0:
        likelihood = np.clip(likelihood, 1e-6, 1.0 - 1e-6)

    complement_likelihood = 1.0 - likelihood
    evidence = likelihood * prior + complement_likelihood * (1.0 - prior)
    if evidence < 1e-12:
        return prior
    return float((likelihood * prior) / evidence)


# ---------------------------------------------------------------------------
# Feature → Likelihood Mappings (clinically grounded heuristics)
# ---------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float, steepness: float = 10.0) -> float:
    """Logistic sigmoid centred at *midpoint*."""
    z = steepness * (x - midpoint)
    return float(1.0 / (1.0 + math.exp(-z)))


# Mapping tables: feature_name → (sigmoid_midpoint, steepness, invert)
# invert=True means *lower* feature values → *higher* likelihood

_DEPRESSION_MAP = {
    # Low HRV → depression   (RMSSD < ~0.04 is concerning in rPPG scale)
    "rmssd":              (0.04, -15.0, False),
    # Reduced RSA
    "rsa":                (0.001, -20.0, False),
    # Elevated spectral entropy
    "spectral_entropy":   (3.0, 5.0, False),
    # High stress reactivity
    "stress_reactivity":  (0.5, 8.0, False),
}

_ANXIETY_MAP = {
    # Elevated LF/HF ratio
    "lf_hf_ratio":        (2.0, 4.0, False),
    # High sympathetic index
    "sympathetic_index":  (0.6, 10.0, False),
    # Low RMSSD
    "rmssd":              (0.04, -15.0, False),
}

_BURNOUT_MAP = {
    # High spectral entropy → dysregulation
    "spectral_entropy":   (3.5, 4.0, False),
    # High stress reactivity
    "stress_reactivity":  (0.6, 8.0, False),
    # Low SDNN
    "sdnn":               (0.04, -12.0, False),
}

_PTSD_MAP = {
    # High PTV → autonomic instability
    "ptv":                (0.15, 10.0, False),
    # Elevated LF/HF
    "lf_hf_ratio":        (2.5, 3.0, False),
    # High stress reactivity
    "stress_reactivity":  (0.65, 8.0, False),
    # Low RSA
    "rsa":                (0.001, -20.0, False),
}

_CONDITION_MAPS = {
    "depression": _DEPRESSION_MAP,
    "anxiety":    _ANXIETY_MAP,
    "burnout":    _BURNOUT_MAP,
    "ptsd":       _PTSD_MAP,
}

# Default (uninformed) priors — prevalence-based
_DEFAULT_PRIORS: Dict[str, float] = {
    "depression": 0.15,
    "anxiety":    0.18,
    "burnout":    0.12,
    "ptsd":       0.07,
}


# ---------------------------------------------------------------------------
# Bayesian Mental Health Engine
# ---------------------------------------------------------------------------

class BayesianMentalHealthEngine:
    """Bayesian inference engine for mental health risk estimation.

    Usage::

        engine = BayesianMentalHealthEngine()
        posteriors = engine.update(features_dict)
        risk_class = engine.classify(posteriors)
    """

    def __init__(self, priors: Optional[Dict[str, float]] = None):
        self.priors = dict(priors or _DEFAULT_PRIORS)

    # ----- public API ------

    def update(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute posterior probabilities for each condition given *features*.

        Parameters
        ----------
        features : dict  Output of ``extract_all_features()``.

        Returns
        -------
        dict  {condition: posterior_probability}
        """
        posteriors: Dict[str, float] = {}
        for condition, mapping in _CONDITION_MAPS.items():
            posterior = self.priors.get(condition, 0.1)
            for feat_name, (midpoint, steepness, _inv) in mapping.items():
                value = features.get(feat_name)
                if value is None or math.isnan(value):
                    continue
                likelihood = _sigmoid(value, midpoint, steepness)
                posterior = bayesian_update(posterior, likelihood)
            posteriors[condition] = posterior
        return posteriors

    def classify(self, posteriors: Dict[str, float]) -> str:
        """Map the highest posterior to a risk class string.

        Returns one of: ``"Low"`` / ``"Moderate"`` / ``"High"`` / ``"Critical"``
        """
        if not posteriors:
            return "Low"
        max_post = max(posteriors.values())
        if max_post >= 0.75:
            return "Critical"
        if max_post >= 0.50:
            return "High"
        if max_post >= 0.25:
            return "Moderate"
        return "Low"

    def explain_results(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Derives clinical evidence contributions (SHAP-like) for the 
        final risk assessment.
        """
        explanation = {
            "Low_HRV": 0.0,
            "High_LFHF": 0.0,
            "High_Entropy": 0.0,
            "Low_RSA": 0.0,
            "High_Stress": 0.0
        }
        
        # HRV Check (RMSSD)
        rmssd = features.get("rmssd", 0.1)
        if rmssd < 0.04:
            explanation["Low_HRV"] = float(np.clip((0.04 - rmssd) * 10, 0, 1))

        # Spectral Balance
        lfhf = features.get("lf_hf_ratio", 1.0)
        if lfhf > 2.0:
            explanation["High_LFHF"] = float(np.clip((lfhf - 2.0) / 3.0, 0, 1))

        # Irregularity
        ent = features.get("spectral_entropy", 2.0)
        if ent > 3.0:
            explanation["High_Entropy"] = float(np.clip((ent - 3.0) / 2.0, 0, 1))
            
        # Autonomic reactivity
        stress = features.get("stress_reactivity", 0.0)
        explanation["High_Stress"] = float(stress)
            
        return explanation

    def full_inference(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Run update + classify + explain and return a combined result dict."""
        try:
            posteriors = self.update(features)
            risk_class = self.classify(posteriors)
            explainability = self.explain_results(features)
            
            return {
                "posteriors": posteriors,
                "risk_class": risk_class,
                "dominant_condition": max(posteriors, key=posteriors.get) if posteriors else None,
                "explainability": explainability
            }
        except Exception as e:
            print(f"[NeuroVitals] [BAYESIAN_ENGINE] ERROR in full_inference: {e}")
            # Fallback to safe defaults
            return {
                "posteriors": self.priors,
                "risk_class": "Low",
                "dominant_condition": None,
                "explainability": self.explain_results(features)
            }


# ---------------------------------------------------------------------------
# Mood & Emotional State Inference (Circumplex Model)
# ---------------------------------------------------------------------------

class MoodInferenceEngine:
    """
    Infers emotional state (Mood) using a circumplex model of Arousal and Valence
    derived from autonomic biomarkers.
    
    Arousal ~ LF/HF ratio & Heart Rate
    Valence ~ RMSSD & RSA (High Vagal Tone -> Positive Valence)
    """

    def infer_mood(self, features: Dict[str, float]) -> Dict[str, Any]:
        rmssd = features.get("rmssd", 0.04)
        lfhf = features.get("lf_hf_ratio", 1.5)
        hr = features.get("heart_rate_bpm", 70)
        entropy = features.get("spectral_entropy", 2.5)

        # 1. Estimate Arousal [0-1]
        # Driven by sympathetic dominance (LF/HF) and tachycardia (HR)
        arousal_lfhf = np.clip((lfhf - 1.0) / 4.0, 0, 1)
        arousal_hr = np.clip((hr - 60) / 60, 0, 1)
        arousal = float(0.6 * arousal_lfhf + 0.4 * arousal_hr)

        # 2. Estimate Valence [0-1]
        # Driven by vagal tone (RMSSD) and signal stability (low entropy)
        valence_hrv = np.clip(rmssd / 0.1, 0, 1)
        valence_stability = 1.0 - np.clip((entropy - 2.0) / 3.0, 0, 1)
        valence = float(0.7 * valence_hrv + 0.3 * valence_stability)

        # 3. Classify State
        # Circumplex quadrants:
        # High Arousal, High Valence -> Alert/Excited/Happy
        # High Arousal, Low Valence  -> Stressed/Anxious/Angry
        # Low Arousal, High Valence  -> Calm/Relaxed/Content
        # Low Arousal, Low Valence   -> Fatigued/Sad/Bored

        if arousal > 0.5:
            state = "Alert" if valence > 0.5 else "Stressed"
        else:
            state = "Calm" if valence > 0.5 else "Fatigued"

        # Refine state based on specific markers
        if state == "Stressed" and entropy > 4.0:
            state = "Agitated"
        if state == "Fatigued" and lfhf < 0.8:
            state = "Exhausted"

        # 4. Behavioral Indicators
        # Readiness for complex tasks (High vagal tone + stable signal)
        cognitive_readiness = float(np.clip(valence * (1.0 - arousal * 0.5), 0, 1))
        
        # Social Engagement Potential (High RSA is a key marker here)
        rsa = features.get("rsa", 0.0)
        social_engagement = float(np.clip(rsa / 0.005, 0, 1))

        return {
            "Arousal": arousal,
            "Valence": valence,
            "MoodState": state,
            "EmotionalStability": float(1.0 - entropy / 6.0),
            "CognitiveReadiness": cognitive_readiness,
            "SocialEngagement": social_engagement
        }
