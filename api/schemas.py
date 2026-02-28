"""
NeuroVitals — API Pydantic Schemas
===================================
Request / response models for every endpoint, aligned with the
NeuroVitals risk-engineering output specification (Section 10).
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "NeuroVitals rPPG API"
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class ExplainabilityDetail(BaseModel):
    Low_HRV: float = Field(0.0, description="SHAP / contribution of low HRV")
    High_LFHF: float = Field(0.0, description="Contribution of elevated LF/HF ratio")
    High_Entropy: float = Field(0.0, description="Contribution of high spectral entropy")
    Low_RSA: float = Field(0.0, description="Contribution of reduced RSA")
    High_Stress: float = Field(0.0, description="Contribution of stress reactivity")


class BayesianResult(BaseModel):
    depression: float = 0.0
    anxiety: float = 0.0
    burnout: float = 0.0
    ptsd: float = 0.0


class ClinicalMetricSet(BaseModel):
    heart_rate_bpm: Optional[float] = None
    rmssd: Optional[float] = None
    sdnn: Optional[float] = None
    lf_hf_ratio: Optional[float] = None
    rsa: Optional[float] = None
    ptv: Optional[float] = None
    spectral_entropy: Optional[float] = None
    sympathetic_index: Optional[float] = None
    stress_reactivity: Optional[float] = None


class MoodResult(BaseModel):
    Arousal: float
    Valence: float
    MoodState: str
    EmotionalStability: float
    CognitiveReadiness: float
    SocialEngagement: float


class AnalysisResponse(BaseModel):
    """Full risk-engineering output object (Section 10 of blueprint)."""
    IdentityVerified: bool = False
    LivenessScore: float = 0.0
    SignalQualityIndex: float = 0.0
    DepressionRiskScore: float = 0.0
    AnxietyRiskScore: float = 0.0
    AutonomicStabilityIndex: float = 0.0
    MentalHealthRiskClass: str = "Low"
    Explainability: ExplainabilityDetail
    Mood: MoodResult
    BayesianPosteriors: BayesianResult = Field(default_factory=BayesianResult)
    ClinicalFeatures: ClinicalMetricSet = Field(default_factory=ClinicalMetricSet)
    ConfidenceScore: float = 0.0
    DominantCondition: Optional[str] = None


# ---------------------------------------------------------------------------
# Enroll / Verify
# ---------------------------------------------------------------------------

class EnrollResponse(BaseModel):
    subject_id: str
    enrolled: bool = True
    embedding_dim: int = 512
    message: str = "Face embedding enrolled successfully"


class VerifyResponse(BaseModel):
    subject_id: str
    identity_verified: bool = False
    similarity_score: float = 0.0
    liveness_score: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class DriftResult(BaseModel):
    statistic: float = 0.0
    p_value: float = 1.0
    drift_detected: bool = False


class CalibrationResult(BaseModel):
    brier_score: float = 0.0
    calibration_pass: bool = True


class GovernanceReport(BaseModel):
    status: str = "generated"
    any_drift_detected: bool = False
    drift: Optional[Dict[str, DriftResult]] = None
    fairness: Optional[Dict] = None
    calibration: Optional[CalibrationResult] = None
    recommendation: str = "No action required"
