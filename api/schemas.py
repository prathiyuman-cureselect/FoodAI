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
    # --- Core Vitals ---
    heart_rate_bpm: Optional[float] = None
    rmssd: Optional[float] = None
    sdnn: Optional[float] = None
    lf_hf_ratio: Optional[float] = None
    rsa: Optional[float] = None
    ptv: Optional[float] = None
    spectral_entropy: Optional[float] = None
    sympathetic_index: Optional[float] = None
    stress_reactivity: Optional[float] = None
    blood_pressure_sys: Optional[float] = None
    blood_pressure_dia: Optional[float] = None
    spo2: Optional[float] = None
    
    # --- Hemodynamic Expansion ---
    mean_arterial_pressure: Optional[float] = None
    cardiac_workload: Optional[float] = None
    pulse_pressure: Optional[float] = None
    ascvd_risk: Optional[str] = "Low"
    high_bp_risk: Optional[float] = 0.0
    
    # --- Respiratory Expansion ---
    breathing_rate: Optional[float] = None
    pulse_respiration_quotient: Optional[float] = None  # PRQ
    
    # --- Metabolic / Hematic (AI Estimated Risks) ---
    hemoglobin_estimated: Optional[float] = None       # hb
    hba1c_estimated: Optional[float] = None            # hbA1c
    glucose_risk: Optional[float] = 0.0                # High Fasting Glucose Risk
    cholesterol_risk: Optional[float] = 0.0            # High Total Cholesterol risk
    anemia_risk: Optional[float] = 0.0                 # Low hb risk
    hba1c_risk: Optional[float] = 0.0                  # High HbA1c risk
    
    # --- Wellness & Safety ---
    heart_age: Optional[int] = None
    wellness_score: Optional[float] = 0.0
    fall_risk: Optional[float] = 0.0
    activity_index: Optional[float] = 0.0              # Movement / Low Activity
    
    hrv_raw: Optional[Dict[str, List[float]]] = None


class ClinicalTrendSet(BaseModel):
    """Time-series data for clinical visualization."""
    heart_rate: List[float] = Field(default_factory=list)
    blood_pressure_sys: List[float] = Field(default_factory=list)
    blood_pressure_dia: List[float] = Field(default_factory=list)
    spo2: List[float] = Field(default_factory=list)
    breathing_signal: List[float] = Field(default_factory=list)
    stress_index: List[float] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)


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
    ClinicalTrends: Optional[ClinicalTrendSet] = None
    ConfidenceScore: float = 0.0
    DominantCondition: Optional[str] = None
    DetectedGender: Optional[str] = None
    DetectedAge: Optional[int] = None


# ---------------------------------------------------------------------------
# Gender Detection
# ---------------------------------------------------------------------------

class GenderDetectionResponse(BaseModel):
    success: bool = False
    gender: Optional[str] = None
    age: Optional[int] = None
    confidence: float = 0.0
    message: str = ""


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


