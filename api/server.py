"""
NeuroVitals rPPG API — Production Server
=========================================
Full endpoint suite for the NeuroVitals Mental Health Phenotyping Engine.

Endpoints
---------
GET  /                   Health check
POST /analyze            Full rPPG analysis pipeline → risk output
POST /enroll             Enroll a face embedding for a subject
POST /verify             Verify identity against enrolled embedding
GET  /governance/report  Governance / drift / bias monitoring report
"""

import os
import sys

# MONKEYPATCH: Prevent mediapipe -> sounddevice -> PortAudio initialization which hangs on some Windows systems
try:
    import sounddevice
    def mock_init(*args, **kwargs): pass
    sounddevice._initialize = mock_init
    print("[NeuroVitals] [PATCH] sounddevice._initialize bypassed successfully.")
except Exception:
    pass

import math
import time
import tempfile
import traceback
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from FACE.sdk.face_processor import FaceProcessor, extract_frames_from_video
from FACE.sdk.rppg_extractor import RPPGExtractor
from FACE.sdk.feature_engineer import (
    extract_all_features, detect_peaks, compute_sqi,
)
from FACE.sdk.anomaly_model import WaveformValidator
from FACE.sdk.bayesian_engine import (
    BayesianMentalHealthEngine, MoodInferenceEngine
)
from FACE.sdk.identity_verifier import (
    EmbeddingModel, verify_identity, cosine_similarity, IdentityVerifier,
    compute_liveness_score,
)
from FACE.sdk.logger import (
    configure_audit_logger, audit_analysis, audit_identity,
    audit_consent, audit_governance, audit_event,
)
from FACE.sdk.security import (
    generate_consent_token, verify_consent_token,
)
from FACE.sdk.governance import GovernanceMonitor
from FACE.api.schemas import (
    HealthResponse,
    AnalysisResponse,
    ExplainabilityDetail,
    BayesianResult,
    ClinicalMetricSet,
    EnrollResponse,
    VerifyResponse,
    GovernanceReport,
    MoodResult,
)


# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

TRACE_LOG = os.path.join(os.path.dirname(__file__), "_logs", "analysis_trace.log")
os.makedirs(os.path.dirname(TRACE_LOG), exist_ok=True)

def trace_log(message: str):
    """Helper to write to both terminal and a dedicated debug file."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] {message}\n"
    print(message, flush=True)  # keep terminal output
    with open(TRACE_LOG, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()

app = FastAPI(
    title="NeuroVitals rPPG API",
    description=(
        "AI-powered contactless facial identification and mental health "
        "phenotyping engine using remote photoplethysmography (rPPG)."
    ),
    version="1.0.0",
)

# ROBUST CORS MIDDLEWARE
# Mirrors the request origin to satisfy browser 'null' checks
from fastapi import Request, Response

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    # Mirror the request's origin EXACTLY (handles 'null' and protocol mismatches)
    origin = request.headers.get("origin", "*")
    
    if request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true"
            }
        )
    
    try:
        response = await call_next(request)
    except Exception as e:
        trace_log(f"[CORS_GATE] Unhandled exception in pipeline: {e}")
        response = JSONResponse(
            status_code=500,
            content={"detail": f"Clinical Engine Failure: {str(e)}"}
        )
    
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

logger = configure_audit_logger()

# In-memory stores (replace with database in production)
_enrolled_embeddings: dict = {}       # subject_id → np.ndarray

# Global singletons for performance — INITIALIZED AT STARTUP
_face_processor = FaceProcessor()
_rppg_extractor = RPPGExtractor(fs=30.0)
_bayesian_engine = BayesianMentalHealthEngine()
_mood_engine = MoodInferenceEngine()
_waveform_validator = WaveformValidator()
_identity_verifier = IdentityVerifier()
_governance_monitor = GovernanceMonitor()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse()


# ---------------------------------------------------------------------------
# POST /analyze  — Full rPPG Analysis Pipeline
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_video(file: UploadFile = File(...)):
    """Accept a short video, run the full rPPG → Bayesian → SHAP pipeline.
    
    Using 'def' instead of 'async def' to offload this CPU-bound work
    to FastAPI's internal threadpool, preventing event loop blockage.
    """

    video_path = None # Initialize video_path outside the try block

    try:
        # Use simple file.file.read() to avoid async file operations in a sync worker
        contents = file.file.read()

        # Write to temp file for cv2
        trace_log(f"[NeuroVitals] Received request. Payload size: {len(contents)} bytes")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        try:
            tmp.write(contents)
            tmp.close()
            video_path = tmp.name
            trace_log(f"[NeuroVitals] Temp video stored: {video_path}")
        except Exception as e:
            trace_log(f"[NeuroVitals] ERROR writing temp file: {e}")
            raise HTTPException(status_code=500, detail="Failed to store uploaded video")

        # --- 1. Frame extraction ---
        trace_log("[NeuroVitals] [STEP 1] Starting frame extraction...")
        frames = extract_frames_from_video(video_path, max_frames=450)
        trace_log(f"[NeuroVitals] [STEP 1] Extracted {len(frames)} frames total.")
        
        if not frames:
            trace_log("[NeuroVitals] [STEP 1] FATAL ERROR: extract_frames_from_video returned 0 frames.")
            raise HTTPException(status_code=400, detail="No frames found in video")

        # --- 2. Face ROI extraction & Landmark Collection ---
        trace_log("[NeuroVitals] [STEP 2] Initializing ROI extraction...")
        roi_buffer = []
        landmarks_buffer = []  # For liveness tracking
        
        # We'll try to limit to 300 frames if it's too large, or just keep going
        frames_to_process = frames[:300] if len(frames) > 300 else frames
        trace_log(f"[NeuroVitals] [STEP 2] Processing {len(frames_to_process)} frames for ROI...")

        try:
            for i, frame in enumerate(frames_to_process):
                if i % 20 == 0:
                    trace_log(f"[NeuroVitals] [STEP 2] ROI Frame {i}/{len(frames_to_process)}... Buffer size: {len(roi_buffer)}")

                # Use a very specific try-except here
                res_roi, res_lm = None, None
                try:
                    res_roi, res_lm = _face_processor.extract_roi_with_landmarks(frame)
                except Exception as sdk_e:
                    trace_log(f"[NeuroVitals] [STEP 2] SDK Error at frame {i}: {sdk_e}")
                
                if res_roi is not None:
                    roi_buffer.append(res_roi)
                    landmarks_buffer.append(res_lm)
                
                # Manual memory check/hint
                if i % 100 == 0:
                    import gc
                    gc.collect()

            trace_log(f"[NeuroVitals] [STEP 2] ROI extraction complete. Valid faces: {len(roi_buffer)}")
        except Exception as loop_e:
            trace_log(f"[NeuroVitals] [STEP 2] FATAL LOOP ERROR: {loop_e}")
            trace_log(traceback.format_exc())
            raise HTTPException(status_code=500, detail="ROI pipeline failure")

        if len(roi_buffer) < 2:  # Reduced from 10 to 2 for extreme resiliency
            trace_log(f"[NeuroVitals] [STEP 2] WARNING: Only {len(roi_buffer)} faces found. Proceeding with limited data.")
            if len(roi_buffer) == 0:
                raise HTTPException(status_code=400, detail="No face detected in video stream")

        # --- 3. rPPG signal extraction (CHROM) ---
        trace_log("[NeuroVitals] [STEP 3] Running CHROM signal extraction...")
        signal = _rppg_extractor.extract(roi_buffer)
        trace_log(f"[NeuroVitals] [STEP 3] Signal extracted. Samples: {len(signal)}")

        # --- 4. Full feature extraction ---
        trace_log("[NeuroVitals] [STEP 4] Computing 10 clinical features...")
        features = extract_all_features(signal, fs=30.0)
        
        # Clean up features: convert NaN to 0.0 for easier consumption if preferred, 
        # but here we log them to see what's actually happening.
        trace_log(f"[NeuroVitals] [STEP 4] RAW Features: {features}")
        
        # Explicitly check for HR
        hr_val = features.get('heart_rate_bpm')
        if hr_val is None or math.isnan(hr_val):
            trace_log("[NeuroVitals] [STEP 4] WARNING: Heart rate is NaN (insufficient peaks)")
        else:
            trace_log(f"[NeuroVitals] [STEP 4] Heart Rate Detected: {hr_val:.2f} BPM")

        # --- 5. Bayesian mental health inference ---
        trace_log("[NeuroVitals] [STEP 5] Running Bayesian Inference Engine...")
        inference = _bayesian_engine.full_inference(features)
        
        posteriors = inference["posteriors"]
        risk_class = inference["risk_class"]
        trace_log(f"[NeuroVitals] [STEP 5] Inference Result: {risk_class}")

        # --- 6. Identity & liveness ---
        trace_log("[NeuroVitals] [STEP 6] Running Identity Verification...")
        # Liveness now uses the real landmarks sequence
        liveness_score = _identity_verifier.track_liveness(landmarks_buffer)
        trace_log(f"[NeuroVitals] [STEP 6] Blink-based Liveness: {liveness_score:.2f}")

        # Verification (Face Embedding)
        # In a real system, we'd pass the best ROI to the embedder
        best_roi = roi_buffer[len(roi_buffer)//2] if roi_buffer else None
        embedding_model = EmbeddingModel()
        embedding = embedding_model.embed(best_roi)
        # Placeholder verification against self
        identity_ok = verify_identity(embedding, embedding, threshold=0.85)

        # --- 7. Build explainability detail ---
        rmssd = features.get("rmssd", 0.0) or 0.0
        lf_hf = features.get("lf_hf_ratio", 0.0) or 0.0
        entropy = features.get("spectral_entropy", 0.0) or 0.0
        rsa = features.get("rsa", 0.0) or 0.0
        stress = features.get("stress_reactivity", 0.0) or 0.0

        explainability = ExplainabilityDetail(
            Low_HRV=float(max(0, 1.0 - rmssd / 0.1)),
            High_LFHF=float(min(lf_hf / 5.0, 1.0)),
            High_Entropy=float(min(entropy / 5.0, 1.0)),
            Low_RSA=float(max(0, 1.0 - rsa / 0.005)),
            High_Stress=float(stress),
        )

        # --- 8. Signal Authenticity (LSTM Anomaly) ---
        authenticity = _waveform_validator.validate_signal(signal)
        trace_log(f"[NeuroVitals] [STEP 8] Waveform Authenticity Score: {authenticity:.2f}")

        # --- 9. Clinical Result Mapping & Explainability ---
        inference = _bayesian_engine.full_inference(features)
        posteriors = inference["posteriors"]
        risk_class = inference["risk_class"]
        explainability_data = inference["explainability"]

        # --- 9.5 Mood Inference ---
        mood_data = _mood_engine.infer_mood(features)
        trace_log(f"[NeuroVitals] [MOOD] Current State: {mood_data['MoodState']}")

        # --- 10. Confidence based on SQI ---
        sqi = features.get("signal_quality_index", 0.5)
        
        # --- 11. Assemble response ---
        # Ensure all NaN features are converted to 0.0 for JSON compatibility
        processed_features = {
            k: (v if not (isinstance(v, (float, np.float64)) and np.isnan(v)) else 0.0)
            for k, v in features.items()
        }

        # Final Liveness is a blend of facial consistency + waveform authenticity
        final_liveness = float(np.clip(0.7 * liveness_score + 0.3 * authenticity, 0.0, 1.0))

        response = AnalysisResponse(
            IdentityVerified=bool(identity_ok),
            LivenessScore=final_liveness,
            SignalQualityIndex=float(sqi),
            DepressionRiskScore=float(posteriors.get("depression", 0.0)),
            AnxietyRiskScore=float(posteriors.get("anxiety", 0.0)),
            AutonomicStabilityIndex=float(1.0 - processed_features.get("stress_reactivity", 0.0)),
            MentalHealthRiskClass=risk_class,
            Explainability=ExplainabilityDetail(**explainability_data),
            Mood=MoodResult(**mood_data),
            BayesianPosteriors=BayesianResult(**posteriors),
            ClinicalFeatures=ClinicalMetricSet(**processed_features),
            ConfidenceScore=float(np.clip(sqi * 0.95, 0.0, 1.0)),
            DominantCondition=max(posteriors, key=posteriors.get) if posteriors else "Unknown",
        )

        audit_analysis(logger, {
            "risk_class": response.MentalHealthRiskClass,
            "confidence": response.ConfidenceScore,
            "sqi": response.SignalQualityIndex,
        })

        return response

    except HTTPException:
        # Re-raise HTTPExceptions as they are intended responses
        raise
    except Exception as e:
        error_msg = f"[NeuroVitals] [FATAL] Critical error during analysis: {str(e)}\n{traceback.format_exc()}"
        trace_log(error_msg)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Cleanup
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as e:
                trace_log(f"[NeuroVitals] WARNING: Failed to delete temp file {video_path}: {e}")


# ---------------------------------------------------------------------------
# POST /enroll  — Enroll Face Embedding
# ---------------------------------------------------------------------------

@app.post("/enroll", response_model=EnrollResponse)
async def enroll_face(
    subject_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Enroll a face image for identity verification."""
    import cv2

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = EmbeddingModel()
    embedding = model.embed(image)
    _enrolled_embeddings[subject_id] = embedding

    audit_identity(logger, {
        "action": "enroll",
        "subject_id": subject_id,
    })

    return EnrollResponse(subject_id=subject_id)


# ---------------------------------------------------------------------------
# POST /verify  — Verify Identity
# ---------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse)
async def verify_face(
    subject_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Verify a face image against an enrolled embedding."""
    import cv2

    if subject_id not in _enrolled_embeddings:
        raise HTTPException(status_code=404,
                            detail=f"Subject '{subject_id}' not enrolled")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = EmbeddingModel()
    probe = model.embed(image)
    enrolled = _enrolled_embeddings[subject_id]

    sim = cosine_similarity(probe, enrolled)
    verified = verify_identity(probe, enrolled, threshold=0.85)

    outcome = "success" if verified else "failure"
    audit_identity(logger, {
        "action": "verify",
        "subject_id": subject_id,
        "similarity": round(sim, 4),
        "verified": verified,
    }, outcome=outcome)

    return VerifyResponse(
        subject_id=subject_id,
        identity_verified=verified,
        similarity_score=round(sim, 4),
        liveness_score=0.9,  # placeholder
        message="Identity verified" if verified else "Identity mismatch",
    )


# ---------------------------------------------------------------------------
# GET /governance/report
# ---------------------------------------------------------------------------

@app.get("/governance/report", response_model=GovernanceReport)
async def governance_report():
    """Return the latest governance / drift / bias monitoring report.

    In production this would use stored reference distributions and
    recent prediction logs.  Here we return a placeholder report.
    """
    report = _governance_monitor.generate_report()

    audit_governance(logger, {"action": "report_generated"})

    recommendation = "No action required"
    if report.get("any_drift_detected"):
        recommendation = "Feature drift detected — model recalibration recommended"

    return GovernanceReport(
        status=report.get("status", "generated"),
        any_drift_detected=report.get("any_drift_detected", False),
        recommendation=recommendation,
    )
