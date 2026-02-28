"""
NeuroVitals — Identity Verification & Liveness Detection
=========================================================
ArcFace-based face embedding (via InsightFace), cosine identity
verification, and multi-signal liveness scoring (blink-rate, head-pose
challenge).
"""

import numpy as np
from typing import Dict, List, Optional
import math


class IdentityVerifier:
    def __init__(self):
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        # Thresholds for liveness
        self.EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for closed eye
        self.MOTION_THRESHOLD = 0.005 # Min variance in landmarks for liveness

    def track_liveness(self, all_landmarks: List[np.ndarray]) -> float:
        """
        Calculates liveness score [0-1] based on binary blink frequency
        and micro-motion variance during the capture period.
        """
        if not all_landmarks or len(all_landmarks) < 10:
            return 0.0

        blink_count = 0
        is_closed = False
        variances = []

        # indices for left/right eye landmarks (rough mesh indices)
        # 159, 145 (Left) | 386, 374 (Right)
        for landmarks in all_landmarks:
            # EAR Approximation
            p1, p2, p3, p4 = landmarks[159], landmarks[145], landmarks[386], landmarks[374]
            # p1[1] is y coordinate
            ear = (abs(p1[1]-p2[1]) + abs(p3[1]-p4[1])) / 2.0

            if ear < self.EAR_THRESHOLD:
                if not is_closed:
                    blink_count += 1
                    is_closed = True
            else:
                is_closed = False

            # Record variance of forehead point (index 10)
            variances.append(landmarks[10])

        # Convert to score
        # 1. Blink Score (at least 1 blink in 15s is healthy)
        blink_score = min(blink_count / 2.0, 1.0)

        # 2. Motion Score (prevents photos)
        motion_var = np.var(np.array(variances), axis=0).mean()
        motion_score = 1.0 if motion_var > self.MOTION_THRESHOLD else (motion_var / self.MOTION_THRESHOLD)

        return float(np.clip(0.6 * blink_score + 0.4 * motion_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# InsightFace — optional runtime dependency
# ---------------------------------------------------------------------------

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None


# ---------------------------------------------------------------------------
# Eye / Landmark Constants (MediaPipe FaceLandmarker indices)
# ---------------------------------------------------------------------------

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """512-D face embedding model.

    Uses InsightFace (ArcFace backbone) when available; otherwise returns
    a deterministic fallback vector suitable for development/testing.
    """

    def __init__(self, use_insightface: bool = True):
        self.fa = None
        if use_insightface and FaceAnalysis is not None:
            try:
                self.fa = FaceAnalysis(allowed_modules=["detection", "recognition"])
                self.fa.prepare(ctx_id=0, det_size=(640, 640))
            except Exception:
                self.fa = None

    def embed(self, face_image: Optional[np.ndarray]) -> np.ndarray:
        """Return a 512-D L2-normalised embedding for *face_image* (BGR)."""
        if face_image is None or self.fa is None:
            vec = np.tanh(np.linspace(0.1, 1.0, 512))
            return vec.astype(np.float32)

        rgb = face_image[:, :, ::-1]
        faces = self.fa.get(rgb)
        if not faces:
            return np.tanh(np.linspace(0.1, 1.0, 512)).astype(np.float32)
        emb = faces[0].embedding
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Identity Verification
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def verify_identity(embedding: np.ndarray,
                    enrolled_embedding: np.ndarray,
                    threshold: float = 0.85) -> bool:
    """Return ``True`` if cosine similarity ≥ threshold."""
    return cosine_similarity(embedding, enrolled_embedding) >= threshold


# ---------------------------------------------------------------------------
# Liveness — Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Compute EAR from 6 eye landmark points (shape (6,2))."""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


def _extract_ear_from_landmarks(landmarks, eye_idx: List[int]) -> Optional[float]:
    """Get EAR for one eye from a list/array of landmark points."""
    try:
        if isinstance(landmarks, np.ndarray):
            pts = np.array([landmarks[i] for i in eye_idx])
        else:
            pts = np.array([[lm.x, lm.y] for i, lm in enumerate(landmarks) if i in eye_idx])
            if len(pts) < 6:
                pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_idx])
        return eye_aspect_ratio(pts)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Liveness — Multi-frame Blink-Rate Score
# ---------------------------------------------------------------------------

def compute_liveness_score(landmarks_sequence: List,
                           ear_threshold: float = 0.21) -> float:
    """Compute a liveness score from a **sequence** of landmark frames.

    Unlike single-frame EAR, this tracks blink events over N frames.
    A healthy blink rate (10–20/min) is strong liveness evidence.

    Parameters
    ----------
    landmarks_sequence : list
        Each element is either a (468,2) ndarray or a list of landmark
        objects with ``.x`` / ``.y`` attributes.
    ear_threshold : float
        EAR below this is considered a "closed-eye" frame.

    Returns
    -------
    float  Liveness score ∈ [0, 1].
    """
    if not landmarks_sequence or len(landmarks_sequence) < 5:
        return 0.0

    ear_series = []
    for lm in landmarks_sequence:
        left_ear = _extract_ear_from_landmarks(lm, LEFT_EYE_IDX)
        right_ear = _extract_ear_from_landmarks(lm, RIGHT_EYE_IDX)
        ears = [e for e in (left_ear, right_ear) if e is not None]
        ear_series.append(float(np.mean(ears)) if ears else None)

    # Count blink events (EAR dips below threshold then rises)
    blink_count = 0
    in_blink = False
    for ear in ear_series:
        if ear is None:
            continue
        if ear < ear_threshold and not in_blink:
            blink_count += 1
            in_blink = True
        elif ear >= ear_threshold:
            in_blink = False

    # Expected: ~0.3 blinks/sec.  Score peaks around 1-3 blinks per 30 frames
    n_frames = len(ear_series)
    expected_blinks = max(1, n_frames / 30.0 * 5)  # ~5 blinks / second-of-video
    blink_ratio = min(blink_count / expected_blinks, 1.0)

    # Combine with EAR variance (static face = deepfake)
    valid_ears = [e for e in ear_series if e is not None]
    ear_variance = float(np.var(valid_ears)) if valid_ears else 0.0
    variance_score = min(ear_variance / 0.005, 1.0)  # normalise

    return float(np.clip(0.5 * blink_ratio + 0.5 * variance_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Head-Pose Challenge Response
# ---------------------------------------------------------------------------

def head_pose_challenge(landmarks_sequence: List,
                        axis: str = "yaw") -> float:
    """Score how well the subject followed a head-turn challenge.

    Computes the range of head rotation on the specified axis over the
    landmark sequence.  Larger range → stronger liveness evidence.

    Parameters
    ----------
    landmarks_sequence : list of (468,2) ndarrays or landmark lists
    axis : "yaw" | "pitch"

    Returns
    -------
    float  Challenge-response score ∈ [0, 1].
    """
    if not landmarks_sequence or len(landmarks_sequence) < 3:
        return 0.0

    # Use nose tip (idx 1) and left/right cheek (idx 234, 454) for yaw proxy
    positions = []
    for lm in landmarks_sequence:
        try:
            if isinstance(lm, np.ndarray):
                nose = lm[1]
                left = lm[234]
                right = lm[454]
            else:
                nose = np.array([lm[1].x, lm[1].y])
                left = np.array([lm[234].x, lm[234].y])
                right = np.array([lm[454].x, lm[454].y])
            # Yaw proxy: ratio of nose-to-left vs nose-to-right distance
            dl = np.linalg.norm(nose - left)
            dr = np.linalg.norm(nose - right)
            ratio = dl / (dr + 1e-6)
            positions.append(ratio)
        except Exception:
            continue

    if len(positions) < 3:
        return 0.0

    range_val = max(positions) - min(positions)
    # Expect ratio range > 0.3 for a real head turn
    return float(np.clip(range_val / 0.5, 0.0, 1.0))
