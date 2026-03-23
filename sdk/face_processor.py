import cv2
import numpy as np
from typing import Optional

# MediaPipe 0.10.x+ uses the Tasks API instead of the legacy mp.solutions API.
# We import the new FaceLandmarker from mediapipe.tasks.
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from insightface.app import FaceAnalysis


# ── model asset download helper ──────────────────────────────────────────
import os, urllib.request

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "_models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model() -> str:
    """Download the face_landmarker model once if it is not already cached."""
    if os.path.isfile(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"[NeuroVitals] Downloading face landmarker model to {_MODEL_PATH} …")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class FaceProcessor:
    """High-reliability face ROI extraction using MediaPipe FaceLandmarker (Tasks API).

    - extracts a forehead ROI mask optimised for rPPG (low-motion area)
    - keeps an internal FaceLandmarker instance to be reused across frames
    """

    def __init__(self, max_faces: int = 1):
        print("[NeuroVitals] [FaceProcessor] Ensuring model exists...")
        model_path = _ensure_model()
        print(f"[NeuroVitals] [FaceProcessor] Using model: {model_path}")
        
        print("[NeuroVitals] [FaceProcessor] Setting up MediaPipe BaseOptions...")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        
        print("[NeuroVitals] [FaceProcessor] Configuring LandmarkerOptions...")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        
        print("[NeuroVitals] [FaceProcessor] Creating FaceLandmarker from options...")
        try:
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            print("[NeuroVitals] [FaceProcessor] FaceLandmarker created successfully.")
        except Exception as e:
            print(f"[NeuroVitals] [FaceProcessor] FATAL: Landmarker creation failed: {e}")
            raise e

        print("[NeuroVitals] [FaceProcessor] InsightFace will be lazy-loaded on first use (buffalo_s).")
        self._demographics_app = None
        self._demographics_initialized = False

    def estimate_demographics(self, frame: np.ndarray):
        """Estimate age and gender using InsightFace (lazy-loaded)."""
        # Lazy-init InsightFace on first call to save startup memory
        if not self._demographics_initialized:
            self._demographics_initialized = True
            try:
                print("[NeuroVitals] [FaceProcessor] Lazy-loading InsightFace buffalo_s...")
                self._demographics_app = FaceAnalysis(
                    name='buffalo_s', root=_MODEL_DIR,
                    providers=['CPUExecutionProvider']
                )
                self._demographics_app.prepare(ctx_id=-1, det_size=(320, 320))
                print("[NeuroVitals] [FaceProcessor] InsightFace initialized successfully.")
            except Exception as e:
                print(f"[NeuroVitals] [FaceProcessor] WARNING: InsightFace failed: {e}")
                self._demographics_app = None

        if self._demographics_app is None:
            return None
        
        try:
            faces = self._demographics_app.get(frame)
            if not faces:
                return None
            
            # Use the first face detected
            face = faces[0]
            gender_val = "male" if face.gender == 1 else "female"
            age_val = int(face.age)
            
            return {
                "gender": gender_val,
                "age": age_val,
                "confidence": float(face.det_score)
            }
        except Exception as e:
            print(f"[NeuroVitals] [FaceProcessor] Demographic estimation error: {e}")
            return None


    def extract_roi_with_landmarks(self, frame: np.ndarray):
        """
        Unified method: Returns both the cropping-optimized Forehead ROI 
        and the raw MediaPipe landmark array for rPPG + Liveness.
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                return None, None

            landmarks = result.face_landmarks[0]  # first face
            h, w, _ = frame.shape
            
            # Convert landmarks to pixel array for liveness tracking
            lm_px = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])

            # ROI for forehead (indices 10, 151, 9, 8)
            forehead_idx = [10, 151, 9, 8, 338, 297, 332]
            forehead_points = [lm_px[i] for i in forehead_idx if i < len(lm_px)]

            if len(forehead_points) < 3:
                return None, None

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(forehead_points), 255)
            
            # Extract full frame with mask for signal
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Crop the ROI similarly to the original extract_forehead_roi logic
            x, y, ww, hh = cv2.boundingRect(np.array(forehead_points))
            roi_cropped = masked_frame[y : y + hh, x : x + ww]

            return roi_cropped, lm_px
        except Exception as e:
            print(f"[NeuroVitals] [FACE_PROCESSOR] ERROR in extract_roi_with_landmarks: {e}")
            return None, None


    def close(self):
        self._landmarker.close()


def extract_frames_from_video(path: str, max_frames: int = 300):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames
