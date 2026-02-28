"""
NeuroVitals — ML Models, Inference Pipeline & Explainability
============================================================
LSTM waveform anomaly detector, XGBoost mental distress classifier,
SHAP-based explainability, and a unified RiskInferencePipeline.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from typing import Any, Dict, Optional

try:
    import shap
except ImportError:
    shap = None  # graceful fallback if shap is not installed

try:
    import joblib
except ImportError:
    joblib = None


# ---------------------------------------------------------------------------
# LSTM Waveform Anomaly Detector
# ---------------------------------------------------------------------------

class LSTMAnomaly(nn.Module):
    """Stacked LSTM for detecting anomalous pulse-waveform patterns.

    Input : (batch, seq_len, input_dim)
    Output: (batch, 1)  — anomaly score
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


# ---------------------------------------------------------------------------
# XGBoost Risk Classifier — train / save / load
# ---------------------------------------------------------------------------

def train_xgb_risk_model(X_train, y_train, **kwargs) -> Any:
    """Train an XGBoost mental-distress risk classifier."""
    params = dict(
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        eval_metric="logloss", use_label_encoder=False,
    )
    params.update(kwargs)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str) -> None:
    """Persist a model to disk (XGBoost JSON or PyTorch .pt)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(model, xgb.XGBClassifier):
        model.save_model(path)
    elif isinstance(model, nn.Module):
        torch.save(model.state_dict(), path)
    elif joblib is not None:
        joblib.dump(model, path)
    else:
        raise ValueError(f"Cannot save model of type {type(model)}")


def load_xgb_model(path: str) -> xgb.XGBClassifier:
    """Load a previously saved XGBoost model."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def load_lstm_model(path: str, input_dim: int = 1,
                    hidden_dim: int = 64,
                    num_layers: int = 2) -> LSTMAnomaly:
    """Load a previously saved LSTM model."""
    model = LSTMAnomaly(input_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------

def explain_with_shap(model, features: Dict[str, float]) -> Dict[str, float]:
    """Generate SHAP feature-contribution values for an XGBoost prediction.

    Returns a dict mapping feature names → SHAP values (signed contribution).
    Falls back to a normalised feature-magnitude heuristic if *shap* is not
    installed or the model is not tree-based.
    """
    feature_names = list(features.keys())
    feature_values = np.array([[features[k] for k in feature_names]])

    if shap is not None and isinstance(model, xgb.XGBClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        sv = shap_values[0]
        return {name: float(val) for name, val in zip(feature_names, sv)}

    # Fallback: normalised absolute feature contributions
    abs_vals = np.abs(feature_values[0])
    total = abs_vals.sum() + 1e-12
    return {name: float(abs_vals[i] / total) for i, name in enumerate(feature_names)}


# ---------------------------------------------------------------------------
# Unified Risk Inference Pipeline
# ---------------------------------------------------------------------------

class RiskInferencePipeline:
    """Orchestrates feature → model prediction → explainability.

    Usage::

        pipe = RiskInferencePipeline(xgb_model=model)
        result = pipe.predict(features_dict)
        # result → {"risk_score": 0.42, "explainability": {...}}
    """

    def __init__(self,
                 xgb_model: Optional[xgb.XGBClassifier] = None,
                 lstm_model: Optional[LSTMAnomaly] = None):
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model

    def predict(self, features: Dict[str, float]) -> Dict:
        """Run inference on a feature dict and return risk score + SHAP."""
        result: Dict[str, Any] = {}

        if self.xgb_model is not None:
            feature_names = sorted(features.keys())
            X = np.array([[features.get(k, 0.0) for k in feature_names]])
            try:
                proba = self.xgb_model.predict_proba(X)
                risk_score = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            except Exception:
                risk_score = float("nan")
            result["risk_score"] = risk_score
            result["explainability"] = explain_with_shap(self.xgb_model, features)
        else:
            # Without a trained model, return heuristic scores
            result["risk_score"] = float("nan")
            result["explainability"] = explain_with_shap(None, features)

        if self.lstm_model is not None:
            # LSTM expects raw signal; skip if only feature dict provided
            result["lstm_anomaly_score"] = float("nan")

        return result
