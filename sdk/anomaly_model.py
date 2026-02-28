"""
NeuroVitals — LSTM Waveform Anomaly Detection
=============================================
Implementation of point 6.1 from the architecture blueprint.
Uses a lightweight RNN to detect temporal inconsistencies, signal dropouts,
or non-biological patterns (e.g. synthetic pulses).
"""

import torch
import torch.nn as nn
import numpy as np

class LSTMAnomalyModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # We only care about the last output for classification
        return torch.sigmoid(self.linear(out[:, -1, :]))

class WaveformValidator:
    def __init__(self):
        self.model = LSTMAnomalyModel()
        # In production, we would load weights here:
        # self.model.load_state_dict(torch.load('models/lstm_anomaly.pth'))
        self.model.eval()

    def validate_signal(self, signal: np.ndarray) -> float:
        """
        Returns an 'authenticity' score [0-1].
        1.0 = highly biological; 0.0 = anomalous (synthetic/noisy).
        """
        if len(signal) < 30:
            return 0.0
            
        # Reshape for LSTM: (1, seq_len, 1)
        # Normalize signal locally
        s_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
        x_tensor = torch.FloatTensor(s_norm).unsqueeze(0).unsqueeze(-1)
        
        with torch.no_grad():
            score = self.model(x_tensor)
            
        return float(score.item())
