"""
DeepShield-NIDS: Real-Time Inference Engine
Team PNS: P. Vinod, B. Pavan, B. Siddhartha (IIITDM Kancheepuram)
"""

import numpy as np
import pandas as pd
from .models import DeepShieldHybrid
from .dataset import FEATURE_NAMES

class DeepShieldPredictor:
    def __init__(self, alpha=0.65, threshold=0.50):
        self.model = DeepShieldHybrid(input_dim=len(FEATURE_NAMES), alpha=alpha)
        self.threshold = threshold
        self.feature_names = FEATURE_NAMES

    def predict_flow(self, feature_vector):
        """
        Takes a 69-dimensional feature array/list.
        Returns:
          label: "BENIGN" or "ATTACK"
          attack_probability: float [0, 1]
          reconstruction_mse: float
          is_zero_day_anomaly: bool
        """
        arr = np.asarray(feature_vector, dtype=np.float32)
        preds, probs, mses = self.model.predict(arr, threshold=self.threshold)
        
        is_attack = bool(preds[0] == 1)
        prob = float(probs[0])
        mse = float(mses[0])
        is_anomaly = bool(mse > self.model.autoencoder.reconstruction_threshold)
        
        return {
            "prediction": "ATTACK" if is_attack else "BENIGN",
            "is_attack": is_attack,
            "attack_probability": round(prob, 4),
            "reconstruction_error": round(mse, 4),
            "zero_day_anomaly_flag": is_anomaly,
            "threat_level": "CRITICAL" if prob > 0.8 else ("HIGH" if prob > 0.5 else ("ELEVATED" if is_anomaly else "NORMAL"))
        }

    def predict_batch(self, df):
        """
        Takes pandas DataFrame with the 69 features.
        Returns original DataFrame enriched with predictions.
        """
        # Ensure only expected features are taken
        existing_cols = [c for c in self.feature_names if c in df.columns]
        if len(existing_cols) < len(self.feature_names):
            # Pad missing with 0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
                    
        X = df[self.feature_names].values.astype(np.float32)
        preds, probs, mses = self.model.predict(X, threshold=self.threshold)
        
        res = df.copy()
        res["Predicted_Label"] = np.where(preds == 1, "ATTACK", "BENIGN")
        res["Attack_Probability"] = np.round(probs, 4)
        res["Reconstruction_MSE"] = np.round(mses, 4)
        res["Zero_Day_Flag"] = (mses > self.model.autoencoder.reconstruction_threshold)
        
        return res
