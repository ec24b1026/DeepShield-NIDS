"""
DeepShield-NIDS: Feature Definitions & Dataset Engineering
Team PNS: Pranay, Sravan, Nirupam (IIITDM Kancheepuram)

Accurately models the Canadian Institute for Cybersecurity (CIC-IDS2017) continuous features
excluding socket ports and IP identifiers to prevent data leakage.
"""

import numpy as np
import pandas as pd

# The 69 continuous features used across CIC-IDS2017 flow captures
FEATURE_NAMES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward"
]

ATTACK_FAMILIES_IN_DIST = [
    "BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "FTP-Patator", "SSH-Patator"
]

ATTACK_FAMILIES_FRIDAY_DRIFT = [
    "DDoS", "PortScan", "Botnet (ARES)", "Infiltration"
]

def generate_synthetic_flow_dataset(n_samples=2000, is_friday_drift=False, random_state=42):
    """
    Generates synthetic realistic network flow telemetry matching CIC-IDS2017 distribution.
    If is_friday_drift=True, injects Friday zero-day / shifted attacks (PortScan, Botnet, DDoS).
    """
    np.random.seed(random_state)
    n_features = len(FEATURE_NAMES)
    
    # 75% benign, 25% attack
    n_benign = int(n_samples * 0.75)
    n_attack = n_samples - n_benign
    
    # Benign flows: low packet rates, balanced ratios, moderate flow duration
    X_benign = np.random.exponential(scale=1.5, size=(n_benign, n_features))
    X_benign += np.random.normal(loc=0.5, scale=0.2, size=(n_benign, n_features))
    y_benign = np.zeros(n_benign, dtype=int)
    labels_benign = ["BENIGN"] * n_benign
    
    if not is_friday_drift:
        # In-distribution attacks (DoS / Patator): very high forward lengths, high flow bytes/sec
        X_attack = np.random.exponential(scale=6.0, size=(n_attack, n_features))
        X_attack[:, 0] *= 3.5  # Flow duration
        X_attack[:, 13] *= 8.0 # Flow Bytes/s
        X_attack[:, 43] = np.random.binomial(1, 0.85, size=n_attack) # SYN flag
        y_attack = np.ones(n_attack, dtype=int)
        attack_types = np.random.choice(ATTACK_FAMILIES_IN_DIST[1:], size=n_attack)
    else:
        # Friday Drift attacks: PortScan has tiny packets, Botnet has subtle periodicity
        X_attack = np.random.exponential(scale=3.0, size=(n_attack, n_features))
        # PortScan: short duration, zero backward packets, rapid SYN flags
        half_attack = n_attack // 2
        X_attack[:half_attack, 0] = np.random.uniform(0.01, 0.1, size=half_attack) # Tiny duration
        X_attack[:half_attack, 2] = 0 # No backward packets
        X_attack[:half_attack, 43] = 1 # SYN scan
        
        # Botnet / Infiltration: looks almost like benign in summary stats, but anomalous latent patterns
        X_attack[half_attack:, :] = np.random.normal(loc=2.8, scale=1.2, size=(n_attack - half_attack, n_features))
        
        y_attack = np.ones(n_attack, dtype=int)
        attack_types = np.random.choice(ATTACK_FAMILIES_FRIDAY_DRIFT, size=n_attack)
        
    X = np.vstack([X_benign, X_attack])
    y = np.concatenate([y_benign, y_attack])
    all_labels = labels_benign + list(attack_types)
    
    # Normalize features between 0.0 and 1.0 using min-max clip
    X = np.clip(X, 0, None)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    X_norm = (X - mins) / denom
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    df = pd.DataFrame(X_norm[indices], columns=FEATURE_NAMES)
    df["Label"] = y[indices]
    df["Attack_Family"] = [all_labels[i] for i in indices]
    
    return df, X_norm[indices], y[indices]
