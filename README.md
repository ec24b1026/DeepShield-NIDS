# 🛡️ DeepShield-NIDS: Deep Hybrid Temporal Autoencoder & Multi-Head Attention Network for Zero-Day Intrusion Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepshield-nids-pns.streamlit.app/)

**Developed by Team PNS (IIITDM Kancheepuram):**
- **Pranay**
- **Sravan**
- **Nirupam**

---

## 📌 Executive Summary

Modern Network Intrusion Detection Systems (NIDS) frequently exhibit near-perfect accuracy (99.9%+) on static, in-distribution cross-validation data. However, as demonstrated in our foundational research (`vinod935/network-intrusion-detection-system`), production models trained on early weekday traffic (Monday–Thursday) **catastrophically collapse** when confronted with unseen temporal distribution shifts (the **"Friday Test"**). Specifically:
- **Baseline HistGradientBoosting Recall on Friday Attacks**: Collapsed to **28.46%**.
- **Botnet (ARES) Attack Recall**: **0.00%** (completely undetected).
- **PortScan Attack Recall**: **1.18%**.

**DeepShield-NIDS** resolves this fundamental flaw by moving beyond shallow tabular algorithms to an end-to-end **Dual-Stream Deep Learning Architecture**:
1. **Stream A (Self-Supervised Deep Autoencoder)**: Models the manifold of benign network behaviors. Zero-day attacks and novel temporal variations produce significant reconstruction error ($\text{MSE} = \frac{1}{D} \sum (x_i - \hat{x}_i)^2$), detecting intrusions even when uncatalogued by supervised labels.
2. **Stream B (Temporal 1D-CNN + BiLSTM with Multi-Head Self-Attention)**: Dynamically extracts hierarchical spatial-temporal correlations across 69 continuous flow metrics extracted via CICFlowMeter.
3. **Adaptive Hybrid Fusion Engine**: Calibrates supervised classification confidence with self-supervised reconstruction energy.

---

## 🚀 Key Benchmark Results

| Model Architecture | In-Dist Validation F1 | Friday Shift Recall | Botnet (ARES) Recall | PortScan Recall | Friday Shift F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 77.58% | 18.40% | 0.00% | 0.45% | 25.45% |
| Random Forest Baseline | 99.75% | 26.10% | 0.00% | 0.95% | 36.75% |
| HistGradientBoosting (*vinod935*) | 99.88% | 28.46% | 0.00% | 1.18% | 39.73% |
| Standard MLP Baseline | 98.42% | 32.80% | 12.40% | 14.50% | 43.40% |
| **DeepShield-NIDS Hybrid (Ours)** | **99.92%** | **94.82%** 🔥 | **91.40%** 🔥 | **95.80%** 🔥 | **93.75%** 🔥 |

---

## 🏗️ Architecture Blueprint

```text
========================================================================================
                      INPUT NETWORK FLOW (69 CONTINUOUS FEATURES)
========================================================================================
                                   |
               +-------------------+-------------------+
               |                                       |
[STREAM A: SELF-SUPERVISED VAE]         [STREAM B: TEMPORAL ATTENTION NETWORK]
               |                                       |
     Dense(69 -> 48) + ReLU                  1D Conv Layer (69 -> 64 filters)
               |                                       |
     Dense(48 -> 24) + ReLU                  Bidirectional LSTM Sequence Block
               |                                       |
   Latent Space Z (12-dim)                   Multi-Head Scaled Dot-Product Attention
               |                                       |
     Dense(12 -> 24) + ReLU                       Attention Feature Pool (64-dim)
               |                                       |
     Dense(24 -> 48) + ReLU                  Dense Classifier + Sigmoid
               |                                       |
     Reconstruction Output (69)              Supervised Probability P_sup
               |                                       |
   MSE = ||X - X_hat||^2 / 69                          |
               |                                       |
   Anomaly Prob P_vae = σ((MSE - θ)/T)                 |
               |                                       |
               +-------------------+-------------------+
                                   |
                        [HYBRID FUSION ENGINE]
         Attack Score S = α * P_sup + (1 - α) * P_vae
                                   |
                           [DECISION THRESHOLD]
                     Final Prediction: BENIGN / ATTACK
========================================================================================
```

---

## 💻 Quick Start & Running the Dashboard

### 1. Clone & Install
```bash
git clone https://github.com/vinod935/DeepShield-NIDS.git
cd DeepShield-NIDS
pip install -r requirements.txt
```

### 2. Launch Streamlit SOC Dashboard
```bash
streamlit run dashboard/app.py
```
Then navigate to `http://localhost:8501`.

---

## 👥 Authors (Team PNS)
- **Pranay** - Deep Learning Architecture & Temporal Drift Research
- **Sravan** - Feature Engineering & Data Pipeline Optimization
- **Nirupam** - Streamlit SOC Dashboard & Benchmark Evaluation
- *Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram*
