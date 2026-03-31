# 🧠 Intelligent Bearing Fault Detection

### Two-Stage Cascaded SVM Pipeline with EWMA Control

> 📄 Research-grade implementation designed for IEEE TII / IM&M submission
> ⚙️ Focus: Early fault detection, diagnosis, and degradation tracking

---

## 📌 Overview

This project presents a **two-stage machine learning framework** for bearing fault detection using vibration signal analysis.

Unlike traditional single-stage classifiers or deep learning approaches, this system:

* Separates **fault detection** and **fault diagnosis**
* Uses **statistical control (EWMA)** for robust decision-making
* Introduces a **K-consecutive alarm filter** to reduce false alarms
* Provides a **Degradation Index (DI)** for monitoring system health over time

---

## 🏗️ Pipeline Architecture

```
Raw Signal → Feature Extraction → mRMR Selection  
        ↓
   Stage-1: Fault Detection (SVM + EWMA)
        ↓ (Gate)
   Stage-2: Fault Diagnosis (Multi-class SVM)
        ↓
   Degradation Index (Isotonic Regression)
```

---

## ⚡ Quick Start (Google Colab)

```python
# 1. Upload all .py files to Colab
# 2. Edit config.py → Paths class with your Drive paths
# 3. Run colab_runner.py step-by-step
```

### 📁 Expected Dataset Structure

```
MyDrive/
├── IMS2/extracted/1st_test/
├── IMS2/extracted/2nd_test/
├── IMS2/extracted/3rd_test/
└── FEMTO/Bearing1_1/   (optional)
```

---

## 📂 Project Structure

| File                       | Description                              |
| -------------------------- | ---------------------------------------- |
| `config.py`                | Central configuration (paths, constants) |
| `01_feature_extraction.py` | Feature extraction + mRMR selection      |
| `02_stage1_detection.py`   | SVM + EWMA + alarm logic                 |
| `03_stage2_diagnosis.py`   | Fault classification + DI                |
| `04_ablation_study.py`     | Model component analysis                 |
| `05_results_figures.py`    | Generates all plots                      |

---

## 🔬 Key Contributions

### 1️⃣ Two-Stage Cascaded Framework

* Separates detection and diagnosis
* Reduces unnecessary computation in Stage-2

### 2️⃣ EWMA-Based Statistical Gating

* Controls false alarm rate using control charts
* More robust than threshold-based methods

### 3️⃣ K-Consecutive Alarm Mechanism

* Ensures **temporal consistency**
* Reduces noisy detections

### 4️⃣ Fault-Type Classification

* Identifies:

  * Inner race fault
  * Outer race fault
  * Rolling element fault

### 5️⃣ Degradation Index (DI)

* Monotonic health indicator
* No need for labeled RUL data

### 6️⃣ Cross-Dataset Generalization

* Validated on:

  * IMS Dataset

---

## 📊 Evaluation Metrics

| Metric                 | Goal     |
| ---------------------- | -------- |
| FAR (False Alarm Rate) | < 1%     |
| Detection Delay (DD)   | Minimize |
| Stage-1 F1 Score       | Maximize |
| Stage-2 Macro F1       | Maximize |
| DI Monotonicity        | > 0.85   |

---

## 📦 Installation

```bash
pip install scikit-learn PyWavelets scipy numpy pandas matplotlib seaborn joblib
```

---

## 📈 Key Equations

**EWMA Statistic**

```
Z_t = λ·P_t + (1−λ)·Z_{t−1}
```

**Control Limit**

```
UCL_t = μ_0 + L·σ_0·√[λ/(2−λ)]
```

**Degradation Index**

```
DI = isotonic(P(fault))
```

---

## 📚 References

* Peng et al. (2005) — mRMR
* Montgomery (2009) — EWMA Control Charts
* Lei et al. (2018) — Prognostics Review
* Nectoux et al. (2012) — FEMTO Dataset

---

## 🎯 Future Work

* Real-time deployment using streaming data
* Integration with edge devices
* Extension to deep hybrid models

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork, raise issues, or submit pull requests.

---

## 📜 License

MIT License 

---

## ⭐ If you found this useful

Give it a star ⭐ — it helps a lot!
