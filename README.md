# Bearing Fault Detection 

A two-stage machine learning system for real-time bearing fault detection using vibration signal analysis. This project focuses on early fault detection and severity classification using the NASA IMS run-to-failure dataset.

---

## Project Overview

Bearings are critical components in rotating machinery, and their failure can lead to costly downtime and safety risks.

This project implements a predictive maintenance system that:

- Detects faults at an early stage  
- Classifies severity of degradation  
- Simulates real-time monitoring  

---

## System Architecture

Raw Vibration Signal  
→ Feature Extraction (Time + Frequency + Envelope)  
→ Stage 1: Fault Detection (SVM)  
→ Stage 2: Severity Classification (Logistic Regression)  
→ Real-Time Alarm Logic  

---

## Dataset Used

**NASA IMS Bearing Dataset**

- Run-to-failure dataset  
- Sampling frequency: 20 kHz  
- Contains vibration signals over time  
- Shows full degradation progression  

---

## Dataset Not Included in Repository

Due to size constraints, the dataset is not included in this repository.

---

## How to Add Dataset

### Step 1: Download Dataset  
Download from:  
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

### Step 2: Extract Files  
Unzip and locate folders such as:  

1st_test/
2nd_test/
3rd_test/


### Step 3: Place Dataset in Project  

Create the following structure:


project-root/
│
├── data/
│ └── IMS/
│ └── 1st_test/
│ ├── 2003.10.22.12.06.24
│ ├── 2003.10.22.12.09.13
│ └── ...


### Step 4: Update Path in Code  

Set the dataset path in your code:

```python
DATA_PATH = "data/IMS/1st_test/"
Feature Engineering

The following features are extracted from vibration signals:

Time-Domain Features
RMS
Standard deviation
Peak-to-peak
Kurtosis
Crest factor
Frequency-Domain Features
FFT
Band energy (4–9.5 kHz)
Envelope Features
Envelope RMS
Envelope Kurtosis
Model Design
Stage 1: Fault Detection
Model: Support Vector Machine (SVM)
Output: Healthy or Faulty
Stage 2: Severity Classification
Model: Logistic Regression
Output: Degraded or Failing
Labeling Strategy

Based on time progression:

Early stage: Healthy
Mid stage: Degraded
Late stage: Failing
Real-Time Detection Simulation

The system simulates real-time processing:

Read vibration files sequentially
Extract features
Predict fault probability
Apply threshold
Trigger alarm after K consecutive detections (K = 3)
Classify severity
Observations
RMS increases as fault severity increases
Kurtosis spikes during defects
Fault probability rises before failure
Early warning signals are detected
Evaluation Strategy
Time-based train-test split
Model comparison (SVM, Random Forest, KNN, etc.)
F1-score as primary metric
Current Status
Feature engineering pipeline completed
Two-stage model implemented
Real-time detection simulation completed
Fault detection and severity classification working
Future Scope
Cross-dataset evaluation (IMS to XJTU-SY)
Domain adaptation for better generalization
Condition-invariant feature learning
Remaining Useful Life (RUL) prediction
Deep learning approaches (CNN, TCN)
Deployment on edge or IoT devices
Real-time monitoring dashboard
Key Learnings
Feature engineering is critical for fault detection
High accuracy does not guarantee generalization
Real-time systems require stable predictions
Domain shift is a major challenge in industrial ML
Tech Stack
Python
NumPy
Pandas
Scikit-learn
SciPy
Matplotlib
Project Structure
project-root/
│
├── data/
│   └── IMS/
│
├── src/
│   ├── feature_extraction.py
│   ├── model_stage1.py
│   ├── model_stage2.py
│   ├── realtime_simulation.py
│
├── notebooks/
├── results/
├── README.md
Contribution

This project is part of a learning and research effort in predictive maintenance and machine learning.
