"""
config.py — Shared configuration for IEEE Bearing Fault Detection Pipeline
===========================================================================
Centralised constants for IMS + FEMTO-ST datasets, bearing geometry,
EWMA parameters, and path management.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS  (edit these to match your environment)
# ─────────────────────────────────────────────────────────────
class Paths:
    # ── Google Colab (default)
    BASE              = Path('/content/drive/MyDrive/IMS2')
    IMS_ROOT          = BASE / 'extracted'
    XJTU_ROOT        = BASE / 'XJTU/extracted/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1'          # PRONOSTIA dataset root

    # ── Output directories (auto-created)
    OUTPUT            = BASE / 'bearing_results'
    FEATURES_DIR      = OUTPUT / '01_features'
    DETECTION_DIR     = OUTPUT / '02_detection'
    DIAGNOSIS_DIR     = OUTPUT / '03_diagnosis'
    RESULTS_DIR       = OUTPUT / '04_results'
    FIGURES_DIR       = OUTPUT / '05_figures'

    @classmethod
    def make_all(cls):
        for attr in ['FEATURES_DIR','DETECTION_DIR','DIAGNOSIS_DIR',
                     'RESULTS_DIR','FIGURES_DIR']:
            Path(getattr(cls, attr)).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SIGNAL PARAMETERS
# ─────────────────────────────────────────────────────────────
class Signal:
    FS          = 20_000      # Sampling frequency  (IMS)
    FS_FEMTO    = 25_600      # Sampling frequency  (FEMTO-ST)
    N_SAMPLES   = 20_480      # Points per file     (IMS)
    N_SAMPLES_F = 2_560       # Points per file     (FEMTO-ST)
    RPM         = 2_000       # Shaft speed         (IMS)
    RPM_FEMTO   = 1_800       # Shaft speed         (FEMTO-ST, Condition 1)
    SHAFT_FREQ  = RPM / 60    # ~33.33 Hz


# ─────────────────────────────────────────────────────────────
# BEARING GEOMETRY — Rexnord ZA-2115 (IMS)
# ─────────────────────────────────────────────────────────────
class BearingIMS:
    """Characteristic defect frequencies (Hz) at 2000 RPM."""
    BPFO = 236.4   # Ball Pass Frequency Outer race
    BPFI = 296.9   # Ball Pass Frequency Inner race
    BSF  = 139.9   # Ball Spin Frequency
    FTF  =  14.9   # Fundamental Train Frequency (cage)
    FREQS = {'BPFO': BPFO, 'BPFI': BPFI, 'BSF': BSF, 'FTF': FTF}


# ─────────────────────────────────────────────────────────────
# BEARING GEOMETRY — SKF 6203 (FEMTO-ST)
# ─────────────────────────────────────────────────────────────
class BearingFEMTO:
    """Characteristic defect frequencies (Hz) at 1800 RPM."""
    BPFO = 236.4   # approx — update with exact SKF 6203 geometry
    BPFI = 296.9
    BSF  = 139.9
    FTF  =  14.9
    FREQS = {'BPFO': BPFO, 'BPFI': BPFI, 'BSF': BSF, 'FTF': FTF}


# ─────────────────────────────────────────────────────────────
# IMS DATASET METADATA
# ─────────────────────────────────────────────────────────────
IMS_DATASETS = {
    1: {
        'path':     '1st_test',
        'n_ch':     8,
        'interval': 10,   # minutes
        'bearing_map': {
            'B1x':0,'B1y':1,'B2x':2,'B2y':3,
            'B3x':4,'B3y':5,'B4x':6,'B4y':7,
        },
        # Ground-truth failure bearings and fault types
        'failures': {
            'B3': 'inner_race',
            'B4': 'roller_element',
        },
    },
    2: {
        'path':     '2nd_test',
        'n_ch':     4,
        'interval': 10,
        'bearing_map': {'B1':0,'B2':1,'B3':2,'B4':3},
        'failures': {'B1': 'outer_race'},
    },
    3: {
        'path':     '3rd_test',
        'n_ch':     4,
        'interval': 10,
        'bearing_map': {'B1':0,'B2':1,'B3':2,'B4':3},
        'failures': {'B3': 'outer_race'},
    },
}

# ─────────────────────────────────────────────────────────────
# EWMA CONTROL CHART PARAMETERS
# ─────────────────────────────────────────────────────────────
class EWMA:
    LAMBDA      = 0.2     # Smoothing factor (0<λ≤1); smaller = more smoothing
    L           = 3.0     # Control limit multiplier (L=3 → ~0.27% FAR under normality)
    TARGET_FAR  = 0.01    # 1% False Alarm Rate target for UCL tuning
    WARMUP      = 30      # Files before EWMA is considered stable


# ─────────────────────────────────────────────────────────────
# K-CONSECUTIVE ALARM LOGIC
# ─────────────────────────────────────────────────────────────
class Alarm:
    K_DEFAULT   = 5       # Consecutive exceedances before alarm triggers
    K_RANGE     = range(1, 16)   # Range for K sensitivity analysis


# ─────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────
class FeatureSelection:
    METHOD      = 'mRMR'  # Maximum Relevance Minimum Redundancy
    N_FEATURES  = 20      # Final feature count after selection
    N_FEATURES_S1 = 15    # Features for Stage-1 SVM
    N_FEATURES_S2 = 20    # Features for Stage-2 SVM


# ─────────────────────────────────────────────────────────────
# SVM HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
class SVMConfig:
    # Stage-1 (binary, calibrated)
    S1_C        = 10.0
    S1_GAMMA    = 'scale'
    S1_KERNEL   = 'rbf'
    CV_FOLDS    = 5

    # Stage-2 (multi-class fault-type)
    S2_C        = 50.0
    S2_GAMMA    = 'scale'
    S2_KERNEL   = 'rbf'
    S2_DECISION = 'ovo'   # One-vs-One for multi-class


# ─────────────────────────────────────────────────────────────
# FAULT TYPE LABELS
# ─────────────────────────────────────────────────────────────
FAULT_TYPES = {
    'healthy':        0,
    'outer_race':     1,
    'inner_race':     2,
    'roller_element': 3,
}
FAULT_NAMES = {v: k for k, v in FAULT_TYPES.items()}

STAGE1_LABELS = {0: 'Healthy', 1: 'Fault'}
STAGE2_LABELS = {
    0: 'Healthy',
    1: 'Outer Race Fault',
    2: 'Inner Race Fault',
    3: 'Roller Element Fault',
}

RANDOM_STATE = 42
