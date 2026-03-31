"""
02_stage1_detection.py
======================
Stage-1: Calibrated RBF-SVM fault detector with:
  1. Platt-scaled probability output  P(fault) ∈ [0,1]
  2. EWMA control chart smoothing     (statistically principled, tunable λ)
  3. K-consecutive alarm logic        (controls FAR vs Detection Delay tradeoff)

This is the first novelty claim of the paper:
  "Unlike fixed-threshold or moving-average smoothing used in prior work,
   we employ an EWMA control chart with statistically guaranteed FAR control
   as the inter-stage gate."

Outputs:
  stage1_model.pkl           — trained calibrated SVM
  stage1_results.csv         — P(fault), EWMA stat, alarm flags per timestep
  stage1_metrics.csv         — FAR, Detection Delay, F1 per dataset
  ewma_alarm_plot_*.png      — detection timeline plots
  k_sensitivity.png          — FAR vs DD tradeoff curve (paper Figure)
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, precision_recall_curve,
                              average_precision_score)
from sklearn.inspection import permutation_importance

from config import (Paths, EWMA, Alarm, SVMConfig, RANDOM_STATE,
                    STAGE1_LABELS)

META_COLS = ['source','dataset_id','file_idx','timestamp','bearing',
             'time_norm','rul_norm','is_failing','fault_type','fault_str']


# ─────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────

def load_data(features_csv: str, mrmr_csv: str):
    df  = pd.read_csv(features_csv)
    rnk = pd.read_csv(mrmr_csv)

    feat_cols = rnk['feature_stage1'].tolist()
    # Keep only features present in dataframe
    feat_cols = [f for f in feat_cols if f in df.columns]

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['is_failing'].astype(int)

    print(f"Stage-1 dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"Class balance  : {y.value_counts().to_dict()}")
    return X, y, df[META_COLS], feat_cols


def research_split(meta: pd.DataFrame):
    """
    Train on IMS datasets 1 & 2 + FEMTO training bearings.
    Test  on IMS dataset 3   + FEMTO test bearings.
    Prevents any temporal or cross-experiment leakage.
    """
    train_mask = (
        ((meta['source'] == 'IMS')   & meta['dataset_id'].isin([1, 2])) |
        ((meta['source'] == 'FEMTO') & meta['bearing'].str.contains('1_1|1_2|2_1'))
    )
    test_mask = ~train_mask
    return train_mask.values, test_mask.values


# ─────────────────────────────────────────────────────────────
# CALIBRATED SVM (Stage-1)
# ─────────────────────────────────────────────────────────────

def build_stage1_pipeline():
    """
    Platt scaling (CalibratedClassifierCV with cv=5) gives well-calibrated
    P(fault) probabilities suitable for EWMA charting.
    Raw SVM decision scores are NOT probabilities — calibration is essential.
    """
    base_svm = SVC(
        C=SVMConfig.S1_C,
        gamma=SVMConfig.S1_GAMMA,
        kernel=SVMConfig.S1_KERNEL,
        class_weight='balanced',
        probability=False,      # We apply Platt scaling externally
        random_state=RANDOM_STATE,
    )
    calibrated = CalibratedClassifierCV(
        estimator=base_svm,
        cv=SVMConfig.CV_FOLDS,
        method='sigmoid',        # Platt scaling
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    calibrated),
    ])
    return pipeline


def train_stage1(X_train, y_train, output_dir: str):
    print("\n" + "="*65)
    print("STAGE-1: Training Calibrated RBF-SVM (Platt Scaling)")
    print("="*65)

    pipe = build_stage1_pipeline()

    # 5-fold stratified CV on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipe.fit(X_train, y_train)

    model_path = os.path.join(output_dir, 'stage1_model.pkl')
    joblib.dump(pipe, model_path)
    print(f"  Model saved → {model_path}")
    return pipe


# ─────────────────────────────────────────────────────────────
# EWMA CONTROL CHART
# ─────────────────────────────────────────────────────────────

class EWMAChart:
    """
    EWMA (Exponentially Weighted Moving Average) control chart.

    Z_t = λ·P_t + (1−λ)·Z_{t-1}

    Upper Control Limit (UCL):
      UCL = μ₀ + L·σ₀·√(λ/(2−λ)·(1−(1−λ)^{2t}))

    Where μ₀, σ₀ are estimated from healthy (in-control) data.

    Properties:
      - More sensitive to small, sustained shifts than Shewhart charts
      - λ controls memory: small λ = long memory, detects gradual degradation
      - L controls false alarm rate: L=3 → ~0.27% FAR under normality
    """

    def __init__(self, lam: float = EWMA.LAMBDA, L: float = EWMA.L):
        self.lam = lam
        self.L   = L
        self.mu0 = None
        self.sig0 = None

    def fit(self, p_healthy: np.ndarray):
        """Estimate in-control mean and std from healthy-state probabilities."""
        self.mu0  = np.mean(p_healthy)
        self.sig0 = np.std(p_healthy)
        return self

    def transform(self, p: np.ndarray) -> tuple:
        """
        Apply EWMA smoothing and compute UCL.
        Returns: (ewma_stats, ucl_array, out_of_control_flags)
        """
        assert self.mu0 is not None, "Call fit() first on healthy data."
        n     = len(p)
        Z     = np.zeros(n)
        UCL   = np.zeros(n)
        lam   = self.lam
        L     = self.L
        mu0   = self.mu0
        sig0  = self.sig0

        Z[0] = lam * p[0] + (1 - lam) * mu0
        for t in range(1, n):
            Z[t] = lam * p[t] + (1 - lam) * Z[t-1]

        for t in range(n):
            variance_factor = (lam / (2 - lam)) * (1 - (1 - lam)**(2*(t+1)))
            UCL[t] = mu0 + L * sig0 * np.sqrt(variance_factor)

        ooc = (Z > UCL).astype(int)
        return Z, UCL, ooc


# ─────────────────────────────────────────────────────────────
# K-CONSECUTIVE ALARM LOGIC
# ─────────────────────────────────────────────────────────────

def k_consecutive_alarm(ooc_flags: np.ndarray, K: int) -> np.ndarray:
    """
    Alarm at time t only if ooc_flags[t-K+1 : t+1] are ALL 1.
    This is the persistence filter that suppresses transient false alarms.

    Returns an array of alarm flags (0/1), same length as ooc_flags.
    """
    n      = len(ooc_flags)
    alarms = np.zeros(n, dtype=int)
    for t in range(K-1, n):
        if np.all(ooc_flags[t-K+1:t+1] == 1):
            alarms[t] = 1
    return alarms


def first_alarm_index(alarms: np.ndarray) -> int:
    """Return index of first alarm trigger (-1 if never triggered)."""
    idx = np.where(alarms == 1)[0]
    return int(idx[0]) if len(idx) > 0 else -1


# ─────────────────────────────────────────────────────────────
# EVALUATION: FAR, DETECTION DELAY, F1
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, alarms: np.ndarray,
                    time_norm: np.ndarray, fault_onset_norm: float = 0.6):
    """
    y_true      : binary ground truth (0=healthy, 1=fault)
    alarms      : binary alarm flags from K-consecutive logic
    time_norm   : normalized time index [0,1]
    fault_onset_norm: time_norm threshold after which a fault is considered present
    """
    # False Alarm Rate: alarms raised in TRULY healthy region
    healthy_mask = time_norm < fault_onset_norm
    FAR = float(np.mean(alarms[healthy_mask])) if np.any(healthy_mask) else 0.0

    # Detection Delay: first alarm minus fault onset point
    fault_onset_idx = np.where(time_norm >= fault_onset_norm)[0]
    first_alarm_idx = first_alarm_index(alarms)

    if len(fault_onset_idx) > 0 and first_alarm_idx >= 0:
        onset_idx = fault_onset_idx[0]
        DD = max(0, first_alarm_idx - onset_idx)   # in number of files
    else:
        DD = -1    # missed detection

    # Standard F1 (point-wise)
    F1 = f1_score(y_true, alarms, zero_division=0)

    return {'FAR': FAR, 'DD_files': DD, 'F1': F1}


# ─────────────────────────────────────────────────────────────
# K SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────

def k_sensitivity_analysis(p_fault: np.ndarray, ewma_chart: EWMAChart,
                             y_true: np.ndarray, time_norm: np.ndarray,
                             output_dir: str, bearing_label: str = ''):
    """
    Sweep K from 1 to 15, record FAR and DD.
    Produces the FAR–DD tradeoff curve — a key contribution figure.
    """
    Z, UCL, ooc = ewma_chart.transform(p_fault)
    k_results = []

    for K in Alarm.K_RANGE:
        alarms  = k_consecutive_alarm(ooc, K)
        metrics = compute_metrics(y_true, alarms, time_norm)
        metrics['K'] = K
        k_results.append(metrics)

    df_k = pd.DataFrame(k_results)

    # ── Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df_k['K'], df_k['FAR']*100, 'o-', color='crimson', lw=2)
    axes[0].set_xlabel('K  (consecutive exceedances required)')
    axes[0].set_ylabel('False Alarm Rate (%)')
    axes[0].set_title(f'FAR vs K  {bearing_label}')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1.0, ls='--', color='grey', alpha=0.6, label='1% FAR')
    axes[0].legend()

    # FAR vs DD tradeoff (main paper figure)
    valid = df_k[df_k['DD_files'] >= 0]
    axes[1].plot(valid['FAR']*100, valid['DD_files'], 's-',
                  color='steelblue', lw=2)
    for _, row in valid.iterrows():
        axes[1].annotate(f"K={int(row['K'])}", (row['FAR']*100, row['DD_files']),
                          textcoords='offset points', xytext=(5, 3), fontsize=7)
    axes[1].set_xlabel('False Alarm Rate (%)')
    axes[1].set_ylabel('Detection Delay (files)')
    axes[1].set_title(f'FAR–Detection Delay Tradeoff  {bearing_label}')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('K-Consecutive Alarm Sensitivity Analysis', fontsize=13)
    plt.tight_layout()
    fname = f'k_sensitivity_{bearing_label.replace(" ","_")}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=130)
    plt.close()

    df_k.to_csv(os.path.join(output_dir, f'k_sensitivity_{bearing_label}.csv'), index=False)
    return df_k


# ─────────────────────────────────────────────────────────────
# DETECTION TIMELINE PLOT
# ─────────────────────────────────────────────────────────────

def plot_detection_timeline(p_fault, Z, UCL, alarms, y_true,
                             time_norm, output_dir, label=''):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    x = np.arange(len(time_norm))

    # Panel 1: raw P(fault) from SVM
    axes[0].plot(x, p_fault, lw=1, color='royalblue', alpha=0.8)
    axes[0].set_ylabel('P(fault)  — calibrated SVM')
    axes[0].set_title(f'Stage-1 Detection Pipeline  |  {label}')
    axes[0].set_ylim(-0.05, 1.05)

    # Panel 2: EWMA statistic vs UCL
    axes[1].plot(x, Z,   lw=1.5, color='darkorange', label='EWMA Z_t')
    axes[1].plot(x, UCL, lw=1.2, color='red',        ls='--', label='UCL')
    axes[1].fill_between(x, Z, UCL,
                          where=(Z > UCL), alpha=0.25, color='red',
                          label='Exceedance')
    axes[1].set_ylabel('EWMA Control Chart')
    axes[1].legend(loc='upper left', fontsize=8)

    # Panel 3: alarm flags vs ground truth
    axes[2].fill_between(x, y_true, step='mid', alpha=0.2,
                          color='green', label='Ground truth fault')
    axes[2].step(x, alarms, where='mid', color='red', lw=1.5, label='Alarm')
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Normal', 'Alarm'])
    axes[2].set_xlabel('File Index  (time →)')
    axes[2].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fname = f'detection_timeline_{label.replace(" ","_")}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=130)
    plt.close()
    print(f"  Timeline plot → {fname}")


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────

def evaluate_on_test(pipe, df_test: pd.DataFrame, feat_cols: list,
                      output_dir: str, K: int = Alarm.K_DEFAULT):
    """
    Run full Stage-1 pipeline on each test bearing independently.
    EWMA is fitted on the first EWMA.WARMUP healthy files of each bearing.
    """
    records = []
    all_metrics = []

    for (src, ds_id, bearing), grp in df_test.groupby(
            ['source','dataset_id','bearing']):
        grp = grp.sort_values('file_idx').reset_index(drop=True)

        X_b = grp[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
        y_b = grp['is_failing'].values.astype(int)
        tn  = grp['time_norm'].values

        # Calibrated P(fault)
        p_fault = pipe.predict_proba(X_b)[:, 1]

        # Fit EWMA on healthy warm-up period
        warmup  = min(EWMA.WARMUP, int(0.15 * len(p_fault)))
        p_warm  = p_fault[:warmup]
        chart   = EWMAChart(lam=EWMA.LAMBDA, L=EWMA.L).fit(p_warm)

        Z, UCL, ooc = chart.transform(p_fault)
        alarms      = k_consecutive_alarm(ooc, K)
        metrics     = compute_metrics(y_b, alarms, tn)

        label = f"{src}_ds{ds_id}_{bearing}"
        print(f"  [{label:30s}]  "
              f"FAR={metrics['FAR']*100:5.1f}%  "
              f"DD={metrics['DD_files']:4d}  "
              f"F1={metrics['F1']:.4f}")

        plot_detection_timeline(p_fault, Z, UCL, alarms, y_b, tn,
                                 output_dir, label)
        df_k = k_sensitivity_analysis(p_fault, chart, y_b, tn,
                                       output_dir, label)

        # Append row-level results
        grp_out = grp.copy()
        grp_out['p_fault'] = p_fault
        grp_out['ewma_Z']  = Z
        grp_out['ewma_UCL']= UCL
        grp_out['ooc']     = ooc
        grp_out['alarm']   = alarms
        records.append(grp_out)

        metrics.update({'source':src,'dataset_id':ds_id,'bearing':bearing})
        all_metrics.append(metrics)

    df_results  = pd.concat(records, ignore_index=True)
    df_metrics  = pd.DataFrame(all_metrics)

    df_results.to_csv(os.path.join(output_dir, 'stage1_results.csv'), index=False)
    df_metrics.to_csv(os.path.join(output_dir, 'stage1_metrics.csv'), index=False)

    print(f"\n  Mean FAR : {df_metrics['FAR'].mean()*100:.2f}%")
    print(f"  Mean DD  : {df_metrics[df_metrics['DD_files']>=0]['DD_files'].mean():.1f} files")
    print(f"  Mean F1  : {df_metrics['F1'].mean():.4f}")
    return df_results, df_metrics


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from config import Paths
    Paths.make_all()

    feat_csv  = sys.argv[1] if len(sys.argv)>1 else str(Paths.FEATURES_DIR/'features_combined.csv')
    mrmr_csv  = sys.argv[2] if len(sys.argv)>2 else str(Paths.FEATURES_DIR/'mrmr_ranking.csv')
    out_dir   = str(Paths.DETECTION_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print("="*65)
    print("STEP 2 — Stage-1: Calibrated SVM + EWMA + K-Alarm")
    print("="*65)

    X, y, meta, feat_cols = load_data(feat_csv, mrmr_csv)
    train_mask, test_mask  = research_split(meta)

    X_train, y_train = X.values[train_mask], y.values[train_mask]
    meta_test        = meta[test_mask].reset_index(drop=True)
    df_test_full     = pd.read_csv(feat_csv)
    df_test          = df_test_full[test_mask].reset_index(drop=True)

    # Train
    pipe = train_stage1(X_train, y_train, out_dir)

    # Evaluate
    print(f"\nEvaluating on test set (K={Alarm.K_DEFAULT})...")
    df_results, df_metrics = evaluate_on_test(pipe, df_test, feat_cols,
                                               out_dir, K=Alarm.K_DEFAULT)

    print("\n✅ Stage-1 complete. Run 03_stage2_diagnosis.py next.")
