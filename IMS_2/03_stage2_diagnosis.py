"""
03_stage2_diagnosis.py
======================
Stage-2: Multi-class fault-type SVM (One-vs-One), activated ONLY when
Stage-1 raises an alarm. Diagnoses:
  0 — Healthy          (should be rare here; Stage-1 gates this)
  1 — Outer Race Fault
  2 — Inner Race Fault
  3 — Roller Element Fault

Also computes a Degradation Index (DI) — a monotonic health score
constructed from Stage-2 confidence trajectories using isotonic regression.

Novel contributions in this script:
  "Stage-2 provides fault-TYPE specificity absent from prior two-stage works;
   the Degradation Index offers a continuous, monotone health indicator
   without requiring a separate RUL regressor."

Outputs:
  stage2_model.pkl
  stage2_results.csv       — per-alarm diagnosis + confidence
  degradation_index.csv    — DI over time per bearing
  stage2_confusion_*.png
  degradation_plot_*.png
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score,
                              roc_auc_score)
from sklearn.isotonic import IsotonicRegression

from config import (Paths, SVMConfig, FAULT_TYPES, FAULT_NAMES,
                    STAGE2_LABELS, RANDOM_STATE)

META_COLS = ['source','dataset_id','file_idx','timestamp','bearing',
             'time_norm','rul_norm','is_failing','fault_type','fault_str']

LABEL_NAMES = ['Healthy','Outer Race','Inner Race','Roller Element']


# ─────────────────────────────────────────────────────────────
# DATA PREPARATION FOR STAGE-2
# ─────────────────────────────────────────────────────────────

def load_stage2_data(features_csv: str, mrmr_csv: str,
                      stage1_results_csv: str = None):
    """
    Stage-2 trains on ALL labeled fault data (both healthy and fault types).
    At inference it is only CALLED for alarmed samples, but training uses
    the full distribution for best class boundary estimation.
    """
    df  = pd.read_csv(features_csv)
    rnk = pd.read_csv(mrmr_csv)

    feat_cols = [f for f in rnk['feature_stage2'].tolist() if f in df.columns]

    X = df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df['fault_type'].astype(int)      # 0/1/2/3

    print(f"Stage-2 dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"Class distribution (fault type):")
    for ft, cnt in y.value_counts().sort_index().items():
        print(f"  {STAGE2_LABELS.get(ft, ft):25s}: {cnt:6,}")

    return X, y, df[META_COLS], feat_cols


def research_split(meta: pd.DataFrame):
    train_mask = (
        ((meta['source'] == 'IMS')   & meta['dataset_id'].isin([1, 2])) |
        ((meta['source'] == 'FEMTO') & meta['bearing'].str.contains('1_1|1_2|2_1'))
    )
    return train_mask.values, (~train_mask).values


# ─────────────────────────────────────────────────────────────
# STAGE-2 CALIBRATED SVM
# ─────────────────────────────────────────────────────────────

def build_stage2_pipeline():
    """
    OvO multi-class calibrated SVM.
    SVC with decision_function_shape='ovo' gives per-class confidence scores,
    then Platt scaling maps them to proper probabilities.
    """
    base_svm = SVC(
        C=SVMConfig.S2_C,
        gamma=SVMConfig.S2_GAMMA,
        kernel=SVMConfig.S2_KERNEL,
        decision_function_shape=SVMConfig.S2_DECISION,
        class_weight='balanced',
        probability=False,
        random_state=RANDOM_STATE,
    )
    calibrated = CalibratedClassifierCV(
        estimator=base_svm,
        cv=SVMConfig.CV_FOLDS,
        method='sigmoid',
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    calibrated),
    ])
    return pipeline


def train_stage2(X_train, y_train, output_dir: str):
    print("\n" + "="*65)
    print("STAGE-2: Training Fault-Type Diagnostic SVM (OvO + Platt)")
    print("="*65)

    pipe = build_stage2_pipeline()

    # Only train on classes present in training set
    classes_present = np.unique(y_train)
    print(f"  Classes in training: {[STAGE2_LABELS.get(c,c) for c in classes_present]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f"  CV Macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, os.path.join(output_dir, 'stage2_model.pkl'))
    print(f"  Model saved → {output_dir}/stage2_model.pkl")
    return pipe


# ─────────────────────────────────────────────────────────────
# DEGRADATION INDEX (DI) via Isotonic Regression
# ─────────────────────────────────────────────────────────────

def compute_degradation_index(proba_fault: np.ndarray,
                               time_norm: np.ndarray) -> np.ndarray:
    """
    Constructs a monotonically non-decreasing Degradation Index (DI) from
    the raw P(any fault class) trajectory.

    Isotonic regression enforces monotonicity without assuming a parametric
    model — unlike exponential or polynomial fitting used in prior works.

    DI ∈ [0, 1]:  0 = perfect health,  1 = imminent failure.

    Reference: Barlow et al., "Statistical Inference under Order Restrictions", 1972.
    """
    # P(fault) = 1 - P(healthy)
    p_fault = 1.0 - proba_fault[:, 0] if proba_fault.shape[1] > 1 else proba_fault.flatten()

    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    DI  = iso.fit_transform(time_norm, p_fault)
    DI  = np.clip(DI, 0, 1)
    return DI


# ─────────────────────────────────────────────────────────────
# EVALUATION PER BEARING
# ─────────────────────────────────────────────────────────────

def evaluate_stage2(pipe, df_test: pd.DataFrame, feat_cols: list,
                     stage1_csv: str, output_dir: str):
    """
    Apply Stage-2 only to samples where Stage-1 raised an alarm.
    Compute per-bearing confusion matrices + degradation index.
    """
    # Load Stage-1 alarm flags
    if stage1_csv and os.path.exists(stage1_csv):
        df_s1 = pd.read_csv(stage1_csv)
        df_test = df_test.copy()
        # Merge alarm flags (match on source, dataset_id, bearing, file_idx)
        merge_keys = ['source','dataset_id','bearing','file_idx']
        df_test = df_test.merge(
            df_s1[merge_keys + ['alarm','p_fault','ewma_Z']],
            on=merge_keys, how='left'
        )
        df_test['alarm'] = df_test['alarm'].fillna(0).astype(int)
    else:
        df_test = df_test.copy()
        df_test['alarm'] = 1   # Evaluate all if no Stage-1 results

    all_records  = []
    all_metrics  = []
    y_true_all   = []
    y_pred_all   = []

    for (src, ds_id, bearing), grp in df_test.groupby(
            ['source','dataset_id','bearing']):
        grp   = grp.sort_values('file_idx').reset_index(drop=True)
        label = f"{src}_ds{ds_id}_{bearing}"

        X_b  = grp[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
        y_b  = grp['fault_type'].values.astype(int)
        tn   = grp['time_norm'].values

        # Get probabilities for ALL samples (for DI computation)
        proba_all = pipe.predict_proba(X_b)

        # Degradation Index (computed on full trajectory)
        DI = compute_degradation_index(proba_all, tn)

        # Predictions (all samples, but metric reported on alarm-activated only)
        y_pred_all_b = pipe.predict(X_b)
        alarmed_mask = grp['alarm'].values == 1

        if np.sum(alarmed_mask) == 0:
            print(f"  [{label}] No alarms — skipping Stage-2 metrics")
            continue

        y_true_alarm = y_b[alarmed_mask]
        y_pred_alarm = y_pred_all_b[alarmed_mask]

        f1 = f1_score(y_true_alarm, y_pred_alarm, average='macro', zero_division=0)
        print(f"  [{label:35s}]  alarmed={np.sum(alarmed_mask):4d}  "
              f"Macro-F1={f1:.4f}")

        y_true_all.extend(y_true_alarm)
        y_pred_all.extend(y_pred_alarm)

        # Confusion matrix per bearing
        cm_labels = sorted(set(y_true_alarm) | set(y_pred_alarm))
        cm_names  = [STAGE2_LABELS.get(l, str(l)) for l in cm_labels]
        cm = confusion_matrix(y_true_alarm, y_pred_alarm, labels=cm_labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=cm_names)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'Stage-2 Confusion — {label}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'stage2_cm_{label}.png'), dpi=120)
        plt.close()

        # Degradation index plot
        _plot_degradation(tn, DI, y_b, grp.get('alarm', np.zeros(len(grp))).values,
                           label, output_dir)

        # Build output records
        grp_out = grp.copy()
        grp_out['pred_fault_type'] = y_pred_all_b
        grp_out['pred_fault_str']  = [STAGE2_LABELS.get(p, str(p))
                                       for p in y_pred_all_b]
        for ci, cn in enumerate(STAGE2_LABELS.keys()):
            if ci < proba_all.shape[1]:
                grp_out[f'proba_{STAGE2_LABELS[cn].replace(" ","_").lower()}'] = \
                    proba_all[:, ci]
        grp_out['degradation_index'] = DI
        all_records.append(grp_out)

        metrics = {
            'source': src, 'dataset_id': ds_id, 'bearing': bearing,
            'n_alarmed': int(np.sum(alarmed_mask)),
            'macro_f1': f1,
        }
        report = classification_report(y_true_alarm, y_pred_alarm,
                                        target_names=cm_names,
                                        output_dict=True, zero_division=0)
        for cls_name, cls_metrics in report.items():
            if isinstance(cls_metrics, dict):
                for metric_name, val in cls_metrics.items():
                    metrics[f'{cls_name}_{metric_name}'] = val
        all_metrics.append(metrics)

    # Global confusion matrix
    if y_true_all:
        cm_global = confusion_matrix(y_true_all, y_pred_all)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(
            cm_global,
            display_labels=[STAGE2_LABELS.get(i,'?') for i in range(len(STAGE2_LABELS))]
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Stage-2 Global Confusion Matrix (Test Set)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stage2_cm_global.png'), dpi=130)
        plt.close()

    df_results = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    df_metrics = pd.DataFrame(all_metrics)

    if not df_results.empty:
        df_results.to_csv(os.path.join(output_dir, 'stage2_results.csv'), index=False)
    if not df_metrics.empty:
        df_metrics.to_csv(os.path.join(output_dir, 'stage2_metrics.csv'), index=False)
        print(f"\n  Overall Macro-F1 (alarmed samples): "
              f"{df_metrics['macro_f1'].mean():.4f}")

    return df_results, df_metrics


def _plot_degradation(time_norm, DI, y_true, alarms, label, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    x = np.arange(len(time_norm))

    # DI panel
    axes[0].plot(x, DI, color='darkorange', lw=2, label='Degradation Index (DI)')
    axes[0].fill_between(x, DI, alpha=0.15, color='darkorange')
    axes[0].axhspan(0.7, 1.0, alpha=0.08, color='red',    label='Critical zone')
    axes[0].axhspan(0.4, 0.7, alpha=0.06, color='orange', label='Warning zone')
    axes[0].axhline(0.7, ls='--', color='red',    lw=1.2)
    axes[0].axhline(0.4, ls='--', color='orange', lw=1.2)
    axes[0].set_ylabel('Degradation Index')
    axes[0].set_title(f'Bearing Degradation Trajectory  |  {label}')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)

    # Fault ground truth + alarms
    axes[1].fill_between(x, (y_true > 0).astype(int), step='mid',
                          alpha=0.2, color='green', label='True fault region')
    if np.any(alarms > 0):
        axes[1].vlines(np.where(alarms > 0)[0], 0, 1,
                        color='red', alpha=0.4, lw=0.8, label='Stage-1 alarm')
    axes[1].set_ylabel('Fault / Alarm')
    axes[1].set_xlabel('File Index  (time →)')
    axes[1].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fname = f'degradation_{label.replace(" ","_")}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=130)
    plt.close()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from config import Paths
    Paths.make_all()

    feat_csv   = sys.argv[1] if len(sys.argv)>1 else str(Paths.FEATURES_DIR/'features_combined.csv')
    mrmr_csv   = sys.argv[2] if len(sys.argv)>2 else str(Paths.FEATURES_DIR/'mrmr_ranking.csv')
    s1_csv     = sys.argv[3] if len(sys.argv)>3 else str(Paths.DETECTION_DIR/'stage1_results.csv')
    out_dir    = str(Paths.DIAGNOSIS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print("="*65)
    print("STEP 3 — Stage-2: Fault-Type SVM + Degradation Index")
    print("="*65)

    X, y, meta, feat_cols = load_stage2_data(feat_csv, mrmr_csv, s1_csv)
    train_mask, test_mask  = research_split(meta)

    X_train, y_train = X.values[train_mask], y.values[train_mask]
    df_test = pd.read_csv(feat_csv)
    df_test = df_test[test_mask].reset_index(drop=True)

    pipe = train_stage2(X_train, y_train, out_dir)

    print(f"\nEvaluating Stage-2 on test set...")
    df_results, df_metrics = evaluate_stage2(
        pipe, df_test, feat_cols, s1_csv, out_dir
    )

    print("\n✅ Stage-2 complete. Run 04_ablation_study.py next.")
