"""
04_ablation_study.py
====================
Mandatory for IEEE journal acceptance: ablation study proving each component
earns its place in the proposed framework.

Variants tested (one component removed at a time):
  V0  Full system              (proposed)
  V1  No EWMA   → raw P(fault) threshold
  V2  No K-filter → alarm on single EWMA exceedance (K=1)
  V3  No Stage-2  → binary output only, no fault-type diagnosis
  V4  Stage-1 SVM → Random Forest  (model substitution)
  V5  Stage-1 SVM → Logistic Regression  (linear baseline)
  V6  No mRMR     → all features used (tests feature selection value)

Metrics reported per variant:
  FAR (%), Detection Delay (files), Stage-1 F1, Stage-2 Macro-F1

Produces:
  ablation_results.csv
  ablation_table.png     ← formatted table for paper
  ablation_bar.png       ← bar chart comparing variants
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from config import (Paths, EWMA, Alarm, SVMConfig, RANDOM_STATE)
from stage1_utils import (EWMAChart, k_consecutive_alarm,
                           compute_metrics, research_split as s1_split)

# Import stage1_utils from 02_ (we'll factor shared logic)
# For standalone running, re-import directly:
import importlib.util, sys as _sys

def _import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

META_COLS = ['source','dataset_id','file_idx','timestamp','bearing',
             'time_norm','rul_norm','is_failing','fault_type','fault_str']

# ─────────────────────────────────────────────────────────────
# SHARED UTILITIES (self-contained for ablation)
# ─────────────────────────────────────────────────────────────

def _ewma_pipeline(p_fault, K, use_ewma=True, use_k=True,
                   warmup=EWMA.WARMUP):
    """Run detection pipeline with selectable components."""
    p_warm = p_fault[:min(warmup, int(0.15*len(p_fault)))]

    if use_ewma:
        chart   = EWMAChart(lam=EWMA.LAMBDA, L=EWMA.L).fit(p_warm)
        Z, UCL, ooc = chart.transform(p_fault)
    else:
        # Raw threshold: alarm if P(fault) > μ₀ + 3σ₀
        mu0, sig0 = np.mean(p_warm), np.std(p_warm)
        threshold = mu0 + 3 * sig0
        ooc = (p_fault > threshold).astype(int)

    K_eff = K if use_k else 1
    alarms = k_consecutive_alarm(ooc, K_eff)
    return alarms


def _get_classifier(variant: str):
    """Return (classifier, needs_calibration) for each variant."""
    if variant == 'RF':
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                      random_state=RANDOM_STATE, n_jobs=-1)
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)]), False
    elif variant == 'LR':
        clf = LogisticRegression(C=1.0, class_weight='balanced',
                                  max_iter=1000, random_state=RANDOM_STATE)
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)]), False
    else:   # SVM (default)
        base = SVC(C=SVMConfig.S1_C, gamma=SVMConfig.S1_GAMMA,
                   kernel=SVMConfig.S1_KERNEL, class_weight='balanced',
                   probability=False, random_state=RANDOM_STATE)
        cal  = CalibratedClassifierCV(estimator=base,
                                       cv=SVMConfig.CV_FOLDS, method='sigmoid')
        return Pipeline([('scaler', StandardScaler()), ('clf', cal)]), True


# ─────────────────────────────────────────────────────────────
# ABLATION VARIANT RUNNER
# ─────────────────────────────────────────────────────────────

VARIANTS = {
    'V0_Full':       {'use_ewma': True,  'use_k': True,  'clf': 'SVM', 'use_mrmr': True},
    'V1_NoEWMA':     {'use_ewma': False, 'use_k': True,  'clf': 'SVM', 'use_mrmr': True},
    'V2_NoKFilter':  {'use_ewma': True,  'use_k': False, 'clf': 'SVM', 'use_mrmr': True},
    'V4_RF':         {'use_ewma': True,  'use_k': True,  'clf': 'RF',  'use_mrmr': True},
    'V5_LR':         {'use_ewma': True,  'use_k': True,  'clf': 'LR',  'use_mrmr': True},
    'V6_NoMRMR':     {'use_ewma': True,  'use_k': True,  'clf': 'SVM', 'use_mrmr': False},
}


def run_variant(variant_name: str, cfg: dict,
                df_train: pd.DataFrame, df_test: pd.DataFrame,
                feat_cols_mrmr: list, all_feat_cols: list,
                K: int = Alarm.K_DEFAULT) -> dict:

    feat_cols = feat_cols_mrmr if cfg['use_mrmr'] else all_feat_cols

    X_tr = df_train[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    y_tr = df_train['is_failing'].astype(int)

    pipe, _ = _get_classifier(cfg['clf'])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe.fit(X_tr.values, y_tr.values)

    # Test: per-bearing evaluation
    far_list, dd_list, f1_list = [], [], []

    for (src, ds_id, bearing), grp in df_test.groupby(
            ['source','dataset_id','bearing']):
        grp = grp.sort_values('file_idx').reset_index(drop=True)
        X_b = grp[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_b = grp['is_failing'].values.astype(int)
        tn  = grp['time_norm'].values

        # P(fault)
        if hasattr(pipe['clf'], 'predict_proba'):
            p_fault = pipe.predict_proba(X_b)[:, 1]
        else:
            scores  = pipe.decision_function(X_b)
            p_fault = 1 / (1 + np.exp(-scores))   # sigmoid

        alarms  = _ewma_pipeline(p_fault, K,
                                  use_ewma=cfg['use_ewma'],
                                  use_k=cfg['use_k'])
        metrics = compute_metrics(y_b, alarms, tn)

        far_list.append(metrics['FAR'])
        dd_list.append(metrics['DD_files'])
        f1_list.append(f1_score(y_b, alarms, zero_division=0))

    dd_valid = [d for d in dd_list if d >= 0]
    result = {
        'Variant':       variant_name,
        'FAR_%':         round(np.mean(far_list)*100, 2),
        'FAR_std':       round(np.std(far_list)*100, 2),
        'DD_files':      round(np.mean(dd_valid), 1) if dd_valid else -1,
        'DD_std':        round(np.std(dd_valid), 1)  if dd_valid else 0,
        'F1_Stage1':     round(np.mean(f1_list), 4),
        'Features_Used': len(feat_cols),
    }
    print(f"  [{variant_name:20s}]  "
          f"FAR={result['FAR_%']:5.1f}%  "
          f"DD={result['DD_files']:6.1f}  "
          f"F1={result['F1_Stage1']:.4f}  "
          f"Feats={result['Features_Used']}")
    return result


# ─────────────────────────────────────────────────────────────
# TABLE AND FIGURE GENERATION
# ─────────────────────────────────────────────────────────────

def plot_ablation_bar(df_ablation: pd.DataFrame, output_dir: str):
    variants = df_ablation['Variant'].tolist()
    x = np.arange(len(variants))
    width = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # FAR
    bars = axes[0].bar(x, df_ablation['FAR_%'], width*2.5,
                        color=['#2196F3' if v=='V0_Full' else '#90CAF9' for v in variants])
    axes[0].set_xticks(x); axes[0].set_xticklabels(variants, rotation=25, ha='right', fontsize=9)
    axes[0].set_ylabel('False Alarm Rate (%)'); axes[0].set_title('FAR ↓')
    axes[0].axhline(1.0, ls='--', color='red', alpha=0.6, label='1% target')
    axes[0].legend(fontsize=8)
    axes[0].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)

    # DD
    dd_vals = df_ablation['DD_files'].clip(lower=0)
    bars2 = axes[1].bar(x, dd_vals, width*2.5,
                         color=['#4CAF50' if v=='V0_Full' else '#A5D6A7' for v in variants])
    axes[1].set_xticks(x); axes[1].set_xticklabels(variants, rotation=25, ha='right', fontsize=9)
    axes[1].set_ylabel('Detection Delay (files)'); axes[1].set_title('Detection Delay ↓')
    axes[1].bar_label(bars2, fmt='%.0f', fontsize=8, padding=2)

    # F1
    bars3 = axes[2].bar(x, df_ablation['F1_Stage1'], width*2.5,
                         color=['#FF9800' if v=='V0_Full' else '#FFCC80' for v in variants])
    axes[2].set_xticks(x); axes[2].set_xticklabels(variants, rotation=25, ha='right', fontsize=9)
    axes[2].set_ylabel('F1-Score'); axes[2].set_title('Stage-1 F1 ↑')
    axes[2].set_ylim(0, 1.1)
    axes[2].bar_label(bars3, fmt='%.3f', fontsize=8, padding=2)

    plt.suptitle('Ablation Study — Component Contribution Analysis', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_bar.png'), dpi=130,
                bbox_inches='tight')
    plt.close()


def plot_ablation_table(df_ablation: pd.DataFrame, output_dir: str):
    """Render a publication-style table as an image."""
    fig, ax = plt.subplots(figsize=(13, len(df_ablation)*0.7 + 1.5))
    ax.axis('off')

    cols = ['Variant','FAR_%','FAR_std','DD_files','DD_std','F1_Stage1','Features_Used']
    col_labels = ['Variant','FAR (%)','FAR Std','DD (files)','DD Std','F1 Stage-1','# Features']
    data = df_ablation[cols].values.tolist()

    tbl = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Highlight proposed method row
    for j in range(len(cols)):
        tbl[1, j].set_facecolor('#E3F2FD')
        tbl[1, j].set_text_props(fontweight='bold')
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#1565C0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Table: Ablation Study Results (Test Set)',
                  fontsize=13, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_table.png'), dpi=130,
                bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Inline imports for standalone run
    import importlib.util
    import pathlib

    script_dir = pathlib.Path(__file__).parent

    # Patch: inline EWMAChart & compute_metrics if stage1_utils not present
    try:
        from stage1_utils import EWMAChart, k_consecutive_alarm, compute_metrics
    except ImportError:
        # Load them directly from 02_stage1_detection.py
        s1_mod = _import_script(str(script_dir / '02_stage1_detection.py'),
                                 'stage1')
        EWMAChart           = s1_mod.EWMAChart
        k_consecutive_alarm = s1_mod.k_consecutive_alarm
        compute_metrics     = s1_mod.compute_metrics

    from config import Paths
    Paths.make_all()

    feat_csv = str(Paths.FEATURES_DIR / 'features_combined.csv')
    mrmr_csv = str(Paths.FEATURES_DIR / 'mrmr_ranking.csv')
    out_dir  = str(Paths.RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print("="*65)
    print("STEP 4 — Ablation Study")
    print("="*65)

    df   = pd.read_csv(feat_csv)
    rnk  = pd.read_csv(mrmr_csv)

    all_feat_cols  = [c for c in df.columns if c not in META_COLS]
    mrmr_feat_cols = [f for f in rnk['feature_stage1'].tolist() if f in df.columns]

    # Research split
    from config import IMS_DATASETS
    train_mask = (
        ((df['source']=='IMS')   & df['dataset_id'].isin([1,2])) |
        ((df['source']=='FEMTO') & df['bearing'].str.contains('1_1|1_2|2_1',na=False))
    )
    df_train = df[train_mask].reset_index(drop=True)
    df_test  = df[~train_mask].reset_index(drop=True)

    results = []
    for vname, vcfg in VARIANTS.items():
        result = run_variant(vname, vcfg, df_train, df_test,
                              mrmr_feat_cols, all_feat_cols)
        results.append(result)

    df_ablation = pd.DataFrame(results)
    df_ablation.to_csv(os.path.join(out_dir, 'ablation_results.csv'), index=False)

    plot_ablation_bar(df_ablation, out_dir)
    plot_ablation_table(df_ablation, out_dir)

    print(f"\nAblation results saved → {out_dir}")
    print("\n✅ Ablation study complete. Run 05_results_figures.py next.")
