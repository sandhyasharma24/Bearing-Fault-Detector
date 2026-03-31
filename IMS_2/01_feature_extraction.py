"""
01_feature_extraction.py
========================
Extracts 55+ physically-motivated features from raw vibration signals
for both IMS and FEMTO-ST datasets, then applies mRMR feature selection.

Feature groups:
  A) Time-domain statistical features      (14)
  B) Frequency-domain features             (14)
  C) Bearing defect frequency energies     (16)  ← dataset-specific geometry
  D) Envelope analysis (Hilbert)           (9)
  E) Wavelet energy sub-bands              (5)
  ─────────────────────────────────────────────
  Total per channel per snapshot:         ~58

mRMR (Maximum Relevance Minimum Redundancy):
  - Selects features that are maximally relevant to the target label
    while minimally redundant with each other.
  - Produces a ranked list used consistently in Stage-1 and Stage-2.
  - Reference: Peng et al., IEEE TPAMI, 2005.

Outputs:
  features_ims.csv
  features_femto.csv
  features_combined.csv
  mrmr_ranking.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal as sp_signal
from scipy.fft import fft, fftfreq
import pywt                        # pip install PyWavelets
warnings.filterwarnings('ignore')

from config import (Paths, Signal, BearingIMS, BearingFEMTO,
                    IMS_DATASETS, FAULT_TYPES, RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# A) TIME-DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────

def time_features(x: np.ndarray) -> dict:
    rms  = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x)) + 1e-12
    sqrt_mean_sqrt = (np.mean(np.sqrt(np.abs(x))) + 1e-12)**2

    return {
        'mean':             np.mean(x),
        'std':              np.std(x),
        'rms':              rms,
        'peak':             peak,
        'peak_to_peak':     np.ptp(x),
        'crest_factor':     peak / (rms + 1e-12),
        'skewness':         float(stats.skew(x)),
        'kurtosis':         float(stats.kurtosis(x)),          # excess kurtosis
        'shape_factor':     rms / mean_abs,
        'impulse_factor':   peak / mean_abs,
        'clearance_factor': peak / sqrt_mean_sqrt,
        'variance':         np.var(x),
        'energy':           np.sum(x**2),
        'entropy_sample':   _sample_entropy(x),
    }


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Approximate Sample Entropy — complexity measure sensitive to fault onset."""
    r = r_factor * np.std(x)
    N = len(x)
    # Subsample for speed (2048 points is sufficient)
    x = x[:2048]
    N = len(x)
    def _phi(m):
        count = 0
        for i in range(N - m):
            template = x[i:i+m]
            matches  = np.sum(np.max(np.abs(x[j:j+m] - template)
                              for j in range(N-m) if j != i) < r
                              for _ in [None])  # placeholder
            # Fast vectorised version:
            mat = np.array([x[j:j+m] for j in range(N-m) if j != i])
            if len(mat) == 0: return 1e-12
            count += np.sum(np.max(np.abs(mat - template), axis=1) < r)
        return count / ((N-m) * (N-m-1) + 1e-12)

    try:
        phi_m   = _phi(m)
        phi_m1  = _phi(m+1)
        return -np.log(phi_m1 / (phi_m + 1e-12) + 1e-12)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# B) FREQUENCY-DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────

def freq_features(x: np.ndarray, fs: int) -> dict:
    n     = len(x)
    freqs = fftfreq(n, 1/fs)[:n//2]
    mag   = np.abs(fft(x))[:n//2] * 2 / n
    f_w, psd = sp_signal.welch(x, fs=fs, nperseg=1024)

    total_power = np.sum(mag**2) + 1e-12

    return {
        'spec_mean':      np.mean(mag),
        'spec_std':       np.std(mag),
        'spec_peak':      np.max(mag),
        'spec_centroid':  float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12)),
        'spec_rolloff':   _rolloff(freqs, mag),
        'spec_flatness':  _flatness(psd),
        'spec_kurtosis':  float(stats.kurtosis(mag)),
        'spec_skewness':  float(stats.skew(mag)),
        'band_0_1k':      _band_power(f_w, psd, 0,    1000),
        'band_1_3k':      _band_power(f_w, psd, 1000, 3000),
        'band_3_6k':      _band_power(f_w, psd, 3000, 6000),
        'band_6_10k':     _band_power(f_w, psd, 6000, 10000),
        'freq_rms':       np.sqrt(np.mean(psd)),
        'freq_variance':  np.var(mag),
    }


def _rolloff(freqs, mag, threshold=0.85):
    cum = np.cumsum(mag)
    idx = np.searchsorted(cum, threshold * cum[-1])
    return float(freqs[min(idx, len(freqs)-1)])

def _flatness(psd):
    geo  = np.exp(np.mean(np.log(psd + 1e-12)))
    arith = np.mean(psd)
    return float(geo / (arith + 1e-12))

def _band_power(f, psd, f_low, f_high):
    mask = (f >= f_low) & (f <= f_high)
    return float(np.trapz(psd[mask], f[mask])) if np.any(mask) else 0.0


# ─────────────────────────────────────────────────────────────
# C) BEARING DEFECT FREQUENCY ENERGIES
# ─────────────────────────────────────────────────────────────

def defect_freq_features(x: np.ndarray, fs: int, bearing_freqs: dict,
                          bw: float = 5.0) -> dict:
    """Energy at each defect frequency + 2nd and 3rd harmonics."""
    n     = len(x)
    freqs = fftfreq(n, 1/fs)[:n//2]
    mag   = np.abs(fft(x))[:n//2] * 2 / n
    feats = {}
    for name, f0 in bearing_freqs.items():
        feats[f'def_{name}']    = _freq_energy(freqs, mag, f0,    bw)
        feats[f'def_{name}_h2'] = _freq_energy(freqs, mag, f0*2,  bw)
        feats[f'def_{name}_h3'] = _freq_energy(freqs, mag, f0*3,  bw)
        feats[f'def_{name}_h4'] = _freq_energy(freqs, mag, f0*4,  bw)
    return feats

def _freq_energy(freqs, mag, fc, bw):
    mask = (freqs >= fc-bw) & (freqs <= fc+bw)
    return float(np.sum(mag[mask]**2)) if np.any(mask) else 0.0


# ─────────────────────────────────────────────────────────────
# D) ENVELOPE ANALYSIS (Hilbert transform)
# ─────────────────────────────────────────────────────────────

def envelope_features(x: np.ndarray, fs: int, bearing_freqs: dict) -> dict:
    """
    Classic demodulation: bandpass → Hilbert → envelope spectrum.
    Bandpass region 3–9 kHz captures structural resonances excited by impacts.
    """
    nyq  = fs / 2
    lo   = min(3000, nyq * 0.3)
    hi   = min(9000, nyq * 0.9)
    b, a = sp_signal.butter(4, [lo/nyq, hi/nyq], btype='band')
    filt = sp_signal.filtfilt(b, a, x)

    env  = np.abs(sp_signal.hilbert(filt))

    n      = len(env)
    e_freq = fftfreq(n, 1/fs)[:n//2]
    e_mag  = np.abs(fft(env))[:n//2] * 2 / n

    feats = {
        'env_mean':     np.mean(env),
        'env_std':      np.std(env),
        'env_rms':      float(np.sqrt(np.mean(env**2))),
        'env_kurtosis': float(stats.kurtosis(env)),
        'env_peak':     float(np.max(env)),
        'env_crest':    float(np.max(env) / (np.sqrt(np.mean(env**2)) + 1e-12)),
    }
    for name, f0 in bearing_freqs.items():
        feats[f'env_{name}']    = _freq_energy(e_freq, e_mag, f0,   3.0)
        feats[f'env_{name}_h2'] = _freq_energy(e_freq, e_mag, f0*2, 3.0)
        feats[f'env_{name}_h3'] = _freq_energy(e_freq, e_mag, f0*3, 3.0)
    return feats


# ─────────────────────────────────────────────────────────────
# E) WAVELET ENERGY (Daubechies db4, 5 levels)
# ─────────────────────────────────────────────────────────────

def wavelet_features(x: np.ndarray, wavelet: str = 'db4', level: int = 5) -> dict:
    """
    Discrete wavelet packet energy in each sub-band.
    Captures non-stationary impulsive events better than FFT.
    """
    coeffs = pywt.wavedec(x, wavelet, level=level)
    feats  = {}
    for i, c in enumerate(coeffs):
        feats[f'wt_energy_L{i}'] = float(np.sum(c**2))
        feats[f'wt_entropy_L{i}'] = float(
            -np.sum((c**2 / (np.sum(c**2) + 1e-12)) *
                    np.log(c**2 / (np.sum(c**2) + 1e-12) + 1e-12))
        )
    return feats


# ─────────────────────────────────────────────────────────────
# COMBINED FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_all(x: np.ndarray, fs: int, bearing_freqs: dict) -> dict:
    feats = {}
    feats.update(time_features(x))
    feats.update(freq_features(x, fs))
    feats.update(defect_freq_features(x, fs, bearing_freqs))
    feats.update(envelope_features(x, fs, bearing_freqs))
    feats.update(wavelet_features(x))
    return feats


# ─────────────────────────────────────────────────────────────
# IMS DATASET LOADER
# ─────────────────────────────────────────────────────────────

def process_ims(ims_root: str, output_dir: str) -> pd.DataFrame:
    """Process all 3 IMS datasets and save features."""
    from config import IMS_DATASETS, Signal, BearingIMS, FAULT_TYPES
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []

    for ds_id, cfg in IMS_DATASETS.items():
        data_path = Path(ims_root) / cfg['path']
        if not data_path.exists():
            print(f"[SKIP] Dataset {ds_id} not found: {data_path}")
            continue

        files = sorted(f for f in data_path.iterdir() if f.is_file())
        n_files = len(files)
        print(f"\n[IMS Dataset {ds_id}] {n_files} files | failures: {cfg['failures']}")

        records = []
        for idx, fpath in enumerate(files):
            if idx % 200 == 0:
                print(f"  {idx}/{n_files} ({100*idx//n_files}%)")
            try:
                raw = np.loadtxt(str(fpath))
            except Exception as e:
                continue
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)

            try:
                ts = pd.to_datetime(fpath.name, format='%Y.%m.%d.%H.%M.%S')
            except Exception:
                ts = pd.NaT

            time_norm = idx / max(n_files - 1, 1)   # 0=start, 1=failure
            rul_norm  = 1.0 - time_norm

            for bearing_name, ch_idx in cfg['bearing_map'].items():
                if ch_idx >= raw.shape[1]:
                    continue
                sig = raw[:, ch_idx]

                feats = extract_all(sig, Signal.FS, BearingIMS.FREQS)

                # Determine fault type label
                base = bearing_name[:2]   # e.g. 'B1', 'B3'
                fault_type_str = cfg['failures'].get(base, 'healthy')
                fault_type_int = FAULT_TYPES.get(fault_type_str, 0)
                is_failing     = int(fault_type_str != 'healthy')

                record = {
                    'source':       'IMS',
                    'dataset_id':   ds_id,
                    'file_idx':     idx,
                    'timestamp':    ts,
                    'bearing':      bearing_name,
                    'time_norm':    time_norm,
                    'rul_norm':     rul_norm,
                    'is_failing':   is_failing,
                    'fault_type':   fault_type_int,    # 0/1/2/3
                    'fault_str':    fault_type_str,
                }
                record.update(feats)
                records.append(record)

        df_ds = pd.DataFrame(records)
        out_path = os.path.join(output_dir, f'ims_dataset_{ds_id}.csv')
        df_ds.to_csv(out_path, index=False)
        all_dfs.append(df_ds)
        print(f"  Saved {len(df_ds)} records → {out_path}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(output_dir, 'ims_all.csv'), index=False)
    return df_all


# ─────────────────────────────────────────────────────────────
# FEMTO-ST DATASET LOADER
# ─────────────────────────────────────────────────────────────

def process_femto(femto_root: str, output_dir: str) -> pd.DataFrame:
    """
    Process FEMTO-ST (PRONOSTIA) dataset.
    Expected structure: femto_root/Bearing1_1/acc_*.csv, ...
    Each CSV: col1=time, col2=h-acc, col3=v-acc (2560 samples @ 25.6 kHz)
    """
    from config import Signal, BearingFEMTO, FAULT_TYPES
    os.makedirs(output_dir, exist_ok=True)

    # All bearing folders in the FEMTO root
    bearing_dirs = sorted(Path(femto_root).glob('Bearing*_*'))
    if not bearing_dirs:
        print(f"[WARN] No FEMTO bearings found in {femto_root}")
        return pd.DataFrame()

    records = []
    for b_dir in bearing_dirs:
        bearing_name = b_dir.name
        # Known failure bearings from IEEE PHM 2012 challenge labels
        # (outer race failure for most; update per your label file)
        fault_type_int = FAULT_TYPES.get('outer_race', 1)

        acc_files = sorted(b_dir.glob('acc_*.csv'))
        n_files   = len(acc_files)
        print(f"  [FEMTO] {bearing_name}: {n_files} files")

        for idx, fpath in enumerate(acc_files):
            try:
                raw = pd.read_csv(str(fpath), header=None).values
                sig = raw[:, 1].astype(float)   # horizontal channel
            except Exception:
                continue

            feats     = extract_all(sig, Signal.FS_FEMTO, BearingFEMTO.FREQS)
            time_norm = idx / max(n_files - 1, 1)

            record = {
                'source':     'FEMTO',
                'dataset_id': bearing_name,
                'file_idx':   idx,
                'timestamp':  pd.NaT,
                'bearing':    bearing_name,
                'time_norm':  time_norm,
                'rul_norm':   1.0 - time_norm,
                'is_failing': 1,
                'fault_type': fault_type_int,
                'fault_str':  'outer_race',
            }
            record.update(feats)
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'femto_all.csv'), index=False)
    print(f"FEMTO: {len(df)} records saved.")
    return df


# ─────────────────────────────────────────────────────────────
# mRMR FEATURE SELECTION
# ─────────────────────────────────────────────────────────────

META_COLS = ['source','dataset_id','file_idx','timestamp','bearing',
             'time_norm','rul_norm','is_failing','fault_type','fault_str']

def mrmr_select(df: pd.DataFrame, target_col: str = 'is_failing',
                n_select: int = 20) -> list:
    """
    Maximum Relevance Minimum Redundancy feature selection.

    Score(f) = Relevance(f, target) - (1/|S|) * Σ Redundancy(f, s∈S)

    Relevance  = mutual information between feature and target.
    Redundancy = average mutual information between feature and already-selected features.

    Reference: Peng, Long & Ding, IEEE TPAMI, 2005.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import KBinsDiscretizer

    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df[target_col].values.astype(int)

    print(f"\nmRMR feature selection: {len(feat_cols)} → {n_select} features")
    print(f"  Target: '{target_col}' | Samples: {len(y)}")

    # Mutual information between each feature and target
    mi_target = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    selected_idx = []
    selected_mi  = {}     # idx → MI with target

    # Greedily select features
    remaining = list(range(len(feat_cols)))
    for step in range(n_select):
        best_idx   = None
        best_score = -np.inf

        for i in remaining:
            relevance   = mi_target[i]
            redundancy  = 0.0
            if selected_idx:
                # Mean MI between candidate and already-selected features
                mi_sel = mutual_info_classif(
                    X[:, selected_idx], X[:, i].reshape(-1,1),
                    random_state=RANDOM_STATE
                )
                redundancy = float(np.mean(mi_sel))
            score = relevance - redundancy
            if score > best_score:
                best_score = score
                best_idx   = i

        selected_idx.append(best_idx)
        remaining.remove(best_idx)
        print(f"  [{step+1:2d}] {feat_cols[best_idx]:35s}  "
              f"relevance={mi_target[best_idx]:.4f}  score={best_score:.4f}")

    selected_features = [feat_cols[i] for i in selected_idx]
    return selected_features


def run_mrmr_and_save(df_combined: pd.DataFrame, output_dir: str,
                       n_select: int = 20) -> list:
    """Run mRMR on combined dataset and save the ranked feature list."""
    # Run for Stage-1 target (binary: fault vs healthy)
    selected_s1 = mrmr_select(df_combined, target_col='is_failing',
                               n_select=n_select)
    # Run for Stage-2 target (fault type)
    selected_s2 = mrmr_select(df_combined[df_combined['is_failing']==1],
                               target_col='fault_type', n_select=n_select)

    ranking_df = pd.DataFrame({
        'rank':             range(1, n_select+1),
        'feature_stage1':   selected_s1,
        'feature_stage2':   selected_s2,
    })
    out = os.path.join(output_dir, 'mrmr_ranking.csv')
    ranking_df.to_csv(out, index=False)
    print(f"\nmRMR rankings saved → {out}")
    return selected_s1, selected_s2


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from config import Paths
    Paths.make_all()

    ims_root   = sys.argv[1] if len(sys.argv) > 1 else str(Paths.IMS_ROOT)
    femto_root = sys.argv[2] if len(sys.argv) > 2 else str(Paths.FEMTO_ROOT)
    out_dir    = str(Paths.FEATURES_DIR)

    print("=" * 65)
    print("STEP 1 — Feature Extraction (IMS + FEMTO-ST)")
    print("=" * 65)

    df_ims   = process_ims(ims_root, out_dir)
    df_femto = process_femto(femto_root, out_dir)

    # Combine (common feature set only)
    dfs = [d for d in [df_ims, df_femto] if not d.empty]
    df_all = pd.concat(dfs, ignore_index=True)

    common_feats = [c for c in df_all.columns if c not in META_COLS]
    df_all[common_feats] = df_all[common_feats].replace(
        [np.inf, -np.inf], np.nan).fillna(0)

    master_path = os.path.join(out_dir, 'features_combined.csv')
    df_all.to_csv(master_path, index=False)
    print(f"\nCombined dataset: {len(df_all):,} records × {len(df_all.columns)} columns")
    print(f"Saved → {master_path}")

    # mRMR feature selection
    s1_feats, s2_feats = run_mrmr_and_save(df_all, out_dir, n_select=20)

    print("\n✅ Feature extraction complete. Run 02_stage1_detection.py next.")
