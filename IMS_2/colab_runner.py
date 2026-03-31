# ============================================================
# IMS Bearing — IEEE Pipeline: Google Colab Runner
# Run each cell in order. Runtime: ~40–90 min total.
# ============================================================

# ── CELL 1: Install dependencies
# !pip install scikit-learn xgboost lightgbm PyWavelets scipy \
#             numpy pandas matplotlib seaborn joblib -q

# ── CELL 2: Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# ── CELL 3: Upload pipeline scripts to Colab working directory
# Upload: config.py, 01_feature_extraction.py, 02_stage1_detection.py,
#         03_stage2_diagnosis.py, 04_ablation_study.py, 05_results_figures.py

# ── CELL 4: Verify dataset paths
import os
from pathlib import Path

IMS_ROOT   = '/content/drive/MyDrive/IMS2/extracted'
FEMTO_ROOT = '/content/drive/MyDrive/FEMTO'          # optional
OUTPUT_DIR = '/content/drive/MyDrive/bearing_results'

for name, path in [('IMS Root', IMS_ROOT), ('Output', OUTPUT_DIR)]:
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"{name}: {path} — {'EXISTS ✓' if os.path.exists(path) else 'NOT FOUND ✗'}")

# Check dataset folders
for ds_name in ['1st_test', '2nd_test', '3rd_test']:
    p = Path(IMS_ROOT) / ds_name
    n = len(list(p.iterdir())) if p.exists() else 0
    print(f"  {ds_name}: {n} files")

# ── CELL 5: Patch config.py with your paths
# Edit config.py → Paths class:
#   IMS_ROOT  = Path('/content/drive/MyDrive/IMS2/extracted')
#   FEMTO_ROOT = Path('/content/drive/MyDrive/FEMTO')
#   OUTPUT    = Path('/content/drive/MyDrive/bearing_results')

# ── CELL 6: Step 1 — Feature Extraction (~30–60 min)
print("\n" + "="*60)
print("STEP 1: Feature Extraction")
print("="*60)

import subprocess
result = subprocess.run(
    ['python', '01_feature_extraction.py', IMS_ROOT, FEMTO_ROOT],
    capture_output=True, text=True
)
print(result.stdout[-3000:])   # last 3000 chars
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# ── CELL 7: Step 2 — Stage-1 Detection (~5–10 min)
print("\n" + "="*60)
print("STEP 2: Stage-1 Calibrated SVM + EWMA + K-Alarm")
print("="*60)

FEAT_CSV = f'{OUTPUT_DIR}/01_features/features_combined.csv'
MRMR_CSV = f'{OUTPUT_DIR}/01_features/mrmr_ranking.csv'

result = subprocess.run(
    ['python', '02_stage1_detection.py', FEAT_CSV, MRMR_CSV],
    capture_output=True, text=True
)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# ── CELL 8: Step 3 — Stage-2 Fault Diagnosis (~3–5 min)
print("\n" + "="*60)
print("STEP 3: Stage-2 Fault-Type SVM + Degradation Index")
print("="*60)

S1_CSV = f'{OUTPUT_DIR}/02_detection/stage1_results.csv'

result = subprocess.run(
    ['python', '03_stage2_diagnosis.py', FEAT_CSV, MRMR_CSV, S1_CSV],
    capture_output=True, text=True
)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# ── CELL 9: Step 4 — Ablation Study (~15–25 min)
print("\n" + "="*60)
print("STEP 4: Ablation Study")
print("="*60)

result = subprocess.run(
    ['python', '04_ablation_study.py'],
    capture_output=True, text=True
)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# ── CELL 10: Step 5 — Generate Figures
print("\n" + "="*60)
print("STEP 5: Publication Figures")
print("="*60)

result = subprocess.run(
    ['python', '05_results_figures.py'],
    capture_output=True, text=True
)
print(result.stdout[-2000:])

# ── CELL 11: Show key figures inline
from IPython.display import Image, display
import glob

fig_dir = f'{OUTPUT_DIR}/05_figures'
for fig_path in sorted(glob.glob(f'{fig_dir}/*.png')):
    print(f"\n{os.path.basename(fig_path)}")
    display(Image(fig_path, width=900))

# ── CELL 12: Print final metrics summary
import pandas as pd

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

try:
    s1 = pd.read_csv(f'{OUTPUT_DIR}/02_detection/stage1_metrics.csv')
    print("\nStage-1 (per bearing):")
    print(s1[['source','dataset_id','bearing','FAR','DD_files','F1']].to_string(index=False))
    print(f"\n  Mean FAR : {s1['FAR'].mean()*100:.2f}%")
    print(f"  Mean DD  : {s1[s1['DD_files']>=0]['DD_files'].mean():.1f} files")
    print(f"  Mean F1  : {s1['F1'].mean():.4f}")
except Exception as e:
    print(f"[Stage-1 metrics not found: {e}]")

try:
    s2 = pd.read_csv(f'{OUTPUT_DIR}/03_diagnosis/stage2_metrics.csv')
    print(f"\nStage-2 Macro-F1: {s2['macro_f1'].mean():.4f}")
except Exception as e:
    print(f"[Stage-2 metrics not found: {e}]")

try:
    abl = pd.read_csv(f'{OUTPUT_DIR}/04_results/ablation_results.csv')
    print("\nAblation Study:")
    print(abl[['Variant','FAR_%','DD_files','F1_Stage1']].to_string(index=False))
except Exception as e:
    print(f"[Ablation not found: {e}]")

print("\n✅ All done! Check OUTPUT_DIR for all results and figures.")
