#!/usr/bin/env python3
"""
Simplified Foundation Model Training
Quick baseline training on available data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import glob
import numpy as np
import mne
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("🚀 Starting Simplified Training")
print("="*60)
print()

# Load data
print("📂 Loading EEG data...")
subjects = glob.glob("data/raw/hbn/sub-*/")
print(f"   Found {len(subjects)} subjects")

X_all = []
y_all = []
subject_ids = []

for subj_path in subjects:
    subj_name = Path(subj_path).name
    print(f"   Loading {subj_name}...")
    
    eeg_files = list(Path(subj_path).glob("**/*.set"))
    
    for eeg_file in eeg_files[:3]:  # Limit files per subject
        try:
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
            
            # Simple feature extraction: mean and std per channel
            data = raw.get_data()
            features = np.concatenate([
                data.mean(axis=1),  # Mean per channel
                data.std(axis=1),   # Std per channel
            ])
            
            X_all.append(features)
            # Dummy label for now (1 if "Rest" in filename, 0 otherwise)
            y_all.append(1 if "Rest" in eeg_file.name else 0)
            subject_ids.append(subj_name)
            
        except Exception as e:
            print(f"      ⚠️  Skipping {eeg_file.name}: {e}")

X = np.array(X_all)
y = np.array(y_all)

print()
print(f"✅ Loaded {len(X)} samples")
print(f"   Feature shape: {X.shape}")
print(f"   Label distribution: {np.bincount(y)}")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🤖 Training Random Forest baseline...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
print()
print("📊 Evaluation Results:")
print()
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(classification_report(y_test, y_pred, target_names=["Task", "Rest"]))

try:
    auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"AUROC: {auroc:.3f}")
except:
    print("AUROC: N/A (single class)")

# Save model
output_dir = Path("outputs/simple_baseline")
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / "rf_model.pkl"
joblib.dump(model, model_path)

print()
print(f"✅ Model saved to: {model_path}")
print()
print("="*60)
print("🎉 Training Complete!")
print()
print(f"Summary:")
print(f"  - Samples: {len(X)}")
print(f"  - Features: {X.shape[1]}")
print(f"  - Train accuracy: {model.score(X_train, y_train):.3f}")
print(f"  - Test accuracy: {model.score(X_test, y_test):.3f}")
