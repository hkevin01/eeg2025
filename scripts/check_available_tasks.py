#!/usr/bin/env python3
"""
Check Available HBN Task Data
==============================
Scan the HBN dataset to see which tasks are available for which subjects.
"""
from pathlib import Path
import pandas as pd

print("="*80)
print("📊 HBN TASK AVAILABILITY CHECK")
print("="*80)

data_dir = Path("data/raw/hbn")

# Load participants metadata
participants_file = data_dir / "participants.tsv"
if not participants_file.exists():
    print("❌ participants.tsv not found!")
    exit(1)

participants_df = pd.read_csv(participants_file, sep='\t')
print(f"\n📋 Total participants in metadata: {len(participants_df)}")

# Task names to check
task_names = [
    "RestingState",
    "SuS",  # Sustained attention
    "CCD",  # Contrast Change Detection  
    "MW",   # Movie Watching
    "SL",   # Sequence Learning
    "SyS",  # Symbol Search
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "ThePresent"
]

# Check each subject directory
subjects = []
task_availability = {}

for subject_dir in sorted(data_dir.glob("sub-*")):
    if not subject_dir.is_dir():
        continue
    
    subject_id = subject_dir.name
    eeg_dir = subject_dir / "eeg"
    
    if not eeg_dir.exists():
        continue
    
    subjects.append(subject_id)
    task_availability[subject_id] = {}
    
    # Check each task
    for task in task_names:
        eeg_files = list(eeg_dir.glob(f"*{task}*.set"))
        task_availability[subject_id][task] = len(eeg_files) > 0

print(f"\n📁 Subjects with EEG data: {len(subjects)}")

# Print task availability matrix
print("\n" + "="*80)
print("TASK AVAILABILITY MATRIX")
print("="*80)

# Header
print(f"{'Subject':<20}", end="")
for task in task_names:
    print(f"{task:<15}", end="")
print()

print("-"*80)

# Data rows
for subject_id in sorted(subjects):
    print(f"{subject_id:<20}", end="")
    for task in task_names:
        available = task_availability[subject_id].get(task, False)
        symbol = "✅" if available else "❌"
        print(f"{symbol:<15}", end="")
    print()

# Summary statistics
print("\n" + "="*80)
print("TASK SUMMARY")
print("="*80)

for task in task_names:
    count = sum(1 for subj in subjects if task_availability[subj].get(task, False))
    percentage = (count / len(subjects) * 100) if subjects else 0
    print(f"{task:<20} {count:>3}/{len(subjects):<3} subjects ({percentage:>5.1f}%)")

# Challenge-specific check
print("\n" + "="*80)
print("CHALLENGE READINESS")
print("="*80)

# Challenge 1: SuS → CCD transfer
sus_subjects = [s for s in subjects if task_availability[s].get("SuS", False)]
ccd_subjects = [s for s in subjects if task_availability[s].get("CCD", False)]
both_sus_ccd = [s for s in subjects if task_availability[s].get("SuS", False) and task_availability[s].get("CCD", False)]

print(f"\n🎯 Challenge 1 (SuS → CCD Transfer):")
print(f"   Subjects with SuS: {len(sus_subjects)}")
print(f"   Subjects with CCD: {len(ccd_subjects)}")
print(f"   Subjects with both: {len(both_sus_ccd)}")
if both_sus_ccd:
    print(f"   ✅ Ready to train Challenge 1")
else:
    print(f"   ❌ Need subjects with both SuS and CCD tasks")

# Challenge 2: Multi-task for psychopathology
rs_subjects = [s for s in subjects if task_availability[s].get("RestingState", False)]
print(f"\n🎯 Challenge 2 (Psychopathology Prediction):")
print(f"   Subjects with RestingState: {len(rs_subjects)}")
if rs_subjects:
    print(f"   ✅ Ready to train Challenge 2 (using RestingState)")
else:
    print(f"   ❌ Need subjects with task data")

# Check for clinical scores
print("\n" + "="*80)
print("CLINICAL SCORES CHECK")
print("="*80)

# Check participants.tsv for clinical columns
clinical_columns = ['ehq_total', 'p_factor', 'attention', 'internalizing', 'externalizing']
available_clinical = [col for col in clinical_columns if col in participants_df.columns]

print(f"\nClinical columns in participants.tsv:")
for col in clinical_columns:
    if col in participants_df.columns:
        non_null = participants_df[col].notna().sum()
        print(f"   ✅ {col:<20} {non_null}/{len(participants_df)} values")
    else:
        print(f"   ❌ {col:<20} Not found")

print("\n" + "="*80)
print("✅ Task availability check complete!")
print("="*80)
