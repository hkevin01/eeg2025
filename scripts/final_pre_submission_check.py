#!/usr/bin/env python3
"""
Final Pre-Submission Verification
==================================
Complete checklist before submitting to Codabench.
"""
import os
import sys
from pathlib import Path
import zipfile
import torch
import time

print("="*80)
print("🔍 FINAL PRE-SUBMISSION VERIFICATION")
print("="*80)
print()

checklist = []
warnings = []

def check(name, condition, details=""):
    """Add item to checklist"""
    status = "✅" if condition else "❌"
    checklist.append((name, condition, status))
    print(f"{status} {name}", flush=True)
    if details:
        print(f"   {details}", flush=True)
    return condition

def warn(message):
    """Add warning"""
    warnings.append(message)
    print(f"⚠️  {message}", flush=True)

print("📦 CHECKING SUBMISSION PACKAGE")
print("-" * 80)

# 1. Check ZIP file exists
zip_path = Path("submission_complete.zip")
has_zip = check("Submission ZIP exists", zip_path.exists(), 
                f"Path: {zip_path}")

# 2. Check ZIP structure
if has_zip:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = zf.namelist()
        
        # Check no folders
        has_no_folders = all('/' not in f for f in files)
        check("No nested folders in ZIP", has_no_folders,
              f"Files: {', '.join(files)}")
        
        # Check required files
        check("submission.py present", 'submission.py' in files)
        check("weights_challenge_1.pt present", 'weights_challenge_1.pt' in files)
        check("weights_challenge_2.pt present", 'weights_challenge_2.pt' in files)
        
        # Check file count
        check("Exactly 3 files", len(files) == 3,
              f"Found {len(files)} files")

# 3. Check ZIP size
if has_zip:
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    check("ZIP size < 20 MB", size_mb < 20,
          f"Size: {size_mb:.2f} MB")

print()
print("🧪 CHECKING MODELS")
print("-" * 80)

# 4. Check weight files exist
w1_exists = check("weights_challenge_1.pt exists", 
                  Path("weights_challenge_1.pt").exists())
w2_exists = check("weights_challenge_2.pt exists",
                  Path("weights_challenge_2.pt").exists())

# 5. Check models load
if w1_exists:
    try:
        w1 = torch.load('weights_challenge_1.pt', map_location='cpu', weights_only=True)
        check("Challenge 1 weights load", True,
              f"Parameters: {sum(v.numel() for v in w1.values()):,}")
    except Exception as e:
        check("Challenge 1 weights load", False, f"Error: {e}")

if w2_exists:
    try:
        w2 = torch.load('weights_challenge_2.pt', map_location='cpu', weights_only=True)
        check("Challenge 2 weights load", True,
              f"Parameters: {sum(v.numel() for v in w2.values()):,}")
    except Exception as e:
        check("Challenge 2 weights load", False, f"Error: {e}")

# 6. Check submission.py
sub_exists = check("submission.py exists", Path("submission.py").exists())

if sub_exists:
    try:
        sys.path.insert(0, str(Path.cwd()))
        from submission import Submission
        
        # Test instantiation
        SFREQ = 100
        DEVICE = torch.device('cpu')
        sub = Submission(SFREQ, DEVICE)
        check("Submission class instantiates", True)
        
        # Test model methods
        try:
            model1 = sub.get_model_challenge_1()
            check("get_model_challenge_1() works", True)
        except Exception as e:
            check("get_model_challenge_1() works", False, f"Error: {e}")
        
        try:
            model2 = sub.get_model_challenge_2()
            check("get_model_challenge_2() works", True)
        except Exception as e:
            check("get_model_challenge_2() works", False, f"Error: {e}")
        
        # Test inference
        try:
            X = torch.randn(2, 129, 200)
            with torch.inference_mode():
                out1 = model1(X)
                out2 = model2(X)
            
            check("Both models run inference", True,
                  f"C1: {out1.shape}, C2: {out2.shape}")
            check("Output shapes correct", 
                  out1.shape == (2, 1) and out2.shape == (2, 1))
        except Exception as e:
            check("Both models run inference", False, f"Error: {e}")
            
    except Exception as e:
        check("submission.py imports", False, f"Error: {e}")

print()
print("📄 CHECKING DOCUMENTATION")
print("-" * 80)

# 7. Check methods document
methods_md = check("METHODS_DOCUMENT.md exists",
                   Path("METHODS_DOCUMENT.md").exists())
methods_html = check("METHODS_DOCUMENT.html exists",
                     Path("METHODS_DOCUMENT.html").exists())
methods_pdf = check("METHODS_DOCUMENT.pdf exists",
                    Path("METHODS_DOCUMENT.pdf").exists())

if not methods_pdf:
    warn("PDF methods document not created yet. See PDF_CONVERSION_INSTRUCTIONS.md")

print()
print("📊 CHECKING RESULTS")
print("-" * 80)

# 8. Check checkpoints
check("Challenge 1 checkpoint exists",
      Path("checkpoints/response_time_improved.pth").exists())
check("Challenge 2 checkpoint exists",
      Path("checkpoints/externalizing_model.pth").exists())

# 9. Check results files
check("Challenge 1 results exist",
      Path("results/challenge1_response_time_improved.txt").exists())
check("Challenge 2 results exist",
      Path("results/challenge2_externalizing.txt").exists())

print()
print("🎨 CHECKING VISUALIZATIONS")
print("-" * 80)

viz_dir = Path("results/visualizations")
if viz_dir.exists():
    viz_files = list(viz_dir.glob("*.png"))
    check("Visualization files created", len(viz_files) > 0,
          f"Found {len(viz_files)} files")
else:
    check("Visualization directory exists", False)

print()
print("="*80)
print("📋 SUMMARY")
print("="*80)

passed = sum(1 for _, condition, _ in checklist if condition)
total = len(checklist)

print(f"\nTests Passed: {passed}/{total}")

if warnings:
    print(f"\nWarnings: {len(warnings)}")
    for w in warnings:
        print(f"  ⚠️  {w}")

print()
if passed == total:
    print("✅ ALL CHECKS PASSED - READY TO SUBMIT!")
    print()
    print("Next steps:")
    print("1. Convert METHODS_DOCUMENT.html to PDF (see PDF_CONVERSION_INSTRUCTIONS.md)")
    print("2. Go to https://www.codabench.org/competitions/4287/")
    print("3. Upload submission_complete.zip")
    print("4. Upload METHODS_DOCUMENT.pdf")
    print("5. Monitor leaderboard results")
else:
    print("⚠️  SOME CHECKS FAILED - REVIEW ABOVE")
    print()
    print("Failed checks:")
    for name, condition, status in checklist:
        if not condition:
            print(f"  {status} {name}")

print()
print("="*80)
