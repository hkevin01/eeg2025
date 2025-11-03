#!/usr/bin/env python3
"""
Pre-Submission Validation Script
Catches common issues before uploading to competition
"""

import sys
import re
from pathlib import Path


def check_weights_only(submission_path):
    """Check for the dreaded weights_only parameter"""
    print("\n🔍 Checking for weights_only parameter...")
    
    with open(submission_path, 'r') as f:
        content = f.read()
    
    # Search for weights_only
    matches = re.finditer(r'weights_only\s*=', content)
    found_issues = False
    
    for match in matches:
        found_issues = True
        # Find line number
        line_num = content[:match.start()].count('\n') + 1
        print(f"   ❌ FOUND at line {line_num}: weights_only parameter")
        
        # Show context
        lines = content.split('\n')
        start = max(0, line_num - 2)
        end = min(len(lines), line_num + 2)
        print(f"   Context:")
        for i in range(start, end):
            marker = ">>>" if i == line_num - 1 else "   "
            print(f"      {marker} {i+1}: {lines[i]}")
    
    if not found_issues:
        print("   ✅ No weights_only parameters found")
        return True
    else:
        print("\n   ⚠️  CRITICAL: Remove weights_only parameter before uploading!")
        print("   Competition environment uses PyTorch < 1.13")
        return False


def check_torch_load(submission_path):
    """Check torch.load calls are correct"""
    print("\n🔍 Checking torch.load() calls...")
    
    with open(submission_path, 'r') as f:
        content = f.read()
    
    # Find all torch.load calls
    pattern = r'torch\.load\([^)]+\)'
    matches = list(re.finditer(pattern, content))
    
    if not matches:
        print("   ⚠️  No torch.load() calls found")
        return True
    
    print(f"   Found {len(matches)} torch.load() call(s)")
    all_good = True
    
    for i, match in enumerate(matches, 1):
        call = match.group()
        line_num = content[:match.start()].count('\n') + 1
        
        # Check for weights_only
        if 'weights_only' in call:
            print(f"   ❌ Call {i} (line {line_num}): Contains weights_only")
            all_good = False
        else:
            print(f"   ✅ Call {i} (line {line_num}): OK")
            print(f"      {call[:80]}...")
    
    return all_good


def check_file_size(submission_dir):
    """Check submission size is under limit"""
    print("\n🔍 Checking file sizes...")
    
    total_size = 0
    large_files = []
    
    for file in Path(submission_dir).rglob('*'):
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            
            # Flag files > 1MB
            if size > 1_000_000:
                large_files.append((file.name, size / 1_000_000))
    
    total_mb = total_size / 1_000_000
    print(f"   Total size: {total_mb:.2f} MB")
    
    if large_files:
        print(f"   Large files:")
        for name, size in sorted(large_files, key=lambda x: x[1], reverse=True):
            print(f"      - {name}: {size:.2f} MB")
    
    if total_mb > 10:
        print(f"   ⚠️  WARNING: Submission is {total_mb:.2f} MB (limit is ~10 MB)")
        return False
    else:
        print(f"   ✅ Size OK ({total_mb:.2f} MB < 10 MB limit)")
        return True


def check_required_files(submission_dir):
    """Check required files exist"""
    print("\n🔍 Checking required files...")
    
    required = ['submission.py']
    optional = ['weights_challenge_1.pt', 'weights_challenge_2.pt']
    
    all_good = True
    for filename in required:
        path = Path(submission_dir) / filename
        if path.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} NOT FOUND")
            all_good = False
    
    print("\n   Optional weight files:")
    for filename in optional:
        path = Path(submission_dir) / filename
        if path.exists():
            size = path.stat().st_size / 1_000_000
            print(f"   ✅ {filename} ({size:.2f} MB)")
    
    # Check for multi-seed weights (V16 pattern)
    seed_weights = list(Path(submission_dir).glob('weights_challenge_1_seed*.pt'))
    if seed_weights:
        print(f"\n   Found {len(seed_weights)} ensemble model(s):")
        for w in sorted(seed_weights):
            size = w.stat().st_size / 1_000_000
            print(f"      ✅ {w.name} ({size:.2f} MB)")
    
    return all_good


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_directory>")
        print("Example: python validate_submission.py submissions/phase1_v16/")
        sys.exit(1)
    
    submission_dir = Path(sys.argv[1])
    
    if not submission_dir.exists():
        print(f"❌ Directory not found: {submission_dir}")
        sys.exit(1)
    
    submission_py = submission_dir / 'submission.py'
    if not submission_py.exists():
        print(f"❌ submission.py not found in {submission_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("🧪 EEG Challenge Pre-Submission Validation")
    print("=" * 70)
    print(f"📁 Directory: {submission_dir}")
    
    results = []
    
    # Run all checks
    results.append(("weights_only check", check_weights_only(submission_py)))
    results.append(("torch.load check", check_torch_load(submission_py)))
    results.append(("Required files", check_required_files(submission_dir)))
    results.append(("File size", check_file_size(submission_dir)))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - SAFE TO UPLOAD!")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE UPLOADING!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
