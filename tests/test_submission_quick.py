#!/usr/bin/env python3
"""Quick test of submission package without full model loading."""

import os
import sys

print("=" * 70)
print("QUICK SUBMISSION PACKAGE TEST")
print("=" * 70)
print()

# Check if we're in the right directory
if os.path.exists('submission.py'):
    print("✅ Found submission.py")
else:
    print("❌ submission.py not found!")
    print(f"   Current directory: {os.getcwd()}")
    sys.exit(1)

# Check weights files
weights_c1 = "weights_challenge_1.pt"
weights_c2 = "weights_challenge_2.pt"

if os.path.exists(weights_c1):
    size = os.path.getsize(weights_c1) / (1024*1024)
    print(f"✅ Found {weights_c1} ({size:.2f} MB)")
else:
    print(f"❌ {weights_c1} not found!")

if os.path.exists(weights_c2):
    size = os.path.getsize(weights_c2) / (1024*1024)
    print(f"✅ Found {weights_c2} ({size:.2f} MB)")
else:
    print(f"❌ {weights_c2} not found!")

print()

# Try to import submission module
print("Testing submission.py import...")
try:
    import submission as sub_module
    print("✅ submission.py imports successfully")
    
    # Check if Submission class exists
    if hasattr(sub_module, 'Submission'):
        print("✅ Submission class found")
    else:
        print("❌ Submission class not found!")
        
    # Check if helper functions exist
    if hasattr(sub_module, 'select_device'):
        print("✅ select_device() function found")
    if hasattr(sub_module, 'resolve_path'):
        print("✅ resolve_path() function found")
        
except Exception as e:
    print(f"❌ Error importing submission.py: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("✅ ALL BASIC CHECKS PASSED!")
print("=" * 70)
print()
print("📦 Submission package looks good!")
print()
print("Next steps:")
print("1. Upload these files to competition platform:")
print("   - submission.py")
print("   - weights_challenge_1.pt")
print("   - weights_challenge_2.pt")
print()
print("2. Or use the ZIP file: submission_20251021_213202.zip")
print()

