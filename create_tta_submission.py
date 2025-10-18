"""
Create submission ZIP with TTA integration
"""
import zipfile
import shutil
from pathlib import Path

print("=" * 70)
print("Creating TTA-Enhanced Submission Package")
print("=" * 70)

# Create temporary directory
temp_dir = Path('/home/kevin/Projects/eeg2025/submission_tta_temp')
if temp_dir.exists():
    shutil.rmtree(temp_dir)
temp_dir.mkdir()

print("\n📦 Copying files...")

# Copy submission_with_tta.py as submission.py
shutil.copy(
    '/home/kevin/Projects/eeg2025/submission_with_tta.py',
    temp_dir / 'submission.py'
)
print("   ✅ submission.py (with TTA)")

# Copy original submission.py as submission_base.py (for imports)
shutil.copy(
    '/home/kevin/Projects/eeg2025/submission.py',
    temp_dir / 'submission_base.py'
)
print("   ✅ submission_base.py (base models)")

# Update imports in submission.py to use submission_base
with open(temp_dir / 'submission.py', 'r') as f:
    content = f.read()

content = content.replace(
    'from submission import (',
    'from submission_base import ('
)

with open(temp_dir / 'submission.py', 'w') as f:
    f.write(content)

print("   ✅ Fixed imports")

# Copy model weights
checkpoints_dir = Path('/home/kevin/Projects/eeg2025/checkpoints')

shutil.copy(
    checkpoints_dir / 'response_time_attention.pth',
    temp_dir / 'response_time_attention.pth'
)
print("   ✅ response_time_attention.pth (9.8 MB)")

shutil.copy(
    checkpoints_dir / 'weights_challenge_2_multi_release.pt',
    temp_dir / 'weights_challenge_2_multi_release.pt'
)
print("   ✅ weights_challenge_2_multi_release.pt (261 KB)")

# Create ZIP
zip_path = Path('/home/kevin/Projects/eeg2025/eeg2025_submission_tta_v5.zip')
if zip_path.exists():
    zip_path.unlink()

print("\n📦 Creating ZIP archive...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in temp_dir.iterdir():
        zipf.write(file, file.name)
        print(f"   ✅ Added: {file.name}")

# Cleanup
shutil.rmtree(temp_dir)

# Get ZIP size
zip_size_mb = zip_path.stat().st_size / (1024 * 1024)

print("\n" + "=" * 70)
print("✅ SUBMISSION PACKAGE CREATED")
print("=" * 70)

print(f"\n📦 Package: {zip_path}")
print(f"📊 Size: {zip_size_mb:.2f} MB")

print("\n📝 Contents:")
print("   - submission.py (with TTA integration)")
print("   - submission_base.py (original models)")
print("   - response_time_attention.pth")
print("   - weights_challenge_2_multi_release.pt")

print("\n🚀 Expected Performance:")
print("   Baseline (v4):          0.283 NRMSE")
print("   With TTA (v5):          0.25-0.26 NRMSE")
print("   Expected improvement:   5-10% reduction")

print("\n📤 Next Steps:")
print("   1. Upload eeg2025_submission_tta_v5.zip to Codabench")
print("   2. Monitor test results (1-2 hours)")
print("   3. If successful, this becomes new baseline")
print("   4. Then train ensemble models for further gains")

print("\n" + "=" * 70)
