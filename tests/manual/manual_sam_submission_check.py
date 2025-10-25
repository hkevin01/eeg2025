"""
Local test for SAM combined submission
"""
import importlib
import sys
from pathlib import Path

import torch

# Ensure we can import the archived submission implementation
BASE_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = BASE_DIR / "submissions" / "versions"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

Submission = importlib.import_module("submission_sam_final").Submission

def test_submission():
    """Test the combined SAM submission locally"""

    print("="*80)
    print("🧪 Testing Combined SAM Submission Locally")
    print("="*80)
    print()

    # Setup
    SFREQ = 100
    DEVICE = 'cpu'  # Use CPU for local testing

    # Create submission instance
    print("📦 Creating submission instance...")
    submission = Submission(SFREQ, DEVICE)
    print()

    # Test Challenge 1
    print("🧪 Testing Challenge 1...")
    print("-" * 80)

    # Create dummy input
    batch_size = 4
    n_chans = 129
    n_times = 200
    X_c1 = torch.randn(batch_size, n_chans, n_times)

    print(f"Input shape: {X_c1.shape}")

    # Make predictions
    try:
        preds_c1 = submission(X_c1, challenge_number=1)
        print(f"Output shape: {preds_c1.shape}")
        print(f"Sample predictions: {preds_c1[:3, 0].tolist()}")
        print("✅ Challenge 1 test PASSED")
    except Exception as e:
        print(f"❌ Challenge 1 test FAILED: {e}")
        return False

    print()

    # Test Challenge 2
    print("🧪 Testing Challenge 2...")
    print("-" * 80)

    # Create dummy input (same shape)
    X_c2 = torch.randn(batch_size, n_chans, n_times)

    print(f"Input shape: {X_c2.shape}")

    # Make predictions
    try:
        preds_c2 = submission(X_c2, challenge_number=2)
        print(f"Output shape: {preds_c2.shape}")
        print(f"Sample predictions: {preds_c2[:3, 0].tolist()}")
        print("✅ Challenge 2 test PASSED")
    except Exception as e:
        print(f"❌ Challenge 2 test FAILED: {e}")
        return False

    print()
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print()
    archive_path = BASE_DIR / "submissions" / "archives" / "submission_sam_combined.zip"
    if archive_path.exists():
        size_kb = archive_path.stat().st_size // 1024
        size_repr = f"{size_kb}K"
    else:
        size_repr = "missing"
    print(f"📦 Submission package: {archive_path.name} ({size_repr})")
    print("   Files:")
    print("   - submission_sam_final.py (6.6K)")
    print("   - weights_challenge_1_sam.pt (259K, val 0.3008)")
    print("   - weights_challenge_2_sam.pt (257K, val 0.2042)")
    print()
    print("🚀 Ready to upload to Codabench!")
    print()

    return True

if __name__ == "__main__":
    success = test_submission()
    sys.exit(0 if success else 1)
