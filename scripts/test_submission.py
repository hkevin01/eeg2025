#!/usr/bin/env python3
"""
Test the submission.py file locally before uploading
"""

import numpy as np
import torch
from submission import Submission

def test_submission():
    print("=" * 80)
    print("Testing EEG 2025 Submission")
    print("=" * 80)
    
    # Initialize submission
    print("\n1. Initializing Submission class...")
    try:
        submission = Submission()
        print("✅ Submission initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Test Challenge 1 (Response Time)
    print("\n2. Testing Challenge 1 (Response Time Prediction)...")
    try:
        # Create dummy EEG data (batch_size=4, channels=129, samples=200)
        dummy_eeg_c1 = np.random.randn(4, 129, 200).astype(np.float32)
        
        predictions_c1 = submission.predict_response_time(dummy_eeg_c1)
        
        print(f"   Input shape: {dummy_eeg_c1.shape}")
        print(f"   Output shape: {predictions_c1.shape}")
        print(f"   Sample predictions: {predictions_c1[:2]}")
        
        assert predictions_c1.shape == (4,), f"Expected shape (4,), got {predictions_c1.shape}"
        assert not np.isnan(predictions_c1).any(), "NaN values in predictions"
        assert not np.isinf(predictions_c1).any(), "Inf values in predictions"
        
        print("✅ Challenge 1 predictions successful")
    except Exception as e:
        print(f"❌ Challenge 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Challenge 2 (Externalizing)
    print("\n3. Testing Challenge 2 (Externalizing Prediction)...")
    try:
        # Create dummy EEG data (batch_size=4, channels=129, samples=200)
        dummy_eeg_c2 = np.random.randn(4, 129, 200).astype(np.float32)
        
        predictions_c2 = submission.predict_externalizing(dummy_eeg_c2)
        
        print(f"   Input shape: {dummy_eeg_c2.shape}")
        print(f"   Output shape: {predictions_c2.shape}")
        print(f"   Sample predictions: {predictions_c2[:2]}")
        
        assert predictions_c2.shape == (4,), f"Expected shape (4,), got {predictions_c2.shape}"
        assert not np.isnan(predictions_c2).any(), "NaN values in predictions"
        assert not np.isinf(predictions_c2).any(), "Inf values in predictions"
        
        print("✅ Challenge 2 predictions successful")
    except Exception as e:
        print(f"❌ Challenge 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test batch processing
    print("\n4. Testing batch processing...")
    try:
        batch_sizes = [1, 8, 16, 32]
        for bs in batch_sizes:
            dummy_data = np.random.randn(bs, 129, 200).astype(np.float32)
            preds_c1 = submission.predict_response_time(dummy_data)
            preds_c2 = submission.predict_externalizing(dummy_data)
            
            assert preds_c1.shape == (bs,), f"Batch size {bs} failed for C1"
            assert preds_c2.shape == (bs,), f"Batch size {bs} failed for C2"
        
        print(f"   Tested batch sizes: {batch_sizes}")
        print("✅ Batch processing successful")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False
    
    # Model info
    print("\n5. Model Information:")
    print(f"   Challenge 1 params: {sum(p.numel() for p in submission.model_response_time.parameters()):,}")
    print(f"   Challenge 2 params: {sum(p.numel() for p in submission.model_externalizing.parameters()):,}")
    print(f"   Device: {submission.device}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_submission()
    exit(0 if success else 1)
