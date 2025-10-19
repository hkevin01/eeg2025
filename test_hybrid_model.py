"""Quick sanity check for hybrid model."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.hybrid_cnn import HybridNeuroModel

print("="*80)
print("�� TESTING HYBRID MODEL")
print("="*80)

# Test model creation
print("\n1. Creating model...")
model = HybridNeuroModel(
    num_channels=129,
    seq_length=200,
    dropout=0.4,
    use_neuro_features=True
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✅ Model created successfully")
print(f"   Parameters: {n_params:,}")

# Test forward pass
print("\n2. Testing forward pass...")
batch_size = 4
x = torch.randn(batch_size, 129, 200)

try:
    output = model(x)
    print(f"   ✅ Forward pass successful")
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    sys.exit(1)

# Test with neuroscience features disabled
print("\n3. Testing CNN-only mode (no neuro features)...")
model_cnn_only = HybridNeuroModel(
    num_channels=129,
    seq_length=200,
    dropout=0.4,
    use_neuro_features=False
)

try:
    output_cnn = model_cnn_only(x)
    print(f"   ✅ CNN-only mode works")
    print(f"   Output shape: {output_cnn.shape}")
except Exception as e:
    print(f"   ❌ CNN-only mode failed: {e}")
    sys.exit(1)

# Test gradient flow
print("\n4. Testing backward pass...")
try:
    loss = output.mean()
    loss.backward()
    print(f"   ✅ Backward pass successful")
    print(f"   Gradients computed")
except Exception as e:
    print(f"   ❌ Backward pass failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - MODEL READY FOR TRAINING")
print("="*80)
print("\nNext steps:")
print("  1. Run: python scripts/training/challenge1/train_hybrid_hdf5.py")
print("  2. Monitor training progress")
print("  3. Compare to baseline (0.26 NRMSE)")
print()
