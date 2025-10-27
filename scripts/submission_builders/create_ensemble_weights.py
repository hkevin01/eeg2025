"""
Create ensemble weights for v9 submission
Combines CompactCNN (baseline) + TCN weights
"""

import torch

print("Creating ensemble weights for submission v9...")
print("=" * 60)

# Load CompactCNN weights (from baseline Oct 16)
print("\n📦 Loading CompactCNN weights...")
try:
    compact_weights = torch.load('checkpoints/baseline_cnn.pth', 
                                map_location='cpu',
                                weights_only=False)
    print(f"✅ CompactCNN loaded: {len(compact_weights)} keys")
    print(f"   Sample keys: {list(compact_weights.keys())[:3]}")
except Exception as e:
    print(f"❌ Error loading CompactCNN: {e}")
    compact_weights = None

# Load TCN weights (from Oct 17)
print("\n📦 Loading TCN weights...")
try:
    tcn_weights = torch.load('checkpoints/challenge1_tcn_competition_best.pth',
                            map_location='cpu',
                            weights_only=False)
    print(f"✅ TCN loaded: {len(tcn_weights)} keys")
    print(f"   Sample keys: {list(tcn_weights.keys())[:3]}")
except Exception as e:
    print(f"❌ Error loading TCN: {e}")
    tcn_weights = None

# Create ensemble weights dictionary
print("\n🔧 Creating ensemble weights...")
ensemble_weights = {}

if compact_weights is not None:
    # Add CompactCNN weights with prefix
    for key, value in compact_weights.items():
        new_key = f"compact_cnn.{key}"
        ensemble_weights[new_key] = value
    print(f"✅ Added {len(compact_weights)} CompactCNN weights")

if tcn_weights is not None:
    # Add TCN weights with prefix
    for key, value in tcn_weights.items():
        new_key = f"tcn.{key}"
        ensemble_weights[new_key] = value
    print(f"✅ Added {len(tcn_weights)} TCN weights")

print(f"\n📊 Total ensemble weights: {len(ensemble_weights)} keys")

# Save ensemble weights
print("\n💾 Saving ensemble weights...")
torch.save(ensemble_weights, 'weights_c1_ensemble.pt')
print("✅ Saved to: weights_c1_ensemble.pt")

# Also save individual weight files for fallback
if compact_weights is not None:
    torch.save(compact_weights, 'weights_c1_compact.pt')
    print("✅ Saved CompactCNN to: weights_c1_compact.pt")

if tcn_weights is not None:
    torch.save(tcn_weights, 'weights_c1_tcn.pt')
    print("✅ Saved TCN to: weights_c1_tcn.pt")

print("\n" + "=" * 60)
print("✅ Ensemble weights created successfully!")
print("\nFiles created:")
print("  - weights_c1_ensemble.pt (combined)")
print("  - weights_c1_compact.pt (CompactCNN only)")
print("  - weights_c1_tcn.pt (TCN only)")
