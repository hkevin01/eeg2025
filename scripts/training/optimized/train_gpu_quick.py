#!/usr/bin/env python3
"""
Quick GPU Training with Real-Time Progress
- Unbuffered output for immediate feedback
- Timeout protection
- Minimal processing
"""

import os
import sys

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

from pathlib import Path

# ROCm environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Quick GPU Training", flush=True)
print("=" * 50, flush=True)

def progress(msg):
    """Print with immediate flush"""
    print(msg, flush=True)

class TinyModel(nn.Module):
    """Minimal model for testing"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(129, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def main():
    progress("\n📊 Step 1: Loading dataset...")

    try:
        from scripts.models.eeg_dataset_simple import SimpleEEGDataset

        dataset = SimpleEEGDataset(
            data_dir=Path(__file__).parent.parent / "data" / "raw" / "hbn",
            max_subjects=2,
            window_size=1000,
            verbose=True
        )
        progress(f"   ✅ Loaded {len(dataset)} windows")

        # Use only 20 samples for quick test
        dataset = Subset(dataset, list(range(min(20, len(dataset)))))
        progress(f"   Using {len(dataset)} samples for quick test")

    except Exception as e:
        progress(f"   ❌ Dataset error: {e}")
        return

    progress("\n🔧 Step 2: GPU setup...")

    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            progress(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")

            # Test GPU with small tensor
            test = torch.randn(10, 10).to(device)
            result = test @ test.T
            progress(f"   ✅ GPU test passed: {result.shape}")

        else:
            device = torch.device('cpu')
            progress("   ⚠️  No GPU, using CPU")

    except Exception as e:
        progress(f"   ❌ GPU error: {e}")
        device = torch.device('cpu')
        progress("   Falling back to CPU")

    progress(f"\n🧠 Step 3: Creating model on {device}...")

    try:
        model = TinyModel().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        progress(f"   ✅ Model created: {n_params:,} parameters")

    except Exception as e:
        progress(f"   ❌ Model error: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return

    progress("\n📦 Step 4: Creating data loader...")

    try:
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        progress(f"   ✅ DataLoader ready: {len(loader)} batches")

    except Exception as e:
        progress(f"   ❌ Loader error: {e}")
        return

    progress("\n🏋️  Step 5: Training...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        model.train()

        for epoch in range(2):
            progress(f"\nEpoch {epoch+1}/2")
            epoch_loss = 0

            for i, (data, target) in enumerate(loader):
                progress(f"  Batch {i+1}/{len(loader)}", end='')

                try:
                    data = data.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    progress(f" - loss: {loss.item():.4f}")

                    # Clear cache after each batch
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        progress(f" - OOM! Skipping...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise

            avg_loss = epoch_loss / len(loader)
            progress(f"  Epoch loss: {avg_loss:.4f}")

        progress("\n✅ Training completed successfully!")

        # Save model
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        save_path = checkpoint_dir / "gpu_quick_model.pth"
        torch.save(model.state_dict(), save_path)
        progress(f"💾 Model saved: {save_path}")

    except Exception as e:
        progress(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        progress("\n🧹 Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        progress("\n⚠️  Interrupted by user")
    except Exception as e:
        progress(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
