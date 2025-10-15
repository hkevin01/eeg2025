#!/usr/bin/env python3
"""
GPU Training with Timeout Protection
- Each GPU operation has a timeout
- Automatic fallback to CPU if GPU hangs
- Real-time progress output
"""

import os
import sys
import signal
from pathlib import Path
from multiprocessing import Process, Queue
import time

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# ROCm environment
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

def progress(msg):
    """Print with immediate flush"""
    print(msg, flush=True)

def test_gpu_with_timeout(queue, timeout=10):
    """Test GPU in separate process with timeout"""
    try:
        import torch
        
        progress("   Testing GPU availability...")
        if not torch.cuda.is_available():
            queue.put(("cpu", "GPU not available"))
            return
        
        progress(f"   GPU detected: {torch.cuda.get_device_name(0)}")
        
        progress("   Testing small tensor operation...")
        device = torch.device('cuda:0')
        test = torch.randn(10, 10).to(device)
        result = test @ test.T
        result_cpu = result.cpu()
        
        progress(f"   ✅ GPU test passed: {result_cpu.shape}")
        queue.put(("cuda", None))
        
    except Exception as e:
        progress(f"   ❌ GPU error: {e}")
        queue.put(("cpu", str(e)))

def safe_gpu_check(timeout=15):
    """Check GPU with timeout protection"""
    progress("\n🔧 Step 2: Safe GPU check (with timeout)...")
    
    queue = Queue()
    p = Process(target=test_gpu_with_timeout, args=(queue, timeout))
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        progress(f"   ⚠️  GPU test timed out after {timeout}s - killing process")
        p.terminate()
        p.join()
        return "cpu", "GPU test timeout"
    
    if not queue.empty():
        device_type, error = queue.get()
        if error:
            progress(f"   Falling back to CPU: {error}")
        return device_type, error
    else:
        progress("   ⚠️  No response from GPU test - using CPU")
        return "cpu", "No response"

def train_on_device(device_type):
    """Train model on specified device"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.models.eeg_dataset_simple import SimpleEEGDataset
    
    progress(f"\n🏋️  Training on {device_type.upper()}...")
    
    device = torch.device(device_type + (':0' if device_type == 'cuda' else ''))
    
    # Tiny model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(129, 16, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, 2)
        
        def forward(self, x):
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    progress("\n📊 Loading minimal dataset...")
    dataset = SimpleEEGDataset(
        data_dir=Path(__file__).parent.parent / "data" / "raw" / "hbn",
        max_subjects=2,
        window_size=1000,
        verbose=False
    )
    
    # Use only 10 samples for ultra-fast test
    dataset = Subset(dataset, list(range(min(10, len(dataset)))))
    progress(f"   Using {len(dataset)} samples")
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    progress(f"\n🧠 Creating model on {device}...")
    model = TinyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    progress(f"\n🚀 Training for 2 epochs...")
    model.train()
    
    for epoch in range(2):
        progress(f"\nEpoch {epoch+1}/2")
        epoch_loss = 0
        batch_count = 0
        
        for i, (data, target) in enumerate(loader):
            try:
                data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                progress(f"  Batch {i+1}/{len(loader)} - loss: {loss.item():.4f}")
                
                # Clear cache after each batch on GPU
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    progress(f"  OOM! Skipping batch...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            progress(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    
    progress("\n✅ Training completed successfully!")
    
    # Save model
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    save_path = checkpoint_dir / f"{device_type}_timeout_model.pth"
    torch.save(model.state_dict(), save_path)
    progress(f"💾 Model saved: {save_path}")

def main():
    progress("🚀 GPU Training with Timeout Protection")
    progress("=" * 50)
    
    # Check GPU with timeout
    device_type, error = safe_gpu_check(timeout=15)
    
    if device_type == "cpu":
        progress("\n⚠️  Using CPU for training")
        if error:
            progress(f"   Reason: {error}")
    else:
        progress("\n✅ Using GPU for training")
    
    # Train
    try:
        train_on_device(device_type)
    except Exception as e:
        progress(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        if device_type == "cuda":
            progress("\n🔄 Retrying on CPU...")
            try:
                train_on_device("cpu")
            except Exception as e2:
                progress(f"❌ CPU training also failed: {e2}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        progress("\n⚠️  Interrupted by user")
    except Exception as e:
        progress(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
