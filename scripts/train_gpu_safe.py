#!/usr/bin/env python3
"""
Safe GPU Training Script with Memory Management
Supports both NVIDIA CUDA and AMD ROCm
"""

import os
import sys
from pathlib import Path

# Critical ROCm environment variables for Navi 10 (RX 5700 XT) stability
# Must be set BEFORE importing torch
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['GPU_MAX_HW_QUEUES'] = '4'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import mne

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Safe GPU Training Script")
print("=" * 60)

# GPU Setup with safety checks
def setup_device():
    """Setup GPU device with safety checks and memory limits"""
    if not torch.cuda.is_available():
        print("⚠️  No GPU available, using CPU")
        return torch.device("cpu"), {}

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    # Get GPU properties
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / (1024**3)  # GB

    print(f"\n✅ GPU detected: {gpu_name}")
    print(f"   Total memory: {total_memory:.2f} GB")
    print(f"   Compute capability: {props.major}.{props.minor}")

    # Set memory fraction to prevent OOM (use max 80% of GPU memory)
    torch.cuda.set_per_process_memory_fraction(0.8, device=device)

    # Enable TF32 for better performance on Ampere+ (if available)
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    # ROCm specific optimizations
    is_rocm = 'rocm' in torch.__version__.lower() or torch.version.hip is not None
    if is_rocm:
        print(f"   ROCm version: {torch.version.hip}")
        # Set environment variables for stability
        os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512'

    config = {
        'device': device,
        'gpu_name': gpu_name,
        'total_memory_gb': total_memory,
        'is_rocm': is_rocm,
        'mixed_precision': True,  # Use FP16 to save memory
    }

    return device, config

# Simple EEG Dataset
class SimpleEEGDataset(Dataset):
    def __init__(self, data_dir, max_subjects=None):
        self.data_dir = Path(data_dir)
        self.windows = []
        self.labels = []

        print(f"\n📂 Loading EEG data from: {data_dir}")

        # Find all subjects
        subjects = sorted(list(self.data_dir.glob("sub-NDAR*")))
        if max_subjects:
            subjects = subjects[:max_subjects]

        print(f"   Found {len(subjects)} subjects")

        for subj_dir in tqdm(subjects, desc="Loading subjects"):
            eeg_dir = subj_dir / "eeg"
            if not eeg_dir.exists():
                continue

            # Load .set files
            for set_file in eeg_dir.glob("*.set"):
                try:
                    # Load with MNE
                    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)

                    # Preprocessing
                    raw.filter(0.5, 45, verbose=False)
                    raw.notch_filter(60, verbose=False)

                    # Get data and create windows
                    data = raw.get_data()  # (n_channels, n_samples)
                    sfreq = raw.info['sfreq']
                    window_samples = int(2.0 * sfreq)  # 2 second windows

                    # Create overlapping windows (50% overlap)
                    step = window_samples // 2
                    for start in range(0, data.shape[1] - window_samples, step):
                        window = data[:, start:start + window_samples]

                        # Store window
                        self.windows.append(window)
                        # Dummy label (0 or 1)
                        self.labels.append(np.random.randint(0, 2))

                except Exception as e:
                    print(f"      Error loading {set_file.name}: {e}")
                    continue

        print(f"\n✅ Loaded {len(self.windows)} windows")
        if len(self.windows) > 0:
            print(f"   Window shape: {self.windows[0].shape}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Convert to tensor
        x = torch.FloatTensor(self.windows[idx])
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y

# Simple transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=256,
                 n_heads=8, n_layers=4, n_classes=2):
        super().__init__()

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(n_channels, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        batch_size = x.size(0)

        # Reshape to (batch, time, channels)
        x = x.transpose(1, 2)

        # Project to hidden dim
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classify
        out = self.classifier(x)
        return out

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        # Move to device
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

        # Clear cache periodically
        if batch_idx % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    # Setup device
    device, gpu_config = setup_device()

    print(f"\n📊 Configuration:")
    print(f"   Device: {device}")
    print(f"   Mixed precision: {gpu_config.get('mixed_precision', False)}")

    # Load dataset with limit
    print("\n" + "="*60)
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"

    # Start with small dataset to test
    max_subjects = 3  # Start small to prevent OOM
    dataset = SimpleEEGDataset(data_dir, max_subjects=max_subjects)

    if len(dataset) == 0:
        print("❌ No data loaded! Exiting.")
        return

    # Small batch size for safety
    batch_size = 4 if device.type == 'cuda' else 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )

    print(f"\n📦 Data loader:")
    print(f"   Total windows: {len(dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {len(dataloader)}")

    # Create model
    print("\n" + "="*60)
    print("🧠 Creating model...")

    # Get sample to determine dimensions
    sample_data, _ = dataset[0]
    n_channels, seq_len = sample_data.shape

    print(f"   Input shape: ({n_channels}, {seq_len})")

    # Smaller model for safety
    model = SimpleTransformer(
        n_channels=n_channels,
        seq_len=seq_len,
        hidden_dim=128,  # Smaller to save memory
        n_heads=4,
        n_layers=2,
        n_classes=2
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    print(f"   Estimated memory: ~{n_params * 4 / (1024**2):.1f} MB")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Mixed precision scaler (for GPU only)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and gpu_config.get('mixed_precision') else None

    # Training loop
    print("\n" + "="*60)
    print("🏋️  Starting training...")
    print("=" * 60)

    n_epochs = 5  # Small number for testing

    try:
        for epoch in range(n_epochs):
            print(f"\n📈 Epoch {epoch+1}/{n_epochs}")

            # Train
            train_loss, train_acc = train_epoch(
                model, dataloader, criterion, optimizer, device, scaler
            )

            print(f"   Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

            # Memory stats
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        print("\n" + "="*60)
        print("✅ Training completed successfully!")

        # Save model
        output_dir = Path(__file__).parent.parent / "outputs" / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "transformer_gpu.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'n_channels': n_channels,
                'seq_len': seq_len,
                'hidden_dim': 128,
                'n_heads': 4,
                'n_layers': 2,
            }
        }, model_path)

        print(f"💾 Model saved to: {model_path}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ GPU Out of Memory!")
            print("   Try reducing:")
            print("   - batch_size (currently {})".format(batch_size))
            print("   - hidden_dim (currently 128)")
            print("   - n_layers (currently 2)")
            print("   - max_subjects (currently {})".format(max_subjects))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 Cleanup complete")

if __name__ == "__main__":
    main()
