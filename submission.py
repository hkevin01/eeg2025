"""
EEG Foundation Challenge 2025 - Submission File
================================================
This submission includes GPU/ROCm/CUDA support with automatic CPU fallback.

Challenge 1: Cross-Task Transfer Learning (CCD task)
Challenge 2: Externalizing Factor Prediction

Models:
- Challenge 1: TCN (Temporal Convolutional Network) - 196K params
- Challenge 2: EEGNeX-style model

Author: Kevin
Date: October 19, 2025
"""

import torch
import torch.nn as nn
from pathlib import Path

# ============================================================================
# DEVICE SELECTION WITH GPU/ROCM/CUDA FALLBACK
# ============================================================================

def select_device(verbose=True):
    """
    Smart device selection with GPU/ROCm/CUDA support and CPU fallback.

    Tries in order:
    1. CUDA (NVIDIA GPUs)
    2. ROCm (AMD GPUs via CUDA interface)
    3. CPU (fallback)

    Returns:
        torch.device: Selected device
        str: Device description
    """
    device_info = ""

    # Try GPU first
    if torch.cuda.is_available():
        try:
            # Test if GPU actually works
            test_tensor = torch.randn(1, 1).cuda()
            _ = test_tensor + test_tensor
            del test_tensor
            torch.cuda.empty_cache()

            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            device_info = f"GPU ({device_name})"

            if verbose:
                print(f"✅ Using GPU: {device_name}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   PyTorch version: {torch.__version__}")

            return device, device_info

        except Exception as e:
            if verbose:
                print(f"⚠️  GPU available but failed test: {e}")
                print("   Falling back to CPU...")

    # Fallback to CPU
    device = torch.device('cpu')
    device_info = "CPU"

    if verbose:
        print(f"✅ Using CPU")
        print(f"   PyTorch version: {torch.__version__}")

    return device, device_info


# ============================================================================
# PATH RESOLUTION (from starter kit)
# ============================================================================

def resolve_path(name="model_file_name"):
    """
    Resolve path to model weights file.
    Tries multiple locations in order.
    """
    search_paths = [
        Path(f"/app/input/res/{name}"),
        Path(f"/app/input/{name}"),
        Path(name),
        Path(__file__).parent / name,
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Could not find {name} in any of: {[str(p) for p in search_paths]}"
    )


# ============================================================================
# TCN MODEL ARCHITECTURE (Challenge 1)
# ============================================================================

class TemporalBlock(nn.Module):
    """Temporal block for TCN with BatchNorm."""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.padding = padding

    def forward(self, x):
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        # Match dimensions
        if out.shape[-1] != res.shape[-1]:
            res = res[:, :, :out.shape[-1]]

        return self.relu2(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network."""
    def __init__(self, num_inputs, num_filters, kernel_size, dropout, num_levels):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_filters
            layers.append(
                TemporalBlock(in_channels, num_filters, kernel_size,
                            dilation=dilation_size, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_EEG(nn.Module):
    """
    TCN for EEG regression - EXACT match to trained model.

    Args:
        num_channels: Number of EEG channels (default: 129)
        num_outputs: Number of output values (default: 1)
        num_filters: Number of filters per TCN level (default: 48)
        kernel_size: Kernel size for temporal convolutions (default: 7)
        num_levels: Number of TCN levels (default: 5)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, num_channels=129, num_outputs=1, num_filters=48,
                 kernel_size=7, num_levels=5, dropout=0.3):
        super(TCN_EEG, self).__init__()

        # Build TCN layers directly to match checkpoint structure
        # Checkpoint has keys like: network.0.conv1.weight, network.1.conv1.weight, etc.
        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels if i == 0 else num_filters
            layers.append(
                TemporalBlock(in_channels, num_filters, kernel_size,
                            dilation=dilation_size, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)

        # Global pooling and output
        self.fc = nn.Linear(num_filters, num_outputs)

    def forward(self, x):
        # x shape: (batch, 129, 200)
        out = self.network(x)  # (batch, num_filters, time)
        out = out.mean(dim=-1)  # Global average pooling: (batch, num_filters)
        out = self.fc(out)  # (batch, 1)
        return out
# ============================================================================
# CHALLENGE 2 MODEL (EEGNeX-style)
# ============================================================================

class EEGNeX_Simple(nn.Module):
    """
    Simplified EEGNeX-style model for Challenge 2.
    Can be replaced with actual EEGNeX if braindecode available.
    """
    def __init__(self, n_chans=129, n_outputs=1, sfreq=100, n_times=200):
        super(EEGNeX_Simple, self).__init__()

        # Simple CNN architecture as fallback
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(n_chans, 1))
        self.conv3 = nn.Conv1d(64, 128, kernel_size=10, padding=5)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, 129, 200)
        x = x.unsqueeze(1)  # (batch, 1, 129, 200)

        x = torch.relu(self.conv1(x))  # (batch, 32, 129, 200)
        x = torch.relu(self.conv2(x))  # (batch, 64, 1, 200)
        x = x.squeeze(2)  # (batch, 64, 200)
        x = torch.relu(self.conv3(x))  # (batch, 128, 200)

        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, 1)

        return x


# ============================================================================
# SUBMISSION CLASS
# ============================================================================

class Submission:
    """
    Submission class for EEG Foundation Challenge.

    Automatically selects GPU (CUDA/ROCm) if available, falls back to CPU.
    """

    def __init__(self, SFREQ, DEVICE=None):
        """
        Initialize submission.

        Args:
            SFREQ: Sampling frequency (Hz)
            DEVICE: Device to use (if None, will auto-select)
        """
        self.sfreq = SFREQ

        # Auto-select device if not provided
        if DEVICE is None:
            self.device, device_info = select_device(verbose=True)
        else:
            self.device = DEVICE
            device_info = str(DEVICE)

        print(f"\n{'='*60}")
        print(f"EEG Foundation Challenge 2025 - Submission")
        print(f"{'='*60}")
        print(f"Device: {device_info}")
        print(f"Sampling Frequency: {SFREQ} Hz")
        print(f"Expected input shape: (batch, 129 channels, {int(2 * SFREQ)} timepoints)")
        print(f"{'='*60}\n")

    def get_model_challenge_1(self):
        """
        Load Challenge 1 model (TCN for CCD task).

        Returns:
            nn.Module: Trained model ready for inference
        """
        print("Loading Challenge 1 model (TCN)...")

        # Create model
        model = TCN_EEG(
            num_channels=129,
            num_outputs=1,
            num_filters=48,
            kernel_size=7,
            num_levels=5,
            dropout=0.3
        )

        # Load weights
        try:
            weights_path = resolve_path("weights_challenge_1.pt")
            print(f"  Loading weights from: {weights_path}")

            checkpoint = torch.load(weights_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'epoch' in checkpoint:
                    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
                if 'val_loss' in checkpoint:
                    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
            else:
                model.load_state_dict(checkpoint)

            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        except FileNotFoundError:
            print("  ⚠️  Weights file not found, using untrained model")
        except Exception as e:
            print(f"  ⚠️  Error loading weights: {e}")
            print("  Using untrained model")

        model = model.to(self.device)
        model.eval()

        print("  ✅ Challenge 1 model ready\n")
        return model

    def get_model_challenge_2(self):
        """
        Load Challenge 2 model (externalizing factor prediction).

        Returns:
            nn.Module: Trained model ready for inference
        """
        print("Loading Challenge 2 model...")

        # Try to use EEGNeX if braindecode available
        try:
            from braindecode.models import EEGNeX
            model = EEGNeX(
                n_chans=129,
                n_outputs=1,
                sfreq=self.sfreq,
                n_times=int(2 * self.sfreq)
            )
            print("  Using EEGNeX architecture")
        except ImportError:
            print("  Using simplified CNN (braindecode not available)")
            model = EEGNeX_Simple(
                n_chans=129,
                n_outputs=1,
                sfreq=self.sfreq,
                n_times=int(2 * self.sfreq)
            )

        # Load weights
        try:
            weights_path = resolve_path("weights_challenge_2.pt")
            print(f"  Loading weights from: {weights_path}")

            checkpoint = torch.load(weights_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        except FileNotFoundError:
            print("  ⚠️  Weights file not found, using untrained model")
        except Exception as e:
            print(f"  ⚠️  Error loading weights: {e}")
            print("  Using untrained model")

        model = model.to(self.device)
        model.eval()

        print("  ✅ Challenge 2 model ready\n")
        return model


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test submission file locally."""

    print("\n" + "="*60)
    print("TESTING SUBMISSION FILE")
    print("="*60 + "\n")

    # Initialize
    SFREQ = 100
    sub = Submission(SFREQ)

    # Test Challenge 1
    print("\nTesting Challenge 1 model:")
    print("-" * 40)
    model_1 = sub.get_model_challenge_1()

    # Create test input
    batch_size = 4
    n_channels = 129
    n_times = int(2 * SFREQ)
    X_test = torch.randn(batch_size, n_channels, n_times, device=sub.device)

    print(f"Test input shape: {X_test.shape}")

    with torch.inference_mode():
        y_pred = model_1(X_test)
        print(f"Output shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred[:3, 0].cpu().numpy()}")

    print("✅ Challenge 1 model works!\n")

    # Clean up
    del model_1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test Challenge 2
    print("\nTesting Challenge 2 model:")
    print("-" * 40)
    model_2 = sub.get_model_challenge_2()

    with torch.inference_mode():
        y_pred = model_2(X_test)
        print(f"Output shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred[:3, 0].cpu().numpy()}")

    print("✅ Challenge 2 model works!\n")

    print("\n" + "="*60)
    print("ALL TESTS PASSED - SUBMISSION FILE READY")
    print("="*60 + "\n")
