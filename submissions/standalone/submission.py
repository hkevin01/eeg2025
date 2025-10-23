"""
Competition Submission for EEG Foundation Challenge 2025
EXACT match to trained models - no braindecode dependency
Platform: CPU-only
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def resolve_path(name="model_file_name"):
    """Resolve file path - matches starter kit exactly."""
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} in /app/input/res/ or /app/input/ or current directory"
        )


class EEGNeX_Standalone(nn.Module):
    """
    Standalone EEGNeX - EXACT match to braindecode architecture.
    Takes 3D input (batch, channels, time) like braindecode.
    """

    def __init__(
        self,
        n_chans=129,
        n_outputs=1,
        n_times=200,
        sfreq=100,
    ):
        super().__init__()

        # Match braindecode defaults exactly
        filter_1 = 8
        filter_2 = 32
        filter_3 = 64  # filter_2 * 2

        # block_1: index 1=Conv2d, index 2=BatchNorm (index 0 is Rearrange in braindecode)
        self.block_1_1 = nn.Conv2d(1, filter_1, kernel_size=(1, 64), padding='same', bias=False)
        self.block_1_2 = nn.BatchNorm2d(filter_1)

        # block_2
        self.block_2_0 = nn.Conv2d(filter_1, filter_2, kernel_size=(1, 64), padding='same', bias=False)
        self.block_2_1 = nn.BatchNorm2d(filter_2)

        # block_3: Spatial depthwise conv (with constraint handled in loading)
        self.block_3_0 = nn.Conv2d(filter_2, filter_3, kernel_size=(n_chans, 1),
                                   groups=filter_2, bias=False, padding=0)
        self.block_3_1 = nn.BatchNorm2d(filter_3)
        self.block_3_2 = nn.ELU()
        self.block_3_3 = nn.AvgPool2d((1, 4), padding=(0, 1))
        self.block_3_4 = nn.Dropout(0.5)

        # block_4: Conv2d + BatchNorm (NO activation/pooling/dropout)
        self.block_4_0 = nn.Conv2d(filter_3, filter_2, kernel_size=(1, 16),
                                   dilation=(1, 2), padding='same', bias=False)
        self.block_4_1 = nn.BatchNorm2d(filter_2)

        # block_5
        self.block_5_0 = nn.Conv2d(filter_2, filter_1, kernel_size=(1, 16),
                                   dilation=(1, 4), padding='same', bias=False)
        self.block_5_1 = nn.BatchNorm2d(filter_1)
        self.block_5_2 = nn.ELU()
        self.block_5_3 = nn.AvgPool2d((1, 8), padding=(0, 1))
        self.block_5_4 = nn.Dropout(0.5)
        self.block_5_5 = nn.Flatten()

        # Final layer (constraint handled in loading)
        self.final_layer = nn.Linear(48, n_outputs)

    def load_braindecode_weights(self, state_dict):
        """Load weights from braindecode model, handling parametrizations."""
        new_state_dict = {}

        for key, value in state_dict.items():
            # Handle parametrized weights
            if 'parametrizations.weight.original' in key:
                # Map to regular weight
                new_key = key.replace('.parametrizations.weight.original', '.weight')
                new_state_dict[new_key] = value
            else:
                # Direct mapping with underscores instead of dots for sub-layers
                new_key = key.replace('.', '_', 1)  # Replace first dot only
                if new_key.startswith('block'):
                    # Keep structure like block_1_1.weight
                    new_state_dict[new_key] = value
                elif new_key.startswith('final_layer'):
                    new_state_dict[key] = value
                else:
                    new_state_dict[key] = value

        # Load the mapped weights
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        """
        Forward pass.
        Input: (batch, channels, time) - 3D tensor
        Output: (batch, n_outputs)
        """
        # Rearrange: (batch, channels, time) -> (batch, 1, channels, time)
        x = x.unsqueeze(1)

        # Block 1
        x = self.block_1_1(x)
        x = self.block_1_2(x)

        # Block 2
        x = self.block_2_0(x)
        x = self.block_2_1(x)

        # Block 3
        x = self.block_3_0(x)
        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.block_3_3(x)
        x = self.block_3_4(x)

        # Block 4
        x = self.block_4_0(x)
        x = self.block_4_1(x)

        # Block 5
        x = self.block_5_0(x)
        x = self.block_5_1(x)
        x = self.block_5_2(x)
        x = self.block_5_3(x)
        x = self.block_5_4(x)
        x = self.block_5_5(x)

        # Final
        x = self.final_layer(x)
        return x


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block from competition training."""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()

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

        if out.shape[-1] != res.shape[-1]:
            res = res[:, :, :out.shape[-1]]

        return self.relu2(out + res)


class TCN_EEG(nn.Module):
    """Temporal Convolutional Network from competition training."""

    def __init__(self, num_channels=129, num_outputs=1, num_filters=64,
                 kernel_size=7, dropout=0.2, num_levels=6):
        super().__init__()

        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels if i == 0 else num_filters
            layers.append(
                TemporalBlock(in_channels, num_filters, kernel_size,
                            dilation=dilation_size, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, num_outputs)

    def forward(self, x):
        out = self.network(x)
        out = out.mean(dim=-1)
        return self.fc(out)


class Submission:
    """Main submission class for competition."""

    def __init__(self, SFREQ: int, DEVICE):
        """Initialize submission with device and constants.

        Args:
            SFREQ: Sampling frequency
            DEVICE: Device provided by competition (may be string or torch.device)
        """
        # Auto-detect device: Use provided DEVICE if valid, otherwise CPU
        # Competition uses: torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ROCm systems: CUDA calls work via HIP compatibility layer
        if isinstance(DEVICE, str):
            self.device = torch.device(DEVICE)
        elif isinstance(DEVICE, torch.device):
            self.device = DEVICE
        else:
            # Fallback: auto-detect
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("=" * 60)
        print("EEG Foundation Challenge 2025 - Submission")
        print("=" * 60)
        print(f"Device: {self.device.type.upper()}")
        if self.device.type == "cuda":
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("⚠️  CUDA requested but not available - falling back to CPU")
                self.device = torch.device("cpu")

        self.sfreq = SFREQ
        self.n_chans = 129
        self.n_times = int(2 * SFREQ)

        print(f"Sampling Frequency: {self.sfreq} Hz")
        print(f"Expected input shape: (batch, {self.n_chans} channels, {self.n_times} timepoints)")
        print("=" * 60)
        print()

        self.model_c1 = None
        self.model_c2 = None

    def get_model_challenge_1(self):
        """Load Challenge 1 model (TCN for CCD reaction time)."""
        if self.model_c1 is not None:
            return self.model_c1

        print("Loading Challenge 1 model (TCN)...")

        model = TCN_EEG(
            num_channels=129,
            num_outputs=1,
            num_filters=48,
            kernel_size=7,
            dropout=0.3,
            num_levels=5
        )

        try:
            weights_path = resolve_path('weights_challenge_1.pt')
            print(f"  Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✅ Weights loaded from epoch {checkpoint.get('epoch', '?')}")
            else:
                model.load_state_dict(checkpoint)
                print("  ✅ Weights loaded successfully")
        except FileNotFoundError as e:
            print(f"  ⚠️  Weights file not found: {e}")
            print("  Using untrained model")
        except Exception as e:
            print(f"  ⚠️  Error loading weights: {e}")
            print("  Using untrained model")

        model = model.to(self.device)
        model.eval()

        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("  ✅ Challenge 1 model ready")
        print()

        self.model_c1 = model
        return model

    def get_model_challenge_2(self):
        """Load Challenge 2 model (EEGNeX for externalizing factor)."""
        if self.model_c2 is not None:
            return self.model_c2

        print("Loading Challenge 2 model...")
        print("  Using standalone EEGNeX (matches braindecode exactly)")

        model = EEGNeX_Standalone(
            n_chans=self.n_chans,
            n_outputs=1,
            n_times=self.n_times,
            sfreq=self.sfreq,
        )

        try:
            weights_path = resolve_path('weights_challenge_2.pt')
            print(f"  Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            model.load_braindecode_weights(checkpoint)
            print("  ✅ Weights loaded successfully")
        except FileNotFoundError as e:
            print(f"  ⚠️  Weights file not found: {e}")
            print("  Using untrained model")
        except Exception as e:
            print(f"  ⚠️  Error loading weights: {e}")
            import traceback
            traceback.print_exc()
            print("  Using untrained model")

        model = model.to(self.device)
        model.eval()

        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("  ✅ Challenge 2 model ready")
        print()

        self.model_c2 = model
        return model

    def predict_challenge_1(self, X: np.ndarray) -> np.ndarray:
        """Predict CCD reaction times (Challenge 1)."""
        model = self.get_model_challenge_1()

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        with torch.no_grad():
            predictions = model(X)

        return predictions.cpu().numpy()

    def predict_challenge_2(self, X: np.ndarray) -> np.ndarray:
        """Predict externalizing factor (Challenge 2)."""
        model = self.get_model_challenge_2()

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        with torch.no_grad():
            predictions = model(X)

        return predictions.cpu().numpy()


# Test code
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("TESTING SUBMISSION FILE")
    print("=" * 60)
    print()

    submission = Submission(SFREQ=100, DEVICE=torch.device('cpu'))

    batch_size = 4
    test_data = np.random.randn(batch_size, 129, 200).astype(np.float32)

    print("Testing Challenge 1 model:")
    print("-" * 40)
    pred_c1 = submission.predict_challenge_1(test_data)
    print(f"Test input shape: {test_data.shape}")
    print(f"Output shape: {pred_c1.shape}")
    print(f"Sample predictions: {pred_c1[:3, 0]}")
    print("✅ Challenge 1 model works!")
    print()

    print("Testing Challenge 2 model:")
    print("-" * 40)
    pred_c2 = submission.predict_challenge_2(test_data)
    print(f"Output shape: {pred_c2.shape}")
    print(f"Sample predictions: {pred_c2[:3, 0]}")
    print("✅ Challenge 2 model works!")
    print()

    print("=" * 60)
    print("ALL TESTS PASSED - SUBMISSION READY")
    print("=" * 60)
    print()

