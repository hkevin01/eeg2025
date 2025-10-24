"""
EEG Foundation Challenge 2025 - Improved Submission
====================================================
Both challenges use EEGNeX architecture with anti-overfitting measures
Platform: CPU/GPU compatible
Dependencies: torch, braindecode (available on platform)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def resolve_path(name="model_file_name"):
    """Resolve file path - works for local testing and competition platform."""
    search_paths = [
        f"/app/input/res/{name}",
        f"/app/input/{name}",
        f"{name}",
        str(Path(__file__).parent / name),
    ]

    for path in search_paths:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        f"Could not find {name} in: {', '.join(search_paths)}"
    )


class Submission:
    """Main submission class - both challenges use EEGNeX."""

    def __init__(self, SFREQ: int, DEVICE):
        """Initialize submission with device and constants.

        Args:
            SFREQ: Sampling frequency (Hz)
            DEVICE: Device for inference (string or torch.device)
        """
        # Device handling
        if isinstance(DEVICE, str):
            self.device = torch.device(DEVICE)
        elif isinstance(DEVICE, torch.device):
            self.device = DEVICE
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("=" * 70)
        print("🧠 EEG Foundation Challenge 2025 - Improved Submission")
        print("=" * 70)
        print(f"Device: {self.device.type.upper()}")

        if self.device.type == "cuda":
            if torch.cuda.is_available():
                try:
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"GPU Memory: {memory_gb:.1f} GB")
                except:
                    print("GPU: Available (details unavailable)")
            else:
                print("⚠️  CUDA requested but not available - using CPU")
                self.device = torch.device("cpu")

        # EEG parameters
        self.sfreq = SFREQ
        self.n_chans = 129
        self.n_times = int(2 * SFREQ)  # 2-second windows

        print(f"Sampling Frequency: {self.sfreq} Hz")
        print(f"Input shape: (batch, {self.n_chans} channels, {self.n_times} timepoints)")
        print(f"Architecture: EEGNeX (both challenges)")
        print("=" * 70)
        print()

        # Models (lazy loading)
        self.model_c1 = None
        self.model_c2 = None

    def get_model_challenge_1(self):
        """Load Challenge 1 model - EEGNeX for response time prediction.

        Returns:
            torch.nn.Module: Trained EEGNeX model
        """
        if self.model_c1 is not None:
            return self.model_c1

        print("📦 Loading Challenge 1 Model")
        print("-" * 70)
        print("Task: Response Time Prediction (Contrast Change Detection)")
        print("Architecture: EEGNeX")
        print("Training: Anti-overfitting strategy (augmentation + regularization)")
        print()

        try:
            # Import braindecode (available on competition platform)
            from braindecode.models import EEGNeX

            # Create model
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,  # Response time (single value)
                sfreq=self.sfreq,
            )

            # Load trained weights
            weights_path = resolve_path('weights_challenge_1.pt')
            print(f"Loading weights: {Path(weights_path).name}")

            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Handle both checkpoint format and direct state_dict format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)

            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()

            # Model info
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}")
            print("✅ Challenge 1 model ready")
            print()

            self.model_c1 = model
            return model

        except FileNotFoundError as e:
            print(f"❌ Weights file not found: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading Challenge 1 model: {e}")
            raise

    def get_model_challenge_2(self):
        """Load Challenge 2 model - EEGNeX for externalizing factor prediction.

        Returns:
            torch.nn.Module: Trained EEGNeX model
        """
        if self.model_c2 is not None:
            return self.model_c2

        print("📦 Loading Challenge 2 Model")
        print("-" * 70)
        print("Task: Externalizing Factor Prediction")
        print("Architecture: EEGNeX")
        print("Training: Anti-overfitting strategy (augmentation + regularization)")
        print()

        try:
            # Import braindecode (available on competition platform)
            from braindecode.models import EEGNeX

            # Create model
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,  # Externalizing factor (single value)
                sfreq=self.sfreq,
            )

            # Load trained weights
            weights_path = resolve_path('weights_challenge_2.pt')
            print(f"Loading weights: {Path(weights_path).name}")

            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Handle both checkpoint format and direct state_dict format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()

            # Model info
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}")
            print("✅ Challenge 2 model ready")
            print()

            self.model_c2 = model
            return model

        except FileNotFoundError as e:
            print(f"❌ Weights file not found: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading Challenge 2 model: {e}")
            raise

    def preprocess_eeg(self, eeg_data: np.ndarray) -> torch.Tensor:
        """Preprocess EEG data for inference.

        Args:
            eeg_data: numpy array of shape (batch, channels, timepoints)

        Returns:
            torch.Tensor: Preprocessed EEG tensor
        """
        # Convert to tensor
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.from_numpy(eeg_data).float()

        # Ensure correct shape
        if eeg_data.ndim == 2:
            eeg_data = eeg_data.unsqueeze(0)  # Add batch dimension

        # Move to device
        eeg_data = eeg_data.to(self.device)

        return eeg_data

    def challenge_1(self, eeg_data: np.ndarray) -> np.ndarray:
        """Challenge 1: Predict response time from EEG.

        Args:
            eeg_data: numpy array of shape (batch, 129, 200) or (129, 200)

        Returns:
            numpy array of predicted response times (batch,) or scalar
        """
        model = self.get_model_challenge_1()

        # Preprocess
        eeg_tensor = self.preprocess_eeg(eeg_data)

        # Inference
        with torch.no_grad():
            predictions = model(eeg_tensor)

        # Convert to numpy
        predictions = predictions.cpu().numpy().squeeze()

        return predictions

    def challenge_2(self, eeg_data: np.ndarray) -> np.ndarray:
        """Challenge 2: Predict externalizing factor from EEG.

        Args:
            eeg_data: numpy array of shape (batch, 129, 200) or (129, 200)

        Returns:
            numpy array of predicted externalizing factors (batch,) or scalar
        """
        model = self.get_model_challenge_2()

        # Preprocess
        eeg_tensor = self.preprocess_eeg(eeg_data)

        # Inference
        with torch.no_grad():
            predictions = model(eeg_tensor)

        # Convert to numpy
        predictions = predictions.cpu().numpy().squeeze()

        return predictions


# Competition entry point
def main():
    """Test submission locally."""
    print("Testing submission locally...")
    print()

    # Initialize submission (mimic competition environment)
    SFREQ = 100  # 100 Hz sampling rate
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    submission = Submission(SFREQ=SFREQ, DEVICE=DEVICE)

    # Test Challenge 1
    print("🧪 Testing Challenge 1...")
    test_eeg_c1 = np.random.randn(2, 129, 200).astype(np.float32)
    predictions_c1 = submission.challenge_1(test_eeg_c1)
    print(f"Input shape: {test_eeg_c1.shape}")
    print(f"Output shape: {predictions_c1.shape}")
    print(f"Sample predictions: {predictions_c1}")
    print("✅ Challenge 1 working")
    print()

    # Test Challenge 2
    print("🧪 Testing Challenge 2...")
    test_eeg_c2 = np.random.randn(2, 129, 200).astype(np.float32)
    predictions_c2 = submission.challenge_2(test_eeg_c2)
    print(f"Input shape: {test_eeg_c2.shape}")
    print(f"Output shape: {predictions_c2.shape}")
    print(f"Sample predictions: {predictions_c2}")
    print("✅ Challenge 2 working")
    print()

    print("=" * 70)
    print("✅ All tests passed! Submission is ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
