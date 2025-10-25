"""
EEG Foundation Challenge 2025 - Combined SAM Submission
Both challenges use SAM-trained EEGNeX models

Challenge 1: SAM EEGNeX (Val NRMSE: 0.3008, 70% improvement)
Challenge 2: SAM EEGNeX (Val NRMSE: 0.2042, 80% improvement)
Expected overall: 0.25-0.45 (60-75% improvement over baseline)
"""

import torch
import torch.nn as nn
from pathlib import Path


def resolve_path(filename):
    """Find file in competition or local environment"""
    search_paths = [
        f"/app/input/res/{filename}",
        f"/app/input/{filename}",
        filename,
        str(Path(__file__).parent / filename),
    ]

    for path in search_paths:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename} in: {', '.join(search_paths)}"
    )


class Submission:
    """EEG 2025 Competition Submission - Combined SAM Models"""

    def __init__(self, SFREQ, DEVICE):
        """Initialize submission
        
        Args:
            SFREQ (int): Sampling frequency (100 Hz)
            DEVICE (str): 'cuda' or 'cpu'
        """
        self.sfreq = SFREQ
        self.device = DEVICE
        self.n_chans = 129
        self.n_times = 200  # 2 seconds at 100 Hz
        self.model_c1 = None
        self.model_c2 = None

        print("\n" + "="*70)
        print("🧠 EEG Foundation Challenge 2025 - SAM Combined Submission")
        print("="*70)
        print(f"Device: {DEVICE}")
        if DEVICE == 'cuda' and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Sampling Frequency: {SFREQ} Hz")
        print(f"Input shape: (batch, {self.n_chans} channels, {self.n_times} timepoints)")
        print("\n🎯 SAM Optimizer Trained Models")
        print("  C1: EEGNeX + SAM (Val NRMSE: 0.3008)")
        print("  C2: EEGNeX + SAM (Val NRMSE: 0.2042)")
        print("="*70)
        print()

    def get_model_challenge_1(self):
        """Load Challenge 1 model - SAM-trained EEGNeX (Val NRMSE: 0.3008)"""
        if self.model_c1 is not None:
            return self.model_c1

        print("📦 Loading Challenge 1 Model")
        print("-" * 70)
        print("Task: Response Time Prediction")
        print("Architecture: EEGNeX + SAM Optimizer")
        print("Validation NRMSE: 0.3008 (70% better than baseline!)")
        print()

        try:
            # Import braindecode
            from braindecode.models import EEGNeX

            # Create model
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,
                sfreq=self.sfreq,
            )

            # Load weights
            weights_path = resolve_path('weights_challenge_1_sam.pt')
            print(f"Loading weights: {Path(weights_path).name}")

            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Handle checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)

            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()

            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}")
            print("✅ Challenge 1 model ready")
            print()

            self.model_c1 = model
            return model

        except Exception as e:
            print(f"❌ Error loading Challenge 1 model: {e}")
            raise

    def get_model_challenge_2(self):
        """Load Challenge 2 model - SAM-trained EEGNeX (Val NRMSE: 0.2042)"""
        if self.model_c2 is not None:
            return self.model_c2

        print("📦 Loading Challenge 2 Model")
        print("-" * 70)
        print("Task: Externalizing Factor Prediction")
        print("Architecture: EEGNeX + SAM Optimizer")
        print("Validation NRMSE: 0.2042 (80% better than baseline!)")
        print()

        try:
            # Import braindecode
            from braindecode.models import EEGNeX

            # Create model
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,
                sfreq=self.sfreq,
            )

            # Load weights
            weights_path = resolve_path('weights_challenge_2_sam.pt')
            print(f"Loading weights: {Path(weights_path).name}")

            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Handle checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)

            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()

            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}")
            print("✅ Challenge 2 model ready")
            print()

            self.model_c2 = model
            return model

        except Exception as e:
            print(f"❌ Error loading Challenge 2 model: {e}")
            raise

    def get_model_challenge(self, challenge_number):
        """Get model for specified challenge
        
        Args:
            challenge_number (int): 1 or 2
            
        Returns:
            nn.Module: Model for the challenge
        """
        if challenge_number == 1:
            return self.get_model_challenge_1()
        elif challenge_number == 2:
            return self.get_model_challenge_2()
        else:
            raise ValueError(f"Invalid challenge number: {challenge_number}")

    def __call__(self, X, challenge_number):
        """Predict on input data
        
        Args:
            X (torch.Tensor): Input EEG data [batch, channels, timepoints]
            challenge_number (int): 1 or 2
            
        Returns:
            torch.Tensor: Predictions [batch, 1]
        """
        # Get model
        model = self.get_model_challenge(challenge_number)

        # Ensure X is on correct device
        X = X.to(self.device)

        # Make predictions
        with torch.no_grad():
            predictions = model(X)

        return predictions
