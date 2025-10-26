"""
EEG Foundation Challenge 2025 - Combined SAM Submission (ROBUST)
Both challenges use SAM-trained EEGNeX models

Challenge 1: SAM EEGNeX (Val NRMSE: 0.3008, 70% improvement)
Challenge 2: SAM EEGNeX (Val NRMSE: 0.2042, 80% improvement)
Expected overall: 0.25-0.45 (60-75% improvement over baseline)

FIXED: Added robust error handling and logging
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import traceback


def resolve_path(filename):
    """Find file in competition or local environment"""
    search_paths = [
        f"/app/input/res/{filename}",
        f"/app/input/{filename}",
        filename,
        str(Path(__file__).parent / filename),
    ]

    print(f"🔍 Searching for {filename}:")
    for path in search_paths:
        print(f"  Checking: {path}")
        if Path(path).exists():
            print(f"  ✅ Found at: {path}")
            return path

    print(f"❌ File not found in any location")
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
        print("\n" + "="*70)
        print("🧠 EEG Foundation Challenge 2025 - SAM Combined Submission (ROBUST)")
        print("="*70)

        try:
            self.sfreq = SFREQ
            self.device = DEVICE
            self.n_chans = 129
            self.n_times = 200  # 2 seconds at 100 Hz
            self.model_c1 = None
            self.model_c2 = None

            print(f"✅ Device: {DEVICE}")
            if DEVICE == 'cuda' and torch.cuda.is_available():
                print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ Sampling Frequency: {SFREQ} Hz")
            print(f"✅ Input shape: (batch, {self.n_chans} channels, {self.n_times} timepoints)")
            print("\n🎯 SAM Optimizer Trained Models")
            print("  C1: EEGNeX + SAM (Val NRMSE: 0.3008)")
            print("  C2: EEGNeX + SAM (Val NRMSE: 0.2042)")
            print("="*70)
            print()

        except Exception as e:
            print(f"\n❌ INITIALIZATION ERROR: {e}")
            traceback.print_exc()
            raise

    def get_model_challenge_1(self):
        """Load Challenge 1 model - SAM-trained EEGNeX (Val NRMSE: 0.3008)"""
        if self.model_c1 is not None:
            print("✅ Returning cached Challenge 1 model")
            return self.model_c1

        print("\n📦 Loading Challenge 1 Model")
        print("-" * 70)
        print("Task: Response Time Prediction")
        print("Architecture: EEGNeX + SAM Optimizer")
        print("Validation NRMSE: 0.3008 (70% better than baseline!)")
        print()

        try:
            print("Step 1: Importing braindecode...")
            try:
                from braindecode.models import EEGNeX
                print("✅ braindecode imported successfully")
            except ImportError as e:
                print(f"❌ Failed to import braindecode: {e}")
                print("Available packages:")
                import pip
                installed_packages = [pkg.project_name for pkg in pip.get_installed_distributions()]
                print(installed_packages[:20])
                raise

            print("\nStep 2: Creating model...")
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,
                sfreq=self.sfreq,
            )
            print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

            print("\nStep 3: Finding weights file...")
            weights_path = resolve_path('weights_challenge_1_sam.pt')

            print("\nStep 4: Loading checkpoint...")
            # Try safe loading first
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
                print("✅ Loaded with weights_only=True")
            except:
                print("⚠️  weights_only=True failed, trying weights_only=False...")
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                print("✅ Loaded with weights_only=False")

            print(f"   Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"   Checkpoint keys: {list(checkpoint.keys())}")

            print("\nStep 5: Extracting state dict...")
            # Handle checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✅ Extracted 'model_state_dict' with {len(state_dict)} entries")
            else:
                state_dict = checkpoint
                print(f"✅ Using checkpoint directly with {len(state_dict)} entries")

            print("\nStep 6: Loading weights into model...")
            model.load_state_dict(state_dict)
            print("✅ Weights loaded successfully")

            print("\nStep 7: Moving to device and setting eval mode...")
            model = model.to(self.device)
            model.eval()
            print(f"✅ Model on {self.device} in eval mode")

            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n✅ Challenge 1 model ready ({n_params:,} parameters)")
            print()

            self.model_c1 = model
            return model

        except Exception as e:
            print(f"\n❌ ERROR LOADING CHALLENGE 1 MODEL")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise

    def get_model_challenge_2(self):
        """Load Challenge 2 model - SAM-trained EEGNeX (Val NRMSE: 0.2042)"""
        if self.model_c2 is not None:
            print("✅ Returning cached Challenge 2 model")
            return self.model_c2

        print("\n📦 Loading Challenge 2 Model")
        print("-" * 70)
        print("Task: Externalizing Factor Prediction")
        print("Architecture: EEGNeX + SAM Optimizer")
        print("Validation NRMSE: 0.2042 (80% better than baseline!)")
        print()

        try:
            print("Step 1: Importing braindecode...")
            from braindecode.models import EEGNeX
            print("✅ braindecode imported")

            print("\nStep 2: Creating model...")
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,
                sfreq=self.sfreq,
            )
            print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

            print("\nStep 3: Finding weights file...")
            weights_path = resolve_path('weights_challenge_2_sam.pt')

            print("\nStep 4: Loading checkpoint...")
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
                print("✅ Loaded with weights_only=True")
            except:
                print("⚠️  weights_only=True failed, trying weights_only=False...")
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                print("✅ Loaded with weights_only=False")

            print("\nStep 5: Extracting state dict...")
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✅ Extracted 'model_state_dict' with {len(state_dict)} entries")
            else:
                state_dict = checkpoint
                print(f"✅ Using checkpoint directly with {len(state_dict)} entries")

            print("\nStep 6: Loading weights into model...")
            model.load_state_dict(state_dict)
            print("✅ Weights loaded successfully")

            print("\nStep 7: Moving to device and setting eval mode...")
            model = model.to(self.device)
            model.eval()
            print(f"✅ Model on {self.device} in eval mode")

            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n✅ Challenge 2 model ready ({n_params:,} parameters)")
            print()

            self.model_c2 = model
            return model

        except Exception as e:
            print(f"\n❌ ERROR LOADING CHALLENGE 2 MODEL")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise

    def get_model_challenge(self, challenge_number):
        """Get model for specified challenge

        Args:
            challenge_number (int): 1 or 2

        Returns:
            nn.Module: Model for the challenge
        """
        try:
            if challenge_number == 1:
                return self.get_model_challenge_1()
            elif challenge_number == 2:
                return self.get_model_challenge_2()
            else:
                raise ValueError(f"Invalid challenge number: {challenge_number}")
        except Exception as e:
            print(f"\n❌ ERROR in get_model_challenge({challenge_number})")
            traceback.print_exc()
            raise

    def challenge_1(self, X):
        """Challenge 1: Predict response time from EEG

        Args:
            X (torch.Tensor): Input EEG data [batch, channels, timepoints]

        Returns:
            torch.Tensor: Predictions [batch, 1] or [batch,]
        """
        try:
            print(f"\n🔮 Challenge 1: Response Time Prediction")
            print(f"   Input shape: {X.shape}")

            # Get model
            model = self.get_model_challenge_1()

            # Ensure X is on correct device
            X = X.to(self.device)
            print(f"   Input on device: {X.device}")

            # Make predictions
            with torch.no_grad():
                predictions = model(X).squeeze(-1)

            print(f"   Output shape: {predictions.shape}")
            print(f"✅ Challenge 1 predictions complete")

            return predictions

        except Exception as e:
            print(f"\n❌ ERROR in challenge_1")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print(f"Input shape: {X.shape if 'X' in locals() else 'Not defined'}")
            traceback.print_exc()
            raise

    def challenge_2(self, X):
        """Challenge 2: Predict externalizing factor from EEG

        Args:
            X (torch.Tensor): Input EEG data [batch, channels, timepoints]

        Returns:
            torch.Tensor: Predictions [batch, 1] or [batch,]
        """
        try:
            print(f"\n🔮 Challenge 2: Externalizing Factor Prediction")
            print(f"   Input shape: {X.shape}")

            # Get model
            model = self.get_model_challenge_2()

            # Ensure X is on correct device
            X = X.to(self.device)
            print(f"   Input on device: {X.device}")

            # Make predictions
            with torch.no_grad():
                predictions = model(X).squeeze(-1)

            print(f"   Output shape: {predictions.shape}")
            print(f"✅ Challenge 2 predictions complete")

            return predictions

        except Exception as e:
            print(f"\n❌ ERROR in challenge_2")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print(f"Input shape: {X.shape if 'X' in locals() else 'Not defined'}")
            traceback.print_exc()
            raise
