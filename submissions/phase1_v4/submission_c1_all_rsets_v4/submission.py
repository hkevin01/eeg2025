"""
EEG Foundation Challenge 2025 - Phase 1 Subject-Aware Submission
- Challenge 1: CompactCNN with subject-aware training (Val NRMSE ~0.1766)
- Challenge 2: EEGNeX from quick_fix (Test score 1.00867)
"""

import torch
import torch.nn as nn
from pathlib import Path


def select_device(verbose=True):
    """Select best available device with GPU/ROCm fallback."""
    device_info = ""

    if torch.cuda.is_available():
        try:
            test_tensor = torch.randn(1, 1, device='cuda')
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
        except Exception as exc:
            if verbose:
                print(f"⚠️  GPU available but failed health check: {exc}")
                print("   Falling back to CPU...")

    device = torch.device('cpu')
    device_info = "CPU"

    if verbose:
        print("✅ Using CPU")
        print(f"   PyTorch version: {torch.__version__}")

    return device, device_info


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
    raise FileNotFoundError(f"Could not find {filename}")


class CompactCNN(nn.Module):
    """Compact CNN for response time prediction (Challenge 1)
    Trained with subject-aware validation (Oct 29, 2025)
    Validation: NRMSE=0.1766 with zero subject overlap
    Architecture: 3 conv layers (kernel=5) + 3 FC layers, 75K params
    """
    def __init__(self, in_channels=129, time_steps=200):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze()


class Submission:
    """EEG 2025 Competition Submission - Phase 1 Subject-Aware"""
    
    def __init__(self, SFREQ, DEVICE):
        # Device handling - convert to torch.device if needed
        if isinstance(DEVICE, str):
            self.device = torch.device(DEVICE)
        elif isinstance(DEVICE, torch.device):
            self.device = DEVICE
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sfreq = SFREQ
        self.n_chans = 129
        self.n_times = int(2 * SFREQ)  # Should be 200 for SFREQ=100
        self.model_c1 = None
        self.model_c2 = None
        
        print("\n" + "="*70)
        print("🧠 EEG Foundation Challenge 2025 - Phase 1 Subject-Aware")
        print("="*70)
        print(f"Device: {self.device.type.upper()}")
        print(f"Sampling Frequency: {self.sfreq} Hz")
        print(f"Input shape: (batch, {self.n_chans} channels, {self.n_times} timepoints)")
        print("Challenge 1: CompactCNN with subject-aware validation")
        print("Challenge 2: EEGNeX (from quick_fix, score 1.00867)")
        print()
    
    def get_model_challenge_1(self):
        """Load Challenge 1 model - Subject-Aware CompactCNN"""
        if self.model_c1 is not None:
            return self.model_c1
        
        print("📦 Loading Challenge 1 Model")
        print("-" * 70)
        print("Task: Response Time Prediction")
        print("Architecture: CompactCNN (3 conv + 3 FC layers, 83K params)")
        print("Training: Subject-aware validation (zero overlap)")
        print("Metrics: Val NRMSE ~0.1766")
        print()
        
        try:
            model = CompactCNN(in_channels=self.n_chans, time_steps=self.n_times)
            
            weights_path = resolve_path('weights_challenge_1.pt')
            print(f"Loading weights: {Path(weights_path).name}")
            
            weights = torch.load(weights_path, map_location=self.device, weights_only=False)
            
            # Handle both checkpoint format and direct state_dict format
            if isinstance(weights, dict) and 'model_state_dict' in weights:
                state_dict = weights['model_state_dict']
            else:
                state_dict = weights
            
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            
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
            print(f"❌ ERROR loading C1 model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_model_challenge_2(self):
        """Load Challenge 2 model - EEGNeX"""
        if self.model_c2 is not None:
            return self.model_c2
        
        print("📦 Loading Challenge 2 Model")
        print("-" * 70)
        print("Task: Externalizing Factor Prediction")
        print("Architecture: EEGNeX")
        print("Score: 1.00867 (from quick_fix)")
        print()
        
        try:
            from braindecode.models import EEGNeX
            
            model = EEGNeX(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=1,
                sfreq=self.sfreq,
            )
            
            weights_path = resolve_path('weights_challenge_2.pt')
            print(f"Loading weights: {Path(weights_path).name}")
            
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            
            # Handle checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            
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
            print(f"❌ ERROR loading C2 model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def __call__(self, X, challenge):
        """
        Make predictions for challenge 1 or 2.
        
        Args:
            X: EEG data, shape (batch, n_chans, n_times)
            challenge: 1 for response time, 2 for externalizing factor
        
        Returns:
            predictions: shape (batch,)
        """
        if challenge == 1:
            model = self.get_model_challenge_1()
        elif challenge == 2:
            model = self.get_model_challenge_2()
        else:
            raise ValueError(f"Unknown challenge: {challenge}")
        
        # Ensure X is on the correct device
        X = X.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = model(X)
            
            # Handle different output shapes
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
        
        return predictions
