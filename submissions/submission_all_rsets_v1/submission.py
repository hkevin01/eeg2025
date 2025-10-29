"""
EEG Foundation Challenge 2025 - v10 Submission (FINAL)
- Challenge 1: CompactCNN cross-R-set ensemble (best NRMSE 0.1625, Pearson r≈0.20)
- Challenge 2: EEGNeX from quick_fix (score 1.0087)
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


class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction (Challenge 1)
    Trained Oct 27-28 with cross-R-set folds (R1,R2,R3 ➜ R4 best)
    Validation: NRMSE=0.1625 (competition metric), Pearson r≈0.20
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


class Submission:
    """EEG 2025 Competition Submission - v10 FINAL"""
    
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.n_chans = 129
        self.n_times = 200
        self.model_c1 = None
        self.model_c2 = None
        
        print("\n" + "="*70)
        print("🧠 EEG Foundation Challenge 2025 - v10 FINAL Submission")
        print("="*70)
        print("Challenge 1: CompactCNN cross-R-set model (dropout 0.2)")
        print("Challenge 2: EEGNeX (from quick_fix)")
        print()
    
    def get_model_challenge_1(self):
        """Load Challenge 1 model - Trained CompactCNN"""
        if self.model_c1 is not None:
            return self.model_c1
        
        print("📦 Loading Challenge 1 Model")
        print("-" * 70)
        print("Task: Response Time Prediction")
        print("Architecture: CompactCNN (3 conv layers, 75K params)")
        print("Training: Cross-R-set folds (R1,R2,R3 ➜ R4 best)")
        print("Metrics: NRMSE 0.1625 | Pearson r≈0.20")
        print()
        
        try:
            model = CompactResponseTimeCNN()
            weight_candidates = [
                'compact_cnn_c1_cross_r123_val4_state.pt',
                'weights_challenge_1.pt',
            ]

            weights_path = None
            for candidate in weight_candidates:
                try:
                    weights_path = resolve_path(candidate)
                    break
                except FileNotFoundError:
                    continue

            if weights_path is None:
                raise FileNotFoundError(
                    "Could not locate Challenge 1 weights. Searched: "
                    + ", ".join(weight_candidates)
                )

            print(f"Loading weights: {Path(weights_path).name}")
            
            weights = torch.load(weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(weights)
            
            model = model.to(self.device)
            model.eval()
            
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}")
            print("✅ Challenge 1 model ready")
            print()
            
            self.model_c1 = model
            return model
        except Exception as e:
            print(f"❌ ERROR: {e}")
            raise
    
    def get_model_challenge_2(self):
        """Load Challenge 2 model - EEGNeX"""
        if self.model_c2 is not None:
            return self.model_c2
        
        print("📦 Loading Challenge 2 Model")
        print("-" * 70)
        print("Task: Externalizing Factor Prediction")
        print("Architecture: EEGNeX")
        print("Score: 1.0087 (from quick_fix)")
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
        except Exception as e:
            print(f"❌ ERROR: {e}")
            raise
    
    def __call__(self, X_test_c1, X_test_c2):
        """Make predictions on test sets"""
        with torch.no_grad():
            # Challenge 1
            if X_test_c1 is not None and len(X_test_c1) > 0:
                model_c1 = self.get_model_challenge_1()
                X_c1 = torch.tensor(X_test_c1, dtype=torch.float32, device=self.device)
                pred_c1 = model_c1(X_c1).cpu().numpy()
                print(f"Challenge 1: Generated {len(pred_c1)} predictions")
            else:
                pred_c1 = None
            
            # Challenge 2
            if X_test_c2 is not None and len(X_test_c2) > 0:
                model_c2 = self.get_model_challenge_2()
                X_c2 = torch.tensor(X_test_c2, dtype=torch.float32, device=self.device)
                pred_c2 = model_c2(X_c2).cpu().numpy()
                print(f"Challenge 2: Generated {len(pred_c2)} predictions")
            else:
                pred_c2 = None
        
        return pred_c1, pred_c2


# Test code
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING SUBMISSION")
    print("="*70)
    
    submission = Submission(SFREQ=100, DEVICE='cpu')
    
    # Test C1
    print("\nTesting Challenge 1...")
    model_c1 = submission.get_model_challenge_1()
    x_c1 = torch.randn(4, 129, 200)
    with torch.no_grad():
        pred_c1 = model_c1(x_c1)
    print(f"Output shape: {pred_c1.shape}")
    print(f"Sample predictions: {pred_c1[:3].squeeze().tolist()}")
    print("✅ Challenge 1 PASS\n")
    
    # Test C2
    print("Testing Challenge 2...")
    model_c2 = submission.get_model_challenge_2()
    x_c2 = torch.randn(4, 129, 200)
    with torch.no_grad():
        pred_c2 = model_c2(x_c2)
    print(f"Output shape: {pred_c2.shape}")
    print(f"Sample predictions: {pred_c2[:3].squeeze().tolist()}")
    print("✅ Challenge 2 PASS\n")
    
    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
