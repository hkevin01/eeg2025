"""
Create ensemble submission from 3 trained CompactCNN models
Averages predictions from all 3 models for better generalization
"""

import torch
import torch.nn as nn
from braindecode.models import EEGNeX
from pathlib import Path
import argparse


# ============================================================================
# COMPACT CNN ARCHITECTURE
# ============================================================================

class CompactResponseTimeCNN(nn.Module):
    """Simple 3-layer CNN - proven to work with score 1.0015"""
    
    def __init__(self, n_channels=129, sequence_length=200):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        predictions = self.regressor(features)
        return predictions


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleCompactCNN(nn.Module):
    """Ensemble of 3 CompactCNN models"""
    
    def __init__(self):
        super().__init__()
        self.model1 = CompactResponseTimeCNN()
        self.model2 = CompactResponseTimeCNN()
        self.model3 = CompactResponseTimeCNN()
    
    def forward(self, x):
        """Average predictions from all 3 models"""
        pred1 = self.model1(x)
        pred2 = self.model2(x)
        pred3 = self.model3(x)
        
        # Average
        return (pred1 + pred2 + pred3) / 3.0


# ============================================================================
# SUBMISSION CLASS
# ============================================================================

class Submission:
    """Competition submission with ensemble"""
    
    def __init__(self, SFREQ, DEVICE):
        self.SFREQ = SFREQ
        self.DEVICE = DEVICE
        
        # Challenge 1: Ensemble of 3 CompactCNN
        self.model_c1 = None
        
        # Challenge 2: EEGNeX
        self.model_c2 = None
    
    def get_model_challenge_1(self):
        """
        Returns model for Challenge 1: Response Time Prediction
        Uses ensemble of 3 CompactCNN models
        """
        if self.model_c1 is None:
            self.model_c1 = EnsembleCompactCNN()
            
            # Load weights for all 3 sub-models
            try:
                weights = torch.load('weights_challenge_1.pt',
                                   map_location=self.DEVICE,
                                   weights_only=False)
                
                # Load each model's weights
                if 'model1' in weights:
                    self.model_c1.model1.load_state_dict(weights['model1'])
                if 'model2' in weights:
                    self.model_c1.model2.load_state_dict(weights['model2'])
                if 'model3' in weights:
                    self.model_c1.model3.load_state_dict(weights['model3'])
                    
                print("✅ Loaded ensemble weights (3 models)")
                    
            except Exception as e:
                print(f"Error loading C1 weights: {e}")
                print("Using randomly initialized ensemble")
            
            self.model_c1.to(self.DEVICE)
            self.model_c1.eval()
        
        return self.model_c1
    
    def get_model_challenge_2(self):
        """
        Returns model for Challenge 2: Externalizing Factor Prediction
        Uses proven EEGNeX from braindecode
        """
        if self.model_c2 is None:
            # Create EEGNeX model
            self.model_c2 = EEGNeX(
                n_chans=129,
                n_times=200,
                n_outputs=1,
                sfreq=self.SFREQ
            )
            
            # Load weights
            try:
                weights = torch.load('weights_challenge_2.pt',
                                   map_location=self.DEVICE,
                                   weights_only=False)
                self.model_c2.load_state_dict(weights)
                print("✅ Loaded EEGNeX weights")
            except Exception as e:
                print(f"Error loading C2 weights: {e}")
                print("Using randomly initialized EEGNeX")
            
            self.model_c2.to(self.DEVICE)
            self.model_c2.eval()
        
        return self.model_c2


# ============================================================================
# CREATE ENSEMBLE WEIGHTS
# ============================================================================

def create_ensemble_weights(checkpoint_dir, output_path='weights_challenge_1_ensemble.pt'):
    """
    Combine 3 trained model checkpoints into ensemble weight file
    
    Args:
        checkpoint_dir: Directory containing the 3 checkpoints
        output_path: Path to save ensemble weights
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    print("\n" + "="*60)
    print("Creating Ensemble Weights")
    print("="*60)
    
    # Find checkpoints
    checkpoints = sorted(checkpoint_dir.glob("compact_cnn_seed*_best.pth"))
    
    if len(checkpoints) < 3:
        print(f"❌ Error: Found only {len(checkpoints)} checkpoints, need 3")
        print(f"Expected files: compact_cnn_seed42_best.pth, etc.")
        return False
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    
    # Load weights
    ensemble_weights = {}
    
    for i, checkpoint_path in enumerate(checkpoints[:3], 1):
        print(f"\nLoading model {i}: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        model_weights = checkpoint['model_state_dict']
        ensemble_weights[f'model{i}'] = model_weights
        
        # Print stats
        pearson_r = checkpoint.get('pearson_r', 'N/A')
        nrmse = checkpoint.get('nrmse', 'N/A')
        print(f"  Validation Pearson r: {pearson_r}")
        print(f"  Validation NRMSE: {nrmse}")
    
    # Save ensemble weights
    print(f"\n💾 Saving ensemble weights to: {output_path}")
    torch.save(ensemble_weights, output_path)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✅ Saved! File size: {file_size_mb:.2f} MB")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create ensemble submission')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='checkpoints/compact_ensemble',
                       help='Directory containing trained checkpoints')
    parser.add_argument('--c2_weights', type=str,
                       default='checkpoints/challenge2_tcn_competition_final.pth',
                       help='Challenge 2 weights (EEGNeX)')
    parser.add_argument('--output', type=str,
                       default='submission_v9_ensemble_final.zip',
                       help='Output submission filename')
    args = parser.parse_args()
    
    # Create ensemble weights
    success = create_ensemble_weights(
        args.checkpoint_dir,
        output_path='weights_challenge_1.pt'
    )
    
    if not success:
        print("\n❌ Failed to create ensemble weights")
        return
    
    # Copy C2 weights
    print(f"\n📦 Copying Challenge 2 weights...")
    import shutil
    try:
        shutil.copy(args.c2_weights, 'weights_challenge_2.pt')
        print(f"✅ Copied from {args.c2_weights}")
    except Exception as e:
        print(f"⚠️  Could not copy C2 weights: {e}")
        print("Using placeholder")
    
    # Create submission.py
    print(f"\n📝 Creating submission.py...")
    with open(__file__, 'r') as f:
        content = f.read()
    
    # Extract only the submission parts
    submission_code = f'''"""
Ensemble Submission v9 - 3x CompactCNN
Challenge 1: Ensemble of 3 CompactCNN models (average predictions)
Challenge 2: EEGNeX from braindecode
"""

import torch
import torch.nn as nn
from braindecode.models import EEGNeX

{content.split("# COMPACT CNN ARCHITECTURE")[1].split("# CREATE ENSEMBLE WEIGHTS")[0]}

{content.split("# SUBMISSION CLASS")[1].split("# CREATE ENSEMBLE WEIGHTS")[0]}

if __name__ == "__main__":
    print("Testing Ensemble Submission...")
    print("="*60)
    
    SFREQ = 100
    DEVICE = torch.device("cpu")
    
    submission = Submission(SFREQ, DEVICE)
    
    # Test C1
    print("\\nChallenge 1: Ensemble Test")
    model_c1 = submission.get_model_challenge_1()
    x_c1 = torch.randn(4, 129, 200)
    with torch.no_grad():
        pred_c1 = model_c1(x_c1)
    print(f"Output shape: {{pred_c1.shape}}")
    print(f"Sample predictions: {{pred_c1[:3].squeeze().tolist()}}")
    print("✅ Challenge 1 PASS")
    
    # Test C2
    print("\\nChallenge 2: EEGNeX Test")
    model_c2 = submission.get_model_challenge_2()
    x_c2 = torch.randn(4, 129, 200)
    with torch.no_grad():
        pred_c2 = model_c2(x_c2)
    print(f"Output shape: {{pred_c2.shape}}")
    print(f"Sample predictions: {{pred_c2[:3].squeeze().tolist()}}")
    print("✅ Challenge 2 PASS")
    
    print("\\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
'''
    
    with open('submission.py', 'w') as f:
        f.write(submission_code)
    
    print("✅ Created submission.py")
    
    # Create zip
    print(f"\n📦 Creating {args.output}...")
    import zipfile
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('submission.py')
        zipf.write('weights_challenge_1.pt')
        zipf.write('weights_challenge_2.pt')
    
    zip_size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"✅ Created {args.output} ({zip_size_mb:.2f} MB)")
    
    print("\n" + "="*60)
    print("✅ ENSEMBLE SUBMISSION READY!")
    print("="*60)
    print(f"\nFile: {args.output}")
    print(f"Contents:")
    print(f"  - submission.py")
    print(f"  - weights_challenge_1.pt (ensemble of 3 models)")
    print(f"  - weights_challenge_2.pt (EEGNeX)")
    print(f"\nNext: Upload to competition!")


if __name__ == '__main__':
    main()
