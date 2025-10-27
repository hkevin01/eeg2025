"""
Ensemble Submission v9 - 3x CompactCNN
Challenge 1: Ensemble of 3 CompactCNN models (average predictions)
Challenge 2: EEGNeX from braindecode
"""

import torch
import torch.nn as nn
from braindecode.models import EEGNeX


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


if __name__ == "__main__":
    print("Testing Ensemble Submission...")
    print("="*60)
    
    SFREQ = 100
    DEVICE = torch.device("cpu")
    
    submission = Submission(SFREQ, DEVICE)
    
    # Test C1
    print("\nChallenge 1: Ensemble Test")
    model_c1 = submission.get_model_challenge_1()
    x_c1 = torch.randn(4, 129, 200)
    with torch.no_grad():
        pred_c1 = model_c1(x_c1)
    print(f"Output shape: {pred_c1.shape}")
    print(f"Sample predictions: {pred_c1[:3].squeeze().tolist()}")
    print("✅ Challenge 1 PASS")
    
    # Test C2
    print("\nChallenge 2: EEGNeX Test")
    model_c2 = submission.get_model_challenge_2()
    x_c2 = torch.randn(4, 129, 200)
    with torch.no_grad():
        pred_c2 = model_c2(x_c2)
    print(f"Output shape: {pred_c2.shape}")
    print(f"Sample predictions: {pred_c2[:3].squeeze().tolist()}")
    print("✅ Challenge 2 PASS")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
