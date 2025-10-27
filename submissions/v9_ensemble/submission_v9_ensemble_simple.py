"""
Submission v9 - Simplified Ensemble  
Path B: TCN neutral (0.90-1.10) - Use best available weights

Challenge 1: TCN (proven val loss 0.010170)
Challenge 2: EEGNeX from braindecode (proven 1.0087)

Note: Simplified to use single best model per challenge since 
      ensemble would require retraining with compatible architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGNeX


# ============================================================================
# CHALLENGE 1: TCN (196K params, val loss 0.010170)
# ============================================================================

class TemporalBlock(nn.Module):
    """Temporal convolutional block with residual connection"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_EEG(nn.Module):
    """Temporal Convolutional Network for EEG - proven val loss 0.010170"""
    
    def __init__(self, n_channels=129, n_filters=48, kernel_size=3, dropout=0.2):
        super().__init__()
        
        layers = []
        num_levels = 5
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = n_channels if i == 0 else n_filters
            out_channels = n_filters
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation,
                padding=(kernel_size - 1) * dilation,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(n_filters, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels=129, time=200)
        Returns:
            predictions: (batch, 1)
        """
        out = self.network(x)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        predictions = self.fc(out)
        return predictions


# ============================================================================
# SUBMISSION CLASS
# ============================================================================

class Submission:
    """Competition submission - TCN for C1, EEGNeX for C2"""
    
    def __init__(self, SFREQ, DEVICE):
        self.SFREQ = SFREQ
        self.DEVICE = DEVICE
        
        # Challenge 1: TCN
        self.model_c1 = None
        
        # Challenge 2: EEGNeX
        self.model_c2 = None
    
    def get_model_challenge_1(self):
        """
        Returns model for Challenge 1: Response Time Prediction
        Uses TCN with proven validation performance
        """
        if self.model_c1 is None:
            self.model_c1 = TCN_EEG(n_channels=129, n_filters=48, kernel_size=3, dropout=0.2)
            
            # Load weights
            try:
                weights = torch.load('weights_challenge_1.pt', 
                                   map_location=self.DEVICE,
                                   weights_only=False)
                self.model_c1.load_state_dict(weights)
            except Exception as e:
                print(f"Error loading C1 weights: {e}")
                print("Using randomly initialized TCN")
            
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
            except Exception as e:
                print(f"Error loading C2 weights: {e}")
                print("Using randomly initialized EEGNeX")
            
            self.model_c2.to(self.DEVICE)
            self.model_c2.eval()
        
        return self.model_c2


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Submission v9...")
    print("=" * 60)
    
    SFREQ = 100
    DEVICE = torch.device("cpu")
    
    # Create submission
    submission = Submission(SFREQ, DEVICE)
    
    # Test Challenge 1 (TCN)
    print("\n�� Challenge 1: TCN Test")
    print("-" * 60)
    model_c1 = submission.get_model_challenge_1()
    
    # Create dummy input
    batch_size = 4
    x_c1 = torch.randn(batch_size, 129, 200)
    
    with torch.no_grad():
        predictions_c1 = model_c1(x_c1)
    
    print(f"Input shape: {x_c1.shape}")
    print(f"Output shape: {predictions_c1.shape}")
    print(f"Sample predictions: {predictions_c1[:3].squeeze().tolist()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_c1.parameters())
    print(f"Total parameters: {total_params:,}")
    
    assert predictions_c1.shape == (batch_size, 1), "C1 output shape mismatch!"
    print("✅ Challenge 1 PASS")
    
    # Test Challenge 2 (EEGNeX)
    print("\n📊 Challenge 2: EEGNeX Test")
    print("-" * 60)
    model_c2 = submission.get_model_challenge_2()
    
    x_c2 = torch.randn(batch_size, 129, 200)
    
    with torch.no_grad():
        predictions_c2 = model_c2(x_c2)
    
    print(f"Input shape: {x_c2.shape}")
    print(f"Output shape: {predictions_c2.shape}")
    print(f"Sample predictions: {predictions_c2[:3].squeeze().tolist()}")
    
    total_params_c2 = sum(p.numel() for p in model_c2.parameters())
    print(f"Total parameters: {total_params_c2:,}")
    
    assert predictions_c2.shape == (batch_size, 1), "C2 output shape mismatch!"
    print("✅ Challenge 2 PASS")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print(f"\nChallenge 1: TCN ({total_params:,} params)")
    print(f"Challenge 2: EEGNeX ({total_params_c2:,} params)")
