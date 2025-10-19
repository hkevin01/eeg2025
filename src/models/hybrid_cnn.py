"""
Hybrid CNN Model: Combines sparse attention CNN with neuroscience features.

Architecture:
- Sparse attention CNN backbone (learns general patterns)
- Neuroscience feature extractor (domain knowledge)
- Fusion layer (combines both)

Anti-overfitting measures:
- Strong dropout (0.4)
- L2 regularization via weight decay
- Small feature network
- Batch normalization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict

# Import neuroscience features
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from features.neuroscience_features import extract_all_neuro_features


class ChannelAttention(nn.Module):
    """Spatial attention over EEG channels."""
    
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, channels, time)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        return x * self.sigmoid(out).unsqueeze(-1)


class SparseMultiHeadAttention(nn.Module):
    """O(N) complexity sparse multi-head attention."""
    
    def __init__(self, hidden_size, scale_factor=0.5, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # x: (batch, seq, hidden)
        batch_size, seq_len, hidden = x.size()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Sparse attention: sample tokens
        num_samples = max(1, int(seq_len * self.scale_factor))
        indices = torch.randperm(seq_len, device=x.device)[:num_samples]
        
        q_sparse = q[:, indices, :]
        k_sparse = k[:, indices, :]
        v_sparse = v[:, indices, :]
        
        # Scaled dot-product attention
        scores = torch.matmul(q_sparse, k_sparse.transpose(-2, -1)) / (hidden ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v_sparse)
        
        # Scatter back to original positions
        full_out = torch.zeros_like(x)
        full_out[:, indices, :] = out
        
        return self.output(full_out), attn


class NeuroFeatureExtractor(nn.Module):
    """
    Small network to process neuroscience features.
    
    Input: 6 features (p300_amp, p300_lat, motor_slope, motor_amp, n200_amp, alpha_supp)
    Output: 16-dimensional embedding
    
    Kept small to avoid overfitting.
    """
    
    def __init__(self, input_dim=6, output_dim=16, dropout=0.4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: (batch, 6) neuroscience features
        return self.network(x)


class HybridNeuroModel(nn.Module):
    """
    Hybrid model combining sparse attention CNN with neuroscience features.
    
    Two parallel pathways:
    1. CNN backbone: Learns general patterns from raw EEG
    2. Neuro features: Explicit domain knowledge (P300, motor prep, etc.)
    
    Features are fused before final prediction.
    
    Args:
        num_channels: Number of EEG channels (default 129)
        seq_length: Sequence length (default 200)
        dropout: Dropout rate (default 0.4 for regularization)
        use_neuro_features: Whether to use neuroscience features (default True)
        sfreq: Sampling frequency for feature extraction (default 100 Hz)
    """
    
    def __init__(
        self,
        num_channels=129,
        seq_length=200,
        dropout=0.4,
        use_neuro_features=True,
        sfreq=100.0,
    ):
        super().__init__()
        
        self.use_neuro_features = use_neuro_features
        self.sfreq = sfreq
        self.num_channels = num_channels
        
        # ===== CNN Backbone (same as baseline) =====
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio=16)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)  # 200 -> 100
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)  # 100 -> 50
        )
        
        # Sparse attention
        self.attention = SparseMultiHeadAttention(
            hidden_size=256,
            scale_factor=0.5,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(256)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Dropout(dropout)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # ===== Neuroscience Feature Pathway =====
        if use_neuro_features:
            self.neuro_extractor = NeuroFeatureExtractor(
                input_dim=6,
                output_dim=16,
                dropout=dropout
            )
            fusion_dim = 256 + 16  # CNN features + neuro features
        else:
            fusion_dim = 256
        
        # ===== Fusion and Prediction =====
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Cache for channel indices (computed once per forward pass)
        self._channel_groups_cache = None
        
    def extract_neuro_features_batch(self, x):
        """
        Extract neuroscience features for a batch of EEG signals.
        
        Args:
            x: (batch, channels, time) EEG data
            
        Returns:
            (batch, 6) neuroscience features
        """
        batch_size = x.shape[0]
        features = torch.zeros((batch_size, 6), dtype=x.dtype, device=x.device)
        
        # Convert to numpy for feature extraction (once per batch)
        x_np = x.detach().cpu().numpy()
        
        for i in range(batch_size):
            try:
                # Extract features (already normalized in extract_all_neuro_features)
                feats = extract_all_neuro_features(
                    x_np[i],
                    sfreq=self.sfreq,
                    channel_groups=None,  # Will use fallbacks
                    stimulus_time=0.0,  # Assume stimulus at start
                    normalize=True
                )
                features[i] = torch.from_numpy(feats)
            except Exception as e:
                # Fallback to zeros if extraction fails
                features[i] = 0.0
        
        return features
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, channels, time) EEG signal
            
        Returns:
            (batch, 1) predicted response time
        """
        # ===== CNN Pathway =====
        # Channel attention
        x_cnn = self.channel_attention(x)
        
        # Convolutional layers
        x_cnn = self.conv1(x_cnn)  # (batch, 128, 100)
        x_cnn = self.conv2(x_cnn)  # (batch, 256, 50)
        
        # Sparse attention
        x_cnn = x_cnn.transpose(1, 2)  # (batch, 50, 256)
        attn_out, _ = self.attention(x_cnn)
        x_cnn = self.norm(x_cnn + attn_out)
        ffn_out = self.ffn(x_cnn)
        x_cnn = x_cnn + ffn_out
        
        # Global pooling
        x_cnn = x_cnn.transpose(1, 2)  # (batch, 256, 50)
        x_cnn = self.global_avg_pool(x_cnn).squeeze(-1)  # (batch, 256)
        
        # ===== Neuroscience Feature Pathway =====
        if self.use_neuro_features:
            # Extract neuroscience features
            neuro_features = self.extract_neuro_features_batch(x)  # (batch, 6)
            
            # Process through small network
            neuro_embedded = self.neuro_extractor(neuro_features)  # (batch, 16)
            
            # Fuse CNN and neuro features
            combined = torch.cat([x_cnn, neuro_embedded], dim=-1)  # (batch, 272)
        else:
            combined = x_cnn
        
        # ===== Final Prediction =====
        output = self.fc(combined)  # (batch, 1)
        
        return output


# For compatibility with existing training scripts
class HybridCNN(HybridNeuroModel):
    """Alias for backwards compatibility."""
    pass


if __name__ == "__main__":
    # Test model creation
    print("Testing HybridNeuroModel...")
    
    model = HybridNeuroModel(
        num_channels=129,
        seq_length=200,
        dropout=0.4,
        use_neuro_features=True
    )
    
    print(f"Model created successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(2, 129, 200)
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    print("\nModel ready for training!")
