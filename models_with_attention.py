"""
Enhanced CNN architectures with Multi-Head Self-Attention
For EEG 2025 Competition - Challenge 1 Improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer for temporal feature learning
    
    This helps the model capture long-range dependencies in EEG signals
    that pure CNNs might miss.
    """
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time) tensor
        Returns:
            attended: (batch, channels, time) tensor
        """
        batch_size, channels, seq_len = x.shape
        
        # Reshape to (batch, time, channels) for attention
        x = x.transpose(1, 2)  # (batch, time, channels)
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch, time, 3*channels)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, time, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        attended = torch.matmul(attn, v)  # (batch, heads, time, head_dim)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()  # (batch, time, heads, head_dim)
        attended = attended.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        attended = self.proj(attended)
        attended = self.dropout(attended)
        
        # Reshape back to (batch, channels, time)
        attended = attended.transpose(1, 2)
        
        return attended


class AttentionCNN_ResponseTime(nn.Module):
    """
    Enhanced CNN with Multi-Head Self-Attention for response time prediction
    
    Architecture:
    1. Conv layers to extract local features
    2. Multi-Head Self-Attention to capture long-range dependencies
    3. More conv layers for refinement
    4. Regressor for final prediction
    
    This combines the local pattern detection of CNNs with the global
    context modeling of attention mechanisms.
    """
    
    def __init__(self, num_heads=4, attention_dropout=0.1):
        super().__init__()
        
        # Initial convolution blocks (extract local features)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Multi-Head Self-Attention layer
        # At this point: (batch, 64, 50) - 50 time steps with 64 features
        self.attention = MultiHeadSelfAttention(
            embed_dim=64,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        
        # Layer norm after attention
        self.norm = nn.LayerNorm(64)
        
        # Final convolution blocks (refine features)
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Initial conv blocks
        x = self.conv_block1(x)  # (batch, 32, 100)
        x = self.conv_block2(x)  # (batch, 64, 50)
        
        # Attention with residual connection
        identity = x
        x_attended = self.attention(x)
        
        # Layer norm (apply on time dimension)
        x_attended = x_attended.transpose(1, 2)  # (batch, time, channels)
        x_attended = self.norm(x_attended)
        x_attended = x_attended.transpose(1, 2)  # (batch, channels, time)
        
        # Residual connection
        x = identity + x_attended
        
        # Final conv and pooling
        x = self.conv_block3(x)  # (batch, 128, 25)
        x = self.pool(x)  # (batch, 128, 1)
        
        # Regression
        output = self.regressor(x)
        
        return output


class LightweightAttentionCNN(nn.Module):
    """
    Lightweight version with fewer parameters
    
    For cases where we want to stay close to the original 75K params
    but still benefit from attention.
    """
    
    def __init__(self, num_heads=4):
        super().__init__()
        
        # Conv blocks (same as original)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Lightweight attention (fewer heads, less dropout)
        self.attention = MultiHeadSelfAttention(
            embed_dim=64,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Skip the layer norm to save params
        
        # Final conv block
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),  # 96 instead of 128
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Smaller regressor
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(48, 1)
        )
        
    def forward(self, x):
        # Conv blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Attention with residual
        identity = x
        x = self.attention(x)
        x = identity + x  # Residual connection
        
        # Final layers
        x = self.conv_block3(x)
        x = self.pool(x)
        output = self.regressor(x)
        
        return output


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING ATTENTION-ENHANCED CNN ARCHITECTURES")
    print("=" * 70)
    
    # Test input
    batch_size = 2
    channels = 129
    seq_len = 200
    x = torch.randn(batch_size, channels, seq_len)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input: (batch={batch_size}, channels={channels}, time={seq_len})")
    
    # Test full attention model
    print("\n" + "=" * 70)
    print("1. AttentionCNN_ResponseTime (Full Model)")
    print("=" * 70)
    model1 = AttentionCNN_ResponseTime(num_heads=4)
    params1 = sum(p.numel() for p in model1.parameters())
    out1 = model1(x)
    print(f"Parameters: {params1:,}")
    print(f"Output shape: {out1.shape}")
    
    # Test lightweight model
    print("\n" + "=" * 70)
    print("2. LightweightAttentionCNN")
    print("=" * 70)
    model2 = LightweightAttentionCNN(num_heads=4)
    params2 = sum(p.numel() for p in model2.parameters())
    out2 = model2(x)
    print(f"Parameters: {params2:,}")
    print(f"Output shape: {out2.shape}")
    
    # Compare with original
    print("\n" + "=" * 70)
    print("COMPARISON WITH ORIGINAL")
    print("=" * 70)
    print(f"Original CompactResponseTimeCNN: 74,753 params")
    print(f"AttentionCNN_ResponseTime:      {params1:,} params (+{params1-74753:,})")
    print(f"LightweightAttentionCNN:         {params2:,} params (+{params2-74753:,})")
    
    print("\n" + "=" * 70)
    print("✅ ALL MODELS TESTED SUCCESSFULLY")
    print("=" * 70)

