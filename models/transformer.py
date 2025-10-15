"""
Simple EEG Transformer Model
A transformer-based model for EEG analysis
"""

import torch
import torch.nn as nn
import math

class EEGTransformer(nn.Module):
    """Transformer model for EEG data"""
    
    def __init__(
        self,
        n_channels: int = 129,
        seq_len: int = 1000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 128  # Embedding dimension
    ):
        """
        Args:
            n_channels: Number of EEG channels
            seq_len: Sequence length (time steps)
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection: (batch, channels, seq_len) -> (batch, seq_len, d_model)
        self.input_proj = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        
        Returns:
            embeddings: Output tensor of shape (batch, output_dim)
        """
        # Transpose to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output projection
        x = self.output_proj(x)  # (batch, output_dim)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def test_model():
    """Test the model"""
    print("=" * 60)
    print("Testing EEGTransformer")
    print("=" * 60)
    
    # Create model
    model = EEGTransformer(
        n_channels=129,
        seq_len=1000,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        output_dim=128
    )
    
    print()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 129, 1000)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output stats: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")
    
    print()
    print("=" * 60)
    print("✅ Model test complete!")


if __name__ == "__main__":
    test_model()
