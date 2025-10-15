#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced GPU-Optimized Neural Network Layers
===========================================

High-performance neural network layers optimized for both NVIDIA and AMD GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu.enhanced_gpu_optimizer import get_enhanced_optimizer

class EnhancedLinear(nn.Module):
    """Enhanced linear layer with GPU optimization"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 use_enhanced_ops: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_enhanced_ops = use_enhanced_ops
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
        # Get enhanced optimizer
        if use_enhanced_ops:
            self.gpu_opt = get_enhanced_optimizer()
        
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced GPU optimization"""
        if self.use_enhanced_ops:
            # Use enhanced matrix multiplication
            result = self.gpu_opt.optimized_matmul(x, self.weight, transpose_b=True)
            if self.bias is not None:
                result = result + self.bias
            return result
        else:
            # Standard PyTorch implementation
            return F.linear(x, self.weight, self.bias)

class EnhancedMultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with GPU optimization"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_enhanced_ops: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_enhanced_ops = use_enhanced_ops
        
        # Linear projections
        self.w_q = EnhancedLinear(d_model, d_model, use_enhanced_ops=use_enhanced_ops)
        self.w_k = EnhancedLinear(d_model, d_model, use_enhanced_ops=use_enhanced_ops)
        self.w_v = EnhancedLinear(d_model, d_model, use_enhanced_ops=use_enhanced_ops)
        self.w_o = EnhancedLinear(d_model, d_model, use_enhanced_ops=use_enhanced_ops)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_enhanced_ops:
            self.gpu_opt = get_enhanced_optimizer()
            
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with enhanced attention computation"""
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        if self.use_enhanced_ops:
            # Enhanced matrix multiplication for attention scores
            scores = self.gpu_opt.optimized_matmul(Q, K, transpose_b=True)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1))
            
        scores = scores / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        if self.use_enhanced_ops:
            context = self.gpu_opt.optimized_matmul(attention_weights, V)
        else:
            context = torch.matmul(attention_weights, V)
            
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(context)

class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with GPU optimization"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_enhanced_ops: bool = True):
        super().__init__()
        self.use_enhanced_ops = use_enhanced_ops
        
        # Multi-head attention
        self.attention = EnhancedMultiHeadAttention(
            d_model, n_heads, dropout, use_enhanced_ops
        )
        
        # Feed-forward network
        self.ff1 = EnhancedLinear(d_model, d_ff, use_enhanced_ops=use_enhanced_ops)
        self.ff2 = EnhancedLinear(d_ff, d_model, use_enhanced_ops=use_enhanced_ops)
        
        # Layer normalization and dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff2(F.relu(self.ff1(x)))
        x = self.ln2(x + self.dropout(ff_output))
        
        return x

class EnhancedEEGFoundationModel(nn.Module):
    """Enhanced EEG foundation model with GPU optimization"""
    
    def __init__(self, n_channels: int = 129, seq_len: int = 1000, 
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 6,
                 d_ff: int = 512, dropout: float = 0.1, use_enhanced_ops: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_enhanced_ops = use_enhanced_ops
        
        # Input projection
        self.input_proj = EnhancedLinear(
            n_channels, d_model, use_enhanced_ops=use_enhanced_ops
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(d_model, n_heads, d_ff, dropout, use_enhanced_ops)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.ln_final = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        if use_enhanced_ops:
            self.gpu_opt = get_enhanced_optimizer()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the foundation model"""
        # x shape: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Final layer norm
        x = self.ln_final(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return x

class EnhancedEEGClassifier(nn.Module):
    """Enhanced EEG classifier with GPU optimization"""
    
    def __init__(self, backbone: nn.Module, d_model: int = 128, 
                 num_classes: int = 1, dropout: float = 0.1,
                 use_enhanced_ops: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_enhanced_ops = use_enhanced_ops
        
        # Classification head
        self.classifier = nn.Sequential(
            EnhancedLinear(d_model, d_model // 2, use_enhanced_ops=use_enhanced_ops),
            nn.ReLU(),
            nn.Dropout(dropout),
            EnhancedLinear(d_model // 2, d_model // 4, use_enhanced_ops=use_enhanced_ops),
            nn.ReLU(),
            nn.Dropout(dropout),
            EnhancedLinear(d_model // 4, num_classes, use_enhanced_ops=use_enhanced_ops)
        )
        
        # For binary classification
        self.use_sigmoid = (num_classes == 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier"""
        # Get features from backbone
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        if self.use_sigmoid:
            return torch.sigmoid(logits.squeeze(-1))
        else:
            return logits

class EnhancedSpectralBlock(nn.Module):
    """Enhanced spectral processing block with safe FFT operations"""
    
    def __init__(self, n_channels: int, use_enhanced_ops: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.use_enhanced_ops = use_enhanced_ops
        
        if use_enhanced_ops:
            self.gpu_opt = get_enhanced_optimizer()
            
        # Learnable frequency weights
        self.freq_weights = nn.Parameter(torch.ones(n_channels, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spectral features safely"""
        # x shape: (batch, channels, time)
        
        if self.use_enhanced_ops:
            # Use enhanced FFT operations
            with self.gpu_opt.memory_management("spectral"):
                X = self.gpu_opt.safe_fft(x, dim=-1)
        else:
            # Standard FFT
            X = torch.fft.rfft(x, dim=-1)
            
        # Compute power spectral density
        power = torch.abs(X) ** 2
        
        # Apply learnable frequency weights
        power = power * self.freq_weights.unsqueeze(0)
        
        # Log power for stability
        log_power = torch.log(power + 1e-8)
        
        # Frequency band features (alpha, beta, gamma, etc.)
        # Assuming 250 Hz sampling rate
        freq_bins = log_power.shape[-1]
        
        # Define frequency bands (normalized to freq_bins)
        alpha_idx = int(0.08 * freq_bins)   # ~8-13 Hz
        beta_idx = int(0.13 * freq_bins)    # ~13-30 Hz  
        gamma_idx = int(0.30 * freq_bins)   # ~30-50 Hz
        
        alpha_power = log_power[..., :alpha_idx].mean(dim=-1)
        beta_power = log_power[..., alpha_idx:beta_idx].mean(dim=-1)
        gamma_power = log_power[..., beta_idx:gamma_idx].mean(dim=-1)
        
        # Combine features
        spectral_features = torch.stack([alpha_power, beta_power, gamma_power], dim=-1)
        
        return spectral_features  # (batch, channels, 3)

def create_enhanced_eeg_model(n_channels: int = 129, num_classes: int = 1,
                             d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                             use_enhanced_ops: bool = True) -> nn.Module:
    """Create enhanced EEG model with GPU optimization"""
    
    # Create foundation model
    backbone = EnhancedEEGFoundationModel(
        n_channels=n_channels,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        use_enhanced_ops=use_enhanced_ops
    )
    
    # Create classifier
    model = EnhancedEEGClassifier(
        backbone=backbone,
        d_model=d_model,
        num_classes=num_classes,
        use_enhanced_ops=use_enhanced_ops
    )
    
    return model
