# Challenge 1 Improvement Plan: Target < 0.8 NRMSE

**Current Status:** C1 = 1.00019 (Rank #72)  
**Target:** C1 < 0.8 (Top tier performance)  
**Gap:** 0.20019 (20% improvement needed)  
**Top Performer:** brain&ai with 0.91222 (8.8% better than baseline)

---

## 🎯 What We Need to Understand

### Top Performers Analysis

| Rank | Team | C1 Score | Strategy Clues |
|------|------|----------|----------------|
| #1 | brain&ai | 0.91222 | "brainmodule" - likely foundation model |
| #2 | CyberBobBeta | 0.92169 | "v5" - multiple iterations |
| #4 | Sigma Nova | 0.92245 | "SigmaNovaV2" - version 2 |
| #5 | MBZUAI | 0.92684 | Multiple submissions |
| #10 | MBZUAI | 0.92775 | Ensemble of approaches |

**Key Observation:** Top teams achieve 8-10% better than baseline

### Our Current Approach (V10)

```python
# EnhancedCompactCNN
- 3 Conv layers (129→32→64→128)
- Heavy dropout (0.6-0.7)
- Spatial attention
- Single model inference
- Score: 1.00019 (barely above baseline)
```

**Problems:**
1. Too simple architecture (only 120K params)
2. Not using temporal information effectively
3. No frequency domain features
4. Limited receptive field
5. No multi-scale processing

---

## 📊 Current Performance Breakdown

### What's Working (Keep)
✅ Subject-aware splits (prevents leakage)
✅ EMA tracking (smooth parameters)
✅ Heavy dropout (prevents overfitting)
✅ Multi-seed training (variance reduction)

### What's Not Working (Fix)
❌ Simple CNN - missing temporal patterns
❌ No frequency features - EEG is inherently rhythmic
❌ Small receptive field - can't see long-range patterns
❌ No attention over time - all timepoints treated equally
❌ No multi-scale features - missing fine and coarse patterns

---

## 🔬 Advanced Techniques to Implement

### 1. Temporal Convolutional Networks (TCN)
**Why:** Captures long-range dependencies better than standard CNN

```python
class TemporalBlock(nn.Module):
    """Dilated causal convolution block"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCNModel(nn.Module):
    """Multi-scale temporal convolution network"""
    def __init__(self, num_channels=129, num_classes=1):
        super().__init__()
        # Dilated convolutions: receptive field grows exponentially
        self.tcn = nn.ModuleList([
            TemporalBlock(num_channels, 64, kernel_size=3, stride=1, dilation=1),   # RF: 3
            TemporalBlock(64, 64, kernel_size=3, stride=1, dilation=2),             # RF: 7
            TemporalBlock(64, 128, kernel_size=3, stride=1, dilation=4),            # RF: 15
            TemporalBlock(128, 128, kernel_size=3, stride=1, dilation=8),           # RF: 31
            TemporalBlock(128, 256, kernel_size=3, stride=1, dilation=16),          # RF: 63
        ])
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        for block in self.tcn:
            x = block(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        return self.fc(x)
```

**Expected Improvement:** 3-5% (C1: 1.00019 → 0.96-0.97)

### 2. Frequency Domain Features
**Why:** EEG is fundamentally oscillatory (alpha, beta, theta, gamma)

```python
class FrequencyEncoder(nn.Module):
    """Extract frequency domain features"""
    def __init__(self, n_fft=64, n_mels=32):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        
    def forward(self, x):
        # x: (batch, channels, time)
        batch, channels, time = x.shape
        
        # Compute STFT for each channel
        freqs = []
        for c in range(channels):
            # Short-time Fourier transform
            stft = torch.stft(x[:, c, :], n_fft=self.n_fft, 
                             hop_length=self.n_fft//4, return_complex=True)
            magnitude = torch.abs(stft)
            freqs.append(magnitude)
        
        # Stack all channels: (batch, channels, freq_bins, time_frames)
        freq_features = torch.stack(freqs, dim=1)
        return freq_features

class TimeFrequencyModel(nn.Module):
    """Combines time and frequency features"""
    def __init__(self):
        super().__init__()
        # Time domain branch
        self.time_conv = nn.Sequential(
            nn.Conv1d(129, 64, kernel_size=7, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Frequency domain branch
        self.freq_encoder = FrequencyEncoder()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        # Time features
        time_feat = self.time_conv(x)
        time_feat = torch.mean(time_feat, dim=2)
        
        # Frequency features
        freq_feat = self.freq_encoder(x)
        freq_feat = self.freq_conv(freq_feat).squeeze(-1).squeeze(-1)
        
        # Combine
        combined = torch.cat([time_feat, freq_feat], dim=1)
        return self.fc(combined)
```

**Expected Improvement:** 4-6% (C1: 1.00019 → 0.94-0.96)

### 3. Transformer with Temporal Attention
**Why:** Learn which time windows matter most for RT prediction

```python
class TemporalAttention(nn.Module):
    """Multi-head attention over time"""
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, time, features)
        attn_out, attn_weights = self.attention(x, x, x)
        return self.norm(x + attn_out), attn_weights

class TransformerModel(nn.Module):
    """Transformer for EEG sequence modeling"""
    def __init__(self, n_channels=129, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        # Embed channels to d_model
        self.channel_embed = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, d_model))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TemporalAttention(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Output
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        # x: (batch, channels, time) → (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Embed and add positional encoding
        x = self.channel_embed(x) + self.pos_encoding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x, _ = layer(x)
        
        # Global average pooling over time
        x = torch.mean(x, dim=1)
        
        return self.fc(x)
```

**Expected Improvement:** 5-8% (C1: 1.00019 → 0.92-0.95)

### 4. Multi-Scale Feature Extraction
**Why:** RT prediction may depend on both fast (millisecond) and slow (second) dynamics

```python
class MultiScaleBlock(nn.Module):
    """Extract features at multiple temporal scales"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Different kernel sizes for different scales
        self.scale1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)   # Fine
        self.scale2 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)   # Medium
        self.scale3 = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7)  # Coarse
        
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        
        # Concatenate all scales
        out = torch.cat([s1, s2, s3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out

class MultiScaleModel(nn.Module):
    """CNN with multi-scale feature extraction"""
    def __init__(self):
        super().__init__()
        self.block1 = MultiScaleBlock(129, 32)     # → 96 channels
        self.block2 = MultiScaleBlock(96, 64)      # → 192 channels
        self.block3 = MultiScaleBlock(192, 128)    # → 384 channels
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 1),
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.fc(x)
```

**Expected Improvement:** 2-4% (C1: 1.00019 → 0.96-0.98)

### 5. Contrastive Pre-training (Self-Supervised)
**Why:** Learn rich representations from unlabeled data

```python
class ContrastiveModel(nn.Module):
    """Self-supervised contrastive learning"""
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
    def forward(self, x1, x2):
        # x1, x2: augmented versions of same trial
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Similarity matrix
        similarity = torch.matmul(z1, z2.T) / temperature
        
        # InfoNCE loss
        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = F.cross_entropy(similarity, labels)
        return loss

# Pre-training augmentations
def augment_eeg(x):
    """Apply augmentations for contrastive learning"""
    # Random time shift
    shift = torch.randint(-10, 10, (1,)).item()
    x_aug1 = torch.roll(x, shifts=shift, dims=2)
    
    # Random amplitude scaling
    scale = torch.rand(1).item() * 0.3 + 0.85  # [0.85, 1.15]
    x_aug2 = x * scale
    
    # Random time masking
    mask_len = torch.randint(5, 20, (1,)).item()
    mask_start = torch.randint(0, 200 - mask_len, (1,)).item()
    x_aug2[:, :, mask_start:mask_start+mask_len] = 0
    
    return x_aug1, x_aug2
```

**Expected Improvement:** 3-5% when fine-tuned (C1: 1.00019 → 0.95-0.97)

### 6. Advanced Data Augmentation
**Why:** More training diversity → better generalization

```python
class EEGAugmentation:
    """Comprehensive EEG augmentation suite"""
    
    @staticmethod
    def time_warp(x, sigma=0.2):
        """Non-linear time warping"""
        time_steps = x.shape[2]
        warp = np.cumsum(np.random.randn(time_steps) * sigma)
        warp = (warp - warp[0]) / (warp[-1] - warp[0]) * time_steps
        warp = np.clip(warp, 0, time_steps - 1).astype(int)
        return x[:, :, warp]
    
    @staticmethod
    def channel_dropout(x, p=0.1):
        """Randomly drop channels"""
        mask = torch.rand(x.shape[1]) > p
        x_aug = x.clone()
        x_aug[:, ~mask, :] = 0
        return x_aug
    
    @staticmethod
    def gaussian_noise(x, std=0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * std
        return x + noise
    
    @staticmethod
    def frequency_mask(x, max_mask=10):
        """Mask random frequency bands"""
        # Convert to frequency domain
        fft = torch.fft.rfft(x, dim=2)
        
        # Random frequency mask
        freq_bins = fft.shape[2]
        mask_size = torch.randint(1, max_mask, (1,)).item()
        mask_start = torch.randint(0, freq_bins - mask_size, (1,)).item()
        fft[:, :, mask_start:mask_start+mask_size] = 0
        
        # Back to time domain
        return torch.fft.irfft(fft, n=x.shape[2], dim=2)
    
    @staticmethod
    def mixup(x1, y1, x2, y2, alpha=0.2):
        """Mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = lam * y1 + (1 - lam) * y2
        return x_mix, y_mix
```

**Expected Improvement:** 1-2% (helps prevent overfitting)

---

## 🎯 Recommended Implementation Strategy

### Phase 1: Quick Wins (Target: C1 < 0.95)
**Timeline:** 1-2 days

1. ✅ **Multi-Scale CNN**
   - Easy to implement
   - Low risk
   - Expected: 0.96-0.98

2. ✅ **Better Data Augmentation**
   - Time warping
   - Channel dropout
   - Frequency masking
   - Expected: 0.95-0.97

3. ✅ **Larger Ensemble**
   - Train 10 seeds instead of 5
   - More TTA variations
   - Expected: 0.94-0.96

### Phase 2: Advanced Methods (Target: C1 < 0.92)
**Timeline:** 2-3 days

4. ✅ **Temporal Convolutional Network**
   - Dilated convolutions
   - Large receptive field
   - Expected: 0.92-0.94

5. ✅ **Time-Frequency Model**
   - Dual-branch architecture
   - STFT features
   - Expected: 0.91-0.93

### Phase 3: State-of-Art (Target: C1 < 0.8)
**Timeline:** 3-5 days

6. ✅ **Transformer Architecture**
   - Temporal attention
   - Learn relevant time windows
   - Expected: 0.85-0.90

7. ✅ **Contrastive Pre-training**
   - Self-supervised learning
   - Fine-tune on labeled data
   - Expected: 0.80-0.85

8. ✅ **Model Ensemble**
   - Combine TCN + Transformer + Time-Freq
   - Weighted averaging
   - Expected: 0.78-0.82

---

## 📊 Expected Progress

| Phase | Methods | Expected C1 | Time | Risk |
|-------|---------|-------------|------|------|
| **Current** | Simple CNN | 1.00019 | - | - |
| **Phase 1** | Multi-scale + Aug + Ensemble | 0.94-0.96 | 1-2 days | Low |
| **Phase 2** | TCN + Time-Freq | 0.91-0.93 | 2-3 days | Medium |
| **Phase 3** | Transformer + Contrastive | 0.78-0.85 | 3-5 days | High |
| **🎯 TARGET** | Full ensemble | **< 0.8** | 6-10 days | Medium |

---

## 🔬 Research to Explore

### Papers to Study
1. **"EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain–Machine Interfaces"** (2021)
   - TCN for EEG classification
   - State-of-art on motor imagery

2. **"EEG-Conformer: Convolutional Transformer for EEG Decoding and Visualization"** (2022)
   - Combines CNN + Transformer
   - Excellent for temporal patterns

3. **"Self-supervised Learning for EEG-based Emotion Recognition"** (2023)
   - Contrastive pre-training
   - Transfer learning

4. **"Deep Learning for EEG-based Brain-Computer Interfaces"** (Review, 2023)
   - Comprehensive overview
   - Best practices

### Codebases to Reference
- **braindecode** - EEG-specific architectures
- **MNE-Python** - Signal processing
- **PyTorch-TS** - Temporal models
- **timm** - Vision transformers (adapt for EEG)

---

## 💻 Implementation Checklist

```markdown
Phase 1 (Quick Wins):
- [ ] Implement MultiScaleModel
- [ ] Add augmentation pipeline
- [ ] Train 10-seed ensemble
- [ ] Test on validation set
- [ ] Submit if C1 < 0.95

Phase 2 (Advanced):
- [ ] Implement TCN architecture
- [ ] Implement Time-Frequency model
- [ ] Train both architectures
- [ ] Ensemble TCN + Time-Freq
- [ ] Submit if C1 < 0.92

Phase 3 (State-of-Art):
- [ ] Implement Transformer
- [ ] Pre-train with contrastive loss
- [ ] Fine-tune on labeled data
- [ ] Create mega-ensemble
- [ ] Submit if C1 < 0.8
```

---

## 🎯 Success Criteria

**Minimum Viable:**
- C1 < 0.95 (Phase 1 complete)
- Rank improvement: #72 → #40-50

**Target:**
- C1 < 0.92 (Phase 2 complete)
- Rank improvement: #72 → #20-30

**Stretch Goal:**
- C1 < 0.8 (Phase 3 complete)
- Rank improvement: #72 → Top 10

---

**Let's start with Phase 1 implementations!** 🚀
