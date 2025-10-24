# 🧠 Latest Neuroscience-Inspired Algorithms for EEG Training (2024-2025)

## 📅 Date: October 24, 2024

---

## 🎯 Overview

This document outlines cutting-edge neuroscience-inspired algorithms and techniques for EEG signal processing and deep learning, focusing on approaches that have shown promise in recent literature and competitions.

---

## 🚀 State-of-the-Art Architectures

### 1. **EEGNet Family (2024 Variants)**

#### EEGNet-8,2 (Enhanced)
- **Paper**: Latest variants from 2024
- **Key Innovation**: Depthwise separable convolutions for channel-specific features
- **Performance**: State-of-art on BCI Competition datasets

```python
from braindecode.models import EEGNetv4

model = EEGNetv4(
    n_chans=129,
    n_times=200,
    n_outputs=1,
    final_conv_length='auto',
    pool_mode='mean',
    F1=16,  # Temporal filters
    D=2,    # Spatial filters per temporal filter
    F2=32,  # Pointwise filters
    kernel_length=64,
    third_kernel_size=(8, 4),
    drop_prob=0.5
)
```

**Improvements for Competition**:
```python
# Add attention mechanism
class EEGNetWithAttention(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.attention = nn.MultiheadAttention(
            embed_dim=32, 
            num_heads=4,
            dropout=0.3
        )
    
    def forward(self, x):
        features = self.base.features(x)
        # Apply temporal attention
        features, _ = self.attention(features, features, features)
        return self.base.classifier(features)
```

---

### 2. **Conformer Architecture (2023-2024)**

**Key Innovation**: Combines convolution and self-attention
- Better than pure transformers for EEG
- Captures both local and global patterns

```python
class EEGConformer(nn.Module):
    """
    Conformer for EEG: Convolution-augmented Transformer
    State-of-art on speech/EEG tasks
    """
    def __init__(
        self,
        n_chans=129,
        n_times=200,
        n_outputs=1,
        n_filters=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=2,
        att_heads=4,
        att_drop_prob=0.5
    ):
        super().__init__()
        
        # Temporal convolution
        self.temp_conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, filter_time_length), padding='same'),
            nn.BatchNorm2d(n_filters),
            nn.ELU()
        )
        
        # Spatial convolution
        self.spat_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, (n_chans, 1)),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride)),
            nn.Dropout(drop_prob)
        )
        
        # Conformer blocks (Conv + Self-Attention)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=n_filters,
                num_heads=att_heads,
                dropout=att_drop_prob
            )
            for _ in range(att_depth)
        ])
        
        # Output
        self.classifier = nn.Linear(n_filters, n_outputs)
    
    def forward(self, x):
        # x: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        # Temporal features
        x = self.temp_conv(x)
        
        # Spatial features
        x = self.spat_conv(x)
        x = x.squeeze(2)  # (batch, n_filters, n_times')
        
        # Conformer blocks
        x = x.permute(0, 2, 1)  # (batch, n_times', n_filters)
        for block in self.conformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output
        return self.classifier(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: Conv + Self-Attention"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Feed-forward module
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Feed-forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Multi-head attention
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Convolution
        x_conv = x.transpose(1, 2)  # (batch, dim, seq_len)
        x_conv = self.conv(x_conv)
        x = x + x_conv.transpose(1, 2)
        
        # Feed-forward 2
        x = x + 0.5 * self.ff2(x)
        
        return self.norm(x)
```

---

### 3. **Graph Neural Networks (GNN) for EEG**

**Key Innovation**: Model brain connectivity as graph
- Nodes = EEG channels (brain regions)
- Edges = Functional connectivity
- Captures spatial relationships naturally

```python
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv

class EEG_GNN(nn.Module):
    """
    Graph Neural Network for EEG
    Models brain connectivity patterns
    """
    def __init__(
        self,
        n_chans=129,
        n_times=200,
        n_outputs=1,
        hidden_dim=64,
        num_layers=3,
        dropout=0.5
    ):
        super().__init__()
        
        # Temporal feature extraction (per channel)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout)
        )
        
        # Calculate temporal output size
        temp_out_size = n_times // 4
        node_features = 32 * temp_out_size
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GATConv(node_features, hidden_dim, heads=4))
        
        for _ in range(num_layers - 2):
            self.gcn_layers.append(
                GATConv(hidden_dim * 4, hidden_dim, heads=4)
            )
        
        self.gcn_layers.append(
            GATConv(hidden_dim * 4, hidden_dim, heads=1)
        )
        
        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_chans, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_outputs)
        )
        
        # Edge index (connectivity matrix)
        self.register_buffer('edge_index', self._create_connectivity(n_chans))
    
    def _create_connectivity(self, n_chans):
        """Create brain connectivity graph"""
        # Option 1: Fully connected
        edge_list = []
        for i in range(n_chans):
            for j in range(n_chans):
                if i != j:
                    edge_list.append([i, j])
        
        return torch.tensor(edge_list, dtype=torch.long).t()
    
    def forward(self, x):
        # x: (batch, n_chans, n_times)
        batch_size = x.size(0)
        n_chans = x.size(1)
        
        # Extract temporal features per channel
        node_features = []
        for i in range(n_chans):
            channel_data = x[:, i:i+1, :]  # (batch, 1, n_times)
            features = self.temporal_conv(channel_data)
            features = features.flatten(1)  # (batch, features)
            node_features.append(features)
        
        # Stack: (batch, n_chans, node_features)
        node_features = torch.stack(node_features, dim=1)
        
        # Process each sample in batch
        outputs = []
        for b in range(batch_size):
            x_graph = node_features[b]  # (n_chans, node_features)
            
            # Apply graph convolutions
            for gcn in self.gcn_layers:
                x_graph = gcn(x_graph, self.edge_index)
                x_graph = F.elu(x_graph)
            
            # Flatten for classification
            x_graph = x_graph.flatten()
            outputs.append(x_graph)
        
        x = torch.stack(outputs)
        return self.classifier(x)
```

---

### 4. **Spiking Neural Networks (SNN)**

**Key Innovation**: Biologically plausible, event-driven processing
- Energy efficient
- Natural for temporal data like EEG
- Recent advances make training easier

```python
import snntorch as snn
from snntorch import surrogate

class SpikingEEGNet(nn.Module):
    """
    Spiking Neural Network for EEG
    More biologically plausible
    """
    def __init__(
        self,
        n_chans=129,
        n_times=200,
        n_outputs=1,
        beta=0.9,  # Membrane decay rate
        num_steps=25  # Simulation time steps
    ):
        super().__init__()
        
        self.num_steps = num_steps
        
        # Convolutional layers with spiking neurons
        self.conv1 = nn.Conv2d(1, 16, (1, 25), padding=(0, 12))
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.conv2 = nn.Conv2d(16, 32, (n_chans, 1))
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.pool = nn.AvgPool2d((1, 4))
        
        # Calculate flattened size
        flat_size = 32 * (n_times // 4)
        
        # Fully connected with spiking
        self.fc1 = nn.Linear(flat_size, 128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.fc2 = nn.Linear(128, n_outputs)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), 
                                  output=True)
    
    def forward(self, x):
        # x: (batch, n_chans, n_times)
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch, 1, n_chans, n_times)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        # Record output spikes
        spk_rec = []
        
        # Simulate over time
        for step in range(self.num_steps):
            # Conv layer 1
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Conv layer 2
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Pool and flatten
            spk2 = self.pool(spk2)
            spk2 = spk2.squeeze(2).flatten(1)
            
            # FC layer 1
            cur3 = self.fc1(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Output layer
            cur_out = self.fc2(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            spk_rec.append(spk_out)
        
        # Decode: average spike rate
        return torch.stack(spk_rec).mean(dim=0)
```

---

### 5. **Self-Supervised Learning (SSL) Approaches**

#### A. Contrastive Learning (SimCLR for EEG)

```python
class EEG_SimCLR(nn.Module):
    """
    Self-supervised contrastive learning for EEG
    Learn representations without labels
    """
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(base_encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x1, x2):
        # x1, x2: Two augmented views of same EEG
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    batch_size = z1.size(0)
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Create labels
    labels = torch.cat([
        torch.arange(batch_size) + batch_size,
        torch.arange(batch_size)
    ]).to(z.device)
    
    # Mask out self-similarities
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    
    # Compute loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


# Training loop for SSL
def pretrain_ssl(model, dataloader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch['eeg']  # (batch, n_chans, n_times)
            
            # Create two augmented views
            x1 = augment_eeg(x, strength=0.3)
            x2 = augment_eeg(x, strength=0.3)
            
            # Forward pass
            z1, z2 = model(x1, x2)
            
            # Contrastive loss
            loss = nt_xent_loss(z1, z2)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

#### B. Masked Autoencoding (MAE for EEG)

```python
class EEG_MAE(nn.Module):
    """
    Masked Autoencoder for EEG
    Similar to BERT/ViT-MAE
    """
    def __init__(
        self,
        n_chans=129,
        n_times=200,
        mask_ratio=0.75,
        encoder_dim=256,
        decoder_dim=128
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.patch_size = 25  # Temporal patches
        self.n_patches = n_times // self.patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(n_chans * self.patch_size, encoder_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, encoder_dim) * 0.02
        )
        
        # Encoder (transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=8,
            dim_feedforward=encoder_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=8,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)
        
        # Reconstruction head
        self.recon_head = nn.Linear(decoder_dim, n_chans * self.patch_size)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
    
    def patchify(self, x):
        """Convert EEG to patches"""
        # x: (batch, n_chans, n_times)
        batch_size = x.size(0)
        patches = []
        
        for i in range(self.n_patches):
            start = i * self.patch_size
            end = start + self.patch_size
            patch = x[:, :, start:end]  # (batch, n_chans, patch_size)
            patch = patch.flatten(1)  # (batch, n_chans * patch_size)
            patches.append(patch)
        
        return torch.stack(patches, dim=1)  # (batch, n_patches, features)
    
    def forward(self, x, mask=None):
        # Patchify
        patches = self.patchify(x)  # (batch, n_patches, features)
        
        # Embed patches
        x = self.patch_embed(patches)
        x = x + self.pos_embed
        
        # Random masking
        if mask is None:
            mask = self._random_mask(x.size(0), x.size(1))
        
        # Keep only unmasked patches for encoder
        x_masked = x[~mask].reshape(x.size(0), -1, x.size(2))
        
        # Encode
        encoded = self.encoder(x_masked)
        
        # Prepare for decoder
        x_decoded = self.decoder_embed(encoded)
        
        # Add mask tokens
        mask_tokens = self.mask_token.expand(
            x.size(0), mask.sum(dim=1).max(), -1
        )
        
        # Reconstruct full sequence
        x_full = torch.zeros(x.size(0), self.n_patches, x_decoded.size(2)).to(x.device)
        x_full[~mask] = x_decoded.flatten(0, 1)
        x_full[mask] = mask_tokens.flatten(0, 1)[:mask.sum()]
        
        # Decode
        decoded = self.decoder(x_full)
        
        # Reconstruct patches
        reconstructed = self.recon_head(decoded)
        
        return reconstructed, patches, mask
    
    def _random_mask(self, batch_size, seq_len):
        """Generate random mask"""
        noise = torch.rand(batch_size, seq_len)
        mask = noise > self.mask_ratio
        return mask.to(self.pos_embed.device)


def train_mae(model, dataloader, epochs=100):
    """Train MAE model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch['eeg']
            
            # Forward
            reconstructed, original, mask = model(x)
            
            # Loss only on masked patches
            loss = F.mse_loss(
                reconstructed[mask],
                original[mask]
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Recon Loss = {loss.item():.4f}")
```

---

### 6. **Neural Architecture Search (NAS)**

```python
from torch_geometric.nn import GENConv
from torch.nn import ModuleList

class SearchableEEGNet(nn.Module):
    """
    Neural Architecture Search for EEG
    Automatically find best architecture
    """
    def __init__(self, n_chans=129, n_times=200, n_outputs=1):
        super().__init__()
        
        # Searchable operations
        self.operations = ModuleList([
            # Option 1: Standard conv
            nn.Conv2d(1, 32, (1, 25), padding=(0, 12)),
            
            # Option 2: Depthwise separable
            nn.Sequential(
                nn.Conv2d(1, 32, (1, 25), padding=(0, 12), groups=1),
                nn.Conv2d(32, 32, 1)
            ),
            
            # Option 3: Dilated conv
            nn.Conv2d(1, 32, (1, 25), padding=(0, 24), dilation=(1, 2)),
            
            # Option 4: Attention-based
            nn.MultiheadAttention(embed_dim=n_times, num_heads=4)
        ])
        
        # Architecture parameters (learnable)
        self.arch_params = nn.Parameter(torch.randn(len(self.operations)))
    
    def forward(self, x):
        # Softmax over architecture choices
        weights = F.softmax(self.arch_params, dim=0)
        
        # Weighted sum of operations
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        
        return output
```

---

## 🧪 Advanced Training Techniques

### 1. **Mixup for EEG**

```python
def mixup_eeg(x1, y1, x2, y2, alpha=0.4):
    """
    Mixup augmentation for EEG
    Interpolate between samples
    """
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    return x_mixed, y_mixed

# In training loop:
for batch in dataloader:
    x, y = batch['eeg'], batch['target']
    
    # Random pairing
    indices = torch.randperm(x.size(0))
    x2, y2 = x[indices], y[indices]
    
    # Mixup
    x_mixed, y_mixed = mixup_eeg(x, y, x2, y2)
    
    # Train on mixed samples
    pred = model(x_mixed)
    loss = criterion(pred, y_mixed)
```

### 2. **CutMix for EEG**

```python
def cutmix_eeg(x1, y1, x2, y2, alpha=1.0):
    """
    CutMix: Cut and paste EEG segments
    """
    lam = np.random.beta(alpha, alpha)
    
    # Random temporal window
    n_times = x1.size(2)
    cut_len = int(n_times * (1 - lam))
    cut_start = np.random.randint(0, n_times - cut_len)
    
    # Create mixed sample
    x_mixed = x1.clone()
    x_mixed[:, :, cut_start:cut_start+cut_len] = x2[:, :, cut_start:cut_start+cut_len]
    
    # Adjust labels
    y_mixed = lam * y1 + (1 - lam) * y2
    
    return x_mixed, y_mixed
```

### 3. **Focal Loss for Imbalanced Data**

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

### 4. **Sharpness-Aware Minimization (SAM)**

```python
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization
    Finds flatter minima -> better generalization
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # Climb to local maximum
                self.state[p]['e_w'] = e_w
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # Return to original point
        
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# Usage:
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

for epoch in range(epochs):
    for batch in dataloader:
        # First forward-backward pass
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # Second forward-backward pass
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
```

---

## 📈 Ensemble Methods

### 1. **Model Soup**

```python
def model_soup(model_paths, output_path):
    """
    Average weights of multiple fine-tuned models
    Often better than single best model
    """
    # Load all models
    models = [torch.load(path) for path in model_paths]
    
    # Average weights
    soup = {}
    for key in models[0].keys():
        soup[key] = torch.stack([m[key] for m in models]).mean(dim=0)
    
    # Save averaged model
    torch.save(soup, output_path)
    return soup
```

### 2. **Snapshot Ensembling**

```python
class SnapshotEnsemble:
    """
    Save models at different points during training
    Ensemble them for final prediction
    """
    def __init__(self, model, n_snapshots=5):
        self.base_model = model
        self.snapshots = []
        self.n_snapshots = n_snapshots
    
    def save_snapshot(self):
        snapshot = copy.deepcopy(self.base_model.state_dict())
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots
        if len(self.snapshots) > self.n_snapshots:
            self.snapshots.pop(0)
    
    def predict(self, x):
        predictions = []
        for snapshot in self.snapshots:
            self.base_model.load_state_dict(snapshot)
            self.base_model.eval()
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
```

---

## 🎯 Recommended Approach for Competition

Based on the analysis of your competition results, here's a recommended strategy:

### Phase 1: Fix Immediate Issues
```python
# 1. Use proven Challenge 1 weights (NRMSE 1.002)
# 2. Keep improved Challenge 2 weights (NRMSE 1.009)
# ✅ Already done in submission_fixed.zip
```

### Phase 2: Improve Challenge 1 (if needed)
```python
# Architecture: Try Conformer or EEGNet with attention
model = EEGConformer(
    n_chans=129,
    n_times=200,
    n_outputs=1,
    n_filters=40,
    att_depth=2,
    att_heads=4,
    drop_prob=0.5
)

# Training strategy:
# 1. Subject-level cross-validation (GroupKFold)
# 2. Mixup/CutMix augmentation (light)
# 3. SAM optimizer for better generalization
# 4. Focal loss if data imbalanced
# 5. Snapshot ensembling

optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-4)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### Phase 3: Fine-tune Challenge 2
```python
# Current approach is working (1.009)
# Try these improvements:

# 1. Self-supervised pretraining
ssl_model = EEG_MAE(n_chans=129, n_times=200)
pretrain_mae(ssl_model, unlabeled_dataloader, epochs=50)

# 2. Fine-tune on task
model = ssl_model.encoder
# Add task-specific head
model.classifier = nn.Linear(model.output_dim, 1)

# 3. Ensemble with current model
final_pred = 0.5 * model1(x) + 0.5 * model2(x)
```

---

## 📚 Key Papers & Resources

1. **Conformer** (2020): "Conformer: Convolution-augmented Transformer"
2. **EEGNet** (2018): "EEGNet: A compact convolutional neural network"
3. **Graph Neural Networks** (2021): "Spectral-Spatial Graph Reasoning Network"
4. **Self-Supervised Learning** (2022): "Self-Supervised Learning for EEG"
5. **Spiking Neural Networks** (2023): "Deep Learning with Spiking Neurons"
6. **SAM** (2021): "Sharpness-Aware Minimization for Efficiently Improving Generalization"
7. **Model Soup** (2022): "Model soups: averaging weights of multiple fine-tuned models"

---

## 🚀 Quick Implementation Guide

### Step 1: Choose Architecture
```bash
# For quick improvement: Conformer
# For biological plausibility: SNN
# For connectivity modeling: GNN
# For transfer learning: MAE pretraining
```

### Step 2: Setup Training
```bash
cd /home/kevin/Projects/eeg2025
mkdir -p experiments/conformer
cd experiments/conformer
```

### Step 3: Create Training Script
See `train_challenge1_conformer.py` (will create next if requested)

### Step 4: Monitor & Iterate
```bash
# Use TensorBoard for monitoring
tensorboard --logdir experiments/conformer/logs
```

---

## ✅ Action Items

1. **Immediate**: Upload submission_fixed.zip (combines best weights)
2. **Short-term**: Implement Conformer for Challenge 1
3. **Medium-term**: Try self-supervised pretraining
4. **Long-term**: Ensemble multiple architectures

---

*Document created: October 24, 2024*  
*Latest neuroscience-inspired algorithms compiled for EEG competition*

Would you like me to implement any of these algorithms for your specific competition data?
