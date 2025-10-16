# Phase 2: Task-Specific Feature Extraction Plan

**Created:** October 16, 2025  
**Status:** Planning (Execute after Phase 1 results)  
**Competition Rules:** ✅ All methods below are explicitly allowed

---

## 📋 Phase 1 vs Phase 2 Strategy

### Phase 1 (CURRENT - Running Now)
- **Strategy**: Multi-release training only
- **Training Data**: R1-R4 (240 datasets)
- **Validation Data**: R5 (60 datasets)
- **Features**: Raw EEG time-series
- **Models**: Compact CNNs (200K/150K params)
- **Expected Results**: Tomorrow morning (~10 AM)
- **Goal**: Fix distribution shift (test on R12 after training on R1-R4)

### Phase 2 (IF NEEDED - After Phase 1 Analysis)
- **Strategy**: Task-specific feature engineering + multi-release
- **Training Data**: Same (R1-R4) but with enhanced features
- **Features**: Task-specific extractions (see below)
- **Models**: Task-aware architectures
- **Goal**: Further improve scores if Phase 1 doesn't reach target

---

## 🎯 Decision Criteria: When to Execute Phase 2

### Execute Phase 2 IF:
- ✅ Challenge 1 NRMSE > 1.0 (target: < 1.0)
- ✅ Challenge 2 NRMSE > 0.4 (target: < 0.4)
- ✅ Overall score > 0.8 (target: < 0.7 for top 3)

### Skip Phase 2 IF:
- ✅ Challenge 1 NRMSE < 0.8 (excellent)
- ✅ Challenge 2 NRMSE < 0.3 (excellent)
- ✅ Overall score < 0.6 (top 3 material)
- ✅ Only 2-3 days until deadline (too risky)

---

## 🧠 Task-Specific Feature Extraction Methods

### Passive Tasks (For Baseline Neural Activity)

#### 1. Resting State (RS)
**Purpose**: Capture baseline neural efficiency and network properties

**Spectral Features**:
```python
# Power Spectral Density (PSD) across frequency bands
bands = {
    'delta': (0.5, 4),   # Deep sleep, unconscious processing
    'theta': (4, 8),      # Memory, meditation
    'alpha': (8, 13),     # Relaxed wakefulness, inhibition
    'beta': (13, 30),     # Active thinking, focus
    'gamma': (30, 50)     # Higher cognitive processing
}

# Extract:
- Band power (absolute and relative)
- Peak frequency per band
- Spectral entropy (signal complexity)
- Frontal alpha asymmetry (emotion regulation)
```

**Connectivity Features**:
```python
# Functional Connectivity
- Phase Locking Value (PLV): Synchronization between channels
- Coherence: Frequency-specific coupling
- Graph metrics:
  * Clustering coefficient
  * Path length
  * Small-world index
  * Hub identification
```

**Architecture**:
```python
class RestingStateEncoder(nn.Module):
    def __init__(self):
        self.spectral_branch = SpectralCNN(bands=5)
        self.connectivity_branch = GraphNN(nodes=129)
        self.fusion = AttentionFusion()
```

#### 2. Surround Suppression (SuS)
**Purpose**: Visual processing efficiency and cortical inhibition

**Event-Related Features**:
```python
# Visual Processing Markers
- Steady-state visual evoked potentials (SSVEP)
- Gamma oscillations (30-50 Hz) in occipital regions
- Alpha suppression during stimulus (visual attention)
- Response amplitude modulation

# Channels of Interest:
occipital = ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8']
parietal = ['P3', 'P4', 'Pz', 'P1', 'P2']
```

**Spatial Attention**:
```python
# Analyze spatial patterns
- Contralateral vs ipsilateral response
- Retinotopic mapping
- Center-surround contrast effects
- Lateral inhibition strength

class SurroundSuppressionEncoder(nn.Module):
    def __init__(self):
        self.visual_cortex_cnn = SpatialCNN(occipital_channels)
        self.gamma_extractor = WaveletTransform(30, 50)
        self.attention_map = SpatialAttention()
```

#### 3. Movie Watching (MW)
**Purpose**: Naturalistic processing and attentional engagement

**Inter-Subject Correlation (ISC)**:
```python
# Measure neural synchrony across subjects
- ISC per channel (how similar are responses?)
- Peak ISC moments (shared attention events)
- ISC in frequency bands (which frequencies sync?)

def compute_isc(subject_signals):
    # Correlate each subject with average of others
    isc_map = []
    for subj in subjects:
        others_avg = mean(subjects - subj)
        correlation = pearsonr(subj, others_avg)
        isc_map.append(correlation)
    return isc_map
```

**Dynamic Connectivity**:
```python
# Time-varying network analysis
- Sliding window connectivity (10s windows)
- State transitions (clustering connectivity patterns)
- Engagement level (overall connectivity strength)
- Frontal-parietal coupling (attention network)

class MovieWatchingEncoder(nn.Module):
    def __init__(self):
        self.temporal_transformer = TemporalTransformer()
        self.dynamic_graph = DynamicGraphNN()
        self.engagement_predictor = AttentionPredictor()
```

### Active Tasks (For Cognitive Performance)

#### 4. Contrast Change Detection (CCD) - **CHALLENGE 1 PRIMARY**
**Purpose**: Response time prediction via decision-making markers

**Event-Related Potentials (ERPs)**:
```python
# Key ERP Components
components = {
    'N1': (80, 120, ['Oz', 'O1', 'O2']),      # Early visual processing
    'P1': (100, 150, ['Oz', 'O1', 'O2']),     # Visual attention
    'N2': (200, 350, ['Fz', 'FCz', 'Cz']),    # Conflict detection
    'P3/P300': (300, 600, ['Pz', 'CPz']),     # Decision-making
    'CNV': (-200, 0, ['Cz', 'FCz']),          # Motor preparation
}

# Extract for each trial:
- Peak amplitude
- Peak latency (correlates with RT!)
- Area under curve
- Slope (faster rise = faster RT)
```

**Oscillatory Dynamics**:
```python
# Time-Frequency Analysis
- Pre-stimulus alpha (8-13 Hz): Baseline attention
- Stimulus-evoked theta (4-8 Hz): Decision processing
- Post-stimulus beta (13-30 Hz): Motor preparation
- Response-locked gamma (30-50 Hz): Motor execution

# Key Insight: Earlier P300 peak = faster response time!

class CCDResponseTimeEncoder(nn.Module):
    def __init__(self):
        # ERP extraction
        self.erp_extractor = ERPNet(components=['P300', 'N2'])
        
        # Oscillatory features
        self.theta_extractor = BandPassNet(4, 8)
        self.beta_extractor = BandPassNet(13, 30)
        
        # Motor preparation
        self.motor_readiness = MotorReadinessNet(['C3', 'C4', 'Cz'])
        
        # Fusion for RT prediction
        self.rt_predictor = nn.Sequential(
            AttentionFusion(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Response time
        )
```

**Critical Channels for CCD**:
```python
visual = ['Oz', 'O1', 'O2', 'PO3', 'PO4']     # Stimulus processing
frontal = ['Fz', 'FCz', 'F3', 'F4']           # Decision-making
central = ['Cz', 'C3', 'C4']                   # Motor preparation
parietal = ['Pz', 'CPz', 'P3', 'P4']          # Integration
```

#### 5. Sequence Learning (SL)
**Purpose**: Working memory capacity assessment

**Memory Load Markers**:
```python
# Theta Oscillations (4-8 Hz)
- Frontal midline theta (Fz, FCz): Working memory load
- Theta power increases with sequence length
- Theta-gamma coupling: Memory encoding/retrieval

# P300 Component
- Amplitude decreases with memory load
- Latency increases with difficulty

class SequenceLearningEncoder(nn.Module):
    def __init__(self):
        self.theta_tracker = FrontalThetaNet()
        self.memory_load = MemoryLoadEstimator()
        self.learning_curve = LearningRatePredictor()
```

**Capacity Estimation**:
```python
# Working Memory Metrics
- Maximum sequence length learned
- Error patterns (serial position effects)
- Learning rate (trials to criterion)
- Theta power as capacity marker
```

#### 6. Symbol Search (SyS)
**Purpose**: Processing speed quantification

**Speed Markers**:
```python
# N2pc Component (200-300ms)
- Posterior contralateral negativity
- Marks attentional shift to target
- Amplitude = attention efficiency

# Gamma Oscillations (30-80 Hz)
- Higher gamma = faster processing
- Posterior channels during search

class SymbolSearchEncoder(nn.Module):
    def __init__(self):
        self.n2pc_extractor = N2pcNet()
        self.gamma_speed = GammaSpeedNet(30, 80)
        self.attention_shift = SpatialAttentionTracker()
```

---

## 🏗️ Phase 2 Architecture Options

### Option 1: Task-Aware CNN (Recommended)
```python
class TaskAwareCNN(nn.Module):
    def __init__(self, tasks=['RS', 'SuS', 'MW', 'CCD', 'SL', 'SyS']):
        super().__init__()
        
        # Task-specific encoders
        self.task_encoders = nn.ModuleDict({
            'RS': RestingStateEncoder(),
            'SuS': SurroundSuppressionEncoder(),
            'MW': MovieWatchingEncoder(),
            'CCD': CCDResponseTimeEncoder(),
            'SL': SequenceLearningEncoder(),
            'SyS': SymbolSearchEncoder(),
        })
        
        # Shared representation
        self.shared_encoder = SharedCNN()
        
        # Task embeddings
        self.task_embedding = nn.Embedding(len(tasks), 128)
        
        # Fusion
        self.fusion = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        
    def forward(self, x, task_name):
        # Task-specific features
        task_features = self.task_encoders[task_name](x)
        
        # Shared features
        shared_features = self.shared_encoder(x)
        
        # Task embedding
        task_emb = self.task_embedding(task_name)
        
        # Fuse
        fused = self.fusion(task_features, shared_features, task_emb)
        return fused
```

### Option 2: Multi-Task Learning
```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared trunk
        self.shared_trunk = SharedEncoder()
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'challenge1': ResponseTimeHead(),
            'challenge2': ExternalizingHead(),
        })
        
        # Auxiliary tasks for pretraining
        self.aux_tasks = nn.ModuleDict({
            'age_prediction': AgeHead(),
            'task_classification': TaskClassifierHead(),
            'sex_prediction': SexHead(),
        })
```

### Option 3: Temporal Transformer
```python
class TemporalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Patch embedding (split time series into patches)
        self.patch_embed = PatchEmbedding(patch_size=20)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding()
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        
        # Channel attention
        self.channel_attention = ChannelAttention(n_channels=129)
```

---

## 📊 Implementation Roadmap

### Step 1: Feature Extraction Library (2-3 hours)
```bash
# Create feature extraction utilities
scripts/features/
├── spectral.py          # PSD, band power, entropy
├── connectivity.py      # PLV, coherence, graph metrics
├── erp.py              # ERP component extraction
├── oscillatory.py      # Time-frequency analysis
└── spatial.py          # Spatial attention maps
```

### Step 2: Task-Specific Datasets (2 hours)
```python
# Modify data loaders to include features
class TaskAwareDataset(Dataset):
    def __init__(self, release, task):
        self.raw_data = load_raw(release, task)
        self.features = extract_task_features(self.raw_data, task)
        
    def __getitem__(self, idx):
        return {
            'raw': self.raw_data[idx],
            'features': self.features[idx],
            'task': self.task_name,
            'target': self.targets[idx]
        }
```

### Step 3: Train Task-Aware Models (6-8 hours)
```bash
# Challenge 1 with task-specific features
python scripts/train_challenge1_task_aware.py \
    --tasks RS SuS MW CCD \
    --features erp oscillatory spatial \
    --releases R1 R2 R3 R4 \
    --validate R5

# Challenge 2 with all tasks
python scripts/train_challenge2_task_aware.py \
    --tasks RS SuS MW CCD SL SyS \
    --features spectral connectivity \
    --releases R1 R2 R3 R4 \
    --validate R5
```

### Step 4: Ensemble (2 hours)
```python
# Combine Phase 1 and Phase 2 models
ensemble = Ensemble([
    phase1_model,          # Raw CNN
    phase2_task_aware,     # Task-specific features
    phase2_spectral,       # Spectral features only
])
```

---

## 🎯 Expected Improvements

### Challenge 1: Response Time Prediction
**Current (Phase 1 expected)**: 1.4 NRMSE  
**With ERP/P300 features**: 0.9-1.1 NRMSE  
**With full task-aware**: 0.7-0.9 NRMSE  
**Target**: < 0.8 NRMSE (top 3)

### Challenge 2: Externalizing Prediction
**Current (Phase 1 expected)**: 0.5 NRMSE  
**With spectral features**: 0.35-0.45 NRMSE  
**With multi-task**: 0.25-0.35 NRMSE  
**Target**: < 0.3 NRMSE (top 3)

### Overall Score
**Phase 1**: ~0.8 NRMSE (competitive)  
**Phase 2**: ~0.5 NRMSE (top 3)  
**Stretch Goal**: < 0.4 NRMSE (top 1-2)

---

## ⚠️ Risks & Mitigation

### Risk 1: Overfitting to Task-Specific Features
**Mitigation**:
- Strong regularization (dropout 0.5, weight decay 1e-4)
- Cross-release validation (R1-R4 train, R5 validate)
- Feature selection (only keep most important)

### Risk 2: Increased Complexity
**Mitigation**:
- Start simple (add one feature type at a time)
- Ablation studies (measure each feature's contribution)
- Keep Phase 1 model as backup

### Risk 3: Time Constraints
**Mitigation**:
- Only execute if Phase 1 results insufficient
- Prioritize highest-impact features (P300 for C1, spectral for C2)
- Have rollback plan to Phase 1 model

### Risk 4: Resource Limits (20 GB inference)
**Mitigation**:
- Pre-compute features, save as cache
- Use lightweight feature extractors
- Distill complex models to smaller ones

---

## 📝 Documentation Requirements

### For Competition Submission
Must document in methods paper:
1. Which tasks used for training
2. Which features extracted per task
3. Architecture modifications
4. Pre-training strategy
5. Any external libraries used

### Example Text:
```
For Challenge 1, we employed task-specific feature extraction 
from the Contrast Change Detection (CCD) task. Specifically, 
we extracted event-related potentials (P300, N2) using MNE-Python 
and computed time-frequency representations using wavelet analysis. 
These features were concatenated with raw EEG and fed into a 
task-aware convolutional neural network with attention mechanisms.

For Challenge 2, we utilized all six cognitive tasks (RS, SuS, MW, 
CCD, SL, SyS) with task-specific preprocessing. Resting state data 
was analyzed using spectral decomposition and graph-based 
connectivity metrics. Active tasks were processed using 
event-related analysis. A multi-task learning framework with 
shared representations was employed to predict externalizing scores.
```

---

## ✅ Competition Rules Compliance

All methods in this plan are explicitly allowed per competition rules:

✅ **Multiple tasks for training**: "Teams can leverage multiple datasets and experimental paradigms"  
✅ **Task-specific preprocessing**: No restrictions on preprocessing methods  
✅ **Advanced architectures**: "Encouraged to use...self-supervised pretraining"  
✅ **Pre-training strategies**: "Utilize unsupervised or self-supervised pretraining"  
✅ **Feature engineering**: No restrictions mentioned  
✅ **Ensemble methods**: Standard practice, allowed  

❌ **NOT allowed**: Training during inference, using test data, exceeding resource limits

---

## 🚦 Go/No-Go Decision Tree

```
Phase 1 Results
    ↓
Is Overall Score < 0.7?
    ↓ YES → Submit Phase 1, celebrate! 🎉
    ↓ NO
    ↓
Is Overall Score < 1.0?
    ↓ YES → Execute Phase 2 (Quick wins: ERP, spectral)
    ↓ NO
    ↓
Is Overall Score > 1.5?
    ↓ YES → Execute Full Phase 2 (All features, ensemble)
    ↓
Days Until Deadline < 3?
    ↓ YES → Submit best we have, avoid risk
    ↓ NO → Execute Phase 2
```

---

## 📚 References & Tools

### Key Libraries
- **MNE-Python**: ERP extraction, time-frequency analysis
- **SciPy**: Signal processing, filtering, FFT
- **NetworkX**: Graph metrics for connectivity
- **PyTorch Geometric**: Graph neural networks (optional)
- **Braindecode**: EEG-specific deep learning utilities

### Key Papers
1. P300 & Response Time: Kutas et al. (1977), Verleger (1997)
2. Theta & Working Memory: Klimesch (1999), Jensen & Tesche (2002)
3. Connectivity Analysis: Bastos & Schoffelen (2016)
4. Multi-Task Learning for EEG: Lawhern et al. (2018)

### Existing Code to Leverage
- `src/features/`: Basic feature extraction (needs extension)
- `src/models/task_aware.py`: Task embedding infrastructure
- `scripts/visualize_features.py`: Feature importance analysis

---

## 🎯 Summary

**Phase 2 is a well-defined, competition-legal enhancement strategy** that:
1. ✅ Builds on Phase 1's multi-release foundation
2. ✅ Leverages neuroscience-grounded features
3. ✅ Has clear decision criteria for execution
4. ✅ Fully compliant with competition rules
5. ✅ Has measured risks with mitigation plans

**Execute Phase 2 only if Phase 1 results indicate it's necessary to reach top 3.**

**Estimated Phase 2 timeline**: 12-16 hours of work over 2-3 days

---

**Next Action**: Wait for Phase 1 results (tomorrow ~10 AM), then decide!
