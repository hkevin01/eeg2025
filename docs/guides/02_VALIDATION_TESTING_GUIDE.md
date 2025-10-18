# Part 2: Validation & Testing Strategy

**Priority**: Critical 🔴  
**Dependencies**: Data acquisition completed  
**Timeline**: 3-5 days

---

## Overview

This guide covers comprehensive testing and validation needed before competition submission.

---

## Testing Pyramid

```
           ┌─────────────────────┐
           │  Integration Tests  │  ← 20% of tests
           │  (End-to-end flow)  │
           ├─────────────────────┤
           │   Component Tests   │  ← 30% of tests
           │  (Model, Data I/O)  │
           ├─────────────────────┤
           │     Unit Tests      │  ← 50% of tests
           │  (Individual funcs) │
           └─────────────────────┘
```

---

## Phase 1: Critical Tests (Must Have)

### 1.1 Data Loading Tests

**File**: `tests/test_data_loading.py`

```python
import pytest
import torch
from pathlib import Path
from src.dataio.hbn_dataset import HBNDataset

def test_dataset_initialization():
    """Test dataset can be initialized."""
    dataset = HBNDataset(
        root_dir="data/raw/hbn",
        task_type="sus",
        split="train"
    )
    assert len(dataset) > 0

def test_data_shape():
    """Test EEG data has correct shape."""
    dataset = HBNDataset(
        root_dir="data/raw/hbn",
        task_type="sus"
    )
    eeg, label = dataset[0]
    
    assert eeg.shape[0] == 128  # Channels
    assert eeg.shape[1] == 1000  # Time points (2s at 500Hz)
    assert eeg.dtype == torch.float32

def test_label_format_challenge1():
    """Test Challenge 1 labels are correct format."""
    dataset = HBNDataset(
        root_dir="data/raw/hbn",
        task_type="sus"
    )
    eeg, label = dataset[0]
    
    assert "rt" in label  # Response time
    assert "success" in label  # Success rate
    assert 0 <= label["success"] <= 1

def test_label_format_challenge2():
    """Test Challenge 2 labels are correct format."""
    dataset = HBNDataset(
        root_dir="data/raw/hbn",
        task_type="rest"
    )
    eeg, label = dataset[0]
    
    required_keys = ["p_factor", "internalizing", 
                     "externalizing", "attention"]
    for key in required_keys:
        assert key in label

def test_missing_data_handling():
    """Test dataset handles missing data gracefully."""
    # Test with subjects that have missing labels
    dataset = HBNDataset(
        root_dir="data/raw/hbn",
        task_type="rest",
        handle_missing="skip"
    )
    # Should not crash
    assert len(dataset) > 0

def test_cross_site_loading():
    """Test data from multiple sites can be loaded."""
    dataset = HBNDataset(
        root_dir="data/raw/hbn",
        sites=["RU", "CBIC", "SI", "CUNY"]
    )
    
    # Check we have data from multiple sites
    sites_found = set()
    for i in range(min(100, len(dataset))):
        _, metadata = dataset.get_metadata(i)
        sites_found.add(metadata.get("site", "unknown"))
    
    assert len(sites_found) > 1

def test_batch_loading():
    """Test DataLoader batching works correctly."""
    from torch.utils.data import DataLoader
    
    dataset = HBNDataset(root_dir="data/raw/hbn")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    batch = next(iter(loader))
    eeg, labels = batch
    
    assert eeg.shape[0] == 16  # Batch size
    assert eeg.shape[1] == 128  # Channels
    assert eeg.shape[2] == 1000  # Time points
```

### 1.2 Model Forward Pass Tests

**File**: `tests/test_model_forward.py`

```python
import pytest
import torch
from src.models.advanced_foundation_model import AdvancedFoundationModel

@pytest.fixture
def model():
    """Create model instance for testing."""
    return AdvancedFoundationModel(
        n_channels=128,
        n_classes=1,
        backbone="transformer"
    )

def test_model_forward_shape(model):
    """Test model output shape is correct."""
    batch_size = 8
    n_channels = 128
    n_timepoints = 1000
    
    x = torch.randn(batch_size, n_channels, n_timepoints)
    output = model(x)
    
    assert output.shape == (batch_size, 1)

def test_model_backward(model):
    """Test backpropagation works."""
    x = torch.randn(1, 128, 1000)
    output = model(x)
    loss = output.mean()
    
    # Should not crash
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

def test_model_inference_mode(model):
    """Test model can run in inference mode."""
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(1, 128, 1000)
        output = model(x)
    
    assert output.requires_grad == False

def test_model_different_seq_lengths(model):
    """Test model handles variable sequence lengths."""
    # Test with different lengths
    for seq_len in [500, 1000, 2000]:
        x = torch.randn(1, 128, seq_len)
        output = model(x)
        assert output.shape[0] == 1

def test_model_multi_task_heads(model):
    """Test multi-task prediction heads."""
    model = AdvancedFoundationModel(
        n_channels=128,
        task_config={
            "p_factor": "regression",
            "internalizing": "regression",
            "externalizing": "regression",
            "attention": "regression",
            "diagnosis": "binary"
        }
    )
    
    x = torch.randn(1, 128, 1000)
    outputs = model(x)
    
    assert "p_factor" in outputs
    assert "diagnosis" in outputs
```

### 1.3 Metrics Validation Tests

**File**: `tests/test_official_metrics.py`

```python
import pytest
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import (
    compute_challenge1_metrics,
    compute_challenge2_metrics
)

def test_challenge1_pearson_correlation():
    """Test Challenge 1 correlation metric."""
    # Perfect correlation
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    corr, _ = pearsonr(y_true, y_pred)
    assert abs(corr - 1.0) < 0.001
    
    # No correlation
    y_pred = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    corr, _ = pearsonr(y_true, y_pred)
    assert abs(corr) < 0.5

def test_challenge1_auroc():
    """Test Challenge 1 AUROC metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    
    auroc = roc_auc_score(y_true, y_pred)
    assert auroc == 1.0

def test_challenge1_combined_score():
    """Test Challenge 1 combined metric."""
    rt_true = np.random.randn(100) * 100 + 500
    rt_pred = rt_true + np.random.randn(100) * 50
    
    success_true = np.random.randint(0, 2, 100)
    success_pred = success_true + np.random.randn(100) * 0.1
    success_pred = np.clip(success_pred, 0, 1)
    
    metrics = compute_challenge1_metrics(
        rt_true, rt_pred,
        success_true, success_pred
    )
    
    assert "rt_correlation" in metrics
    assert "success_auroc" in metrics
    assert "combined_score" in metrics
    assert 0 <= metrics["combined_score"] <= 1

def test_challenge2_correlation():
    """Test Challenge 2 correlation metrics."""
    y_true = {
        "p_factor": np.random.randn(100),
        "internalizing": np.random.randn(100),
        "externalizing": np.random.randn(100),
        "attention": np.random.randn(100)
    }
    
    # Add noise to predictions
    y_pred = {
        k: v + np.random.randn(100) * 0.5 
        for k, v in y_true.items()
    }
    
    metrics = compute_challenge2_metrics(y_true, y_pred)
    
    assert "p_factor_corr" in metrics
    assert "average_correlation" in metrics
    assert -1 <= metrics["average_correlation"] <= 1
```

---

## Phase 2: Performance Tests

### 2.1 Inference Latency Test

**File**: `tests/test_inference_speed.py`

```python
import pytest
import time
import torch
from src.models.advanced_foundation_model import AdvancedFoundationModel

@pytest.fixture
def optimized_model():
    """Create optimized model for inference."""
    model = AdvancedFoundationModel(
        n_channels=128,
        n_classes=1
    )
    model.eval()
    
    # Quantize if CUDA available
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def test_single_inference_latency(optimized_model):
    """Test single sample inference is <50ms."""
    x = torch.randn(1, 128, 1000)
    
    if torch.cuda.is_available():
        x = x.cuda()
    
    # Warmup
    for _ in range(10):
        _ = optimized_model(x)
    
    # Measure
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = optimized_model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time_ms = np.mean(times) * 1000
    p95_time_ms = np.percentile(times, 95) * 1000
    
    print(f"\nInference latency:")
    print(f"  Mean: {avg_time_ms:.2f}ms")
    print(f"  P95: {p95_time_ms:.2f}ms")
    
    assert avg_time_ms < 50, f"Inference too slow: {avg_time_ms:.2f}ms"
    assert p95_time_ms < 75, f"P95 latency too slow: {p95_time_ms:.2f}ms"

def test_batch_inference_throughput(optimized_model):
    """Test batch inference throughput."""
    batch_sizes = [1, 8, 16, 32]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 128, 1000)
        if torch.cuda.is_available():
            x = x.cuda()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = optimized_model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        
        throughput = batch_size / duration
        print(f"Batch {batch_size}: {throughput:.1f} samples/sec")
```

### 2.2 Memory Usage Test

**File**: `tests/test_memory_usage.py`

```python
import pytest
import torch
from src.models.advanced_foundation_model import AdvancedFoundationModel

def test_model_memory_footprint():
    """Test model fits in GPU memory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model = AdvancedFoundationModel(
        n_channels=128,
        n_classes=1
    ).cuda()
    
    # Forward pass
    x = torch.randn(32, 128, 1000).cuda()
    output = model(x)
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print(f"\nPeak GPU memory: {peak_memory_mb:.2f} MB")
    
    # Should fit in 8GB GPU with headroom
    assert peak_memory_mb < 6000, f"Memory usage too high: {peak_memory_mb:.2f}MB"
```

---

## Phase 3: Integration Tests

### 3.1 End-to-End Training Test

**File**: `tests/test_end_to_end_training.py`

```python
import pytest
from pathlib import Path
from src.training.train_cross_task import train_challenge1

def test_challenge1_training_smoke():
    """Test Challenge 1 training runs without crashing."""
    config = {
        "data_dir": "data/raw/hbn",
        "output_dir": "outputs/test_run",
        "epochs": 2,  # Just 2 epochs for testing
        "batch_size": 8,
        "max_samples": 100  # Use small subset
    }
    
    # Should complete without errors
    metrics = train_challenge1(config)
    
    assert "loss" in metrics
    assert "rt_correlation" in metrics
    
def test_challenge2_training_smoke():
    """Test Challenge 2 training runs without crashing."""
    from src.training.train_psych import train_challenge2
    
    config = {
        "data_dir": "data/raw/hbn",
        "output_dir": "outputs/test_run",
        "epochs": 2,
        "batch_size": 8,
        "max_samples": 100
    }
    
    metrics = train_challenge2(config)
    
    assert "loss" in metrics
    assert "p_factor_corr" in metrics
```

---

## Running All Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loading.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Run only critical tests
pytest tests/ -m "critical"
```

---

## Test Coverage Goals

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| Data Loading | 90% | Critical |
| Model Forward | 85% | Critical |
| Metrics | 95% | Critical |
| Preprocessing | 80% | High |
| Training Loop | 70% | High |
| Utilities | 60% | Medium |

---

## Next Steps After Testing

1. ✅ All critical tests passing
2. ✅ Inference latency <50ms
3. ✅ Data loading validated
4. → Move to model training
5. → Run cross-validation
6. → Generate submissions

