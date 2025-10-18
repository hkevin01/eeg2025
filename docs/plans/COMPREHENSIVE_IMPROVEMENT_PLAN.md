# 🚀 Comprehensive Improvement Implementation Plan

**Created**: October 17, 2025  
**Purpose**: Systematic implementation of code quality, testing, and robustness improvements  
**Scope**: 6 major improvement areas across the entire codebase

---

## 📋 Implementation Checklist

```markdown
### Phase 1: Documentation & Architecture ✅
- [x] Enhanced README.md with Mermaid diagrams
- [x] Architecture explanations with WHY justifications
- [x] Performance tables and metrics
- [x] Dark-themed diagram styling
- [ ] Update METHODS_DOCUMENT.pdf with techniques
- [ ] Add mathematical formulations to methods doc

### Phase 2: Methods Analysis & Refactoring
- [ ] Audit all functions in submission.py
- [ ] Audit all training scripts
- [ ] Identify duplicate code patterns
- [ ] Refactor to single-responsibility principle
- [ ] Create reusable utility modules
- [ ] Document all public APIs

### Phase 3: Unit Testing Framework
- [ ] Set up pytest infrastructure
- [ ] Write tests for data loading (nominal cases)
- [ ] Write tests for data loading (off-nominal cases)
- [ ] Write tests for model architectures
- [ ] Write tests for training utilities
- [ ] Write tests for preprocessing functions
- [ ] Achieve >80% code coverage

### Phase 4: Time Units Standardization
- [ ] Audit all time-related variables
- [ ] Document units in docstrings (seconds/ms/samples)
- [ ] Create time conversion utilities
- [ ] Add unit validation in data loading
- [ ] Update all variable names with unit suffixes

### Phase 5: Boundary Condition Handling
- [ ] Add input validation for all public functions
- [ ] Test min/max/edge cases for data shapes
- [ ] Test empty/null/invalid inputs
- [ ] Add defensive assertions
- [ ] Implement graceful error messages

### Phase 6: Persistence & Checkpointing
- [ ] Standardize checkpoint format (JSON schema)
- [ ] Add model metadata to checkpoints
- [ ] Implement retry logic for file I/O
- [ ] Add checkpoint versioning
- [ ] Create checkpoint validation utility

### Phase 7: Error Handling & Logging
- [ ] Set up centralized logging system
- [ ] Add timestamps to all logs
- [ ] Log function entry/exit for key methods
- [ ] Add structured error types
- [ ] Implement crash recovery mechanisms
- [ ] Store crash data for debugging

### Phase 8: Monitoring Tools Organization
- [ ] Create tools/ directory
- [ ] Move all monitoring scripts to tools/
- [ ] Consolidate duplicate monitors
- [ ] Add health check dashboard
- [ ] Create unified monitoring interface
```

---

## 🎯 Part 1: Methods Analysis & Improvement

### Objective
Analyze all methods and functions for efficiency, readability, and maintainability. Refactor to follow best practices.

### Current State Analysis

**Files to Analyze**:
1. `submission.py` (580 lines) - Main submission script
2. `scripts/train_challenge1_attention.py` (~400 lines)
3. `scripts/train_challenge2_multi_release.py` (~500 lines)
4. `src/dataio/*.py` - Data loading modules
5. `src/models/**/*.py` - Model architectures

**Common Issues Found**:
- ❌ Long functions (>100 lines)
- ❌ Duplicate preprocessing code
- ❌ Mixed concerns (I/O + computation + visualization)
- ❌ Unclear variable names
- ❌ Missing docstrings

### Refactoring Strategy

#### 1.1 Extract Common Utilities

**Create `src/utils/preprocessing.py`**:
```python
"""
Common preprocessing utilities for EEG data.

All functions follow these conventions:
- Input: NumPy arrays (channels, time)
- Output: NumPy arrays (same shape unless specified)
- Units: Documented in docstrings
"""

def bandpass_filter_eeg(data: np.ndarray, 
                        fs: float = 100.0,
                        lowcut: float = 0.5, 
                        highcut: float = 50.0) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Args:
        data: EEG data (channels, time_points)
        fs: Sampling frequency in Hz
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        
    Returns:
        Filtered EEG data (channels, time_points)
        
    Raises:
        ValueError: If lowcut >= highcut or fs <= 0
        
    Example:
        >>> eeg = np.random.randn(129, 1000)  # 129 channels, 10 seconds at 100Hz
        >>> filtered = bandpass_filter_eeg(eeg, fs=100.0)
        >>> filtered.shape
        (129, 1000)
    """
    # Validation
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) must be < highcut ({highcut})")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (channels, time), got shape {data.shape}")
    
    # Implementation
    # ... existing bandpass code ...
    
    return filtered_data
```

#### 1.2 Separate Concerns

**Before** (Mixed concerns):
```python
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x, y = batch
        x = x.to(device)  # Device management
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Logging mixed in
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")
    
    return total_loss / len(dataloader)
```

**After** (Separated concerns):
```python
def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                logger: logging.Logger = None) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Single responsibility: Training loop only.
    Logging and device management delegated to caller.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        logger: Optional logger for progress
        
    Returns:
        Dictionary with metrics:
            - 'loss': Average loss over epoch
            - 'samples': Total samples processed
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        # Move data to device (caller's responsibility to set up device)
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        predictions = model(x)
        loss = criterion(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * x.size(0)
        num_samples += x.size(0)
        
        # Optional logging (delegated to logger)
        if logger and batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return {
        'loss': total_loss / num_samples,
        'samples': num_samples
    }
```

#### 1.3 Function Length Guidelines

**Target**: Max 50 lines per function

**Strategy**:
1. Extract helper functions
2. Use early returns for validation
3. Decompose complex logic

**Example Refactoring**:
```python
# Before: 150-line function
def process_and_train_model(data_path, config):
    # Load data (50 lines)
    # Preprocess (40 lines)
    # Create model (20 lines)
    # Train (40 lines)
    pass

# After: Multiple focused functions
def load_dataset(data_path: Path) -> Dict[str, np.ndarray]:
    """Load raw EEG dataset. Single responsibility: I/O only."""
    pass

def preprocess_dataset(raw_data: Dict, config: PreprocessConfig) -> Dict:
    """Preprocess EEG data. Single responsibility: Preprocessing only."""
    pass

def create_model(config: ModelConfig) -> nn.Module:
    """Initialize model. Single responsibility: Model creation only."""
    pass

def train_model(model: nn.Module, data: Dict, config: TrainConfig) -> nn.Module:
    """Train model. Single responsibility: Training orchestration only."""
    pass

def process_and_train_model(data_path: Path, config: Config) -> nn.Module:
    """
    High-level orchestrator. Delegates all work to specialized functions.
    Single responsibility: Coordination only.
    """
    raw_data = load_dataset(data_path)
    processed_data = preprocess_dataset(raw_data, config.preprocess)
    model = create_model(config.model)
    trained_model = train_model(model, processed_data, config.train)
    return trained_model
```

### Action Items

**Week 1** (High Priority):
1. Create `src/utils/` directory structure
2. Extract common preprocessing functions
3. Extract common training utilities
4. Extract common evaluation metrics
5. Add docstrings to all public functions

**Week 2** (Medium Priority):
1. Refactor `submission.py` into modules
2. Refactor training scripts
3. Create unified configuration system
4. Consolidate duplicate code

**Week 3** (Low Priority):
1. Optimize performance bottlenecks
2. Add type hints throughout
3. Create API documentation

---

## 🧪 Part 2: Unit Testing Framework

### Objective
Create comprehensive unit tests for all methods with nominal and off-nominal cases.

### Testing Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── test_data_loading.py           # Data loading tests
├── test_preprocessing.py          # Preprocessing tests
├── test_models.py                 # Model architecture tests
├── test_training.py               # Training utilities tests
├── test_evaluation.py             # Metrics tests
├── test_utils.py                  # Utility function tests
├── test_integration.py            # End-to-end tests
└── fixtures/
    ├── sample_eeg_129ch.npy       # Test data
    ├── sample_metadata.csv
    └── sample_config.yaml
```

### Test Categories

#### 2.1 Nominal Case Tests

```python
# tests/test_preprocessing.py
import pytest
import numpy as np
from src.utils.preprocessing import bandpass_filter_eeg

class TestBandpassFilter:
    """Test suite for bandpass filter function."""
    
    def test_nominal_case_129_channels(self):
        """Test with standard 129 channels, 10 seconds at 100Hz."""
        # Arrange
        data = np.random.randn(129, 1000)
        
        # Act
        filtered = bandpass_filter_eeg(data, fs=100.0, lowcut=0.5, highcut=50.0)
        
        # Assert
        assert filtered.shape == (129, 1000), "Output shape should match input"
        assert not np.array_equal(filtered, data), "Data should be modified"
        assert np.all(np.isfinite(filtered)), "No NaN or Inf values"
    
    def test_different_sampling_rates(self):
        """Test with different sampling rates."""
        data = np.random.randn(129, 500)
        
        for fs in [50, 100, 200, 500]:
            filtered = bandpass_filter_eeg(data, fs=fs)
            assert filtered.shape == data.shape
    
    def test_preserves_dtype(self):
        """Test that float32/float64 is preserved."""
        for dtype in [np.float32, np.float64]:
            data = np.random.randn(129, 1000).astype(dtype)
            filtered = bandpass_filter_eeg(data)
            assert filtered.dtype == dtype
```

#### 2.2 Off-Nominal Case Tests

```python
class TestBandpassFilterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_lowcut_highcut(self):
        """Test that lowcut >= highcut raises ValueError."""
        data = np.random.randn(129, 1000)
        
        with pytest.raises(ValueError, match="lowcut .* must be < highcut"):
            bandpass_filter_eeg(data, lowcut=50.0, highcut=0.5)
    
    def test_invalid_sampling_rate(self):
        """Test that negative/zero sampling rate raises ValueError."""
        data = np.random.randn(129, 1000)
        
        with pytest.raises(ValueError, match="Sampling frequency must be positive"):
            bandpass_filter_eeg(data, fs=0.0)
        
        with pytest.raises(ValueError, match="Sampling frequency must be positive"):
            bandpass_filter_eeg(data, fs=-100.0)
    
    def test_wrong_dimensions(self):
        """Test that 1D or 3D arrays raise ValueError."""
        # 1D array
        data_1d = np.random.randn(1000)
        with pytest.raises(ValueError, match="Expected 2D array"):
            bandpass_filter_eeg(data_1d)
        
        # 3D array
        data_3d = np.random.randn(5, 129, 1000)
        with pytest.raises(ValueError, match="Expected 2D array"):
            bandpass_filter_eeg(data_3d)
    
    def test_empty_array(self):
        """Test with zero-length time dimension."""
        data = np.random.randn(129, 0)
        with pytest.raises(ValueError, match="time"):
            bandpass_filter_eeg(data)
    
    def test_single_timepoint(self):
        """Test with single time point."""
        data = np.random.randn(129, 1)
        with pytest.raises(ValueError, match="Insufficient data"):
            bandpass_filter_eeg(data)
    
    def test_nan_input(self):
        """Test behavior with NaN values."""
        data = np.random.randn(129, 1000)
        data[50, 500] = np.nan
        
        with pytest.raises(ValueError, match="contains NaN"):
            bandpass_filter_eeg(data)
    
    def test_inf_input(self):
        """Test behavior with Inf values."""
        data = np.random.randn(129, 1000)
        data[50, 500] = np.inf
        
        with pytest.raises(ValueError, match="contains Inf"):
            bandpass_filter_eeg(data)
```

#### 2.3 Integration Tests

```python
# tests/test_integration.py
def test_full_pipeline_challenge1():
    """Test complete Challenge 1 pipeline end-to-end."""
    # Load sample data
    data_path = Path("tests/fixtures/sample_eeg_129ch.npy")
    eeg_data = np.load(data_path)
    
    # Preprocess
    filtered = bandpass_filter_eeg(eeg_data)
    normalized = normalize_per_channel(filtered)
    
    # Load model
    model = SparseAttentionResponseTimeCNN()
    model.load_state_dict(torch.load("checkpoints/response_time_attention.pth"))
    model.eval()
    
    # Predict
    with torch.no_grad():
        prediction = model(torch.tensor(normalized).unsqueeze(0))
    
    # Validate
    assert prediction.shape == (1, 1)
    assert 0.0 < prediction.item() < 10.0  # Response times typically 0-10 seconds
    assert not np.isnan(prediction.item())
```

### Coverage Goals

- **Target**: >80% code coverage
- **Critical Paths**: 100% coverage
  - Data loading
  - Model forward passes
  - Metric calculations
- **Nice to Have**: >90% overall coverage

### Continuous Integration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=html --cov-report=term
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## ⏱️ Part 3: Time Units Standardization

### Objective
Standardize all time-related measurements with clear units and validation.

### Current Issues

**Inconsistent Units**:
- Response times: Sometimes seconds, sometimes milliseconds
- Sampling rates: Hz vs samples
- Window lengths: Samples vs seconds vs milliseconds
- Temporal jitter: Samples vs milliseconds

### Standardization Strategy

#### 3.1 Naming Conventions

**Always suffix variable names with units**:

```python
# ❌ Bad: Ambiguous units
response_time = 0.532
window_length = 600
jitter = 5
sampling_rate = 100

# ✅ Good: Clear units
response_time_sec = 0.532
window_length_samples = 600
jitter_ms = 50
sampling_rate_hz = 100
```

#### 3.2 Docstring Standards

```python
def extract_trial_window(eeg_data: np.ndarray,
                         event_time_sec: float,
                         before_sec: float,
                         after_sec: float,
                         sampling_rate_hz: float = 100.0) -> np.ndarray:
    """
    Extract EEG window around event.
    
    Args:
        eeg_data: EEG data (channels, time_samples)
        event_time_sec: Event timestamp in SECONDS
        before_sec: Time before event in SECONDS
        after_sec: Time after event in SECONDS
        sampling_rate_hz: Sampling rate in HERTZ (Hz)
        
    Returns:
        EEG window (channels, window_length_samples)
        
    Example:
        >>> eeg = np.random.randn(129, 10000)  # 100 seconds at 100 Hz
        >>> window = extract_trial_window(eeg, event_time_sec=5.0, 
        ...                                before_sec=0.5, after_sec=1.0,
        ...                                sampling_rate_hz=100.0)
        >>> window.shape
        (129, 150)  # 0.5+1.0 seconds = 1.5 seconds * 100 Hz = 150 samples
    """
    # Convert seconds to samples
    event_sample = int(event_time_sec * sampling_rate_hz)
    before_samples = int(before_sec * sampling_rate_hz)
    after_samples = int(after_sec * sampling_rate_hz)
    
    # Extract window
    start_sample = event_sample - before_samples
    end_sample = event_sample + after_samples
    
    return eeg_data[:, start_sample:end_sample]
```

#### 3.3 Conversion Utilities

```python
# src/utils/time_units.py
"""
Time unit conversion utilities for EEG processing.

Standard units:
- Time: SECONDS (float)
- Sampling rate: HERTZ (float)
- Indices: SAMPLES (int)
"""

def seconds_to_samples(time_sec: float, sampling_rate_hz: float) -> int:
    """
    Convert time in seconds to number of samples.
    
    Args:
        time_sec: Time in seconds
        sampling_rate_hz: Sampling rate in Hz
        
    Returns:
        Number of samples (integer)
        
    Raises:
        ValueError: If inputs are negative
    """
    if time_sec < 0:
        raise ValueError(f"time_sec must be non-negative, got {time_sec}")
    if sampling_rate_hz <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {sampling_rate_hz}")
    
    return int(time_sec * sampling_rate_hz)


def samples_to_seconds(num_samples: int, sampling_rate_hz: float) -> float:
    """
    Convert number of samples to time in seconds.
    
    Args:
        num_samples: Number of samples
        sampling_rate_hz: Sampling rate in Hz
        
    Returns:
        Time in seconds (float)
        
    Raises:
        ValueError: If sampling_rate_hz is not positive
    """
    if sampling_rate_hz <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {sampling_rate_hz}")
    
    return num_samples / sampling_rate_hz


def milliseconds_to_seconds(time_ms: float) -> float:
    """Convert milliseconds to seconds."""
    return time_ms / 1000.0


def seconds_to_milliseconds(time_sec: float) -> float:
    """Convert seconds to milliseconds."""
    return time_sec * 1000.0
```

#### 3.4 Validation Decorators

```python
# src/utils/validation.py
from functools import wraps
import inspect

def validate_time_units(func):
    """
    Decorator to validate time-related parameters have correct units in name.
    
    Checks that parameters ending in _sec, _ms, _hz, _samples have appropriate values.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name, param_value in bound_args.arguments.items():
            # Check _sec parameters
            if param_name.endswith('_sec'):
                if not isinstance(param_value, (int, float)):
                    raise TypeError(f"{param_name} must be numeric, got {type(param_value)}")
                if param_value < 0:
                    raise ValueError(f"{param_name} must be non-negative, got {param_value}")
            
            # Check _hz parameters
            if param_name.endswith('_hz'):
                if not isinstance(param_value, (int, float)):
                    raise TypeError(f"{param_name} must be numeric, got {type(param_value)}")
                if param_value <= 0:
                    raise ValueError(f"{param_name} must be positive, got {param_value}")
            
            # Check _samples parameters
            if param_name.endswith('_samples'):
                if not isinstance(param_value, int):
                    raise TypeError(f"{param_name} must be int, got {type(param_value)}")
                if param_value < 0:
                    raise ValueError(f"{param_name} must be non-negative, got {param_value}")
        
        return func(*args, **kwargs)
    
    return wrapper


# Usage example
@validate_time_units
def process_window(eeg_data: np.ndarray,
                   window_length_sec: float,
                   sampling_rate_hz: float) -> np.ndarray:
    """Process EEG window. Units are validated automatically."""
    window_samples = seconds_to_samples(window_length_sec, sampling_rate_hz)
    return eeg_data[:, :window_samples]
```

### Refactoring Checklist

- [ ] Audit all functions for time-related parameters
- [ ] Rename variables to include unit suffixes
- [ ] Update all docstrings with explicit units
- [ ] Add conversion utilities
- [ ] Add validation decorators
- [ ] Update all training scripts
- [ ] Update submission.py
- [ ] Add unit tests for conversion functions

---

## 🛡️ Part 4: Boundary Condition Handling

### Objective
Add robust boundary condition testing and defensive programming throughout.

### Boundary Conditions to Test

#### 4.1 Data Shape Boundaries

```python
def test_data_shape_boundaries():
    """Test all possible data shape edge cases."""
    
    # Minimum valid case
    assert process_eeg(np.zeros((1, 1))) is not None  # Single channel, single sample
    
    # Maximum expected case
    assert process_eeg(np.zeros((129, 100000))) is not None  # Full resolution
    
    # Empty cases
    with pytest.raises(ValueError):
        process_eeg(np.zeros((0, 100)))  # Zero channels
    
    with pytest.raises(ValueError):
        process_eeg(np.zeros((129, 0)))  # Zero samples
    
    # Wrong dimensions
    with pytest.raises(ValueError):
        process_eeg(np.zeros(129))  # 1D
    
    with pytest.raises(ValueError):
        process_eeg(np.zeros((5, 129, 1000)))  # 3D
```

#### 4.2 Value Range Boundaries

```python
def normalize_eeg(data: np.ndarray, 
                  method: str = 'standard') -> np.ndarray:
    """
    Normalize EEG data with boundary condition handling.
    
    Handles edge cases:
    - All zeros (constant signal)
    - All same value (constant signal)
    - Very small variance (near-constant)
    - NaN/Inf values
    """
    # Check for NaN/Inf
    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains NaN or Inf values")
    
    # Check for constant signal (zero variance)
    if np.allclose(data, data[0, 0]):
        warnings.warn("Data is constant (zero variance). Returning zeros.")
        return np.zeros_like(data)
    
    # Standard normalization
    if method == 'standard':
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        
        # Handle near-zero std
        std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
        
        normalized = (data - mean) / std
        
        # Final sanity check
        if not np.all(np.isfinite(normalized)):
            raise RuntimeError("Normalization produced NaN/Inf values")
        
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
```

#### 4.3 Index Boundaries

```python
def extract_window_safe(data: np.ndarray,
                        start_idx: int,
                        end_idx: int) -> np.ndarray:
    """
    Safely extract window with boundary checking.
    
    Handles:
    - Negative indices
    - Out-of-bounds indices
    - Reversed start/end
    - start == end
    """
    # Validate indices
    if start_idx < 0:
        raise IndexError(f"start_idx must be non-negative, got {start_idx}")
    if end_idx < 0:
        raise IndexError(f"end_idx must be non-negative, got {end_idx}")
    if start_idx >= end_idx:
        raise IndexError(f"start_idx ({start_idx}) must be < end_idx ({end_idx})")
    
    # Check bounds
    if end_idx > data.shape[1]:
        raise IndexError(f"end_idx ({end_idx}) exceeds data length ({data.shape[1]})")
    
    return data[:, start_idx:end_idx]
```

#### 4.4 File I/O Boundaries

```python
def load_eeg_file_robust(file_path: Path,
                         max_retries: int = 3) -> Optional[np.ndarray]:
    """
    Load EEG file with retry logic and error handling.
    
    Handles:
    - File not found
    - Permission errors
    - Corrupted files
    - Network timeouts (if remote)
    - Partial reads
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    # Check existence
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check readability
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File not readable: {file_path}")
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            data = np.load(file_path)
            
            # Validate loaded data
            if data.size == 0:
                raise ValueError("Loaded array is empty")
            
            if not np.all(np.isfinite(data)):
                raise ValueError("Loaded array contains NaN/Inf")
            
            return data
        
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                continue
            else:
                raise RuntimeError(f"Failed to load after {max_retries} attempts: {e}")
    
    return None
```

### Defensive Programming Checklist

- [ ] Add input validation to all public functions
- [ ] Check array shapes before operations
- [ ] Validate value ranges (no NaN/Inf)
- [ ] Check for zero-division cases
- [ ] Handle empty collections gracefully
- [ ] Add boundary checks for indexing
- [ ] Validate file paths before I/O
- [ ] Add type checking for critical parameters
- [ ] Use assertions for invariants
- [ ] Log all validation failures

---

## �� Part 5: Persistence Handling

### Objective
Standardize data persistence with schemas, versioning, and retry logic.

### Checkpoint Schema

```python
# src/utils/persistence.py
from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
from pathlib import Path

@dataclass
class ModelCheckpoint:
    """
    Standard checkpoint format for all models.
    
    Versioning ensures backward compatibility.
    """
    # Metadata
    version: str = "1.0"
    created_at: str = ""  # ISO 8601 timestamp
    model_name: str = ""
    challenge: int = 0  # 1 or 2
    
    # Training info
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_metric: float = 0.0  # NRMSE
    
    # Model config
    model_config: Dict[str, Any] = None
    
    # Training config
    train_config: Dict[str, Any] = None
    
    # Additional metadata
    git_commit: str = ""
    hostname: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelCheckpoint':
        """Load from dictionary."""
        return cls(**data)
    
    def save(self, checkpoint_path: Path, model_state: Dict) -> None:
        """
        Save checkpoint with metadata and model weights.
        
        Creates two files:
        - checkpoint_path.pth: Model weights (PyTorch)
        - checkpoint_path.json: Metadata (JSON)
        """
        import torch
        from datetime import datetime
        import socket
        import subprocess
        
        # Update metadata
        self.created_at = datetime.now().isoformat()
        self.hostname = socket.gethostname()
        
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            self.git_commit = git_commit
        except:
            self.git_commit = "unknown"
        
        # Save model weights
        torch.save(model_state, checkpoint_path)
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, checkpoint_path: Path) -> Tuple['ModelCheckpoint', Dict]:
        """
        Load checkpoint with metadata.
        
        Returns:
            Tuple of (metadata, model_state_dict)
        """
        import torch
        
        # Load metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = cls.from_dict(metadata_dict)
        else:
            # No metadata file, create minimal metadata
            metadata = cls()
        
        # Load model weights
        model_state = torch.load(checkpoint_path, map_location='cpu')
        
        return metadata, model_state


# Usage example
checkpoint = ModelCheckpoint(
    model_name="SparseAttentionResponseTimeCNN",
    challenge=1,
    epoch=25,
    train_loss=0.0532,
    val_loss=0.0621,
    val_metric=0.2632,
    model_config={'num_layers': 4, 'hidden_size': 256},
    train_config={'lr': 0.001, 'batch_size': 32}
)

checkpoint.save(
    Path("checkpoints/model_epoch25.pth"),
    model.state_dict()
)

# Later...
metadata, state_dict = ModelCheckpoint.load(Path("checkpoints/model_epoch25.pth"))
print(f"Loaded model trained at {metadata.created_at}")
print(f"Val NRMSE: {metadata.val_metric:.4f}")
```

### Retry Logic

```python
def save_with_retry(save_func: Callable,
                    max_retries: int = 3,
                    backoff_factor: float = 2.0) -> bool:
    """
    Execute save function with exponential backoff retry.
    
    Args:
        save_func: Function to execute (takes no args)
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for wait time
        
    Returns:
        True if successful, False otherwise
    """
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            save_func()
            logger.info(f"Save successful on attempt {attempt + 1}")
            return True
        
        except Exception as e:
            wait_time = (backoff_factor ** attempt) * 0.1
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"Save failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Save failed after {max_retries} attempts: {e}"
                )
                return False
    
    return False


# Usage
def save_operation():
    torch.save(model.state_dict(), "checkpoints/model.pth")

success = save_with_retry(save_operation, max_retries=5)
if not success:
    raise RuntimeError("Failed to save model after multiple attempts")
```

### Atomic File Operations

```python
import tempfile
import shutil

def atomic_save(data: Any, target_path: Path) -> None:
    """
    Save data atomically to prevent corruption.
    
    Strategy:
    1. Write to temporary file
    2. Verify write successful
    3. Atomic rename to target
    
    Ensures target file is never partially written.
    """
    # Create temporary file in same directory (for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".tmp_{target_path.name}_"
    )
    
    try:
        # Write to temporary file
        with os.fdopen(temp_fd, 'wb') as f:
            if isinstance(data, dict):
                torch.save(data, f)
            else:
                np.save(f, data)
        
        # Verify file exists and has content
        if not Path(temp_path).exists():
            raise IOError("Temporary file not created")
        
        if Path(temp_path).stat().st_size == 0:
            raise IOError("Temporary file is empty")
        
        # Atomic rename
        shutil.move(temp_path, target_path)
        
    except Exception as e:
        # Clean up temporary file on error
        try:
            os.remove(temp_path)
        except:
            pass
        raise e
```

### Persistence Checklist

- [ ] Create checkpoint schema dataclass
- [ ] Add metadata to all saved models
- [ ] Implement retry logic for saves
- [ ] Use atomic file operations
- [ ] Add checkpoint versioning
- [ ] Create checkpoint validation utility
- [ ] Log all persistence operations
- [ ] Add automatic backup system

---

## 🚨 Part 6: Error Handling & Logging

### Objective
Implement centralized error handling and comprehensive logging system.

### Logging System

```python
# src/utils/logging_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str,
                 log_dir: Optional[Path] = None,
                 level: int = logging.INFO,
                 console: bool = True,
                 file: bool = True) -> logging.Logger:
    """
    Set up structured logging with console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | '
        '%(filename)s:%(lineno)d | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all details
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


# Usage
logger = setup_logger(__name__, log_dir=Path("logs"))
logger.info("Training started")
logger.debug(f"Config: {config}")
logger.warning("Low GPU memory")
logger.error("Failed to load checkpoint", exc_info=True)
```

### Structured Error Types

```python
# src/utils/errors.py
class EEG2025Error(Exception):
    """Base exception for EEG2025 project."""
    pass

class DataLoadError(EEG2025Error):
    """Error loading or processing data."""
    pass

class ModelError(EEG2025Error):
    """Error in model architecture or forward pass."""
    pass

class TrainingError(EEG2025Error):
    """Error during training."""
    pass

class ValidationError(EEG2025Error):
    """Data validation error."""
    pass

class CheckpointError(EEG2025Error):
    """Error saving/loading checkpoints."""
    pass


# Usage
def load_dataset(path: Path) -> np.ndarray:
    if not path.exists():
        raise DataLoadError(f"Dataset not found: {path}")
    
    try:
        data = np.load(path)
    except Exception as e:
        raise DataLoadError(f"Failed to load {path}: {e}") from e
    
    if data.shape[0] != 129:
        raise ValidationError(
            f"Expected 129 channels, got {data.shape[0]}"
        )
    
    return data
```

### Crash Recovery

```python
# src/utils/recovery.py
import traceback
import pickle
from datetime import datetime

class CrashHandler:
    """Handle crashes and save debugging information."""
    
    def __init__(self, crash_dir: Path = Path("crash_logs")):
        self.crash_dir = crash_dir
        self.crash_dir.mkdir(parents=True, exist_ok=True)
    
    def save_crash_info(self,
                        exception: Exception,
                        context: Dict[str, Any]) -> Path:
        """
        Save crash information for debugging.
        
        Args:
            exception: The exception that caused the crash
            context: Dictionary with debugging info
                - 'model_state': model.state_dict()
                - 'optimizer_state': optimizer.state_dict()
                - 'batch_data': current batch
                - 'config': training config
                - etc.
        
        Returns:
            Path to saved crash report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_file = self.crash_dir / f"crash_{timestamp}.pkl"
        
        crash_data = {
            'timestamp': timestamp,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        # Save crash data
        with open(crash_file, 'wb') as f:
            pickle.dump(crash_data, f)
        
        # Also save human-readable report
        report_file = crash_file.with_suffix('.txt')
        with open(report_file, 'w') as f:
            f.write(f"Crash Report - {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Exception: {type(exception).__name__}\n")
            f.write(f"Message: {exception}\n\n")
            f.write("Traceback:\n")
            f.write(crash_data['traceback'])
            f.write("\n\nContext:\n")
            for key, value in context.items():
                if key.endswith('_state'):
                    f.write(f"{key}: <state dict>\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        return crash_file


# Usage in training loop
crash_handler = CrashHandler()

try:
    for epoch in range(num_epochs):
        train_epoch(model, dataloader, optimizer)
except Exception as e:
    logger.error("Training crashed!", exc_info=True)
    
    # Save crash info
    context = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': config,
        'last_loss': last_loss
    }
    crash_file = crash_handler.save_crash_info(e, context)
    logger.error(f"Crash info saved to: {crash_file}")
    
    raise
```

### Logging Checklist

- [ ] Set up centralized logging system
- [ ] Add structured log formatters
- [ ] Create custom error types
- [ ] Log function entry/exit for key methods
- [ ] Add crash handler
- [ ] Save crash context for debugging
- [ ] Log all configuration changes
- [ ] Add performance logging (timing)

---

## 🔧 Part 7: Monitoring Tools Organization

### Objective
Consolidate monitoring scripts into organized tools/ directory.

### Tools Directory Structure

```
tools/
├── __init__.py
├── monitor.py                 # Unified monitoring interface
├── health_check.py            # Repository health check
├── training_monitor.py        # Live training monitoring
├── checkpoint_manager.py      # Checkpoint management
├── submission_validator.py    # Validate submission package
└── performance_profiler.py    # Profile model performance
```

### Unified Monitor

```python
# tools/monitor.py
"""
Unified monitoring interface for EEG2025 project.

Usage:
    python tools/monitor.py training --log-file logs/training.log
    python tools/monitor.py health-check
    python tools/monitor.py validate-submission submission.zip
"""

import argparse
from pathlib import Path

def monitor_training(log_file: Path, refresh_sec: int = 5):
    """Monitor training progress from log file."""
    import time
    
    print(f"Monitoring training log: {log_file}")
    print("Press Ctrl+C to stop\n")
    
    with open(log_file, 'r') as f:
        # Go to end
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if line:
                # Parse and display relevant info
                if "Epoch" in line or "Loss" in line or "NRMSE" in line:
                    print(line.strip())
            else:
                time.sleep(refresh_sec)


def health_check():
    """Run repository health check."""
    from tools.health_check import RepositoryHealthCheck
    
    checker = RepositoryHealthCheck()
    results = checker.run_all_checks()
    
    print("\n" + "=" * 80)
    print("HEALTH CHECK RESULTS")
    print("=" * 80 + "\n")
    
    for check_name, result in results.items():
        status = "✅" if result['passed'] else "❌"
        print(f"{status} {check_name}: {result['message']}")


def validate_submission(submission_zip: Path):
    """Validate submission package."""
    from tools.submission_validator import SubmissionValidator
    
    validator = SubmissionValidator()
    results = validator.validate(submission_zip)
    
    if results['valid']:
        print("✅ Submission package is valid!")
    else:
        print("❌ Submission package has issues:")
        for issue in results['issues']:
            print(f"  - {issue}")


def main():
    parser = argparse.ArgumentParser(description="EEG2025 Monitoring Tools")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training monitor
    train_parser = subparsers.add_parser('training', help='Monitor training')
    train_parser.add_argument('--log-file', type=Path, required=True)
    train_parser.add_argument('--refresh', type=int, default=5)
    
    # Health check
    subparsers.add_parser('health-check', help='Run health check')
    
    # Submission validator
    sub_parser = subparsers.add_parser('validate-submission', help='Validate submission')
    sub_parser.add_argument('submission_zip', type=Path)
    
    args = parser.parse_args()
    
    if args.command == 'training':
        monitor_training(args.log_file, args.refresh)
    elif args.command == 'health-check':
        health_check()
    elif args.command == 'validate-submission':
        validate_submission(args.submission_zip)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
```

### Monitoring Checklist

- [ ] Create tools/ directory
- [ ] Move all *.sh monitoring scripts
- [ ] Consolidate duplicate monitors
- [ ] Create unified CLI interface
- [ ] Add health check dashboard
- [ ] Add submission validator
- [ ] Add performance profiler
- [ ] Document all tools in README

---

## 📈 Implementation Timeline

### Week 1: Foundation
- [x] Enhanced README with diagrams
- [ ] Update methods PDF
- [ ] Set up testing infrastructure
- [ ] Create utility modules (preprocessing, time units)
- [ ] Set up logging system

### Week 2: Core Improvements
- [ ] Refactor submission.py
- [ ] Write unit tests (data loading, preprocessing)
- [ ] Add input validation throughout
- [ ] Standardize checkpointing
- [ ] Implement error handling

### Week 3: Testing & Monitoring
- [ ] Write integration tests
- [ ] Achieve >80% coverage
- [ ] Create monitoring tools
- [ ] Add crash recovery
- [ ] Performance profiling

### Week 4: Polish & Documentation
- [ ] Complete all docstrings
- [ ] Update all documentation
- [ ] Code review and cleanup
- [ ] Final validation
- [ ] Prepare for competition submission

---

## 🎯 Success Metrics

- **Code Quality**:
  - ✅ Black formatting passes
  - ✅ Flake8 linting passes
  - ✅ mypy type checking passes
  - ✅ >80% test coverage

- **Robustness**:
  - ✅ All boundary conditions handled
  - ✅ No uncaught exceptions
  - ✅ Comprehensive logging
  - ✅ Crash recovery implemented

- **Maintainability**:
  - ✅ All functions <50 lines
  - ✅ Single responsibility principle
  - ✅ Clear documentation
  - ✅ Reusable utilities

---

**This plan provides a complete roadmap for transforming the codebase into a production-ready, robust, well-tested system. Each part can be implemented incrementally without breaking existing functionality.**
