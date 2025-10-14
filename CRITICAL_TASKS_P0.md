# 🔴 P0 CRITICAL TASKS - Execution Plan

**Created**: October 14, 2025  
**Deadline**: Complete before training starts  
**Status**: IN PROGRESS

---

## 📊 Overview

| Task | Time | Dependencies | Status |
|------|------|--------------|--------|
| 1. Acquire HBN Dataset | 1-2 days | None | ⭕ |
| 2. Core Test Suite | 2-3 days | Data available | ⭕ |
| 3. Validate Pipeline | 1 day | Tests written | ⭕ |
| 4. Inference Latency | 4 hours | Model ready | ⭕ |

**Total Time**: ~5 days  
**Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Task 1: Acquire HBN-EEG Dataset (1-2 days)

### Why Critical
- Cannot train or test without real data
- All other tasks depend on this
- Competition uses HBN dataset

### What To Do
```bash
# Phase 1: Quick Test (30 min)
pip install mne mne-bids boto3 requests tqdm
mkdir -p data/raw/hbn data/processed data/cache
python scripts/download_hbn_data.py --subjects 2 --verify

# Phase 2: Training Set (4-8 hours)
python scripts/download_hbn_data.py --subjects 50 --parallel 4

# Phase 3: Full Dataset (overnight)
python scripts/download_hbn_data.py --all --parallel 8
```

### Success Criteria
- [x] Can download at least 2 subjects
- [x] BIDS structure validated
- [x] Data loads without errors
- [x] Preprocessing works

### Deliverables
- `data/raw/hbn/` with subject data
- `data/logs/download_report.json`
- Verified BIDS structure

---

## Task 2: Core Test Suite (2-3 days)

### Why Critical
- Need confidence code works correctly
- Prevent bugs during competition
- Enable rapid iteration

### What To Do

#### Day 1: Data Tests
```bash
# Create test_data_loading.py
pytest tests/test_data_loading.py -v

# Tests to write:
- test_load_single_subject()
- test_bids_structure_valid()
- test_label_extraction()
- test_data_shape_correct()
- test_missing_data_handling()
```

#### Day 2: Model Tests
```bash
# Create test_model_forward.py
pytest tests/test_model_forward.py -v

# Tests to write:
- test_model_initialization()
- test_forward_pass_shape()
- test_gradient_flow()
- test_model_save_load()
- test_batch_processing()
```

#### Day 3: Integration Tests
```bash
# Create test_challenge_metrics.py
pytest tests/test_challenge_metrics.py -v

# Tests to write:
- test_challenge1_metrics()
- test_challenge2_metrics()
- test_submission_format()
- test_cross_validation()
```

### Success Criteria
- [x] 15+ tests written
- [x] All tests pass
- [x] Code coverage >50%
- [x] CI/CD passes

### Deliverables
- `tests/test_data_loading.py`
- `tests/test_model_forward.py`
- `tests/test_challenge_metrics.py`
- `tests/test_inference_speed.py`

---

## Task 3: Validate Data Pipeline (1 day)

### Why Critical
- Ensure data quality
- Catch preprocessing issues
- Verify challenge compliance

### What To Do

#### Morning: Structure Validation
```bash
python scripts/verify_data_structure.py \
    --data-dir data/raw/hbn \
    --check-bids \
    --check-labels \
    --verbose
```

#### Afternoon: Statistics Validation
```bash
python scripts/validate_data_statistics.py \
    --data-dir data/raw/hbn \
    --plot \
    --compare-baseline
```

#### Evening: End-to-End Test
```bash
python scripts/test_full_pipeline.py \
    --subject NDARAA536PTU \
    --challenge challenge1 \
    --visualize
```

### Success Criteria
- [x] All BIDS validation passes
- [x] Statistics match expected ranges
- [x] End-to-end pipeline works
- [x] No data loading errors

### Deliverables
- `data/validation/structure_report.json`
- `data/validation/statistics_report.json`
- `data/validation/plots/` directory
- Pipeline validation certificate

---

## Task 4: Measure Inference Latency (4 hours)

### Why Critical
- Competition requires <50ms inference
- Need to know if optimization needed
- May affect model architecture choices

### What To Do

#### Step 1: Create Benchmark Script (1 hour)
```python
# tests/test_inference_speed.py
import time
import torch
import numpy as np
from src.models import AdvancedFoundationModel

def test_inference_latency():
    """Verify <50ms inference requirement"""
    model = AdvancedFoundationModel()
    model.eval()
    
    # Test data: 2-second window
    eeg_data = torch.randn(1, 128, 1000)
    
    # Warmup
    for _ in range(10):
        _ = model(eeg_data)
    
    # Measure
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(eeg_data)
        times.append(time.perf_counter() - start)
    
    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    
    print(f"Average: {avg_ms:.2f}ms")
    print(f"P95: {p95_ms:.2f}ms")
    
    assert avg_ms < 50, f"Too slow: {avg_ms:.2f}ms"
```

#### Step 2: Run Benchmark (1 hour)
```bash
# CPU test
pytest tests/test_inference_speed.py -v -s

# GPU test
CUDA_VISIBLE_DEVICES=0 pytest tests/test_inference_speed.py -v -s
```

#### Step 3: Optimize if Needed (2 hours)
```python
# If >50ms, apply optimizations:
- Model quantization (INT8)
- Operator fusion
- Batch processing
- TensorRT compilation
```

### Success Criteria
- [x] Benchmark runs successfully
- [x] Average latency <50ms
- [x] P95 latency <75ms
- [x] Consistent across 100 runs

### Deliverables
- `tests/test_inference_speed.py`
- `benchmarks/inference_report.json`
- Optimization recommendations (if needed)

---

## 📅 Execution Timeline

### Day 1 (Today)
**Focus**: Get data flowing
- [ ] Morning: Install dependencies, setup directories
- [ ] Afternoon: Download 2 sample subjects
- [ ] Evening: Verify data structure, test loading

**Deliverable**: 2 subjects loaded successfully

---

### Day 2
**Focus**: Expand data + start tests
- [ ] Morning: Download 50 subjects (parallel)
- [ ] Afternoon: Write test_data_loading.py (5 tests)
- [ ] Evening: Write test_model_forward.py (5 tests)

**Deliverable**: 10 tests passing + 50 subjects

---

### Day 3
**Focus**: Complete test suite
- [ ] Morning: Write test_challenge_metrics.py
- [ ] Afternoon: Write test_inference_speed.py
- [ ] Evening: Run all tests, fix failures

**Deliverable**: 15+ tests passing, CI/CD green

---

### Day 4
**Focus**: Validation
- [ ] Morning: Validate data structure + statistics
- [ ] Afternoon: End-to-end pipeline test
- [ ] Evening: Measure inference latency

**Deliverable**: All validation passing, latency measured

---

### Day 5 (Buffer)
**Focus**: Fix issues, document results
- [ ] Morning: Address any failures from Day 4
- [ ] Afternoon: Optimize if latency >50ms
- [ ] Evening: Document everything, prepare for training

**Deliverable**: Ready to train

---

## 🎯 Quick Commands Reference

### Data Acquisition
```bash
# Quick test (2 subjects)
python scripts/download_hbn_data.py --subjects 2 --verify

# Training set (50 subjects)
python scripts/download_hbn_data.py --subjects 50 --parallel 4

# Full dataset
python scripts/download_hbn_data.py --all --parallel 8
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_data_loading.py::test_load_single_subject -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Validation
```bash
# Verify structure
python scripts/verify_data_structure.py --data-dir data/raw/hbn

# Check statistics
python scripts/validate_data_statistics.py --data-dir data/raw/hbn

# Full pipeline
python scripts/test_full_pipeline.py --challenge challenge1
```

### Benchmarking
```bash
# Inference speed
pytest tests/test_inference_speed.py -v -s

# Full benchmark suite
python scripts/run_benchmarks.py --output benchmarks/
```

---

## ⚠️ Risk Mitigation

### Risk 1: Data download fails
**Mitigation**: 
- Use retry logic (already in script)
- Download in batches
- Keep verified subjects even if some fail

### Risk 2: Tests reveal critical bugs
**Mitigation**:
- Fix high-priority bugs first
- Document known issues
- Create workarounds if needed

### Risk 3: Inference >50ms
**Mitigation**:
- Quantize model (INT8)
- Use smaller architecture
- Optimize preprocessing

### Risk 4: Running out of time
**Mitigation**:
- Focus on P0 tasks only
- Use minimum viable completions
- Parallelize where possible

---

## 🏁 Definition of Done

All tasks complete when:

✅ **Data**: 
- At least 50 subjects downloaded
- BIDS structure validated
- Can load and preprocess

✅ **Tests**:
- 15+ tests written and passing
- Coverage >50%
- CI/CD green

✅ **Validation**:
- Structure checks pass
- Statistics in expected range
- End-to-end pipeline works

✅ **Performance**:
- Inference <50ms average
- Benchmark report generated
- Optimizations applied if needed

**When all ✅ → Ready to train! 🚀**

---

## 📊 Progress Tracking

Update daily:

```bash
# Add to daily log
echo "$(date): Task X completed, Y remaining" >> PROGRESS.log

# Check status
python << 'EOF'
import os
tasks = {
    "Data (2+ subjects)": os.path.exists("data/raw/hbn/sub-NDARAA536PTU"),
    "Tests (15+)": len([f for f in os.listdir("tests") if f.startswith("test_")]) >= 15 if os.path.exists("tests") else False,
    "Validation scripts": os.path.exists("scripts/verify_data_structure.py"),
    "Benchmark script": os.path.exists("tests/test_inference_speed.py"),
}

print("\n=== P0 Tasks Status ===")
for task, done in tasks.items():
    icon = "✅" if done else "⭕"
    print(f"{icon} {task}")
print()
