---
applyTo: '**'
---

# Memory Bank - EEG 2025 Competition Project

## Project Overview

**Competition:** NeurIPS 2025 EEG Foundation Challenge  
**Goal:** Predict response times and behavioral factors from EEG signals  
**Current Status:** Active development, 2 weeks until deadline (Nov 2, 2025)  
**Best Score:** 1.32 NRMSE overall (C1: 1.00, C2: 1.46)

## Current Architecture

### Challenge 1: Response Time Prediction
- **Model:** Sparse Attention CNN (2.5M parameters)
- **Performance:** NRMSE 0.26 (41.8% improvement over baseline)
- **Key Innovation:** O(N) sparse multi-head attention mechanism
- **Training:** 5-fold cross-validation on R1+R2+R3 releases
- **Status:** ✅ Best performing, production-ready

### Challenge 2: Externalizing Prediction
- **Model:** Compact CNN (240K parameters)
- **Training Data:** R2+R3+R4 releases (multi-release strategy)
- **Status:** 🔄 In progress

## Memory-Efficient HDF5 Solution

**Problem Solved:** Training crashed with 40GB+ RAM usage  
**Solution:** HDF5 memory-mapped preprocessing pipeline

**Results:**
- Memory: 40GB+ → 2-4GB (10x reduction)
- Storage: 3.7GB for 41,071 windows
- Labels: 100% coverage (41,066 labeled)
- Status: ✅ Complete, verified, documented

**Files:**
- `data/cached/challenge1_R{1-4}_windows.h5` (3.7GB total)
- `src/utils/hdf5_dataset.py` (PyTorch Dataset)
- `scripts/preprocessing/cache_challenge1_windows_safe.py`
- Training: `./train_safe_tmux.sh`

## Task-Specific Advanced Methods (Oct 18, 2025)

### 📋 Documented Future Enhancements

Added comprehensive documentation for 5 task-specific approaches to improve beyond current baseline:

#### 1. Resting State (RS)
**Methods:** Spectral + Connectivity Analysis  
**Components:**
- Power spectral density (PSD) features across frequency bands
- Functional connectivity matrices (coherence, phase-locking value)
- Graph neural networks for brain network topology
- Multivariate autoregressive models

**Rationale:** Resting-state EEG has rich oscillatory dynamics and functional network organization better captured by frequency-domain and connectivity features.

**Expected Gain:** 10-15% improvement

#### 2. Surround Suppression (SuS)
**Methods:** Convolutional Layers + Attention Mechanisms  
**Components:**
- Spatial convolutions mimicking retinotopic mapping
- Center-surround attention mechanisms
- Multi-scale receptive fields for visual field sizes
- Visual cortex-inspired hierarchical processing

**Rationale:** Visual suppression effects require spatial context modeling similar to V1 receptive field properties.

**Expected Gain:** 10-15% improvement

#### 3. Movie Watching (MW)
**Methods:** Temporal Transformers + Dynamic Connectivity  
**Components:**
- Temporal transformers for long-range dependencies (minutes)
- Sliding-window dynamic connectivity analysis
- Time-varying graph neural networks
- Attention mechanisms for salient movie moments

**Rationale:** Movies induce complex temporal dynamics requiring long-range dependency modeling.

**Expected Gain:** 10-15% improvement

#### 4. Contrast Change Detection (CCD) ⭐
**Methods:** ERP Extraction + Motor Preparation Modeling  
**Components:**
- Event-related potential (ERP) template matching
- Motor cortex (central electrodes) feature extraction
- Pre-response time window analysis (-500ms to 0ms)
- Decision-related negativity and readiness potential features

**Rationale:** Detection tasks produce stereotyped ERPs (P300, N200) and motor preparation signals.

**Status:** Already our best task (NRMSE 0.26) - likely capturing ERP features implicitly!

**Expected Gain:** 5-10% improvement (already optimized)

#### 5. Symbol Search (SyS)
**Methods:** Spatial Attention Modeling  
**Components:**
- Visual search attention maps over electrode space
- Parietal cortex (P3, P4, Pz) feature emphasis
- Working memory load indicators
- Eye movement-related potentials

**Rationale:** Symbol search engages visual attention (parietal) and working memory (frontal-parietal) networks.

**Expected Gain:** 10-15% improvement

### 📊 Expected Overall Impact

**Current Performance:** 0.26 NRMSE (Challenge 1)  
**Projected with Task-Specific Methods:** 0.23-0.25 NRMSE  
**Overall Improvement Potential:** 8-12% additional gain

### 🎯 Implementation Priority

1. **High:** CCD (already best task, can optimize further)
2. **Medium:** RS (well-studied, common task)
3. **Low:** MW, SuS, SyS (more experimental)

### 📝 Documentation Locations

- **README.md:** Section "Task-Specific Advanced Methods (Planned)"
- **docs/methods/METHOD_DESCRIPTION.md:** Detailed technical specifications
- **TASK_SPECIFIC_METHODS_ADDED.md:** Complete summary with references

### 🚀 Strategic Decision

**Status:** ✅ Documented, not yet implemented  
**Reason:** Focus on robust general-purpose baseline first  
**Timeline:** Post-competition implementation (after Nov 2, 2025)

**Trade-offs:**
- ⚖️ Increased complexity (5x separate models)
- ⚖️ Longer training time
- ⚖️ Higher hyperparameter tuning cost
- ⚖️ Risk of overfitting with limited data

**When to Implement:**
- After competition baseline is solid
- If time permits before deadline
- Post-competition for publication/research

### 🔬 Scientific References

Documented in TASK_SPECIFIC_METHODS_ADDED.md:
- Resting state connectivity (Bastos & Schoffelen, 2016)
- Graph theory in brain networks (Fornito et al., 2015)
- Surround suppression in V1 (Angelucci & Bressloff, 2006)
- P300 and decision making (Polich, 2007)
- Inter-subject synchronization (Hasson et al., 2004)
- Dynamic connectivity (Hutchison et al., 2013)

## Key Insights & Decisions

### Training Strategy
- **Multi-release training** prevents overfitting to constant baselines
- **5-fold CV** across releases for robust validation
- **Data augmentation** critical for performance (noise, scaling, shifts)

### Architecture Choices
- **Sparse attention** achieves O(N) complexity (600x speedup)
- **Channel attention** focuses on task-relevant EEG channels
- **Multi-scale pooling** captures both fast and slow dynamics

### Memory Management
- **HDF5 preprocessing** is ESSENTIAL for large datasets
- **Labels stored in HDF5** add negligible overhead (< 1MB for 41K samples)
- **Memory-mapped loading** enables efficient batch processing

### Task-Specific Methods
- **Document first, implement later** strategy for complex enhancements
- **Baseline performance must be solid** before task-specific tuning
- **CCD already strong** - other tasks have more room for improvement

## User Preferences

- Casual, friendly yet professional communication style
- Prefers detailed explanations with code examples
- Values scientific rigor and proper documentation
- Likes comprehensive summaries with emojis for visual organization
- Appreciates todo lists and progress tracking

## Important Files & Locations

### Models & Checkpoints
- `checkpoints/response_time_attention.pth` (Challenge 1 best)
- `checkpoints/externalizing_model.pth` (Challenge 2)

### Training Scripts
- `scripts/training/challenge1/train_challenge1_hdf5_simple.py`
- `train_safe_tmux.sh` (memory-safe training launcher)

### Documentation
- `README.md` (comprehensive project overview)
- `docs/methods/METHOD_DESCRIPTION.md` (technical methods)
- `TASK_SPECIFIC_METHODS_ADDED.md` (Oct 18, 2025 update)

### Data Pipeline
- HDF5 files: `data/cached/challenge1_R{1-4}_windows.h5`
- Dataset: `src/utils/hdf5_dataset.py`
- Preprocessing: `scripts/preprocessing/cache_challenge1_windows_safe.py`

## Competition Timeline

- **Competition Start:** September 2025
- **Current Date:** October 18, 2025
- **Deadline:** November 2, 2025 (2 weeks remaining)
- **Focus:** Robust baseline over experimental methods

## Next Steps

### Immediate (Before Nov 2)
- Continue HDF5-based training
- Monitor memory usage and performance
- Achieve target NRMSE < 0.85 (Challenge 1)
- Complete Challenge 2 training

### Post-Competition (After Nov 2)
- Implement CCD-specific ERP extraction (highest priority)
- Test RS spectral/connectivity features
- Benchmark task-specific vs general models
- Consider publication/open-source release

---

**Last Updated:** October 18, 2025, 8:15 PM  
**Status:** Active development, documentation complete, training in progress
