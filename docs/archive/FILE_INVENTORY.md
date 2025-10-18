# 📂 EEG2025 Project - Complete File Inventory
**Generated:** October 17, 2025  
**Purpose:** Track all files, their locations, and purposes

---

## 🎯 CORE SUBMISSION FILES

### Root Level - Keep These
```
submission.py                           # Official competition submission script (12 KB)
                                       # Contains: SparseAttentionResponseTimeCNN, ExternalizingCNN
                                       # Status: Ready for Codabench submission

README.md                              # Main project documentation
requirements.txt                       # Python dependencies
requirements-dev.txt                   # Development dependencies
setup.py                               # Package installation script
pyproject.toml                         # Modern Python project config
LICENSE                                # MIT License
Makefile                               # Build automation commands
.gitignore                             # Git ignore rules
```

---

## 🧠 MODEL WEIGHTS & CHECKPOINTS

### Current Best Models (checkpoints/)
```
checkpoints/response_time_attention.pth    # Challenge 1 BEST (9.8 MB)
                                           # Model: SparseAttentionResponseTimeCNN
                                           # NRMSE: 0.2632 ± 0.0368 (5-fold CV)
                                           # Date: Oct 17, 2025
                                           # Status: ⭐ READY FOR SUBMISSION

checkpoints/response_time_improved.pth     # Challenge 1 older (3.1 MB)
                                           # NRMSE: ~0.45
                                           # Status: Superseded by attention model

checkpoints/externalizing_model.pth        # Challenge 2 baseline (949 KB)
                                           # NRMSE: 0.0808 (likely overfit)
                                           # Status: Being replaced by multi-release model

checkpoints/challenge2_clinical.pth        # Challenge 2 older (951 KB)
                                           # Status: Historical
```

### Ensemble Models (checkpoints/ensemble/)
```
response_time_model_seed42.pth         # Ensemble member 1
response_time_model_seed123.pth        # Ensemble member 2  
response_time_model_seed456.pth        # Ensemble member 3
                                       # Status: Experimental, not used in final submission
```

### Historical Weights (checkpoints/)
```
baseline_cnn.pth                       # Original baseline (181 KB)
improved_model.pth                     # Early improvement (487 KB)
response_time_model.pth                # Mid-development (949 KB)
simple_cnn_age.pth                     # Age prediction experiment (423 KB)
cpu_timeout_model.pth                  # CPU-only training (27 KB)
```

### Submission Packages
```
prediction_result/weights_challenge_1.pt   # Oct 16 submission (3.1 MB)
prediction_result/weights_challenge_2.pt   # Oct 16 submission (949 KB)
                                           # Status: Previous submission, kept for reference

response_time_attention.pth                # Also in root (9.8 MB) - duplicate
weights_challenge_2_multi_release.pt       # In root (size TBD) - from training
```

### Archives (archive/weights_*)
```
archive/weights_20251016_154323/weights_challenge_1_multi_release.pt
archive/weights_20251016_154323/weights_challenge_2_multi_release.pt
archive/weights_challenge_1_improved.pt
```

---

## 🔬 TRAINING & EXPERIMENT SCRIPTS

### Challenge 1 Training (scripts/)
```
train_challenge1_attention.py          # ⭐ CURRENT BEST - Sparse attention model
                                       # 5-fold CV, NRMSE: 0.2632
                                       # Time: ~2 minutes per fold

train_challenge1_multi_release.py      # Multi-release baseline training
                                       # Uses: R1+R2+R3 combined
                                       # NRMSE: ~0.45

train_challenge1_improved.py           # Older improved version
                                       # With augmentation

cross_validate_challenge1.py           # 5-fold cross-validation runner
                                       # Outputs: Fold NRMSEs, mean ± std

train_ensemble_challenge1.py           # Ensemble training (3 seeds)
                                       # Status: Experimental
```

### Challenge 2 Training (scripts/)
```
train_challenge2_multi_release.py      # 🔄 CURRENTLY RUNNING
                                       # Strategy: R2+R3+R4 combined
                                       # Expected: NRMSE < 0.35
                                       # Time: ~2-3 hours

train_challenge2_externalizing.py      # Older single-release training
                                       # Status: Superseded

train_challenge2_clinical.py           # Clinical prediction variant
```

### Validation & Testing (scripts/)
```
validate_models.py                     # Model weight validation
                                       # Checks: File exists, loadable, correct architecture

final_pre_submission_check.py          # 25-point submission verification
                                       # Validates: Code, weights, format

validate_setup.py                      # Environment setup validation
validate_repository.py                 # Repository structure check
validate_data_statistics.py            # Data distribution analysis
verify_data_integrity.py               # Corruption detection
verify_data_structure.py               # File structure verification
```

### Analysis & Visualization (scripts/)
```
visualize_features.py                  # Feature importance plots
                                       # Outputs: PNG files to results/visualizations/

analyze_predictions.py                 # Prediction analysis
compare_models.py                      # Model comparison utilities
```

### GPU & Infrastructure (scripts/)
```
train_gpu_safeguarded.py               # GPU training with safety checks
train_gpu_safe.py                      # Safe GPU training
train_gpu_timeout.py                   # Timeout-protected GPU training
train_unified_gpu.py                   # Unified GPU/CPU training
train_with_monitoring.py               # Training with monitoring
train_multiprocess_cpu.py              # Multi-process CPU training
train_cpu_safe.py                      # CPU-only safe training
```

### Utilities (scripts/)
```
train_grouped_cv_template.py           # Template for grouped CV
train_improved_cpu.py                  # CPU-optimized training
train_improved_quick.py                # Quick training for testing
train_minimal.py                       # Minimal working example
train_simple.py                        # Simple baseline
train_transformer.py                   # Transformer experiments
train.py                               # Original training script
```

---

## 📊 RESULTS & LOGS

### Training Logs (logs/)
```
challenge1_attention_20251017_140303.log   # ⭐ Latest C1 attention training
                                           # Contains: 5-fold CV results, 0.2632 NRMSE

challenge2_r234_final.log                  # 🔄 Current C2 training log
                                           # Status: In progress

challenge2_fresh_start.log                 # Previous C2 training
challenge1_multi_release.log               # C1 multi-release training
train_c2_multi.log                         # C2 multi-release training

challenge2_crash_*.log                     # Crash reports (multiple)
challenge2_training_*.log                  # Various C2 training attempts
challenge2_expanded_*.log                  # Expanded dataset experiments
```

### Results Files (results/)
```
challenge1_response_time.txt           # C1 baseline results
challenge2_externalizing.txt           # C2 baseline results
challenge1_crossval.txt                # Cross-validation results
challenge1_ensemble.txt                # Ensemble results
```

### Visualizations (results/visualizations/)
```
feature_importance_*.png               # Feature importance plots
attention_weights_*.png                # Attention visualization
channel_importance_*.png               # Channel importance maps
```

---

## 📚 DOCUMENTATION (51 files in root + docs/)

### Root Level - Status Documents (TO BE MOVED)
```
CURRENT_STATUS.md                      # Latest training status (Oct 17)
FINAL_STATUS_REPORT.md                 # Comprehensive report (Oct 16)
EXECUTIVE_SUMMARY.md                   # Position #47 analysis
TODO.md                                # Action items and todo list
ROADMAP_TO_RANK1.md                    # Strategy to reach #1 (656 lines!)

PHASE1_COMPLETE.md                     # Phase 1 completion report
PHASE1_RESULTS.md                      # Phase 1 results
PHASE1_STATUS.md                       # Phase 1 status
PHASE1_TRAINING_STATUS.md              # Phase 1 training details
PHASE2_PROGRESS.md                     # Phase 2 progress

TRAINING_STATUS.md                     # General training status
TRAINING_STATUS_CURRENT.md             # Current training status
TRAINING_STATUS_20251016_1431.md       # Timestamped status
TRAINING_COMPLETE_SUMMARY.md           # Training completion
TRAINING_FIX_IMPLEMENTATION.md         # Training fixes

CHALLENGE2_ANALYSIS.md                 # Challenge 2 analysis
CHALLENGE2_IMPROVEMENT_TODO.md         # Challenge 2 improvements
SUBMISSION_CHECKLIST.md                # Submission checklist
SUBMISSION_READY.md                    # Submission ready status
SUBMISSION_SUMMARY.md                  # Submission summary
FINAL_SUBMISSION_CHECKLIST.md          # Final submission checklist
FINAL_SUBMISSION_REPORT.md             # Final submission report

GPU_TRAINING_STATUS.md                 # GPU training status
GPU_USAGE_GUIDE.md                     # GPU usage guide
ACTIVE_TRAINING_STATUS.md              # Active training status
COMPETITION_ANALYSIS.md                # Competition analysis
METHODS_COMPARISON.md                  # Methods comparison
SCORE_COMPARISON.md                    # Score comparison
ANSWERS_TO_YOUR_QUESTIONS.md           # Q&A document

IMPROVEMENT_ROADMAP.md                 # Improvement roadmap
IMPROVEMENT_STRATEGY.md                # Improvement strategy
INTEGRATED_IMPROVEMENT_PLAN.md         # Integrated plan
ADVANCED_METHODS_PLAN.md               # Advanced methods
NEXT_STEPS.md                          # Next steps

EXTRACTION_WORKING.md                  # Metadata extraction
METADATA_EXTRACTION_SOLUTION.md        # Metadata solution
P300_EXTRACTION_STATUS.md              # P300 extraction
VECTORIZED_EXTRACTION_STATUS.md        # Vectorized extraction
FINAL_FIX_GETITEM.md                   # Dataset fix
FRESH_START_TRAINING.md                # Fresh training start

ORGANIZATION_COMPLETE.md               # Organization status
IMPLEMENTATION_COMPLETE.md             # Implementation status
ROCM_FIX_DOCUMENTATION.md              # ROCm fix
GITIGNORE_UPDATE.md                    # Gitignore update
SPARSE_ATTENTION_IMPLEMENTATION.md     # Sparse attention details
PROGRESS_SUMMARY.md                    # Progress summary

OVERNIGHT_README.md                    # Overnight training guide
QUICK_UPLOAD_GUIDE.txt                 # Quick upload guide
READY_TO_SUBMIT.txt                    # Ready to submit marker
RESULTS_SUMMARY.txt                    # Results summary
```

### Methods Documentation (Root + docs/methods/)
```
METHODS_DOCUMENT.md                    # Official methods (Markdown)
METHOD_DESCRIPTION.md                  # Method description
METHODS_DOCUMENT.pdf                   # Competition submission PDF
METHOD_DESCRIPTION.pdf                 # Method description PDF
METHOD_DESCRIPTION.html                # HTML version
```

### docs/ Subdirectories
```
docs/status/                           # Status reports (71 files!)
docs/methods/                          # Methods documentation
docs/guides/                           # How-to guides
docs/planning/                         # Project planning
docs/plans/                            # Detailed plans
docs/summaries/                        # Summaries
docs/archive/                          # Historical docs
```

---

## 🗂️ SOURCE CODE

### Core Models (src/models/)
```
src/models/backbone/eeg_transformer.py             # Transformer backbone
src/models/adapters/task_aware.py                  # Task-aware adapter
src/models/heads/temporal_regression.py            # Regression head
src/models/compression_ssl/augmentation.py         # Augmentation module
```

### Training Infrastructure (src/training/)
```
src/training/trainers/ssl_trainer.py               # Self-supervised trainer
```

### Data Processing (src/dataio/)
```
src/dataio/hbn_dataset.py                          # HBN dataset loader
```

### GPU Optimization (src/gpu/)
```
src/gpu/triton/fused_ops.py                        # Triton kernels
```

---

## 🔧 UTILITY SCRIPTS (Root Level)

### Monitoring Scripts
```
monitor_training.sh                    # Basic training monitor
monitor_training_enhanced.sh           # Enhanced monitor with metrics
monitor_overnight.sh                   # Overnight monitoring
monitor_simple.sh                      # Simple monitor
monitor_debug.sh                       # Debug monitoring
monitor_p300_extraction.sh             # P300 extraction monitor
watch_p300.sh                          # P300 watcher

check_training_status.sh               # Training status checker
status.sh                              # General status
quick_status.sh                        # Quick status check
```

### Training Launchers
```
launch_training.sh                     # Launch training
start_independent_training.sh          # Independent training
restart_training.sh                    # Restart training
restart_training_cpu.sh                # Restart CPU training
restart_training_hybrid.sh             # Restart hybrid training
run_overnight_training.sh              # Overnight training
```

### Utilities
```
organize_files.sh                      # File organization
test_gpu.sh                            # GPU testing
test_metadata.py                       # Metadata testing
test_submission.py                     # Submission testing
submission_backup.py                   # Submission backup
```

---

## 📦 SUBMISSION PACKAGES & ARCHIVES

### Submission Zips
```
eeg2025_submission.zip                 # Main submission (9.3 MB)
submission_complete.zip                # Complete submission (3.8 MB)
submission_final_20251017_1314.zip     # Final submission (3.1 MB)
submission.zip                         # Basic submission (588 KB)
prediction_result.zip                  # Prediction results (588 KB)
scoring_result.zip                     # Scoring results (357 bytes)
```

### Submission Directories
```
submission_package/                    # Submission package dir
submission_v2/                         # Version 2
submission_final/                      # Final version
submissions/                           # All submissions
prediction_result/                     # Prediction outputs
scoring_result/                        # Scoring outputs
```

### Archives
```
archive/                               # Historical files
├─ weights_20251016_154323/            # Oct 16 weights backup
├─ weights_challenge_1_improved.pt     # Improved weights
└─ [various old scripts and docs]
```

---

## 🎨 FRONTEND & BACKEND

### Web Interface (web/)
```
web/                                   # Web dashboard (if exists)
```

### Backend Services (backend/)
```
backend/                               # Backend services (if exists)
```

---

## 📓 NOTEBOOKS & CONFIGS

### Jupyter Notebooks (notebooks/)
```
notebooks/                             # Analysis notebooks
├─ data_exploration.ipynb              # Data exploration
├─ model_analysis.ipynb                # Model analysis
└─ visualization.ipynb                 # Visualizations
```

### Configuration Files
```
config/                                # Configuration directory
configs/                               # Alternative configs
.vscode/                               # VS Code settings
.github/                               # GitHub Actions
docker/                                # Docker configs
```

---

## 📦 DATA & OUTPUTS

### Data Directories
```
data/                                  # Raw data directory
├─ raw/hbn_ccd_mini/                   # Mini dataset
└─ processed/                          # Processed data
```

### Output Directories
```
outputs/                               # Model outputs
models/                                # Saved models
assets/                                # Static assets
```

---

## 🧪 TESTS

### Test Files (tests/)
```
tests/simple_validation.py             # Simple validation tests
tests/test_demo_integration.py         # Demo integration tests
tests/test_demo_integration_improved.py # Improved demo tests
tests/test_cross_metrics.py            # Cross-task metrics tests
test_cross_task_simple.py              # Simple cross-task tests
```

---

## 🔍 HIDDEN & SYSTEM FILES

### Git & Version Control
```
.git/                                  # Git repository
.gitignore                             # Git ignore rules
```

### Python Cache
```
__pycache__/                           # Python bytecode cache
.pytest_cache/                         # Pytest cache
.mypy_cache/                           # MyPy cache
```

### Environment
```
venv/                                  # Virtual environment
.editorconfig                          # Editor configuration
.copilot/                              # GitHub Copilot settings
```

### Home Directory Link
```
~/                                     # Home directory symlink (?)
```

---

## 📋 FILE ORGANIZATION PLAN

### Step 1: Create New Structure
```bash
mkdir -p docs/{status,planning,analysis,guides,historical,methods}
mkdir -p archive/{old_scripts,old_weights,old_docs}
mkdir -p submission_history
```

### Step 2: Move Status Documents
```bash
# Move to docs/status/
mv PHASE1_*.md docs/status/
mv TRAINING_STATUS*.md docs/status/
mv CHALLENGE2_*.md docs/status/
mv FINAL_STATUS*.md docs/status/
mv ACTIVE_TRAINING_STATUS.md docs/status/
mv CURRENT_STATUS.md docs/status/
mv GPU_TRAINING_STATUS.md docs/status/
mv SUBMISSION_*.md docs/status/
```

### Step 3: Move Planning Documents
```bash
# Move to docs/planning/
mv TODO.md docs/planning/
mv ROADMAP_TO_RANK1.md docs/planning/
mv NEXT_STEPS.md docs/planning/
mv IMPROVEMENT_*.md docs/planning/
mv ADVANCED_METHODS_PLAN.md docs/planning/
mv INTEGRATED_IMPROVEMENT_PLAN.md docs/planning/
```

### Step 4: Move Analysis Documents
```bash
# Move to docs/analysis/
mv EXECUTIVE_SUMMARY.md docs/analysis/
mv COMPETITION_ANALYSIS.md docs/analysis/
mv METHODS_COMPARISON.md docs/analysis/
mv SCORE_COMPARISON.md docs/analysis/
mv ANSWERS_TO_YOUR_QUESTIONS.md docs/analysis/
mv PROGRESS_SUMMARY.md docs/analysis/
```

### Step 5: Move Guides
```bash
# Move to docs/guides/
mv GPU_USAGE_GUIDE.md docs/guides/
mv OVERNIGHT_README.md docs/guides/
mv QUICK_UPLOAD_GUIDE.txt docs/guides/
mv METADATA_EXTRACTION_SOLUTION.md docs/guides/
```

### Step 6: Move Historical/Completed
```bash
# Move to docs/historical/
mv EXTRACTION_WORKING.md docs/historical/
mv P300_EXTRACTION_STATUS.md docs/historical/
mv VECTORIZED_EXTRACTION_STATUS.md docs/historical/
mv ORGANIZATION_COMPLETE.md docs/historical/
mv IMPLEMENTATION_COMPLETE.md docs/historical/
mv ROCM_FIX_DOCUMENTATION.md docs/historical/
mv GITIGNORE_UPDATE.md docs/historical/
mv SPARSE_ATTENTION_IMPLEMENTATION.md docs/historical/
mv FINAL_FIX_GETITEM.md docs/historical/
mv FRESH_START_TRAINING.md docs/historical/
mv PHASE2_PROGRESS.md docs/historical/
```

### Step 7: Archive Old Submissions
```bash
# Move to submission_history/
mv submission_complete.zip submission_history/
mv submission.zip submission_history/
mv prediction_result.zip submission_history/
mv scoring_result.zip submission_history/
mv submission_v2/ submission_history/
mv submission_package/ submission_history/
mv prediction_result/ submission_history/
mv scoring_result/ submission_history/
```

### Step 8: Keep in Root (Clean!)
```
Root Directory (Final):
├─ README.md
├─ PROJECT_ANALYSIS_OCT17.md          # This analysis! ⭐
├─ FILE_INVENTORY.md                   # This inventory! ⭐
├─ submission.py
├─ requirements.txt
├─ requirements-dev.txt
├─ setup.py
├─ pyproject.toml
├─ LICENSE
├─ Makefile
├─ .gitignore
├─ METHODS_DOCUMENT.pdf                # Competition methods
├─ checkpoints/                        # Active model weights
├─ scripts/                            # Training scripts
├─ src/                                # Source code
├─ tests/                              # Test files
├─ logs/                               # Training logs
├─ results/                            # Results
├─ docs/                               # Organized documentation
├─ archive/                            # Historical files
└─ submission_history/                 # Old submissions

Total: 16 items in root (down from 100+!)
```

---

## 🎯 PRIORITY FILES FOR SUBMISSION

### Must Have (Next Submission)
```
1. submission.py                       # Latest version with sparse attention
2. checkpoints/response_time_attention.pth  # Challenge 1 (9.8 MB)
3. [C2 weights from current training]  # Challenge 2 (TBD)
4. METHODS_DOCUMENT.pdf                # Competition requirement
```

### Nice to Have
```
5. README.md                           # Documentation
6. requirements.txt                    # Dependencies
```

---

**Created:** October 17, 2025  
**Purpose:** Complete file tracking before cleanup  
**Next Action:** Execute cleanup plan to organize root directory  
**Status:** 📂 READY FOR ORGANIZATION
