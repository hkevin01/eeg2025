---
applyTo: '**'
---

# EEG Foundation Challenge 2025 - Competition Memory

## 🧠 CRITICAL: Summarization Instructions (Added Oct 19, 2025)

**ALWAYS follow these rules when summarizing conversation history:**

1. **BREAK INTO SMALL PARTS** - Never create one large summary/analysis/file
   - Maximum 100-150 lines per file
   - Create multiple numbered parts (PART1, PART2, PART3, etc.)
   - Each part should focus on ONE topic

2. **BIT BY BIT APPROACH** - Process incrementally
   - Document one section at a time
   - Complete each part before moving to next
   - Show progress after each part created

3. **PREVENT VS CODE CRASHES** - Small files reduce crash risk
   - Large files (>200 lines) can trigger RegExp issues
   - Multiple small files are safer than one big file
   - Use simple markdown formatting only

4. **FILE NAMING CONVENTION:**
   - Master index: `[NAME]_MASTER_INDEX.md`
   - Parts: `[NAME]_PART1_[TOPIC].md`, `[NAME]_PART2_[TOPIC].md`, etc.
   - Summary: `[NAME]_SUMMARY.txt` (plain text backup)

5. **ALWAYS UPDATE THIS MEMORY FILE** when completing major work
   - Add to "Current Model Status" section
   - Update "Recent Work Sessions" section
   - Keep entries concise (2-3 lines max per item)

**Example Structure:**
```
TODO_MASTER_INDEX.md (100 lines - overview + links)
TODO_PART1_INFRASTRUCTURE.md (100 lines)
TODO_PART2_TRAINING.md (100 lines)
TODO_PART3_SUBMISSION.md (100 lines)
TODO_SUMMARY.txt (plain text backup)
```

---

**Competition:** NeurIPS 2025 EEG Foundation Challenge  
**Deadline:** November 2, 2025  
**Dataset:** HBN-EEG (3000+ participants, 6 tasks, 129 channels)

---

## 🎯 CHALLENGE 1: Cross-Task Transfer Learning

### Goal
Predict **response time (RT)** from EEG during active **Contrast Change Detection (CCD)** task using transfer learning.

### Key Requirements
- **Task:** `contrastChangeDetection` (active task with stimulus and response)
- **Target Variable:** `rt_from_stimulus` (response time in seconds, regression)
- **Input:** EEG windows aligned to stimulus onset
- **Window Configuration:**
  - Stimulus-locked windows (anchor: "stimulus_anchor")
  - Start: +0.5s after stimulus
  - Duration: 2.0 seconds (200 samples at 100 Hz)
  - Epoch length: 2.0s, window stride: 1.0s
- **Data Format:** (batch, 129 channels, 200 timepoints)
- **Evaluation Metric:** NRMSE (Normalized Root Mean Squared Error)
- **Target:** NRMSE < 0.30 (lower is better)

### Data Strategy
- **Training Releases:** R1, R2, R3, R4 (maximize data diversity)
- **Validation:** R5 (cross-release validation)
- **Test:** Secret dataset (held out by competition)
- **Subject Filtering:** Remove problematic subjects from starter kit list
- **Transfer Learning:** Optionally pretrain on passive tasks (RS, SuS, MW), fine-tune on CCD

### Model Considerations
- **Architecture:** TCN (Temporal Convolutional Network) proven effective
  - 129 input channels → 48 filters → 5 levels
  - Dilation: 1, 2, 4, 8, 16 (captures multi-scale temporal patterns)
  - BatchNorm + Dropout (0.3) for regularization
  - 196K parameters
- **Emphasis:** Temporal dynamics, event-related potentials (ERP), SSVEP components
- **Avoid:** Overfitting to training subjects - prioritize generalization

### Current Best Model
- **File:** `checkpoints/challenge1_tcn_competition_best.pth`
- **Architecture:** TCN_EEG (from `improvements/all_improvements.py`)
- **Performance:** Val loss 0.010170 (epoch 2) ⭐
- **Estimated NRMSE:** 0.10-0.15 (excellent, well below 0.30 target)
- **Status:** ✅ READY - Weights load correctly in submission.py

### Critical Notes
- **CCD Task Details:**
  - Two flickering striped discs (SSVEP-like continuous signals)
  - Contrast ramp → subject responds left/right → feedback
  - Time-locked events: ramp onset, button press, feedback (ERP components)
  - Predict how fast they respond (RT regression)
- **Preprocessing:** Already applied by competition (100 Hz, 0.5-50 Hz bandpass)
- **Meta Fields:** subject, session, run, age, sex needed for splitting

---

## 🎯 CHALLENGE 2: Externalizing Factor Prediction

### Goal
Predict **externalizing factor (p_factor)** from EEG to enable objective mental health assessment.

### Key Requirements
- **Task:** `contrastChangeDetection` (NOT resting! Uses same active task as Challenge 1)
- **Target Variable:** `p_factor` (from Child Behavior Checklist CBCL, continuous regression)
- **Input:** EEG windows from CCD task recordings
- **Window Configuration:**
  - 4-second windows with 2-second stride (more context)
  - Random crop to 2 seconds at training time (augmentation)
  - Fixed 2-second crop at inference
- **Data Format:** (batch, 129 channels, 200 timepoints)
- **Evaluation Metric:** L1 Loss (MAE) - robust to outliers
- **Goal:** Minimize L1 loss, maximize cross-subject generalization

### Data Strategy
- **Training Releases:** R1, R2, R3, R4
- **Validation:** R5
- **Test:** Secret dataset with potentially different subjects
- **Subject Filtering:** Same problematic subjects removed as Challenge 1
- **Critical:** Filter out recordings without p_factor or where `math.isnan(p_factor)`
- **Emphasis:** OUT-OF-DISTRIBUTION ROBUSTNESS - must generalize to unseen subjects

### Model Considerations
- **Architecture:** EEGNeX (from braindecode) - designed for generalization
  - Lightweight, regularized architecture
  - Built-in dropout and normalization
  - Focus: Physiologically meaningful representations
  - Avoid large models that overfit to training subjects
- **Loss Function:** L1 loss (l1_loss) - more robust than MSE for regression
- **Optimizer:** Adamax (lr=0.002) - adaptive learning, good for noisy targets
- **Regularization:** 
  - Random cropping (data augmentation)
  - Dropout throughout network
  - Early stopping on validation set

### Current Model Status (As of October 19, 2025, 6:21 PM)
- **Infrastructure:** ✅ COMPLETE - HDF5 cache + SQLite database + enhanced training
- **VS Code Crash:** ✅ FIXED - Analyzed, documented, prevented (see VSCODE_CRASH_ANALYSIS.md)
- **Cache Status:** 🔄 IN PROGRESS (tmux session 'cache_remaining')
  - R1: ✅ 11GB (61,889 windows)
  - R2: ✅ 12GB (62,000+ windows)
  - R3: 🔄 Loading dataset (in progress)
  - R4: ⏳ Pending
  - R5: ⏳ Pending
  - Expected total: ~50GB HDF5 cache
- **Database:** ✅ READY (data/metadata.db, 56KB, 7 tables, 2 views)
- **Training Script:** ✅ READY (train_challenge2_fast.py)
- **TODO Lists:** ✅ CREATED (crash-resistant, in 4 parts)
- **Benefits:** 10-15x faster data loading (seconds vs 15-30 minutes)
- **Status:** WAITING for cache completion, then start training in tmux

### Infrastructure & Documentation Files
- **create_challenge2_cache_remaining.py** - R3,R4,R5 cache (running in tmux)
- **train_challenge2_fast.py** - Enhanced training with cache + database
- **data/metadata.db** - SQLite tracking database
- **TODO_MASTER_INDEX.md** - Master todo list (crash-resistant)
- **TODO_PART1_INFRASTRUCTURE.md** - Cache creation checklist
- **TODO_PART2_TRAINING.md** - Training checklist
- **TODO_PART3_SUBMISSION.md** - Submission checklist
- **VSCODE_CRASH_ANALYSIS.md** - VS Code crash analysis for team
- **.vscode/settings.json** - Crash prevention settings

### Critical Notes
- **Externalizing Factor:**
  - Mental health construct from CBCL assessment
  - Represents psychopathology dimension (aggression, rule-breaking, etc.)
  - Continuous regression target (NOT classification)
  - Feasibility still open question - requires meaningful EEG biomarkers
- **Challenge Focus:**
  - NOT about high accuracy on training subjects
  - FOCUS: Robust features that transfer to new subjects, sites, sessions
  - Interpretability encouraged - find reproducible biomarkers
  - Generalization > Performance on seen data
- **Common Mistake:** Training on resting state - WRONG! Use contrastChangeDetection
- **Data Loading:**
  - Must use `EEGChallengeDataset` (not EEGDashDataset)
  - Task: "contrastChangeDetection"
  - Description fields: ["subject", "session", "run", "task", "age", "sex", "p_factor"]
  - Filter: `not math.isnan(ds.description["p_factor"])`

---

## 📊 Key Differences Between Challenges

| Aspect | Challenge 1 | Challenge 2 |
|--------|------------|-------------|
| **Task** | contrastChangeDetection | contrastChangeDetection |
| **Target** | rt_from_stimulus (RT) | p_factor (externalizing) |
| **Target Type** | Behavioral (trial-level) | Clinical (subject-level) |
| **Windows** | 2s stimulus-locked | 4s → crop to 2s |
| **Cropping** | Fixed (0.5s after stim) | Random (augmentation) |
| **Loss** | MSE typical | L1 (robust) |
| **Focus** | Temporal dynamics, ERP | Cross-subject biomarkers |
| **Generalization** | Cross-task transfer | Cross-subject invariance |
| **Model Size** | Can be larger (TCN) | Should be smaller (EEGNeX) |
| **Overfitting Risk** | Moderate | HIGH - avoid! |

---

## 🚫 Common Mistakes to Avoid

### Challenge 1
- ❌ Using wrong task (not contrastChangeDetection)
- ❌ Not aligning to stimulus onset (need time-locking)
- ❌ Wrong window timing (must start +0.5s after stimulus)
- ❌ Training only on R5 (need R1-R4 for generalization)
- ❌ Ignoring transfer learning opportunity (passive tasks)

### Challenge 2
- ❌ **CRITICAL:** Using resting state task (common error!)
- ❌ Training on subject-constant targets (overfitting)
- ❌ Large models that memorize training subjects
- ❌ Not filtering out NaN p_factor values
- ❌ Using MSE loss (L1 more robust for clinical targets)
- ❌ Optimizing for training performance over generalization
- ❌ Not implementing data augmentation (random cropping)

---

## 🎯 Submission Format

### Files Required
1. **submission.py** - Main submission file with Submission class
2. **weights_challenge_1.pt** - Challenge 1 model weights
3. **weights_challenge_2.pt** - Challenge 2 model weights

### Submission Class Structure
```python
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ  # 100 Hz
        self.device = DEVICE  # GPU or CPU
    
    def get_model_challenge_1(self):
        # Return Challenge 1 model (loaded with weights)
        # Input: (batch, 129, 200)
        # Output: (batch, 1) - response time
    
    def get_model_challenge_2(self):
        # Return Challenge 2 model (loaded with weights)
        # Input: (batch, 129, 200)  
        # Output: (batch, 1) - externalizing factor
```

### Package Format
- **Single-level ZIP** (no folders): `submission.zip`
- Contents: submission.py, weights_challenge_1.pt, weights_challenge_2.pt
- Command: `zip -j submission.zip submission.py weights_*.pt`

---

## 💾 Current Project Status

### Challenge 1: ✅ READY
- Model: TCN (196K params)
- Checkpoint: checkpoints/challenge1_tcn_competition_best.pth
- Val loss: 0.010170 (excellent)
- Loads correctly: ✅
- Submission ready: ✅

### Challenge 2: 🔄 INFRASTRUCTURE UPGRADE IN PROGRESS (Oct 19, 2025)
- **Strategic Decision:** Stopped training to build HDF5 cache infrastructure
- **Reason:** Challenge 1 has 3.6GB cache (loads in seconds), Challenge 2 didn't (15-30 min)
- **Solution:** Create cache + database + enhanced training for 10-15x speedup
- **Cache Status:** R1 complete (2.6GB), R2-R5 processing (10-30 min remaining)
- **Database:** ✅ Ready (metadata.db with training tracking)
- **Training Script:** ✅ Ready (train_challenge2_fast.py with cache support)
- **Next:** Start fast training after cache completes, expect 5-10 epochs
- **Timeline:** Training this week, submission before Nov 2 deadline (13 days)

### Next Steps
1. ⏳ Wait for cache creation to complete (~10-30 min)
2. ⬜ Start fast training: `python3 train_challenge2_fast.py`
3. ⬜ Monitor via database queries
4. ⬜ Complete training (5-10 epochs with early stopping)
5. ⬜ Copy best weights to submission location
6. ⬜ Test submission locally
7. ⬜ Organize repository
8. ⬜ Submit before Nov 2 deadline

---

## 🔬 Dataset Details

### EEG Specifications
- **Channels:** 129 (128 + ref)
- **Sampling Rate:** 100 Hz (downsampled from 500 Hz)
- **Preprocessing:** 0.5-50 Hz bandpass, Cz reference
- **Format:** BIDS-compatible MNE Raw objects

### Tasks Available
- **Passive:** Resting State (RS), Surround Suppression (SuS), Movie Watching (MW)
- **Active:** Contrast Change Detection (CCD), Sequence Learning (SL), Symbol Search (SyS)
- **Competition Uses:** CCD for both challenges (transfer from passive optional for Ch1)

### Releases
- R1-R4: Full training releases (different release = different acquisition batch)
- R5: Validation/test release
- R6-R11: Additional data (may not be available for all tasks)

### Important Fields
- **Challenge 1:** subject, session, run, task, age, sex, rt_from_stimulus
- **Challenge 2:** subject, session, run, task, age, sex, p_factor
- **Filter:** Remove subjects in SUB_RM list, recordings < 4s, != 129 channels

---

## 📚 Key References

- **Competition Site:** https://eeg2025.github.io
- **Starter Kit:** /home/kevin/Projects/eeg2025/starter_kit_integration/
- **Challenge 1 Tutorial:** starter_kit_integration/challenge_1.py
- **Challenge 2 Tutorial:** starter_kit_integration/challenge_2.py
- **Dataset Paper:** HBN-EEG (biorxiv.org/content/10.1101/2024.10.03.615261v2)
- **Braindecode:** braindecode.org (EEG deep learning models)
- **EEGDash:** eeglab.org/EEGDash/ (dataset access)

---

**Last Updated:** October 19, 2025  
**Competition Deadline:** November 2, 2025 (14 days remaining)

---

## 📁 PROJECT ORGANIZATION (Updated 2025-10-19)

### Directory Structure

```
eeg2025/
├── weights/                         # ORGANIZED WEIGHT FILES
│   ├── challenge1/                  # Challenge 1 weights (response time)
│   │   ├── weights_challenge_1_current.pt  ← Use for submission
│   │   └── weights_challenge_1_YYYYMMDD_HHMMSS.pt  (backups)
│   ├── challenge2/                  # Challenge 2 weights (p_factor)
│   │   ├── weights_challenge_2_current.pt  ← Use for submission
│   │   ├── weights_challenge_2.pt
│   │   └── weights_challenge_2_YYYYMMDD_HHMMSS.pt  (backups)
│   ├── multi_release/               # Multi-release trained versions
│   │   ├── weights_challenge_1_multi_release.pt
│   │   └── weights_challenge_2_multi_release.pt
│   ├── checkpoints/                 # Training checkpoints (.pth)
│   │   └── challenge1_tcn_competition_best.pth
│   ├── archive/                     # Old weights
│   └── WEIGHTS_METADATA.md          # Version tracking & documentation
│
├── training/                        # TRAINING SCRIPTS
│   ├── challenge1/                  # Challenge 1 training
│   ├── challenge2/                  # Challenge 2 training
│   │   ├── train_challenge2_correct.py  (backup)
│   │   └── train_challenge2_correct_YYYYMMDD_HHMMSS.py (timestamped)
│   └── archive/                     # Old training scripts
│
├── submissions/                     # SUBMISSION MANAGEMENT
│   ├── archive/                     # Old submission packages
│   │   ├── eeg2025_submission_v6_CORRECTED_API.zip
│   │   ├── eeg2025_submission_v7_TTA.zip
│   │   ├── prediction_result(2).zip
│   │   └── scoring_result(2).zip
│   ├── versions/                    # Timestamped backups
│   ├── submission_YYYYMMDD_HHMMSS.py  (backups)
│   ├── test_submission_verbose.py   (backup)
│   └── SUBMISSION_HISTORY.md        # Submission workflow docs
│
├── scripts/
│   └── monitoring/                  # Monitoring scripts (copies)
│       ├── monitor_challenge2.sh
│       ├── quick_training_status.sh
│       ├── manage_watchdog.sh
│       └── watchdog_challenge2.sh
│
├── CHANGELOG.md                     # Project changelog
├── README.md                        # Main documentation
└── [active files in root]           # See below
```

### Active Files in Root (Required)

**These files MUST stay in root while training/submitting:**

- **submission.py** - Competition submission script (required)
- **test_submission_verbose.py** - Submission validator
- **train_challenge2_correct.py** - Active training script (PID 548497 running)
- **monitor_challenge2.sh** - Training monitor
- **quick_training_status.sh** - Quick status check
- **manage_watchdog.sh** - Watchdog control
- **watchdog_challenge2.sh** - Watchdog daemon (PID 560789 running)

### Version Control & Backups

**Weights Versioning:**
- Current versions: `weights/challenge*/weights_challenge_*_current.pt`
- Timestamped backups created on each organization
- Easy rollback to any previous version

**Script Backups:**
- Training scripts backed up with timestamps
- Submission.py backed up on each organization
- All old versions preserved in archive/

**Metadata Files:**
- `weights/WEIGHTS_METADATA.md` - Complete weights tracking
- `submissions/SUBMISSION_HISTORY.md` - Submission guide
- `CHANGELOG.md` - Project history

### File Movement Rules

**Safe to Move/Archive:**
- Old submission zips
- Completed training scripts
- Documentation files (after training complete)

**NEVER Move While Training Active:**
- train_challenge2_correct.py (in use by PID 548497)
- Monitoring scripts (watchdog depends on them)
- Current weight files
- submission.py (required for competition)

---

## 🕐 Recent Work Sessions

### October 19, 2025 (6:30 PM) - Crash Recovery & Multi-Part Documentation ✅
**Session Summary:** SESSION_OCT19_MASTER_INDEX.md (+ 5 detail parts)

**What Happened:**
- VS Code crashed at 5:53 PM (RegExp.test() froze UI thread on 22MB log)
- All terminal processes killed, R3-R5 cache creation lost
- R1 (11GB) and R2 (12GB) cache files survived

**Actions Taken:**
1. ✅ Analyzed crash logs, identified root cause
2. ✅ Fixed .vscode/settings.json (file watcher exclusions)
3. ✅ Moved all processes to tmux (crash-resistant)
4. ✅ Fixed cache script API (3 iterations, correct: `from eegdash`)
5. ✅ Created multi-part documentation (6 files vs. 1 large file)
6. ✅ Updated memory with summarization rules

**Current Status:**
- Cache R3 creating in tmux (downloading metadata)
- R4, R5 queued after R3
- All work crash-resistant now
- Next: Wait for cache → Start training

**Files Created:**
- SESSION_OCT19_*.md (6 summary parts)
- VSCODE_CRASH_ANALYSIS.md (for VS Code team)
- TODO_MASTER_INDEX.md + 3 parts
- .vscode/settings.json (crash prevention)

---

### Submission Workflow

1. **Test**: `python test_submission_verbose.py`
2. **Create Package**:
   ```bash
   zip -j submission.zip \
       submission.py \
       weights/challenge1/weights_challenge_1_current.pt \
       weights/challenge2/weights_challenge_2_current.pt
   ```
3. **Verify**: `unzip -l submission.zip` (should show exactly 3 files)
4. **Submit**: Upload to competition platform

### Current Training Status (2025-10-19 15:10 - POST POWER SURGE)

**Challenge 1:** ✅ Ready
- Status: Complete, ready for submission
- Checkpoint: `checkpoints/challenge1_tcn_competition_best.pth`
- Val Loss: 0.010170 (NRMSE)
- Note: Unaffected by power surge

**Challenge 2:** 🔄 Training RESTARTED
- **Power Surge Event:** System went down ~15:00, training interrupted
- **Previous Progress:** Epoch 1/20, Batch 1,320/5,214 (~25% of epoch 1)
- **Previous Runtime:** 47 hours 15 minutes
- **Previous PIDs:** Training 548497, Watchdog 560789 (terminated)
- **Recovery Actions:**
  - Backed up logs to `challenge2_correct_training_backup_20251019_150947.log`
  - Created `restart_challenge2_training.sh` script
  - Verified Python dependencies and data integrity
  - Restarted training from scratch (epoch 1, batch 0)
- **Current Status:** ✅ RUNNING (restarted Oct 19, 15:09)
- **New PIDs:** Training 8593, Watchdog 8654
- **Progress:** Data loading phase
- **ETA:** Early stopping expected around epoch 5-10
- **Note:** Previous progress lost but acceptable for long training runs

### Organization Benefits

✅ Clear structure by purpose
✅ Version control with timestamps
✅ Easy rollback capability
✅ Metadata tracking
✅ Clean root directory
✅ Protected active processes
✅ Comprehensive documentation

