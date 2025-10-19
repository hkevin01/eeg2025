---
applyTo: '**'
---

# EEG Foundation Challenge 2025 - Competition Memory

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
- **Status:** ✅ Weights load correctly in submission.py

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

### Current Model Status
- **File:** `weights_challenge_2.pt` (261 KB)
- **Issue:** ⚠️ Architecture mismatch - checkpoint from different model
- **Fallback:** Using untrained EEGNeX (won't crash, but poor performance)
- **Action Needed:** Train correct EEGNeX model on CCD task with p_factor target

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

### Challenge 2: ⚠️ NEEDS TRAINING
- Model: Should be EEGNeX
- Current weights: From different architecture (CompactExternalizingCNN)
- Training script: `train_challenge2_correct.py` (following starter kit)
- Status: Training in progress
- Action: Complete training, then update submission

### Next Steps
1. ✅ Complete Challenge 2 training (in progress)
2. ⬜ Copy weights: `cp weights_challenge_2_correct.pt weights_challenge_2.pt`
3. ⬜ Test submission locally
4. ⬜ Recreate submission.zip
5. ⬜ Submit to competition
6. ⬜ Monitor leaderboard results

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
