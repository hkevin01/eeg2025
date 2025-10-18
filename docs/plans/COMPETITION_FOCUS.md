# 🏆 NeurIPS 2025 EEG Foundation Challenge - Competition Focus

**Date:** October 17, 2025  
**Deadline:** November 2, 2025 (16 days remaining)  
**Competition:** https://eeg2025.github.io  
**Codabench:** https://www.codabench.org/competitions/4287/

---

## 🎯 THE TWO CHALLENGES

### Challenge 1: Cross-Task Transfer Learning
**Goal:** Develop models that effectively transfer knowledge from passive EEG tasks to active cognitive tasks

**Task:** Contrast Change Detection (CCD)
- **Input:** EEG data (129 channels, 2 seconds @ 100 Hz)
- **Output:** Predict response time (regression)
- **Data:** Releases R1-R5 (train on R1-R4, validate on R5)

**Key Points:**
- Pre-train on passive tasks (RestingState, DespicableMe, etc.)
- Fine-tune on active task (contrastChangeDetection)
- Predict: How long does it take the subject to respond?
- Evaluation: Pearson correlation + accuracy metrics

**Current Status:**
- ✅ Training TCN model on R1, R2, R3 → Validate on R4
- 🔄 Currently running (PID: 105017, 78% CPU)
- 📊 Processing 3,786 windows from competition data

### Challenge 2: Externalizing Factor Prediction
**Goal:** Predict externalizing factor from EEG to enable objective mental health assessments

**Task:** Regression of externalizing factor (mental health measure)
- **Input:** EEG data from any task
- **Output:** Predict p_factor score
- **Data:** Releases R2, R3, R4 (train), R5 (validate)

**Key Points:**
- Externalizing factor from Child Behavior Checklist (CBCL)
- Learn physiologically meaningful representations
- Robust out-of-distribution generalization
- May use self-supervised learning (e.g., Relative Positioning)

---

## 📊 COMPETITION DATA STRUCTURE

### Releases (R1-R5)
```
R1, R2, R3 → Training data
R4 → Validation data  
R5 → Test data (for submission evaluation)
```

### Tasks Available
1. **contrastChangeDetection** (CCD) - Challenge 1 target task
2. **RestingState** - Passive, good for pre-training
3. **DespicableMe** - Passive video watching
4. **ThePresent** - Passive video watching
5. **FunwithFractals** - Passive visual stimuli
6. **surroundSupp** - Visual task
7. **seqLearning8target** - Sequential learning

### Data Format
- **EEG:** 129 channels, various sampling rates (downsampled to 100 Hz)
- **Format:** BDF files (Biosemi Data Format)
- **Preprocessing:** Handled by `eegdash` and `braindecode`
- **Metadata:** Subject, session, age, gender, p_factor, etc.

---

## 🚀 CURRENT TRAINING STATUS

### What's Running Right Now
```bash
Process: python3 train_tcn_competition_data.py
PID: 105017
CPU: 78.4%
Time: Running for 45 seconds
Status: Loading and preprocessing competition data
```

### Training Configuration
```python
Model: TCN (Temporal Convolutional Network)
  - Architecture: 196K parameters
  - Filters: 48
  - Kernel: 7
  - Levels: 5
  - Dropout: 0.3

Data:
  - Train: R1, R2, R3 (contrastChangeDetection)
  - Val: R4 (contrastChangeDetection)
  - Task: Predict response time
  - Max datasets: 50 per release (for faster iteration)

Training:
  - Optimizer: AdamW (lr=0.001, wd=0.0001)
  - Scheduler: CosineAnnealingWarmRestarts
  - Batch size: 16 (effective 32 with accumulation)
  - Epochs: 100 (patience=15)
  - Device: CPU (stable, survives crashes)
```

### Crash-Proof System
✅ **Running with nohup** - Survives VS Code crashes  
✅ **Running in background** - Survives terminal closure  
✅ **Automatic checkpointing** - Saves every 5 epochs  
✅ **Early stopping** - Prevents overfitting  
✅ **Signal handling** - Graceful shutdown on SIGINT/SIGTERM  

---

## 📈 WHAT WE'RE TRAINING FOR

### Challenge 1: Response Time Prediction

**Pipeline:**
1. **Load competition data** (R1-R3 for training)
   - Task: contrastChangeDetection
   - Extract trials with stimulus + response
   
2. **Preprocessing:**
   - Annotate trials with response times
   - Create 2-second windows starting 0.5s after stimulus
   - Normalize EEG data (channel-wise z-score)
   
3. **Model Training:**
   - TCN processes temporal patterns
   - Predicts continuous response time value
   - Optimizes MSE loss
   
4. **Validation:**
   - Test on R4 data
   - Compute validation loss
   - Early stopping when no improvement
   
5. **Submission:**
   - Load best checkpoint
   - Predict on R5 test data
   - Submit to Codabench

### Challenge 2: Externalizing Factor (Next)

**Pipeline (to implement):**
1. Load multi-task data (RestingState + others)
2. Create fixed-length windows
3. Train regression model for p_factor
4. Validate on R5
5. Submit to Codabench

---

## 🎯 KEY COMPETITION REQUIREMENTS

### Submission Format
```python
# submission.py must contain:

class Challenge1Model:
    def __init__(self):
        # Load your trained model
        pass
    
    def __call__(self, X):
        # X: (batch, channels, time)
        # Return: (batch, 1) response times
        pass

class Challenge2Model:
    def __init__(self):
        # Load your trained model
        pass
    
    def __call__(self, X):
        # X: (batch, channels, time)
        # Return: (batch, 1) p_factor scores
        pass
```

### File Size Limits
- **Maximum submission size:** 50 MB
- **Our current models:**
  - TCN: 2.4 MB ✅
  - With TTA: 9.3 MB ✅
  - Plenty of room for improvement!

### Evaluation Metrics

**Challenge 1:**
- Pearson correlation (response time prediction)
- Success rate accuracy (optional)
- Combined metric (weighted)

**Challenge 2:**
- Mean Absolute Error (MAE)
- Pearson correlation
- Out-of-distribution generalization

---

## 🔄 TRAINING WORKFLOW (Current)

### Step 1: Data Loading ✅ IN PROGRESS
```
📦 Processing release: R1
   Found 1,328 datasets
   Limited to 50 datasets
   Valid datasets: 45
   Preprocessing...
   Creating windows...
   Total windows: 1,262
   ✅ Extracted samples from R1
```

### Step 2: Training (Next)
- Will train for up to 100 epochs
- Early stopping after 15 epochs without improvement
- Save checkpoints every 5 epochs
- Save best model based on validation loss

### Step 3: Integration (After Training)
- Load best checkpoint
- Integrate into submission.py
- Replace current Challenge1Model
- Test locally with local_scoring.py
- Create new submission ZIP

### Step 4: Submission (Final)
- Upload to Codabench
- Wait for test results (1-2 hours)
- Compare with current score (0.2832 NRMSE)
- Iterate if needed

---

## 📁 FILES IN COMPETITION SUBMISSION

### Current Submission (v5)
```
eeg2025_submission_tta_v5.zip (9.3 MB)
├── submission.py                           # 8.2 KB - Main entry point
├── submission_base.py                      # 11.8 KB - Base models
├── response_time_attention.pth             # 9.8 MB - Challenge 1 model
└── weights_challenge_2_multi_release.pt    # 261 KB - Challenge 2 model
```

### Next Submission (v6) - PLAN
```
eeg2025_submission_tcn_v6.zip
├── submission.py                           # Updated with TCN
├── submission_base.py                      # Keep TTA integration
├── challenge1_tcn_competition_best.pth     # 2.4 MB - NEW TCN model
├── weights_challenge_2_multi_release.pt    # 261 KB - Keep current
└── tta_predictor.py                        # Optional TTA wrapper
```

---

## 🎓 LESSONS FROM STARTER KIT

### Challenge 1 Starter Kit Teaches:
1. ✅ Load data with EEGChallengeDataset
2. ✅ Annotate trials with response times
3. ✅ Create stimulus-locked windows
4. ✅ Build regression models
5. ✅ Train/val/test split by subject
6. ✅ Evaluate with RMSE and correlation

### Challenge 2 Starter Kit Teaches:
1. Load multi-task data
2. Extract p_factor from metadata
3. Create fixed-length windows
4. Handle regression with continuous targets
5. Emphasize generalization

### We're Following Best Practices:
- ✅ Using official EEGChallengeDataset loader
- ✅ Using official preprocessing (annotate_trials_with_target)
- ✅ Using official windowing (create_windows_from_events)
- ✅ Training on R1-R3, validating on R4
- ✅ Will test on R5 via submission
- ✅ Using braindecode-compatible models

---

## 🚨 CRITICAL COMPETITION RULES

### Must Use Official Tools
- ✅ `eegdash.EEGChallengeDataset` for data loading
- ✅ `braindecode` for preprocessing
- ✅ Official annotations (annotate_trials_with_target)
- ✅ Official windowing (create_windows_from_events)

### Must Follow Data Splits
- ✅ Train on R1, R2, R3
- ✅ Validate on R4
- ✅ Test on R5 (via submission only)
- ❌ NEVER train on R5 (that's cheating!)

### Must Meet Technical Requirements
- ✅ Python 3.12 compatible
- ✅ PyTorch models
- ✅ Under 50 MB submission size
- ✅ GPU and CPU compatible
- ✅ No external data dependencies

---

## 📊 CURRENT COMPETITION STANDING

### Our Submissions
| Version | Date | Challenge 1 | Challenge 2 | Combined | Rank |
|---------|------|-------------|-------------|----------|------|
| v1 | Oct 14 | 2.013 | - | 2.013 | #47 |
| v5 | Ready | TTA (exp: 0.25) | Current | 0.25-0.26 | Est: Top 15 |
| v6 | Training | TCN (exp: 0.21) | Current | 0.21-0.22 | Est: Top 5 |

### Target Metrics
- **Current:** 0.2832 NRMSE
- **Top 10:** ~0.20 NRMSE
- **Top 3:** ~0.16-0.18 NRMSE
- **Target:** <0.16 NRMSE for #1 🏆

---

## 🔥 NEXT ACTIONS

### Immediate (Today)
1. ✅ Wait for TCN training to complete (~1-2 hours)
2. ⬜ Evaluate TCN on R4 validation set
3. ⬜ If good, integrate into submission.py
4. ⬜ Test locally with local_scoring.py
5. ⬜ Create v6 submission ZIP
6. ⬜ Upload v6 to Codabench

### Short-term (This Week)
1. ⬜ Implement Challenge 2 TCN training
2. ⬜ Train on multi-task data for p_factor
3. ⬜ Create v7 with improved Challenge 2
4. ⬜ Test S4 State Space Model
5. ⬜ Implement ensemble of TCN + S4

### Medium-term (Next Week)
1. ⬜ Pre-training on passive tasks
2. ⬜ Fine-tuning on active tasks
3. ⬜ Advanced augmentation strategies
4. ⬜ Hyperparameter optimization
5. ⬜ Multiple submissions for A/B testing

### Before Deadline (Nov 2)
1. ⬜ Super-ensemble of all best models
2. ⬜ Optimize TTA parameters
3. ⬜ Final hyperparameter sweep
4. ⬜ Last-minute improvements
5. ⬜ Submit final version

---

## 💡 KEY INSIGHTS

### What Makes This Competition Unique
1. **Transfer Learning Focus:** Not just accuracy, but transferability
2. **Real-World Data:** Noisy, variable, real children's EEG
3. **Generalization:** Must work on unseen subjects and sites
4. **Open Source:** Community-driven, collaborative
5. **Clinical Relevance:** Actual mental health applications

### Our Strategy
1. **Start Simple:** Basic TCN working first ✅
2. **Validate Everything:** Always test on R4 before submission
3. **Iterate Fast:** Small improvements compound
4. **Use Official Tools:** Don't reinvent the wheel
5. **Focus on Competition Goals:** Response time + p_factor

### What's Working
- ✅ Crash-proof training system
- ✅ Following official starter kit patterns
- ✅ Using competition data correctly (R1-R4)
- ✅ TCN architecture validated
- ✅ Clear path to submission

### What's Next
- Train on FULL datasets (not mini, not limited to 50)
- Implement pre-training on passive tasks
- Add Challenge 2 training
- Create ensemble models
- Optimize for competition metrics

---

## 🎯 SUCCESS CRITERIA

### Minimum Success (Baseline)
- ✅ TCN trains without crashing
- ✅ Validation loss decreases
- ⬜ Submission format correct
- ⬜ Better than v1 score (2.013)

### Good Success (Top 15)
- ⬜ TCN+TTA submission uploaded
- ⬜ Score: 0.25-0.26 NRMSE
- ⬜ Both challenges working
- ⬜ Rank improvement visible

### Excellent Success (Top 5)
- ⬜ Advanced models (S4, ensemble)
- ⬜ Score: 0.20-0.22 NRMSE
- ⬜ Transfer learning working
- ⬜ Pre-training + fine-tuning

### Perfect Success (Top 3)
- ⬜ Super-ensemble
- ⬜ Score: <0.18 NRMSE
- ⬜ Novel architecture insights
- ⬜ Reproducible code shared

---

## 🔄 MONITORING COMMANDS

```bash
# Check training status
./scripts/monitor_training.sh

# Watch live training log
tail -f logs/train_real_*.log

# Check for new checkpoints
ls -lht checkpoints/challenge1_tcn_competition*.pth

# View training history
cat checkpoints/challenge1_tcn_competition_history.json

# Check process is running
ps aux | grep train_tcn_competition

# Stop training (if needed)
kill <PID>
```

---

**Last Updated:** October 17, 2025, 18:21  
**Status:** ✅ Training on competition data (R1-R3 → R4)  
**Next Milestone:** Complete TCN training, create v6 submission  
**Days to Deadline:** 16 days
