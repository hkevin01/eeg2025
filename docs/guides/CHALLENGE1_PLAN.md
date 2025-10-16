# Challenge 1: Complete Solution Plan

**Due Date:** November 2, 2025 (18 days remaining)  
**Task:** Response Time Prediction from CCD Task  
**Time Required:** 4-6 hours  
**Status:** Starting now! 🚀

---

## Phase 1: Download CCD Data (1-2 hours)

### Step 1.1: Check AWS CLI Installation
```bash
# Check if AWS CLI is installed
which aws

# If not installed:
sudo apt-get update
sudo apt-get install -y awscli
```

### Step 1.2: Download Dataset
```bash
# Create directory
mkdir -p /home/kevin/Projects/eeg2025/data/raw/hbn_ccd

# Download R1_L100_bdf release (contains CCD data)
# This is about 10-20GB, will take time based on internet speed
aws s3 cp --recursive \
  s3://nmdatasets/NeurIPS25/R1_L100_bdf \
  /home/kevin/Projects/eeg2025/data/raw/hbn_ccd \
  --no-sign-request

# Alternative: Download specific release with CCD
# R5 mini release (smaller, for testing)
aws s3 cp --recursive \
  s3://nmdatasets/NeurIPS25/R5_mini_L100_bdf \
  /home/kevin/Projects/eeg2025/data/raw/hbn_ccd_mini \
  --no-sign-request
```

### Step 1.3: Verify Downloaded Data
```bash
# Check directory structure
ls -lh /home/kevin/Projects/eeg2025/data/raw/hbn_ccd/

# Count subjects
ls -d /home/kevin/Projects/eeg2025/data/raw/hbn_ccd/sub-* | wc -l

# Check for CCD files
find /home/kevin/Projects/eeg2025/data/raw/hbn_ccd -name "*contrastChangeDetection*" | head -5

# Verify participants.tsv exists
cat /home/kevin/Projects/eeg2025/data/raw/hbn_ccd/participants.tsv | head -3
```

---

## Phase 2: Explore CCD Data (30 minutes)

### Step 2.1: Analyze Available Subjects
```bash
# Create exploration script
python3 /home/kevin/Projects/eeg2025/scripts/explore_ccd_data.py
```

### Step 2.2: Check Response Time Distribution
- Understand response time range
- Check for outliers
- Verify data quality

---

## Phase 3: Create Training Script (30 minutes)

### Step 3.1: Copy and Adapt Challenge 2 Script
```bash
cp /home/kevin/Projects/eeg2025/scripts/train_challenge2_externalizing.py \
   /home/kevin/Projects/eeg2025/scripts/train_challenge1_response_time.py
```

### Step 3.2: Key Changes Needed:
1. **Target column:** Change from 'externalizing' to response time field
2. **Task filter:** Change from 'RestingState' to 'contrastChangeDetection'
3. **Data extraction:** Parse response time from CCD events
4. **Model output:** Single regression value (response time in seconds)

---

## Phase 4: Train Challenge 1 Model (2-3 hours)

### Step 4.1: Run Training
```bash
cd /home/kevin/Projects/eeg2025

# Start training with progress output
python3 scripts/train_challenge1_response_time.py 2>&1 | tee logs/challenge1_training.log
```

### Expected Results:
- **Target:** NRMSE < 0.5
- **Goal:** NRMSE ~ 0.1-0.2 (based on Challenge 2 success)
- **Training time:** 2-3 hours on CPU
- **Output:** `checkpoints/response_time_model.pth`

### Step 4.2: Convert to Competition Format
```bash
python3 << 'EOFC'
import torch
from pathlib import Path

checkpoint = torch.load("checkpoints/response_time_model.pth", map_location='cpu')
torch.save(checkpoint['model_state_dict'], "weights_challenge_1.pt")
print("✅ Created weights_challenge_1.pt")
EOFC
```

---

## Phase 5: Test Complete Submission (15 minutes)

### Step 5.1: Test Both Models
```bash
# Quick test (CPU, fast)
python3 scripts/test_submission_quick.py

# Full test with both challenges
python3 submission.py
```

### Step 5.2: Verify Files
```bash
ls -lh weights_challenge_1.pt
ls -lh weights_challenge_2.pt
ls -lh submission.py
```

---

## Phase 6: Create Final Submission (15 minutes)

### Step 6.1: Package Files
```bash
cd /home/kevin/Projects/eeg2025

# Clean previous packages
rm -rf submission_package
rm -f submission*.zip

# Create package
mkdir -p submission_package
cp submission.py submission_package/
cp weights_challenge_1.pt submission_package/
cp weights_challenge_2.pt submission_package/

# Create ZIP (single-level depth, NO FOLDERS!)
cd submission_package
zip -r ../submission_complete.zip .
cd ..

# Verify ZIP structure
unzip -l submission_complete.zip
```

### Step 6.2: Final Checks
```bash
# Check ZIP is single-level
unzip -l submission_complete.zip | grep -E "submission.py|weights_challenge"

# File sizes
ls -lh submission_complete.zip
```

---

## Phase 7: Submit to Codabench (5 minutes)

### Step 7.1: Upload
1. Go to: https://www.codabench.org/competitions/4287/
2. Navigate to "Participate" → "Submit"
3. Upload `submission_complete.zip`
4. Add description: "Complete submission - Both challenges"

### Step 7.2: Monitor Results
- Check submission status
- View leaderboard
- Note any errors for iteration

---

## Timeline Checklist

```markdown
**Total Time:** 4-6 hours (with 18 days available)

- [⏰ 0:00-2:00] Download CCD data (1-2 hours)
  - [ ] Install AWS CLI
  - [ ] Download R1 or R5 release
  - [ ] Verify data integrity
  
- [⏰ 2:00-2:30] Explore CCD data (30 minutes)
  - [ ] Count subjects with CCD
  - [ ] Analyze response times
  - [ ] Check data quality
  
- [⏰ 2:30-3:00] Create training script (30 minutes)
  - [ ] Copy Challenge 2 script
  - [ ] Adapt for CCD task
  - [ ] Add response time parsing
  
- [⏰ 3:00-6:00] Train model (2-3 hours)
  - [ ] Run training with progress bars
  - [ ] Monitor NRMSE (target < 0.5)
  - [ ] Save best checkpoint
  
- [⏰ 6:00-6:15] Test submission (15 minutes)
  - [ ] Convert checkpoint format
  - [ ] Test both models
  - [ ] Verify predictions
  
- [⏰ 6:15-6:30] Package & submit (15 minutes)
  - [ ] Create ZIP package
  - [ ] Upload to Codabench
  - [ ] Monitor results
```

---

## Fallback Plans

### If Download Takes Too Long:
- Use mini release (R5_mini) first for testing
- Train on mini, submit for baseline
- Download full release overnight
- Retrain and resubmit

### If NRMSE Target Not Met:
- Increase training epochs
- Try data augmentation
- Ensemble multiple models
- Fine-tune hyperparameters

### If CCD Data Not Available:
- Check alternative releases (R2, R3, R4, etc.)
- Contact organizers on Discord
- Submit Challenge 2 only (already excellent!)

---

**Next Command to Run:**
```bash
# Check if AWS CLI is installed, then start download
which aws && echo "AWS CLI ready!" || echo "Need to install AWS CLI"
```

