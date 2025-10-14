# EEG Foundation Challenge 2025 - Data Acquisition Guide

**Status**: Critical Priority 🔴  
**Timeline**: Complete within 1-2 days  
**Impact**: Blocking all model training

---

## Overview

The EEG Foundation Challenge 2025 requires the **Healthy Brain Network (HBN)** dataset. This guide provides step-by-step instructions for acquiring and preparing the data.

---

## Dataset Information

### Healthy Brain Network (HBN) Dataset

**Source**: Child Mind Institute  
**Size**: ~500GB (compressed), ~1TB (uncompressed)  
**Participants**: 1,500+ children and adolescents (5-21 years)  
**Tasks**: Resting state, SuS task, CCD task  
**Format**: BIDS-compliant EEG data

**Official Links**:
- Main Site: https://childmind.org/science/healthy-brain-network/
- Data Portal: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/
- LORIS Access: http://data.healthybrainnetwork.org/

---

## Step 1: Registration & Access

### Create Account

1. Visit NITRC Portal: https://www.nitrc.org/account/register.php
2. Register with institutional email (recommended)
3. Wait for account approval (usually 24-48 hours)

### Request Data Access

1. Go to HBN Data Use Agreement page
2. Complete institutional review form
3. Submit data access request
4. Approval takes 3-7 business days

**Alternative for Challenge Participants**:
- Check if competition provides direct download link
- Look for challenge-specific data portal
- Contact organizers if special access available

---

## Step 2: Download Strategy

### Option A: AWS S3 (Recommended - Fastest)

```bash
# Install AWS CLI
pip install awscli

# Configure credentials (if provided by challenge)
aws configure

# Download specific subjects (example)
aws s3 sync s3://fcp-indi/data/Projects/HBN/EEG_data/ \
    data/raw/hbn/ \
    --exclude "*" \
    --include "sub-*/ses-*/eeg/*_eeg.fif" \
    --include "sub-*/ses-*/eeg/*_events.tsv" \
    --include "sub-*/ses-*/eeg/*_channels.tsv"
```

### Option B: Direct HTTP Download

```bash
# Use the provided download script
python scripts/download_hbn_data.py \
    --output_dir data/raw/hbn \
    --tasks sus ccd rest \
    --max_subjects 1500 \
    --resume  # Resume interrupted downloads
```

### Option C: Manual Download via Browser

1. Navigate to LORIS portal
2. Select EEG data filter
3. Download subject batches
4. Extract to `data/raw/hbn/`

---

## Step 3: Verify Download

```bash
# Check data integrity
python scripts/verify_data_integrity.py \
    --data_dir data/raw/hbn \
    --output_report data/verification_report.json

# Expected output:
# ✓ Found 1,500 subjects
# ✓ 1,200 with SuS task data
# ✓ 800 with CCD task data  
# ✓ 1,400 with resting state data
# ✓ All files have valid BIDS structure
```

---

## Step 4: Organize Data Structure

Ensure BIDS-compliant structure:

```
data/raw/hbn/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── sub-NDARINV001/
│   └── ses-01/
│       └── eeg/
│           ├── sub-NDARINV001_ses-01_task-sus_eeg.fif
│           ├── sub-NDARINV001_ses-01_task-sus_events.tsv
│           ├── sub-NDARINV001_ses-01_task-sus_channels.tsv
│           └── sub-NDARINV001_ses-01_task-sus_eeg.json
├── sub-NDARINV002/
│   └── ses-01/
│       └── eeg/
│           └── ...
└── derivatives/
    └── labels/
        ├── cross_task_labels.csv
        └── psychopathology_labels.csv
```

---

## Step 5: Quick Data Exploration

```python
# Run quick data exploration
python scripts/explore_hbn_data.py

# This will generate:
# - data/reports/data_summary.html
# - data/reports/channel_info.csv
# - data/reports/task_coverage.csv
# - data/reports/demographic_distribution.png
```

---

## Troubleshooting

### Issue: Download Timeout
**Solution**: Use `--resume` flag or download in batches

### Issue: Insufficient Disk Space
**Solution**: 
- Use external hard drive (1TB+ recommended)
- Download task-specific data only (SuS + CCD = ~300GB)

### Issue: Corrupted Files
**Solution**: 
```bash
# Re-download specific subjects
python scripts/download_hbn_data.py \
    --subjects sub-NDARINV001 sub-NDARINV002 \
    --force_redownload
```

### Issue: BIDS Validation Errors
**Solution**:
```bash
# Install BIDS validator
npm install -g bids-validator

# Validate structure
bids-validator data/raw/hbn/
```

---

## Sample Data for Testing

If full download takes too long, start with sample data:

```bash
# Download first 10 subjects for testing
python scripts/download_hbn_data.py \
    --output_dir data/raw/hbn_sample \
    --max_subjects 10 \
    --tasks sus

# This takes ~5-10 minutes and uses ~5GB
```

---

## Next Steps After Data Acquisition

1. ✅ Run data validation tests
2. ✅ Load sample subjects with existing data loader
3. ✅ Verify preprocessing pipeline works
4. ✅ Check label alignment (cross_task, psychopathology)
5. ✅ Train small test model to validate end-to-end

---

## Estimated Timeline

| Step | Duration | Depends On |
|------|----------|------------|
| Registration | 1-2 days | Account approval |
| Access Request | 3-7 days | Institutional review |
| Download Setup | 30 min | AWS/credentials |
| Data Download | 4-12 hours | Internet speed |
| Verification | 1-2 hours | Data integrity checks |
| **Total** | **5-10 days** | Including approval time |

**Recommendation**: Start registration process immediately while working on other tasks.

---

## Competition-Specific Notes

**Check with Challenge Organizers**:
- Do they provide pre-processed data?
- Is there a smaller challenge subset?
- Are there download credentials?
- What is the train/test split?

**Challenge Deadlines**:
- Early bird submission: TBD
- Final submission: TBD
- Make sure to leave 2 weeks for training after data acquisition

---

## Resources

- HBN Documentation: http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Pheno_Documentation.html
- BIDS Specification: https://bids-specification.readthedocs.io/
- MNE-BIDS Tutorial: https://mne.tools/mne-bids/stable/index.html

