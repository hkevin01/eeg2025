# EEG 2025 Competition - Final TODO

## Summary

✅ **Challenge 2 is READY** - NRMSE: 0.0808 (6x better than target!)  
⭕ **Challenge 1 needs CCD data** - Download and train required

---

## Option 1: Submit Challenge 2 Now (15 minutes) ⚡

```markdown
- [ ] Create submission package
- [ ] Zip files
- [ ] Upload to Codabench
- [ ] Monitor results
```

### Commands:
```bash
cd /home/kevin/Projects/eeg2025
mkdir -p submission_package
cp submission.py submission_package/
cp weights_challenge_2.pt submission_package/
cd submission_package
zip -r ../submission_challenge2.zip .
cd ..

# Upload submission_challenge2.zip to:
# https://www.codabench.org/competitions/4287/
```

---

## Option 2: Complete Both Challenges (4-6 hours) 🎯

### Phase 1: Download CCD Data (1-2 hours)
```markdown
- [ ] Install AWS CLI if needed
- [ ] Download R1_L100_bdf release
- [ ] Verify CCD task files
- [ ] Update participants.tsv path
```

### Commands:
```bash
# Install AWS CLI (if needed)
sudo apt-get install awscli

# Download data
mkdir -p data/raw/hbn_ccd
aws s3 cp --recursive \
  s3://nmdatasets/NeurIPS25/R1_L100_bdf \
  data/raw/hbn_ccd \
  --no-sign-request

# Verify
ls -lh data/raw/hbn_ccd/
```

### Phase 2: Train Challenge 1 (2-3 hours)
```markdown
- [ ] Create train_challenge1_response_time.py
- [ ] Run training (with progress bars!)
- [ ] Save weights_challenge_1.pt
- [ ] Test submission.py with both models
```

### Phase 3: Submit Complete Package (30 minutes)
```markdown
- [ ] Copy both weight files
- [ ] Create full submission.zip
- [ ] Upload to Codabench
- [ ] Monitor leaderboard
```

---

## Key Files Status

| File | Status | Notes |
|------|--------|-------|
| `submission.py` | ✅ Ready | Official format, tested |
| `weights_challenge_2.pt` | ✅ Ready | NRMSE: 0.0808 |
| `weights_challenge_1.pt` | ⭕ Pending | Need CCD data |
| `optimized_dataloader.py` | ✅ Ready | Fast, progress bars |
| `train_challenge2_externalizing.py` | ✅ Ready | Enhanced output |

---

## Quick Links

- **Codabench Submission:** https://www.codabench.org/competitions/4287/
- **Competition Website:** https://eeg2025.github.io/
- **Starter Kit:** https://github.com/eeg2025/startkit
- **Data Download:** https://eeg2025.github.io/data/
- **Discord Support:** https://discord.gg/8jd7nVKwsc

---

## Testing Commands

### Quick Tests (< 1 minute each):
```bash
# Test submission format
python3 scripts/test_submission_quick.py

# Test optimized loader
python3 scripts/optimized_dataloader.py

# Verify files exist
ls -lh weights_challenge_2.pt
ls -lh submission.py
```

### Full Test (if time permits):
```bash
# Test with full dataset loading
python3 submission.py
```

---

## Decision Matrix

| Scenario | Best Option | Time | Result |
|----------|-------------|------|--------|
| **Want leaderboard score ASAP** | Option 1 | 15 min | Challenge 2 only |
| **Want complete submission** | Option 2 | 4-6 hrs | Both challenges |
| **Data download issues** | Option 1 | 15 min | Submit what works |
| **Plenty of time available** | Option 2 | 4-6 hrs | Full solution |

---

**Recommendation:** Start with Option 1 to get a baseline score, then work on Option 2 to improve leaderboard position.

