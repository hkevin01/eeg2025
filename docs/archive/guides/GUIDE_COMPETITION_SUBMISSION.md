# Guide: Competition Submission

## Quick Validation

```bash
# Check files
ls -lh submissions/

# Verify format
head submissions/challenge1_predictions.csv
# Expected: participant_id,age_prediction

# Count rows
wc -l submissions/*.csv
# Expected: 401 each (1 header + 400 predictions)
```

## Submission Platforms

### Kaggle
```bash
pip install kaggle
kaggle competitions submit -c <competition-name> \
  -f submissions/challenge1_predictions.csv \
  -m "Transfer learning approach"
```

### CodaLab
1. Zip submission: `zip c1.zip challenge1_predictions.csv`
2. Upload via web interface
3. Wait for evaluation

## Current Status

**Challenge 1:**
- Your score: Pearson r = 0.0593 (random labels)
- Target: r > 0.3
- **Fix:** Get real ages (see GUIDE_IMPROVE_CHALLENGE1.md)

**Challenge 2:**
- Status: Not completed
- Target: AUROC > 0.7
- **Action:** Run training (see GUIDE_CHALLENGE2.md)

## Priority Actions

1. **Fix Challenge 1** (30 min) ⭐⭐⭐
   - Get real age labels from participants.tsv
   - Re-train → Expected r > 0.3 ✅

2. **Complete Challenge 2** (30 min) ⭐⭐⭐
   - Train sex classification model
   - Expected AUROC > 0.7 ✅

3. **Use full dataset** (3-5 hours) ⭐⭐
   - 38K samples instead of 5K
   - 2x performance improvement

## Checklist

- [ ] All test samples predicted
- [ ] No missing/NaN values
- [ ] Values in valid range
- [ ] CSV format correct
- [ ] Submission description ready

Good luck! 🚀
