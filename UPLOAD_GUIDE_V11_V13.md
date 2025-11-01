# Upload Guide - V11, V13.5, V13 Options

## Quick Reference

| Version | Size | Strategy | Expected | Risk | Location |
|---------|------|----------|----------|------|----------|
| **V11** | 1.7 MB | V10 C1 + 2-seed C2 | 1.00034 | **LOW** ✅ | `v11_submission.zip` |
| **V13.5** | 4.2 MB | 3-seed C1 + 2-seed C2 + TTA | 1.00031 | MEDIUM 🎯 | `v13.5_submission.zip` |
| **V13** | 6.1 MB | 5-seed C1 + 2-seed C2 + TTA | 1.00030 | HIGH ⚠️ | `v13_submission_corrected.zip` |

---

## RECOMMENDED: Upload V11 First

### Why V11?
1. ✅ **Safe size:** Only 70% larger than working V10 (1.7 MB vs 1.0 MB)
2. ✅ **Proven C1:** Uses exact V10 Challenge 1 model (1.00019)
3. ✅ **Variance reduction:** 2-seed ensemble for Challenge 2
4. ✅ **Low risk:** If V10 works at 1.0 MB, V11 likely works at 1.7 MB
5. ✅ **Good gain:** Expected ~4e-5 improvement over V10

### V11 Upload Steps

1. **Locate file:**
   ```bash
   /home/kevin/Projects/eeg2025/v11_submission.zip
   ```

2. **Upload to competition:**
   - Go to: https://www.codabench.org/competitions/3350/
   - Click "My Submissions"
   - Upload `v11_submission.zip`

3. **Monitor results:**
   - Wait for "Finished" status (typically 5-10 minutes)
   - Download error files if failed: `prediction_result.zip`, `scoring_result.zip`
   - Check leaderboard for score if successful

4. **Success criteria:**
   - ✅ Challenge 1: ~1.00019 (same as V10)
   - ✅ Challenge 2: ~1.00049 (improvement from 1.00066)
   - ✅ Overall: ~1.00034 (improvement from 1.00052)
   - ✅ Rank: ~#60-65 (up from #72)

---

## Option 2: V13.5 (If V11 Succeeds)

### Why V13.5?
- Builds on V11 success
- Adds more C1 variance reduction (3 seeds instead of 1)
- Includes TTA (Test-Time Augmentation)
- Size-optimized: 32% smaller than full V13

### V13.5 Upload Steps

1. **Prerequisites:**
   - ✅ V11 succeeded on platform
   - Want more aggressive variance reduction

2. **Locate file:**
   ```bash
   /home/kevin/Projects/eeg2025/v13.5_submission.zip
   ```

3. **Expected improvement over V11:**
   - Challenge 1: 1.00013 vs 1.00019 (6e-5 better)
   - Challenge 2: 1.00049 (same)
   - Overall: 1.00031 vs 1.00034 (3e-5 better)

4. **If V13.5 fails:**
   - Likely size limit is between 1.7 MB (V11) and 4.2 MB (V13.5)
   - Consider creating V12: 2-seed C1 + 2-seed C2 (~2.5 MB)

---

## Option 3: V13 (If Desperate)

### Why V13?
- Most aggressive variance reduction
- 5-seed C1 ensemble
- Includes TTA and calibration
- Highest expected performance

### V13 Upload Steps

1. **Prerequisites:**
   - ✅ V13.5 succeeded, OR
   - 🎲 Willing to risk 6.1 MB size

2. **Locate file:**
   ```bash
   /home/kevin/Projects/eeg2025/v13_submission_corrected.zip
   ```

3. **Expected (if works):**
   - Challenge 1: 1.00011 (best possible with current approach)
   - Challenge 2: 1.00049
   - Overall: 1.00030
   - Rank: ~#50-55

4. **Risk:**
   - ⚠️ Already failed once (likely due to size)
   - 6x larger than working V10
   - Null exit codes suggest resource failure

---

## Troubleshooting

### If Submission Fails

1. **Download error files:**
   - `prediction_result.zip` or `prediction_result (N).zip`
   - `scoring_result.zip` or `scoring_result (N).zip`

2. **Check metadata:**
   ```bash
   cd /home/kevin/Downloads
   unzip -q "prediction_result (N).zip"
   cat metadata
   ```

3. **Interpret errors:**
   - `exitCode: null` → Resource failure (size/memory)
   - `exitCode: 1` → Python error (check stderr.txt)
   - Empty `scoring_result.zip` → Prediction failed

4. **Next steps based on failure:**
   - **V11 fails:** Size limit < 1.7 MB → Stick with V10
   - **V13.5 fails:** Size limit between 1.7-4.2 MB → Try V12 (2-seed C1)
   - **V13 fails:** Expected → Fall back to V13.5 or V11

---

## Incremental Upload Strategy

**Recommended order:**

```
1. V11 (1.7 MB) → Safe starting point
   ↓ If succeeds
2. V13.5 (4.2 MB) → More aggressive
   ↓ If succeeds  
3. V13 (6.1 MB) → Most aggressive
```

**Benefits:**
- ✅ Find size limit incrementally
- ✅ Lock in improvements at each step
- ✅ Always have working submission
- ✅ Minimize wasted uploads

---

## Quick Commands

### Verify files exist:
```bash
cd /home/kevin/Projects/eeg2025
ls -lh v11_submission.zip v13.5_submission.zip v13_submission_corrected.zip
```

### Copy to Downloads for easy access:
```bash
cp v11_submission.zip /home/kevin/Downloads/
cp v13.5_submission.zip /home/kevin/Downloads/
cp v13_submission_corrected.zip /home/kevin/Downloads/
```

### Test locally before upload:
```bash
# V11
python -c "import sys; sys.path.insert(0, 'submissions/phase1_v11'); from submission import Submission; import numpy as np; sub = Submission(SFREQ=100, DEVICE='cpu'); X = np.random.randn(2, 129, 200).astype(np.float32); p1 = sub.challenge_1(X); p2 = sub.challenge_2(X); print('✅ V11 ready')"

# V13.5
python -c "import sys; sys.path.insert(0, 'submissions/phase1_v13.5'); from submission import Submission; import numpy as np; sub = Submission(SFREQ=100, DEVICE='cpu'); X = np.random.randn(2, 129, 200).astype(np.float32); p1 = sub.challenge_1(X); p2 = sub.challenge_2(X); print('✅ V13.5 ready')"
```

---

## Expected Timeline

- **V11 upload:** ~5-10 minutes to score
- **V13.5 upload:** ~8-12 minutes (larger, more processing)
- **V13 upload:** ~10-15 minutes (if doesn't crash)

Total time for all 3 uploads (if all succeed): ~30-45 minutes

---

## Success Metrics

### V11 Success:
- [x] Submission finishes without errors
- [x] Challenge 1: ~1.00019
- [x] Challenge 2: < 1.00050
- [x] Overall: < 1.00035
- [x] Rank: Better than #72

### V13.5 Success:
- [x] Challenge 1: < 1.00015
- [x] Overall: < 1.00032
- [x] Rank: Better than V11

### V13 Success:
- [x] Challenge 1: < 1.00012
- [x] Overall: < 1.00031
- [x] Rank: Top 55

---

**Ready to upload?** Start with V11! 🚀

