# P300 Feature Extraction Status

**Last Updated:** October 16, 2025, 19:00  
**Status:** 🔄 **RUNNING** (Restarted with fix)

---

## ⚠️ Issue Fixed

**Problem:** First extraction attempt crashed during window creation
- **Error:** `ValueError: Could not find any of the events you specified`
- **Cause:** Missing `add_aux_anchors` preprocessor (adds event markers)
- **Fix:** Added `Preprocessor(add_aux_anchors, apply_on_array=False)` to preprocessing pipeline

**Status:** ✅ **FIXED** and restarted at 18:55

---

## 🔄 Current Progress

**Process ID:** 1423835  
**Current Stage:** Preprocessing R1 data (filtering datasets)  
**Runtime:** ~5 minutes (restarted)  
**ETA:** 60-90 minutes total

### Preprocessing Pipeline (Now Complete)
```python
1. Pick EEG channels only                          ✅
2. Clip outliers (0.5-99.5 percentile)            ✅
3. Bandpass filter (1-40 Hz)                       🔄 IN PROGRESS
4. Add event markers (add_aux_anchors)             ✅ FIXED
5. Create windows from "contrast_trial_start"      ⏳ Next
6. Extract P300 features from each window          ⏳ Pending
```

---

## 📊 Expected Output

### Cache Files (when complete)
```
data/processed/p300_cache/
├── R1_p300_features.pkl  (~50-100 MB)
├── R2_p300_features.pkl  (~50-100 MB)
└── R3_p300_features.pkl  (~50-100 MB)
```

### Features Per Trial
```python
{
    'p300_peak_latency': float,      # 300-600ms window
    'p300_peak_amplitude': float,    # μV
    'p300_mean_amplitude': float,    # μV
    'p300_area_under_curve': float,  # μV·ms
    'p300_onset_latency': float,     # ms
    'p300_rise_time': float          # ms
}
```

### Summary Statistics (generated at end)
```
📊 EXTRACTION SUMMARY
R1: ~2,000-3,000 trials
R2: ~2,000-3,000 trials
R3: ~2,000-3,000 trials
Total: ~6,000-9,000 trials

📈 P300 STATISTICS
P300 Latency: XXX ± XX ms
P300 Amplitude: X.XX ± X.XX μV
Response Time: XXX ± XX ms

💡 Correlation (P300 latency ↔ RT): 0.XXX
```

---

## 📋 Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Data Download | 5 min | ✅ Complete |
| Preprocessing R1 | 15-20 min | 🔄 In Progress (~25%) |
| Create Windows R1 | 5 min | ⏳ Pending |
| Extract P300 R1 | 5-10 min | ⏳ Pending |
| Process R2 | 20-25 min | ⏳ Pending |
| Process R3 | 20-25 min | ⏳ Pending |
| **TOTAL** | **60-90 min** | **ETA: 20:30** |

---

## 🔍 Monitoring Commands

**Quick Status:**
```bash
./monitor_p300_extraction.sh
```

**Live Continuous (auto-refresh every 5s):**
```bash
./watch_p300.sh
```

**Raw Log:**
```bash
tail -f logs/p300_extraction.log
```

**Check Process:**
```bash
ps aux | pgrep -f extract_p300
```

---

## ✅ What Happens Next

### After Extraction Completes (~20:30)

1. **Verify Feature Quality:**
   - Check P300 latency distribution (should be 300-600ms)
   - Verify no NaN or inf values
   - Inspect feature statistics

2. **Check RT Correlation:**
   ```
   IF correlation > 0.1:
       ✅ Features are useful! Proceed with Phase 2
   ELSE:
       ⚠️ Weak correlation, may not improve results
   ```

3. **Decision Point:**
   - **Strong correlation:** Create Phase 2 training script tonight
   - **Weak correlation:** Reconsider Phase 2 strategy

### Tomorrow (If Correlation Good)

1. **Morning:** Train Phase 2 model with P300 features (~1.5 hours)
2. **Afternoon:** Create ensemble (Phase 1 + Phase 2)
3. **Evening:** Test and submit if validation < 0.60

---

## 🎯 Success Criteria

### Extraction Success
- ✅ All 3 releases processed (R1, R2, R3)
- ✅ No errors or crashes
- ✅ ~6,000-9,000 total trials extracted
- ✅ Cache files saved successfully

### Feature Quality
- ✅ P300 latency: 300-600ms range (physiologically valid)
- ✅ No NaN or infinite values
- ✅ Reasonable variance across trials

### RT Correlation
- 🎯 **Target:** r > 0.1 (useful for prediction)
- 🌟 **Excellent:** r > 0.3 (strong predictor)
- ⚠️ **Weak:** r < 0.05 (may not help)

---

## 🛡️ Backup Plan

If P300 features don't correlate with RT:
- **Option 1:** Try spectral features instead (band power)
- **Option 2:** Use attention mechanisms without features
- **Option 3:** Keep Phase 1 submission (0.65 is already good!)

**Safety Net:** Phase 1 submission saved in `~/Downloads/submission.zip`

---

## 📝 Files Created/Modified

**New Files:**
- ✅ `scripts/features/erp.py` (291 lines) - P300 extractor
- ✅ `scripts/extract_p300_features.py` (263 lines) - Extraction pipeline
- ✅ `monitor_p300_extraction.sh` - Quick status check
- ✅ `watch_p300.sh` - Live continuous monitor
- ✅ `PHASE2_PROGRESS.md` - Strategy document
- ✅ `P300_EXTRACTION_STATUS.md` - This file

**Modified:**
- ✅ `scripts/extract_p300_features.py` - Fixed event marker issue

---

**Process is running smoothly!** Check back in ~1 hour (around 20:30) to verify results.

Run `./watch_p300.sh` for live updates! 🚀
