# Phase 2 Implementation Progress

**Date:** October 16, 2025, 18:10 PM
**Status:** P300 Feature Extraction IN PROGRESS 🟡

---

## Phase 2 Strategy: Feature Engineering + Ensemble

### Goal
Improve Challenge 1 score from **1.00 → 0.75-0.85** to reach **Overall < 0.60** (Top 3!)

### Current Phase 1 Baseline
```
Challenge 1:  1.0030 (borderline, needs improvement)
Challenge 2:  0.2970 (excellent!)
Overall:      0.6500 (Top 5-10 projected)
```

---

## Implementation Todo List

```markdown
### Tonight (Feature Extraction)
- [x] Create P300 feature extractor (`scripts/features/erp.py`) - TESTED ✅
- [x] Test extractor on dummy data - WORKING ✅
- [x] Create extraction script (`scripts/extract_p300_features.py`)
- [🔄] Extract P300 features from R1, R2, R3 - IN PROGRESS
  - [🔄] R1 extraction - LOADING DATA
  - [ ] R2 extraction
  - [ ] R3 extraction
- [ ] Verify feature quality and statistics
- [ ] Cache features to disk

### Tomorrow Morning (Phase 2 Training)
- [ ] Create `train_challenge1_phase2_p300.py` script
- [ ] Modify CNN architecture to accept raw + P300 features
- [ ] Train Phase 2 model (30 epochs, ~1.5 hours)
- [ ] Validate on R3 (target: < 0.85 NRMSE)

### Tomorrow Afternoon (Ensemble & Submission)
- [ ] Create ensemble (Phase 1 + Phase 2)
- [ ] Test ensemble on R3 (target: < 0.80 NRMSE)
- [ ] Calculate overall validation score
- [ ] Decision: Submit if validation < 0.60, else keep Phase 1
- [ ] Create `submission_phase2.zip` (if better)
- [ ] Upload to Codabench
```

---

## P300 Feature Extraction Details

### What We're Extracting
P300 event-related potentials correlate directly with response time:
- `p300_peak_latency` (300-600ms window) - PRIMARY FEATURE
- `p300_peak_amplitude`
- `p300_mean_amplitude`
- `p300_area_under_curve`
- `p300_onset_latency`
- `p300_rise_time`

### Expected Correlation
```
P300 Latency ↔ Response Time: r > 0.1 (expected)
→ If correlation is strong, features will improve prediction!
→ If correlation is weak, may not help (will verify after extraction)
```

### Preprocessing (Same as Training)
```python
# Same preprocessing as train_challenge1_multi_release.py
1. Filter corrupted datasets
2. Pick EEG channels only
3. Clip outliers (0.5-99.5 percentile)
4. Bandpass filter (1-40 Hz)
5. Create windows from events (2-second windows)
6. Extract P300 from each trial window
```

### Extraction Status

**Current Step:** Loading R1 data (~200 datasets × multiple trials)

**Estimated Time:**
- R1: ~10-15 minutes
- R2: ~10-15 minutes  
- R3: ~10-15 minutes
**Total: 30-45 minutes**

**Monitor Command:**
```bash
./monitor_p300_extraction.sh
# Or watch the log:
tail -f logs/p300_extraction.log
```

**Cache Location:**
```
data/processed/p300_cache/
├── R1_p300_features.pkl
├── R2_p300_features.pkl
└── R3_p300_features.pkl
```

---

## Phase 2 Architecture Plan

### Current Phase 1 Model
```python
CompactResponseTimeCNN (200K params)
├── Conv1D layers (temporal features)
├── Spatial attention (channel selection)
└── FC layers → response time prediction
```

### Phase 2 Augmented Model
```python
AugmentedResponseTimeCNN (220K params)
├── Raw CNN branch: CompactResponseTimeCNN (200K params)
│   └── Output: 512-dim embedding
├── P300 Feature branch: Linear (6 → 64)
│   └── Input: 6 P300 features
│   └── Output: 64-dim embedding
├── Fusion Layer: Concat [512 + 64] → 576-dim
└── Output Head: Linear (576 → 256 → 1)
    └── Response time prediction
```

**Why This Works:**
1. Raw CNN learns complex temporal patterns
2. P300 features provide direct RT correlation
3. Fusion layer learns optimal combination
4. Minimal parameter increase (20K new params)

---

## Ensemble Strategy

### Simple Weighted Ensemble
```python
prediction = 0.6 * phase1_pred + 0.4 * phase2_pred

# If validation shows phase2 is better:
prediction = 0.4 * phase1_pred + 0.6 * phase2_pred

# Or train a meta-learner (time permitting)
```

### Decision Criteria
```
IF ensemble_val_score < 0.60:
    → SUBMIT Phase 2 (should reach Top 3!)
ELSE IF ensemble_val_score 0.60-0.65:
    → CONSIDER submitting (marginal improvement)
ELSE:
    → KEEP Phase 1 (don't risk making it worse)
```

---

## Risk Management

✅ **Safety Net:** Phase 1 submission (0.65) is already excellent
- Saved in `~/Downloads/submission.zip`
- Can always fall back if Phase 2 doesn't improve

⚠️ **Time Constraints:**
- Feature extraction: Tonight (in progress)
- Training: Tomorrow morning (~1.5 hours)
- Testing & ensemble: Tomorrow afternoon
- Submission deadline: Tomorrow evening

🎯 **Success Metrics:**
```
Minimum Success:   C1 < 0.90  (better than Phase 1)
Target Success:    C1 < 0.85  (significant improvement)
Stretch Goal:      C1 < 0.80  (excellent!)

Overall Target:    < 0.60     (Top 3!)
```

---

## Next Steps (After Extraction Completes)

1. **Verify feature quality:**
   - Check P300 latency statistics (should be 300-600ms)
   - Verify RT correlation (should be r > 0.1)
   - Inspect feature distributions

2. **Create Phase 2 training script:**
   - Copy `train_challenge1_multi_release.py`
   - Modify to load cached P300 features
   - Update dataset class to return (raw_eeg, p300_features, target)
   - Update model to AugmentedResponseTimeCNN

3. **Train overnight (optional):**
   - Start training tonight if time permits
   - Or wait until tomorrow morning

4. **Test and ensemble tomorrow:**
   - Validate Phase 2 model
   - Create ensemble
   - Make final submission decision

---

## Monitoring

**Check extraction progress:**
```bash
./monitor_p300_extraction.sh
```

**Check if process is still running:**
```bash
ps aux | grep extract_p300_features
```

**View live log:**
```bash
tail -f logs/p300_extraction.log
```

**Expected final output:**
```
📊 EXTRACTION SUMMARY
======================================================================
R1: X,XXX trials
R2: X,XXX trials
R3: X,XXX trials
Total: XX,XXX trials

📈 P300 STATISTICS
======================================================================
P300 Latency:  XXX.X ± XX.X ms
P300 Amplitude: X.XX ± X.XX μV
Response Time:  XXX.X ± XX.X ms

💡 Correlation (P300 latency ↔ RT): 0.XXX
   ✅ Good! P300 features should help prediction

✅ FEATURE EXTRACTION COMPLETE!
```

---

**Last Updated:** 2025-10-16 18:10 PM  
**Status:** Extraction in progress, R1 loading data...  
**ETA:** ~30-45 minutes for complete extraction
