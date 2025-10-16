# Competition TODO List - Part 2: Pending Tasks (UPDATED)

---

## ✅ COMPLETED

### Documentation
- [x] Write 2-page methods document (REQUIRED for submission) ✅
  - File: METHODS_DOCUMENT.md (ready for PDF conversion)

### Testing & Validation
- [x] Create comprehensive validation script ✅
- [x] Test with different random seeds ✅
- [x] Verify submission.zip structure ✅
- [x] Check memory usage < 20GB (54 MB used!) ✅
- [x] Time inference speed (C1: 3.9ms, C2: 2.1ms avg) ✅

### Model Improvements
- [x] Feature visualization / saliency maps ✅
  - Generated 6 visualization files
  - Identified top important EEG channels
  - Temporal attention patterns visualized

---

## 🟡 IN PROGRESS (Optional)

### Model Improvements (Scripts Created, Ready to Run)
- [ ] Cross-validation for Challenge 1 (5-fold) 📝 Script ready
- [ ] Ensemble multiple models (3 seeds) 📝 Script ready
- [ ] Test-time augmentation
- [ ] Hyperparameter tuning

### Data Improvements
- [ ] Download more CCD subjects (if available)
- [ ] Advanced preprocessing (ICA artifact removal)
- [ ] Explore other tasks for Challenge 2

### Code Quality
- [ ] Add unit tests
- [ ] Code documentation
- [ ] Type hints
- [ ] Linting cleanup

---

## 📝 SCRIPTS CREATED & READY

1. **scripts/cross_validate_challenge1.py** - 5-fold CV testing
2. **scripts/train_ensemble_challenge1.py** - Train 3 models with different seeds
3. **scripts/visualize_features.py** - Feature importance (DONE ✅)
4. **scripts/validate_comprehensive.py** - Full validation suite (DONE ✅)

---

## 🚫 NOT ALLOWED / OUT OF SCOPE

- Training during inference (prohibited by rules)
- Using test data for training (data leakage)
- Multiple GPU usage (limited to 1 GPU)
- External foundation models without documentation
- Unlimited leaderboard submissions (limited per day)

---

## 🎯 RECOMMENDATION

**Current Status**: Both models exceed targets, well-tested, and documented.

**Options**:
1. **Submit Now** - You're ready! Focus on leaderboard feedback
2. **Run CV** - If you want more robustness validation (~30 min)
3. **Train Ensemble** - If you want ensemble averaging (~1 hour)

**Best Strategy**: Submit baseline now, iterate based on leaderboard results.
