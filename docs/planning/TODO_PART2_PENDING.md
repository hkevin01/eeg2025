# Competition TODO List - Part 2: Pending Tasks

---

## 🟡 IN PROGRESS

### Documentation
- [x] Write 2-page methods document (REQUIRED for submission) ✅
  - [x] Introduction and motivation
  - [x] Data preprocessing details
  - [x] Model architectures
  - [x] Training procedures
  - [x] Results and discussion
  - **File**: METHODS_DOCUMENT.md (ready for PDF conversion)

### Testing & Validation
- [x] Create comprehensive validation script ✅
- [x] Test with different random seeds ✅
- [x] Verify submission.zip structure ✅
- [x] Check memory usage < 20GB (54 MB used!) ✅
- [x] Time inference speed (C1: 3.9ms, C2: 2.1ms avg) ✅

---

## ⭕ PENDING (Optional Improvements)

### Model Improvements
- [ ] Cross-validation for Challenge 1 (5-fold)
- [ ] Ensemble multiple models
- [ ] Feature visualization / saliency maps
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

## 🚫 NOT ALLOWED / OUT OF SCOPE

- Training during inference (prohibited by rules)
- Using test data for training (data leakage)
- Multiple GPU usage (limited to 1 GPU)
- External foundation models without documentation
- Unlimited leaderboard submissions (limited per day)
