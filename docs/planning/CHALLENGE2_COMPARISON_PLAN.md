# 🔬 Challenge 2: Model Comparison Strategy

**Date:** October 18, 2025, 00:05  
**Goal:** Compare CompactCNN vs TCN for Challenge 2

---

## 📊 Model Specifications

### Model 1: CompactExternalizingCNN (Baseline)
- **Parameters:** 64,001
- **Architecture:** 4-layer CNN with BatchNorm + ELU
- **Training:** Multi-release (R2+R3+R4)
- **Val NRMSE:** 0.2917 ✅ Proven performance
- **Checkpoint:** `weights_challenge_2_multi_release.pt`

### Model 2: TCN_EEG (Experimental)
- **Parameters:** 196,225  
- **Architecture:** 5-layer TCN with dilated convolutions
- **Training:** Multi-release (R1+R2+R3), Val on R4
- **Val Loss:** 0.667792 (NRMSE ~0.817) ⚠️ Worse than baseline
- **Checkpoint:** `checkpoints/challenge2_tcn_competition_best.pth`

---

## 🎯 Comparison Strategy

### Option 1: Conservative Approach (Recommended)
**Submission v6:** Use CompactCNN for both challenges
- Challenge 1: TCN (proven 65% improvement)
- Challenge 2: CompactCNN (proven NRMSE 0.2917)
- **Rationale:** Maximize known performance
- **Risk:** Low

### Option 2: Experimental Approach
**Submission v6:** Use CompactCNN, create v7 with TCN
- v6: CompactCNN for Challenge 2
- v7: TCN for Challenge 2 (compare leaderboard scores)
- **Rationale:** Test TCN on real leaderboard
- **Risk:** Medium (TCN might underperform)

### Option 3: Dual Submission
**Create both versions tonight:**
- v6a: TCN (C1) + CompactCNN (C2)
- v6b: TCN (C1) + TCN (C2)
- Upload both, compare results
- **Rationale:** Maximum information gathering
- **Risk:** More work, but best comparison

---

## ✅ Recommended Plan: Option 3 (Dual Submission)

### Submission v6a (Conservative - Primary)
```
Challenge 1: TCN (challenge1_tcn_competition_best.pth)
Challenge 2: CompactCNN (weights_challenge_2_multi_release.pt)
Expected: Strong performance on both challenges
```

### Submission v6b (Experimental - Test)
```
Challenge 1: TCN (challenge1_tcn_competition_best.pth)
Challenge 2: TCN (challenge2_tcn_competition_best.pth)
Expected: Strong C1, unknown C2 (test TCN performance)
```

---

## 📋 Implementation Steps

### Step 1: Create submission_v6a.py (CompactCNN)
- Keep current submission.py as-is
- Challenge 2: Use CompactExternalizingCNN
- Test with dummy data ✅ (already done)

### Step 2: Create submission_v6b.py (TCN)
- Copy submission.py
- Challenge 2: Replace with TCN_EEG
- Update model loading for Challenge 2 TCN
- Test with dummy data

### Step 3: Package Both Submissions
```bash
# v6a (Conservative)
mkdir -p submission_v6a
cp submission.py submission_v6a/
cp challenge1_tcn_competition_best.pth submission_v6a/
cp weights_challenge_2_multi_release.pt submission_v6a/
cd submission_v6a && zip -r ../eeg2025_submission_v6a.zip .
cd ..

# v6b (Experimental)
mkdir -p submission_v6b
cp submission_v6b.py submission_v6b/submission.py
cp challenge1_tcn_competition_best.pth submission_v6b/
cp checkpoints/challenge2_tcn_competition_best.pth submission_v6b/
cd submission_v6b && zip -r ../eeg2025_submission_v6b.zip .
cd ..
```

### Step 4: Upload Both to Codabench
- Upload v6a first (primary submission)
- Upload v6b second (experimental)
- Wait for validation (1-2 hours each)
- Compare leaderboard scores

### Step 5: Analyze Results
Compare on leaderboard:
- If TCN (v6b) > CompactCNN (v6a): Use TCN for future submissions
- If CompactCNN (v6a) > TCN (v6b): Keep CompactCNN, improve TCN training
- Document findings for future improvements

---

## ⏱️ Timeline

**Total Time:** 45 minutes + 2-4 hours validation

- **00:10-00:20** (10 min): Create submission_v6b.py with TCN
- **00:20-00:25** (5 min): Test submission_v6b.py
- **00:25-00:35** (10 min): Package both submissions
- **00:35-00:40** (5 min): Upload v6a to Codabench
- **00:40-00:45** (5 min): Upload v6b to Codabench
- **02:00-04:00** (wait): Check leaderboard results

---

## 🔮 Expected Outcomes

### Best Case
- v6a: NRMSE ~0.15 (C1: 0.10, C2: 0.29)
- v6b: NRMSE ~0.14 (C1: 0.10, C2: 0.25) ✨ TCN wins!

### Likely Case
- v6a: NRMSE ~0.15 (C1: 0.10, C2: 0.29) ✅ Solid
- v6b: NRMSE ~0.17 (C1: 0.10, C2: 0.35) ⚠️ TCN worse

### Worst Case
- v6a: NRMSE ~0.15 (C1: 0.10, C2: 0.29) ✅ Safe
- v6b: NRMSE ~0.20+ (C1: 0.10, C2: 0.50+) ❌ TCN much worse

---

## 📊 Decision Matrix

| Scenario | v6a Score | v6b Score | Action |
|----------|-----------|-----------|--------|
| TCN Wins | 0.15 | 0.14 | Use TCN for v7+ |
| Tie | 0.15 | 0.15 | Use CompactCNN (fewer params) |
| CompactCNN Wins | 0.15 | 0.17+ | Keep CompactCNN, retrain TCN |

---

## ✅ Current Status

- [x] Both models tested successfully
- [x] Comparison script created
- [ ] Create submission_v6b.py
- [ ] Test submission_v6b.py
- [ ] Package both submissions
- [ ] Upload v6a (Conservative)
- [ ] Upload v6b (Experimental)
- [ ] Wait for results
- [ ] Compare and document

---

## 💡 Key Insights

1. **TCN has 3x more parameters** but worse validation loss
2. **CompactCNN is proven** with Val NRMSE 0.2917
3. **Real test:** Leaderboard comparison will reveal true performance
4. **Low risk:** v6a ensures we have solid submission
5. **High reward:** v6b could discover TCN advantage

---

**Next Action:** Create submission_v6b.py with Challenge 2 TCN

