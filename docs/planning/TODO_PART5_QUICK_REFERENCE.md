# Competition TODO List - Part 5: Quick Reference

---

## 🔗 IMPORTANT LINKS

- **Competition**: https://eeg2025.github.io/
- **Codabench**: https://www.codabench.org/competitions/4287/
- **Starter Kit**: https://github.com/eeg2025/startkit
- **Rules**: https://eeg2025.github.io/rules/
- **Submission**: https://eeg2025.github.io/submission/
- **Leaderboard**: https://eeg2025.github.io/leaderboard/

---

## 📁 KEY FILES

### Training Scripts
- `scripts/train_challenge1_response_time.py` - Challenge 1 training
- `scripts/train_challenge2_externalizing.py` - Challenge 2 training

### Models
- `checkpoints/response_time_model.pth` - Challenge 1 checkpoint
- `checkpoints/externalizing_model.pth` - Challenge 2 checkpoint
- `weights_challenge_1.pt` - Challenge 1 submission weights
- `weights_challenge_2.pt` - Challenge 2 submission weights

### Submission
- `submission.py` - Main submission file
- `submission_complete.zip` - Ready to upload

### Documentation
- `LEADERBOARD_STRATEGY.md` - Submission strategy
- `COMPETITION_STATUS.md` - Full status overview
- `CHALLENGE1_PLAN.md` - Challenge 1 execution plan
- `TODO_PART*.md` - This todo list (5 parts)

---

## 🚀 QUICK COMMANDS

### Test Submission
```bash
cd /home/kevin/Projects/eeg2025
python3 scripts/test_submission_quick.py
```

### Check Model Performance
```bash
# Challenge 1
python3 -c "import torch; cp=torch.load('checkpoints/response_time_model.pth', map_location='cpu'); print(f'C1 NRMSE: {cp[\"nrmse\"]:.4f}')"

# Challenge 2
python3 -c "import torch; cp=torch.load('checkpoints/externalizing_model.pth', map_location='cpu'); print(f'C2 NRMSE: {cp[\"nrmse\"]:.4f}')"
```

### Rebuild Submission Package
```bash
cd /home/kevin/Projects/eeg2025
rm -rf submission_package submission_complete.zip
mkdir -p submission_package
cp submission.py weights_challenge_*.pt submission_package/
cd submission_package && zip -r ../submission_complete.zip . && cd ..
ls -lh submission_complete.zip
```

---

## 📊 PERFORMANCE CHECKLIST

Before submitting, verify:
- [x] Challenge 1 NRMSE < 0.5 (achieved: 0.4680)
- [x] Challenge 2 NRMSE < 0.5 (achieved: 0.0808)
- [x] Both models load successfully
- [x] Inference produces reasonable outputs
- [x] Package size < 20 MB (current: 1.8 MB)
- [x] Files at root level (no nested folders)
- [ ] Methods document written
- [ ] Final validation tests pass
- [ ] Code runs on fresh environment

---

## 🎓 KEY INSIGHTS

1. **Challenge 2 is dominant** (70% of score) - already excellent
2. **Challenge 1 matters** (30% of score) - now competitive
3. **Overall estimated NRMSE: 0.1970** - strong position
4. **Limited daily submissions** - test locally first
5. **Methods document required** - start writing now
6. **18 days remaining** - plenty of time for iteration

---

## ✅ SUCCESS CRITERIA

**Minimum Goal**: NRMSE < 0.5 on both challenges ✅ ACHIEVED!

**Stretch Goal**: Top 10 leaderboard - competitive range!

**Competition Goal**: Win/place - depends on other teams

Your current performance puts you in a strong position!
