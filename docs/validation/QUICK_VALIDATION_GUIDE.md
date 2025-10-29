# 🚀 Quick Validation Guide

## 🎯 The Problem

**Validation doesn't predict test!**

```
Val 0.1607 → Test 1.0020  ❌ No correlation
Val 0.1625 → Test 1.1398  ❌ Negative!
Untrained  → Test 1.0015  ✅ BEST!
```

**Why:** Subject leakage (same subject in train + val)

---

## ✅ The Solution

**Subject-Aware Validation**

```python
# ❌ WRONG (current)
train, val = train_test_split(data, test_size=0.1)

# ✅ RIGHT (subject-aware)
subjects = get_unique_subjects(data)
train_subj, val_subj = train_test_split(subjects, test_size=0.1)
train = data[data.subject.isin(train_subj)]
val = data[data.subject.isin(val_subj)]
```

---

## 📋 Quick TODO

```markdown
- [ ] 1. Upload submission_all_rsets_v1.zip (NOW)
- [ ] 2. Wait 10-15 min for results
- [ ] 3. Check C1 score vs 1.0015
- [ ] 4. If C1 > 1.0: implement subject-aware
```

---

## 🎯 Decision Tree

```
C1 < 0.95   → ✅ Works! Optimize hyperparams
C1 = 0.95-1 → ⚠️  Try subject-aware
C1 = 1.0-1.1 → ❌ MUST use subject-aware
C1 > 1.1    → 🚨 Revert to quick_fix
```

---

## 📁 Key Files

**Docs:**
- `VALIDATION_PROBLEM_ANALYSIS.md` - Full analysis
- `VALIDATION_ACTION_PLAN.md` - Step-by-step plan
- `TODO_VALIDATION_IMPROVEMENT.md` - Detailed checklist

**Scripts:**
- `scripts/preprocessing/cache_challenge1_with_subjects.py` - Re-cache with subjects
- `scripts/experiments/train_c1_subject_aware.py` - TO CREATE

**Submission:**
- `submission_all_rsets_v1.zip` - Ready to upload (957 KB)

---

## ⚡ Quick Commands

```bash
# Upload submission
# → https://www.codabench.org/competitions/2948/

# If need subject-aware:
tmux new -s cache_subjects
python scripts/preprocessing/cache_challenge1_with_subjects.py
# Wait 1-2 hours

# Create and train
python scripts/experiments/train_c1_subject_aware.py
# Wait 1 hour

# Submit
python create_submission_subject_aware.py
# Upload to Codabench
```

---

## 🎓 Key Insight

**Validation doesn't have to look good - it has to PREDICT TEST!**

- Val 0.16 → Test 1.14 = USELESS ❌
- Val 1.05 → Test 1.00 = USEFUL ✅

---

## 🚨 NEXT ACTION

**RIGHT NOW:**

1. Upload `submission_all_rsets_v1.zip`
2. Wait for results
3. Decide based on C1 score

**⚠️ DON'T re-cache until we see results!**

---

*Goal: C1 < 0.93 (top 3)*  
*Deadline: Nov 2, 2025 (4 days)*  
*Safety net: quick_fix (1.0065)*
