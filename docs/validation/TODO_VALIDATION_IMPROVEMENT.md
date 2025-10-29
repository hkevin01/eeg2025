# 📋 Validation Improvement - TODO List

**Date:** October 28, 2025 4:30 PM  
**Goal:** Get validation that predicts test performance  
**Deadline:** November 2, 2025 (4 days remaining)  
**Priority:** ⭐⭐⭐ (Critical for competition success)

---

## 🎯 Immediate Actions (NEXT 15 MINUTES)

### Phase 1: Test Current Submission

```markdown
- [ ] **Step 1.1:** Upload submission_all_rsets_v1.zip to Codabench
      URL: https://www.codabench.org/competitions/2948/
      File: submission_all_rsets_v1.zip (957 KB)
      Description: "ALL R-sets training, val NRMSE 0.9954"
      Time: 2 minutes

- [ ] **Step 1.2:** Wait for evaluation results
      Expected time: 10-15 minutes
      Download: scoring_result.zip when complete
      Extract and check C1 score

- [ ] **Step 1.3:** Compare to baseline
      Baseline: quick_fix C1 = 1.0015
      Target: C1 < 0.93 (top 3)
      Decide: Next steps based on result
```

**⏱️ Total time:** 15-20 minutes  
**🎯 Output:** Test C1 score to guide next steps

---

## 🔬 Subject-Aware Validation (IF NEEDED - 3-4 HOURS)

### Phase 2A: Re-Cache Data with Subject IDs

```markdown
- [ ] **Step 2A.1:** Run caching script for R1
      Command: `tmux new -s cache_subjects`
      Script: `python scripts/preprocessing/cache_challenge1_with_subjects.py`
      Expected: data/cached/challenge1_R1_windows_with_subjects.h5 (~900 MB)
      Time: 15-20 minutes

- [ ] **Step 2A.2:** Wait for R2, R3, R4 to complete
      R2: ~20 minutes (~950 MB)
      R3: ~25 minutes (~1.2 GB)
      R4: ~40 minutes (~2.1 GB)
      Total: ~1.5-2 hours
      Monitor: `tmux attach -t cache_subjects`

- [ ] **Step 2A.3:** Verify subject IDs saved correctly
      Command: `python -c "
      import h5py
      for r in ['R1', 'R2', 'R3', 'R4']:
          f = h5py.File(f'data/cached/challenge1_{r}_windows_with_subjects.h5', 'r')
          print(f'{r}: {f['subject_ids'].shape}, {len(set(f['subject_ids'][:]))} subjects')
          f.close()
      "`
      Expected: Subjects properly extracted for all releases
      Time: 1 minute
```

**⏱️ Total time:** 1.5-2 hours  
**🎯 Output:** Cached data WITH subject IDs

---

### Phase 2B: Create Subject-Aware Training Script

```markdown
- [ ] **Step 2B.1:** Create train_c1_subject_aware.py
      Location: scripts/experiments/train_c1_subject_aware.py
      Base on: scripts/experiments/train_c1_all_rsets.py
      Changes:
        - Load subject_ids from HDF5
        - Split by subjects, not samples
        - Verify no subject overlap
      Time: 30 minutes

- [ ] **Step 2B.2:** Test script with mini data
      Command: `python scripts/experiments/train_c1_subject_aware.py --test`
      Verify: Subject split works correctly
      Check: No overlap between train/val subjects
      Time: 5 minutes
```

**⏱️ Total time:** 35 minutes  
**🎯 Output:** Working subject-aware training script

---

### Phase 2C: Train Subject-Aware Model

```markdown
- [ ] **Step 2C.1:** Start training in tmux
      Command: `tmux new -s subject_aware_train`
      Script: `python scripts/experiments/train_c1_subject_aware.py`
      Expected: ~1 hour training time
      Detach: Ctrl+B, then D

- [ ] **Step 2C.2:** Monitor progress
      Check: Every 15 minutes
      Command: `grep -E "Epoch|Best Validation" logs/train_subject_aware_*.log | tail -20`
      Watch: Validation NRMSE (expect 1.0-1.15)

- [ ] **Step 2C.3:** Wait for completion
      Signal: Early stopping triggered
      Output: weights/compact_cnn_subject_aware_state.pt
      Log: checkpoints/c1_subject_aware/training_log_*.json
      Time: ~1 hour
```

**⏱️ Total time:** 1-1.5 hours  
**🎯 Output:** Trained model with subject-aware validation

---

### Phase 2D: Create and Submit

```markdown
- [ ] **Step 2D.1:** Create submission package
      Script: python create_submission_subject_aware.py
      Output: submission_subject_aware_v1.zip
      Contents:
        - submission.py
        - compact_cnn_c1_cross_r123_val4_state.pt (NEW weights)
        - weights_challenge_2.pt (SAME C2 weights)
      Time: 2 minutes

- [ ] **Step 2D.2:** Verify package
      Command: `unzip -l submission_subject_aware_v1.zip`
      Check: All 3 files present, sizes correct
      Time: 1 minute

- [ ] **Step 2D.3:** Upload to Codabench
      URL: https://www.codabench.org/competitions/2948/
      File: submission_subject_aware_v1.zip
      Description: "Subject-aware validation, val NRMSE [actual]"
      Time: 2 minutes

- [ ] **Step 2D.4:** Wait for results
      Time: 10-15 minutes
      Download: scoring_result.zip
      Compare: C1 score vs previous submissions
```

**⏱️ Total time:** 15-20 minutes  
**🎯 Output:** New submission with subject-aware training

---

## 📊 Results Analysis

### Phase 3: Compare Approaches

```markdown
- [ ] **Step 3.1:** Extract test scores
      Command: `unzip scoring_result.zip && cat results.txt`
      Record:
        - all_rsets_v1: C1 = ???
        - subject_aware_v1: C1 = ???
      Time: 2 minutes

- [ ] **Step 3.2:** Compare validation correlation
      Question: Does subject-aware val predict test better?
      Check:
        - Random split: val 0.9954 → test ???
        - Subject-aware: val ??? → test ???
      Conclusion: Which validates better?
      Time: 5 minutes

- [ ] **Step 3.3:** Document findings
      Create: VALIDATION_RESULTS_OCT28.md
      Include:
        - All test scores
        - Val vs test correlation
        - Which approach to use going forward
        - Next optimization steps
      Time: 10 minutes

- [ ] **Step 3.4:** Update status documents
      Files to update:
        - STATUS_SUMMARY_OCT28_3PM.md
        - LEADERBOARD_ANALYSIS_OCT28.md
      Add: Latest test results and decisions
      Time: 5 minutes
```

**⏱️ Total time:** 20-25 minutes  
**🎯 Output:** Clear understanding of best validation strategy

---

## 🎯 Success Criteria

### Must Achieve:

1. **Understand validation correlation** ⭐⭐⭐
   - Know if val 0.9954 predicts test
   - Know if subject-aware improves correlation
   - Can trust validation going forward

2. **Improve C1 score** ⭐⭐⭐
   - Beat baseline C1 = 1.0015
   - Target: C1 < 0.93 (top 3)
   - At least: C1 < 1.0 (some improvement)

3. **Have working validation** ⭐⭐
   - Can iterate faster with confidence
   - Don't need to submit every change
   - Know which changes help before testing

### Nice to Have:

1. **Subject-aware training working**
   - Validated approach for future
   - Better generalization
   - More predictive validation

2. **Documentation complete**
   - Learnings captured
   - Process documented
   - Can reproduce results

---

## ⏱️ Timeline Summary

| Phase | Tasks | Time | When |
|-------|-------|------|------|
| 1 | Upload & wait | 15-20 min | **NOW** |
| 2A | Re-cache data | 1.5-2 hrs | If needed |
| 2B | Create script | 35 min | If needed |
| 2C | Train model | 1-1.5 hrs | If needed |
| 2D | Submit | 15-20 min | If needed |
| 3 | Analyze | 20-25 min | After results |
| **Total** | **Full process** | **4-5 hrs** | **Today** |

---

## 🚨 Decision Points

### After Phase 1 (all_rsets_v1 results):

**If C1 < 0.95:**
- ✅ **STOP** - Current approach works!
- Skip Phase 2 entirely
- Go to Phase 3 analysis
- Focus on hyperparameter tuning

**If C1 = 0.95-1.0:**
- ⚠️ **CONSIDER** - Worth trying subject-aware
- Could push to top 3
- Proceed to Phase 2
- Time investment justified

**If C1 = 1.0-1.1:**
- ❌ **MUST DO** - Need subject-aware validation
- Current val doesn't predict test
- Proceed to Phase 2
- Critical for further progress

**If C1 > 1.1:**
- 🚨 **ABORT** - Training makes things worse
- Revert to quick_fix (1.0065)
- Don't do Phase 2
- Try completely different approach

---

### After Phase 2 (subject_aware_v1 results):

**If subject-aware C1 < random C1:**
- ✅ Subject-aware WORKS!
- Use for all future training
- Can trust validation now

**If subject-aware C1 ≈ random C1:**
- ⚠️ No clear winner
- Both approaches viable
- Choose based on val correlation

**If subject-aware C1 > random C1:**
- ❌ Subject-aware WORSE
- Stick with random split
- Find other optimization strategies

---

## 📁 Files to Track

### Created:
- ✅ VALIDATION_PROBLEM_ANALYSIS.md (comprehensive analysis)
- ✅ VALIDATION_ACTION_PLAN.md (detailed plan)
- ✅ VALIDATION_STRATEGY_SUMMARY.md (quick reference)
- ✅ TODO_VALIDATION_IMPROVEMENT.md (this file)
- ✅ scripts/preprocessing/cache_challenge1_with_subjects.py

### To Create:
- ⏳ scripts/experiments/train_c1_subject_aware.py
- ⏳ create_submission_subject_aware.py
- ⏳ VALIDATION_RESULTS_OCT28.md

### To Update:
- ⏳ STATUS_SUMMARY_OCT28_3PM.md
- ⏳ LEADERBOARD_ANALYSIS_OCT28.md

---

## 🎓 Key Reminders

1. **Upload current submission FIRST** ⭐⭐⭐
   - Don't start re-caching until we know results
   - Might not need subject-aware at all
   - Save 3-4 hours if current works

2. **Keep quick_fix as safety net** ⭐⭐⭐
   - C1 = 1.0015, Overall = 1.0065
   - Already submitted and working
   - Always have backup

3. **Focus on test score, not validation** ⭐⭐
   - Validation might look worse but predict better
   - Only test score matters for competition
   - Trust test, not val

4. **Watch the clock** ⭐⭐
   - 4 days until deadline
   - Each experiment takes time
   - Prioritize high-impact changes

---

## 🚀 NEXT IMMEDIATE ACTION

**RIGHT NOW (in order):**

1. ✅ Read this TODO list
2. ⏳ Go to https://www.codabench.org/competitions/2948/
3. ⏳ Upload submission_all_rsets_v1.zip
4. ⏳ Wait 10-15 minutes
5. ⏳ Download and check C1 score
6. ⏳ Make decision: Continue or implement subject-aware

**DO NOT start Phase 2 until Phase 1 is complete!**

---

*Created: October 28, 2025 4:30 PM*  
*Status: Ready to upload all_rsets_v1*  
*Next: Upload and wait for results*  
*Goal: Top 3 (C1 < 0.93)*
