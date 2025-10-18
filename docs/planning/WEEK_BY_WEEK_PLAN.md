# 📅 Week-by-Week Action Plan

**Start Date**: October 14, 2025  
**Competition Deadline**: ~4 weeks

---

## 🔴 Week 1: Foundation (Oct 14-20)

### Priority 1: Data Setup
- [ ] Day 1: Install dependencies, create directories
- [ ] Day 2: Download 2 sample subjects, verify structure  
- [ ] Day 3: Test data loading pipeline
- [ ] Day 4: Validate preprocessing works

**Success Metric**: Can load and preprocess 2 subjects without errors

### Priority 2: Basic Testing
- [ ] Day 5: Write test_data_loading.py
- [ ] Day 6: Write test_model_forward.py
- [ ] Day 7: Run all tests, fix issues

**Success Metric**: 5+ tests passing, no critical failures

**Time Budget**: 2-3 hours/day
**Deliverable**: Working data pipeline + tests

---

## 🟡 Week 2: Model Training (Oct 21-27)

### Priority 1: Baseline Models
- [ ] Day 1-2: Train Challenge 1 baseline (cross-task)
- [ ] Day 3-4: Train Challenge 2 baseline (psychopathology)  
- [ ] Day 5: Evaluate both on validation set

**Success Metric**: 
- Challenge 1: Combined score >0.40
- Challenge 2: Avg correlation >0.20

### Priority 2: Inference Optimization
- [ ] Day 6: Measure inference latency
- [ ] Day 7: Optimize if >50ms (quantization/pruning)

**Success Metric**: <50ms inference time

**Time Budget**: 3-4 hours/day
**Deliverable**: 2 trained baseline models

---

## 🟢 Week 3: Optimization (Oct 28 - Nov 3)

### Priority 1: Improve Performance
- [ ] Day 1-2: Add artifact detection
- [ ] Day 3-4: Hyperparameter tuning (Optuna)
- [ ] Day 5: Cross-site validation

**Success Metric**: 
- Challenge 1: Score >0.50
- Challenge 2: Correlation >0.25

### Priority 2: Ensemble Preparation
- [ ] Day 6: Train 3-5 models with different seeds
- [ ] Day 7: Test ensemble averaging

**Time Budget**: 4-5 hours/day
**Deliverable**: Improved models + ensemble strategy

---

## 🔵 Week 4: Final Push (Nov 4-10)

### Priority 1: Final Models
- [ ] Day 1-3: Train final ensemble (5-10 models)
- [ ] Day 4: Generate submission files
- [ ] Day 5: Validate submission format

**Success Metric**: Submission files pass all checks

### Priority 2: Submit & Iterate
- [ ] Day 6: Submit to competition
- [ ] Day 7: Analyze feedback, iterate if needed

**Time Budget**: Full focus
**Deliverable**: Competition submission

---

## ⏱️ Daily Schedule Template

### Morning (2 hours)
1. Check overnight training runs (30 min)
2. Analyze results, plan improvements (30 min)
3. Code implementation (60 min)

### Evening (1-2 hours)
1. Start new training run (15 min)
2. Write/run tests (45 min)
3. Document progress (30 min)

---

## 🎯 Key Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| Oct 15 | Data pipeline working | ⭕ |
| Oct 18 | First model trained | ⭕ |
| Oct 22 | Baseline metrics measured | ⭕ |
| Oct 29 | Optimized models ready | ⭕ |
| Nov 5 | Final ensemble complete | ⭕ |
| Nov 8 | Submission uploaded | ⭕ |

---

## 🚨 Risk Mitigation

### If Behind Schedule:
- **Week 1**: Skip full dataset, use 10 subjects only
- **Week 2**: Use default hyperparameters, no tuning
- **Week 3**: Skip ensemble, submit single best model
- **Week 4**: Submit baseline early, iterate if time

### If Ahead of Schedule:
- Experiment with advanced architectures (EEGNet, etc.)
- Add explainability/visualization
- Write detailed documentation
- Create submission video/presentation

---

## 📊 Progress Tracking

Update this daily:

```bash
# Quick status check
echo "Week $(date +%U) Status:" >> progress.log
echo "- Tasks completed: X/Y" >> progress.log
echo "- Model performance: Challenge1=X.XX, Challenge2=X.XX" >> progress.log
echo "- Blockers: [description]" >> progress.log
echo "---" >> progress.log
```

---

**Remember**: Progress > Perfection. Ship working code first, optimize later!
