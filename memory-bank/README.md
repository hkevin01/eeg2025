# EEG 2025 Challenge - Memory Bank

## Purpose

This memory bank enables **instant project recovery** from any interruption:
- Lost chat context
- VS Code crashes  
- System reboots
- Long breaks from project
- New team members joining

## 🚀 START HERE

**If you've lost context:** Read `QUICKSTART.md` first!

**If you want full understanding:** Read documents in this order:
1. `QUICKSTART.md` - Immediate status and actions
2. `app-description.md` - Project overview
3. `implementation-plans/submission-v6.md` - Current work plan
4. `change-log.md` - History of all changes
5. `architecture-decisions/tcn-choice.md` - Model design rationale

## 📁 Structure

```
memory-bank/
├── README.md                        # This file
├── QUICKSTART.md                    # Fast recovery guide
├── app-description.md               # Project overview
├── change-log.md                    # Complete change history
├── implementation-plans/
│   └── submission-v6.md             # Current implementation (ACID)
└── architecture-decisions/
    └── tcn-choice.md                # TCN architecture rationale
```

## 📄 Document Descriptions

### QUICKSTART.md
**Purpose:** Get back to work in < 5 minutes  
**Use When:** Lost context, need status update, don't know what to do next  
**Contains:**
- Current project state
- Immediate action items
- Common recovery scenarios
- Essential commands
- Troubleshooting

### app-description.md  
**Purpose:** Understand what we're building and why  
**Use When:** Need project context, explaining to others, planning changes  
**Contains:**
- Competition structure (Challenge 1 & 2)
- Dataset details (HBN, R1-R5 splits)
- Technical stack (PyTorch, eegdash, etc.)
- Model architecture (TCN)
- Project goals and success metrics

### change-log.md
**Purpose:** Track all modifications with dates and rationale  
**Use When:** Need to understand why something was done, debugging issues  
**Contains:**
- Chronological history (Oct 14-17, 2025)
- Bug fixes (window indexing, dtype, monitor script)
- Feature additions (TCN training, tmux setup)
- Testing results
- Known issues and resolutions

### implementation-plans/submission-v6.md
**Purpose:** ACID breakdown of current work  
**Use When:** Need to continue implementation, track progress  
**Contains:**
- 7 atomic tasks (A1-A7)
- Task dependencies
- Validation criteria
- Timeline
- Risk mitigation
- Completion checklist

### architecture-decisions/tcn-choice.md
**Purpose:** Document why we chose TCN architecture  
**Use When:** Questioning model choice, explaining to others, considering alternatives  
**Contains:**
- Decision rationale
- Alternatives considered (LSTM, Transformer, etc.)
- TCN implementation details
- Validation results
- Trade-offs and consequences

## 🎯 Quick Status (as of Oct 17, 2025 22:35)

### ✅ COMPLETE
- Challenge 1 TCN trained (val loss 0.010170, 65% improvement)
- Challenge 1 integrated into submission.py
- Independent training system (tmux)
- Comprehensive documentation

### 🔄 IN PROGRESS
- Challenge 2 TCN training (epoch 4/100)
- Tmux session: eeg_both_challenges (active)

### ⏳ PENDING
- Challenge 2 integration
- Submission v6 testing
- Package and upload

## 💡 Key Insights

### What Worked
1. **TCN Architecture:** 65% improvement on Challenge 1
2. **Independent Training:** tmux survives all crashes
3. **Documentation First:** Memory bank enables recovery
4. **ACID Planning:** Clear, trackable tasks

### What We Learned
1. **Window indexing bug:** array[0] not array itself
2. **Dtype matters:** Float32 vs Float64 broke training
3. **Monitor logs:** Check which log file you're reading
4. **BatchNorm critical:** Must match trained model architecture

### Current Challenges
1. **Challenge 2 performance:** Val loss 0.668 worse than baseline (0.2917)
   - Expected: Early in training, should improve
   - Monitoring: Every 5 epochs
   - Fallback: Use old Challenge 2 model if doesn't improve

## 🔄 Update Protocol

When making significant changes, update relevant documents:

**Code Changes:** Update `change-log.md`
```markdown
## [Date]

### [Time] - [Component]
**Changes:** What was modified
**Testing:** Validation results
**Impact:** Effect on system
```

**New Features:** Add to `implementation-plans/`
```markdown
## New Plan: [Feature Name]
### ACID Breakdown
- A1: Atomic task 1
- A2: Atomic task 2
...
```

**Architecture Decisions:** Add to `architecture-decisions/`
```markdown
# Decision: [Choice Made]
## Context
## Alternatives
## Rationale
## Consequences
```

## 🆘 Emergency Procedures

### Lost All Context
1. Read `QUICKSTART.md`
2. Run `./check_c2_training.sh`
3. Check `implementation-plans/submission-v6.md` for next task

### Training Crashed
1. Check `logs/train_c2_tcn_*.log` for errors
2. Review `change-log.md` for similar issues
3. Restart via `scripts/train_both_challenges.sh`

### Need to Submit NOW
1. Follow "Scenario 4" in `QUICKSTART.md`
2. Submit with Challenge 1 TCN + old Challenge 2 model
3. No need to wait for Challenge 2 training

### VS Code Won't Start
1. Training continues in tmux - it's safe!
2. SSH to machine: `ssh user@host`
3. Check status: `./check_c2_training.sh`
4. Fix VS Code later, training unaffected

## 📞 Contact Points

- **Competition:** https://www.codabench.org/competitions/4287/
- **Dataset:** https://github.com/eeg2025/downsample-datasets
- **Repository:** eeg2025 (local)

## 🎓 Learning Resources

### TCN Papers
- Original: Bai et al., 2018 "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling"

### EEG Resources
- HBN Dataset documentation
- MNE-Python tutorials
- braindecode examples

### Competition Resources
- Codabench competition page
- Baseline code (if provided)
- Forum discussions

## ✅ Maintenance

### Daily Updates
- [ ] Update `change-log.md` with day's changes
- [ ] Check off completed tasks in `implementation-plans/`
- [ ] Update status in `QUICKSTART.md`

### Before Major Milestones
- [ ] Review all documents for accuracy
- [ ] Update success metrics
- [ ] Document any new decisions

### After Submission
- [ ] Record leaderboard results in `change-log.md`
- [ ] Document lessons learned
- [ ] Archive this memory bank state

## 📊 Success Metrics

### Documentation Success
- [x] Can recover project in < 5 minutes
- [x] New person can understand project from docs
- [x] All decisions documented with rationale
- [x] Change history complete and accurate

### Project Success (Tracked in change-log.md)
- [x] Challenge 1: Val loss 0.010170 ✅
- [ ] Challenge 2: Val loss < 0.30
- [ ] Submission v6: Uploaded
- [ ] Leaderboard: Top 5 ranking

---

**Memory Bank Created:** October 17, 2025 22:35  
**Last Updated:** October 17, 2025 22:35  
**Status:** ✅ Complete and Operational  
**Next Review:** After Challenge 2 completes


---

## Latest Update: October 18, 2025

### Critical Bug Fixes ✅

Three critical bugs fixed in submission.py:
1. **Bug #1:** Broken fallback weight loading (never called torch.load)
2. **Bug #2:** Missing numpy import (used .numpy() without import)
3. **Bug #3:** Wrong API format (didn't match competition starter kit)

### Current Status

✅ All bugs fixed and verified
✅ Final package: eeg2025_submission_CORRECTED_API.zip (2.4 MB)
✅ Workspace organized (50+ files archived)
✅ Documentation complete
🚀 **READY TO UPLOAD TO CODABENCH**

### Expected Performance

- Challenge 1: NRMSE ~0.10 (TCN, 196K params)
- Challenge 2: NRMSE ~0.29 (CompactCNN, 64K params)
- Overall: NRMSE ~0.15-0.18
- Expected Rank: Top 10-15

### Documentation

- **critical-bug-fixes-oct18.md** - Comprehensive bug analysis
- **change-log.md** - Updated with October 18 fixes
- Root directory: CRITICAL_BUGS_FIXED_REPORT.md, WORKSPACE_ORGANIZATION.md, READY_TO_UPLOAD.md

### Next Steps

1. Upload to Codabench: https://www.codabench.org/competitions/4287/
2. Monitor validation (~1-2 hours)
3. Verify results and update memory bank with actual scores
