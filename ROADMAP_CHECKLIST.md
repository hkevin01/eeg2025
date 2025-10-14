# 📋 Roadmap Maintenance Checklist

**Purpose**: Keep implementation roadmap accurate and up-to-date  
**Review Frequency**: Weekly (every Monday)  
**Last Updated**: October 14, 2025

---

## 🔄 Weekly Update Checklist

### Every Monday Morning

- [ ] Review completed tasks from last week
- [ ] Update task status (⭕ → 🟡 → ✅)
- [ ] Update progress percentages
- [ ] Update progress bars in README.md
- [ ] Add completion dates for finished tasks
- [ ] Document any blockers or risks
- [ ] Update next steps section
- [ ] Review and adjust time estimates

### Files to Update

```bash
# 1. Main roadmap
vim IMPLEMENTATION_ROADMAP.md

# 2. README status section
vim README.md

# 3. Competition plan (if needed)
vim docs/competition_implementation_plan.md

# 4. Create weekly summary
vim docs/weekly_updates/week_$(date +%U)_update.md
```

---

## 📊 Progress Tracking Template

### Phase Status Update

```markdown
## Phase PX: [Phase Name] ([STATUS])

**Duration**: [Dates]
**Status**: [⭕/🟡/✅] [X]% [Complete/In Progress/Not Started]
**Last Updated**: [Date]

### Completed This Week
- [x] Task X.1: Description (completed [date])
- [x] Task X.2: Description (completed [date])

### In Progress
- [ ] Task X.3: Description (XX% complete)
  - Blocker: [if any]
  - ETA: [date]

### Blocked
- [ ] Task X.4: Description
  - Reason: [blocker description]
  - Resolution: [plan to unblock]

### Metrics
- Tests: XX passing (was YY last week)
- Coverage: XX% (was YY% last week)
- Performance: [key metric]
```

---

## ✅ Task Completion Checklist

When marking a task complete:

- [ ] Change status: ⭕ → ✅
- [ ] Add completion date
- [ ] Update progress bar
- [ ] Document deliverables
- [ ] Add to "Completed Tasks" section
- [ ] Update dependencies (unblock next tasks)
- [ ] Run verification tests
- [ ] Update metrics
- [ ] Create summary document (if major task)

### Example

**Before**:
```markdown
| P2.1 | Scale Data Acquisition | 3-5 days | ⭕ |
```

**After**:
```markdown
| P2.1 | Scale Data Acquisition | 3 days | ✅ | Oct 18, 2025 |

**Deliverables**:
- 75 subjects downloaded (exceeded 50 target)
- All BIDS-compliant
- Data split: 60/20/20 train/val/test
```

---

## 📈 Progress Bar Update Guide

### Calculate Progress

```python
# For each phase
completed_tasks = [task for task in phase_tasks if task.status == "✅"]
progress = (len(completed_tasks) / len(phase_tasks)) * 100

# Visual bar (20 chars = 100%)
filled = int(progress / 5)
bar = "█" * filled + "░" * (20 - filled)
```

### Update in README.md

```markdown
Phase P2: Advanced Development       ████░░░░░░░░░░░░░░░░  20% 🟡
```

---

## 🎯 Metrics Tracking

### Key Metrics to Update

**Testing**:
```markdown
- Total tests: XX (was YY last week, +ZZ new)
- Passing: XX/XX (100%)
- Coverage: XX% (target: >80%)
- CI/CD: ✅ Green
```

**Data**:
```markdown
- Subjects downloaded: XX (target: 100)
- Total size: XXX GB
- Quality checks: ✅ All passing
```

**Performance**:
```markdown
- Inference latency: XXms (target: <50ms)
- Training time: XX hours per epoch
- GPU utilization: XX%
```

**Models**:
```markdown
- Challenge 1: Pearson r = 0.XX, AUROC = 0.XX
- Challenge 2: Avg Pearson r = 0.XX
- Best checkpoint: epoch XX
```

---

## 🚨 Risk Tracking

### When to Add Risk

Add to risk table if:
- Task delayed >2 days
- Blocker identified
- Resource constraint found
- New technical challenge discovered

### Risk Update Template

```markdown
| Risk | Impact | Probability | Status | Mitigation |
|------|--------|-------------|--------|------------|
| [Description] | 🔴/🟠/🟢 | 🔴/🟡/�� | 🟡 Active | [Action taken] |
```

---

## 📅 Phase Transition Checklist

### When Starting New Phase

- [ ] Mark previous phase as complete (100%)
- [ ] Update phase status: ⭕ → 🟡
- [ ] Create phase kickoff document
- [ ] Review all task dependencies
- [ ] Set up tracking for new tasks
- [ ] Schedule phase review meeting
- [ ] Update timeline if needed

### When Completing Phase

- [ ] Verify all tasks complete (✅)
- [ ] Update progress bar to 100%
- [ ] Set completion date
- [ ] Create phase summary document
- [ ] Document lessons learned
- [ ] Archive phase artifacts
- [ ] Prepare next phase transition

---

## 📝 Documentation Updates

### Required Updates

**IMPLEMENTATION_ROADMAP.md**:
- Task status and progress
- Completion dates
- Deliverables
- Metrics
- Risk assessment

**README.md**:
- Project status table
- Progress bars
- Recent achievements
- Next steps

**competition_implementation_plan.md**:
- Phase summaries
- Challenge-specific updates
- Metric improvements

---

## 🔍 Verification Commands

### Check Current Status

```bash
# Run all verification checks
./check_roadmap_status.sh

# Manual checks
pytest tests/ -v
ls -lh data/raw/hbn/sub-*/ | wc -l
git log --oneline -10
```

### Generate Status Report

```bash
# Create automated status report
python scripts/generate_status_report.py \
  --output docs/weekly_updates/week_$(date +%U)_report.md
```

---

## 🎓 Best Practices

### DO

✅ Update roadmap weekly (minimum)
✅ Be specific with completion dates
✅ Document blockers immediately
✅ Keep metrics accurate
✅ Update progress bars consistently
✅ Link to deliverables
✅ Note lessons learned

### DON'T

❌ Let updates lag >1 week
❌ Mark incomplete tasks as done
❌ Ignore blockers
❌ Skip metric updates
❌ Leave outdated information
❌ Forget to update README
❌ Skip verification tests

---

## 📞 Escalation Process

### When to Escalate

- Task blocked >3 days
- Critical path impacted
- Resources unavailable
- Timeline at risk
- Technical blocker unsolved

### Escalation Template

```markdown
## 🚨 ESCALATION: [Brief Description]

**Priority**: 🔴 Critical / 🟠 High / 🟡 Medium
**Impact**: [Description of impact]
**Task**: [Affected task ID and name]
**Blocker**: [Detailed blocker description]
**Attempted Solutions**: [What was tried]
**Needed**: [What is needed to unblock]
**ETA Impact**: [Days delayed]
**Escalated To**: [Person/team]
**Date**: [Date of escalation]
```

---

## 📋 Quick Reference

### Status Indicators

- ⭕ Not Started
- 🟡 In Progress
- ✅ Complete
- 🔴 Blocked
- 🔄 Needs Review

### Priority Levels

- 🔴 Critical (P0)
- 🟠 High (P1)
- 🟡 Medium (P2)
- 🟢 Low (P3)

### Progress Bars

```
░░░░░░░░░░░░░░░░░░░░   0%
█░░░░░░░░░░░░░░░░░░░   5%
██░░░░░░░░░░░░░░░░░░  10%
████░░░░░░░░░░░░░░░░  20%
██████░░░░░░░░░░░░░░  30%
████████░░░░░░░░░░░░  40%
██████████░░░░░░░░░░  50%
████████████░░░░░░░░  60%
██████████████░░░░░░  70%
████████████████░░░░  80%
██████████████████░░  90%
████████████████████ 100%
```

---

## 🔄 Automation Ideas

### Scripts to Create

1. **Status Checker**: Automatically verify task completion
2. **Progress Calculator**: Update progress percentages
3. **Report Generator**: Create weekly status reports
4. **Metric Tracker**: Log key metrics over time
5. **Blocker Notifier**: Alert on tasks blocked >2 days

### Example Script Stub

```bash
#!/bin/bash
# scripts/update_roadmap_status.sh

# Update test count
TEST_COUNT=$(find tests/ -name "test_*.py" | wc -l)
sed -i "s/Total tests: [0-9]*/Total tests: $TEST_COUNT/" IMPLEMENTATION_ROADMAP.md

# Update data count
DATA_COUNT=$(ls -d data/raw/hbn/sub-*/ 2>/dev/null | wc -l)
sed -i "s/Downloaded: [0-9]*/Downloaded: $DATA_COUNT/" IMPLEMENTATION_ROADMAP.md

# Calculate overall progress
# ... more automation ...

echo "✅ Roadmap status updated!"
```

---

**Last Updated**: October 14, 2025  
**Next Review**: October 21, 2025  
**Maintained By**: Project Team

---

**Quick Commands**:

```bash
# View this checklist
cat ROADMAP_CHECKLIST.md

# Update roadmap
vim IMPLEMENTATION_ROADMAP.md

# Verify status
./check_roadmap_status.sh

# Generate report
python scripts/generate_status_report.py
```
