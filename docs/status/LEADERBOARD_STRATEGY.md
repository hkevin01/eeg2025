# 📊 Leaderboard & Submission Strategy

**Competition:** https://eeg2025.github.io/  
**Codabench:** https://www.codabench.org/competitions/4287/

---

## 🎯 Scoring System

### Overall Ranking:
- **Challenge 1:** 30% of final score (Response Time from CCD)
- **Challenge 2:** 70% of final score (Externalizing Factor)

**Your Current Status:**
- ✅ Challenge 2: NRMSE 0.0808 (excellent! 6x better than target)
- ⏳ Challenge 1: Training in progress

**Estimated Score Contribution:**
- Challenge 2 (70%): You're performing exceptionally well
- Challenge 1 (30%): Target NRMSE < 0.5 (achievable based on C2 success)

---

## 📅 Submission Phases

### 1. Warm-up Phase (June 10 - Oct 10, 2025) [ENDED]
- **Purpose:** Test submissions on validation set (HBN Release 5)
- **Limits:** Unlimited submissions
- **Leaderboard:** Public, validation set performance
- **Status:** This phase has ended

### 2. Final Phase (Oct 10 - Nov 2, 2025) [ACTIVE NOW!]
- **Purpose:** Evaluation on unreleased test set (HBN Release 12)
- **Limits:** LIMITED submissions per day
- **Leaderboard:** Public, test set performance
- **Requirements:** Must submit 2-page methods document
- **Deadline:** November 2, 2025 (18 days remaining)

---

## 🚦 Submission Limits & Strategy

### Key Information:
✅ **Limited submissions per day** - Exact number not specified (likely 2-5)
⚠️  **Avoid overfitting to leaderboard** - Don't submit too frequently
✅ **Code submission only** - No training allowed during inference
✅ **Single GPU, 20GB memory** - Must fit in resource constraints

### Recommended Strategy:

#### ❌ **DON'T:**
- Submit multiple times per day testing small changes
- Overfit to the leaderboard by tweaking for score improvements
- Submit without local testing first
- Use all daily submissions early in the competition

#### ✅ **DO:**
1. **Test thoroughly locally** before submitting
2. **Submit strategically:**
   - Week 1: Submit baseline (Challenge 2 only or both if C1 ready)
   - Week 2: Submit improved version with both challenges
   - Week 3: Final polished submission
3. **Save submission slots** for important iterations
4. **Document your methods** as you go

---

## 📋 Your Optimal Strategy

### Phase 1: Initial Baseline (NOW)
**Option A - Conservative (Recommended):**
```markdown
🎯 Goal: Get on leaderboard with strong Challenge 2
⏰ When: After Challenge 1 training completes
📦 What: Both challenges in one submission
✅ Why: Single submission, maximize score from day 1
```

**Option B - Quick Entry:**
```markdown
🎯 Goal: Get early leaderboard position
⏰ When: Today (15 minutes)
📦 What: Challenge 2 only (NRMSE: 0.0808)
⚠️  Why: Use one submission slot to see where you stand
```

### Phase 2: Refinement (Days 2-10)
```markdown
- Analyze leaderboard feedback
- Train improved models if needed
- Submit major improvements only (not minor tweaks)
- Save 2-3 submission slots for final week
```

### Phase 3: Final Push (Days 11-18)
```markdown
- Polish both models
- Ensemble methods if beneficial
- Final submission 1-2 days before deadline
- Keep 1 submission slot for emergency fixes
```

---

## 📊 Leaderboard Interpretation

### What You'll See:
- **Your rank** among all teams
- **NRMSE scores** for each challenge
- **Overall score** (weighted: 30% C1 + 70% C2)
- **Submission timestamp**

### What It Means:
- **NRMSE < 0.5:** Competitive (meeting target)
- **NRMSE < 0.2:** Strong performance
- **NRMSE < 0.1:** Excellent (like your Challenge 2!)
- **Lower is better** (NRMSE measures error)

### Your Expected Position:
With Challenge 2 at NRMSE 0.0808:
- **70% of score:** Already exceptional
- **30% from C1:** If you match C2 performance (NRMSE ~0.1-0.2)
- **Predicted:** Top 10-20% likely, possibly top 5%

---

## 🎓 Best Practices

### Before Each Submission:

1. **✅ Test Locally**
   ```bash
   python3 scripts/test_submission_quick.py
   python3 submission.py
   ```

2. **✅ Verify File Sizes**
   ```bash
   ls -lh weights_challenge_*.pt
   # Should be < 10MB each for your architecture
   ```

3. **✅ Check ZIP Structure**
   ```bash
   unzip -l submission.zip
   # Should show: submission.py, weights_challenge_1.pt, weights_challenge_2.pt
   # NO folders, single-level only!
   ```

4. **✅ Document Changes**
   - Keep notes on what changed
   - Update methods document
   - Track hyperparameters

### After Each Submission:

1. **📊 Record Results**
   - Leaderboard score
   - Rank position
   - Any error messages

2. **📝 Update Strategy**
   - What worked
   - What to improve
   - Remaining submission slots

3. **⏰ Plan Next Submission**
   - What changes to make
   - When to submit next
   - Resource requirements

---

## 🔢 Submission Count Tracking

```markdown
Estimated Daily Limit: 2-5 submissions (TBD - check Codabench)
Total Days: 18

**Your Submission Log:**
Date | Challenge(s) | NRMSE C1 | NRMSE C2 | Rank | Notes
-----|-------------|----------|----------|------|-------
[Date] | Both | X.XXX | 0.0808 | XX | Initial submission
[Date] | Both | X.XXX | X.XXX | XX | Improved model
...

**Remaining Slots:** Track carefully!
```

---

## 🎯 Success Metrics

### Minimum Goals:
- ✅ Challenge 2: NRMSE < 0.5 → **ACHIEVED (0.0808)**
- ⏳ Challenge 1: NRMSE < 0.5 → Training now
- 🎯 Get on leaderboard → Within 18 days

### Stretch Goals:
- 🌟 Challenge 1: NRMSE < 0.2 (match C2 performance)
- 🌟 Overall: Top 20% on leaderboard
- 🌟 Both challenges: NRMSE < 0.15

### Competition Goals:
- 🏆 Top 10: Code will be released (open source contribution!)
- 🏆 Top 3: Prize money
- 🏆 Top 1: Present at NeurIPS 2025!

---

## ⚠️  Common Pitfalls to Avoid

1. **Submission Spam:** Don't waste slots on minor tweaks
2. **Leaderboard Overfitting:** Don't optimize purely for leaderboard score
3. **Untested Submissions:** Always test locally first
4. **Poor Documentation:** Keep methods document updated
5. **Resource Violations:** Must run on single GPU with 20GB RAM
6. **Wrong Format:** ZIP must be single-level (no folders)
7. **Late Submission:** Don't wait until deadline day (server load!)

---

## 📝 Methods Document Requirement

All submissions must include a **2-page methods document** with:

### Required Sections:
1. **Methods:**
   - Architecture description
   - Training procedures
   - Hyperparameters
   - Data preprocessing

2. **Analysis:**
   - Performance analysis
   - What worked / didn't work
   - Challenges encountered

3. **Discussion:**
   - Key insights
   - Limitations
   - Future directions

### Tips:
- Write as you develop (easier than retroactive)
- Include architecture diagrams
- Cite key techniques used
- Be concise but complete

---

## 🚀 Next Actions

### Today (Challenge 1 Training):
```bash
# Start training Challenge 1
python3 scripts/train_challenge1_response_time.py 2>&1 | tee logs/challenge1_training.log

# Expected: 2-3 hours
# Target: NRMSE < 0.5 (goal: < 0.2)
```

### Tomorrow (First Submission):
```bash
# Test both models locally
python3 scripts/test_submission_quick.py

# Create submission package
./scripts/create_submission_package.sh

# Upload to Codabench
# Record results in submission log
```

### Ongoing:
- Monitor leaderboard daily
- Update methods document
- Plan improvements based on feedback
- Save submission slots for important iterations

---

**Summary:** You have 18 days and limited daily submissions. Focus on quality over quantity. Your Challenge 2 is already excellent - once Challenge 1 is trained, you'll have a strong complete submission. Don't rush to submit multiple times; test thoroughly and submit strategically!

