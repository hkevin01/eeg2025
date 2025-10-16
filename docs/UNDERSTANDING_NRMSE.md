# Understanding NRMSE Scores

## What is NRMSE?

**NRMSE** = Normalized Root Mean Squared Error

```python
from numpy import std
from sklearn.metrics import root_mean_squared_error as rmse

NRMSE = rmse(y_true, y_pred) / std(y_true)
```

### What It Means

- **Lower is better** (closer to 0 = more accurate)
- **Normalized by standard deviation** - makes it comparable across different scales
- **Dimensionless metric** - no units, just a ratio

### Interpretation Guide

| NRMSE Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.1 | Excellent prediction |
| 0.1 - 0.3 | Good prediction |
| 0.3 - 0.5 | Fair prediction |
| 0.5 - 0.7 | Poor prediction |
| > 0.7 | Very poor prediction |
| > 1.0 | Worse than predicting mean |

**Note:** NRMSE > 1.0 means your model is worse than just predicting the mean value!

---

## Our Scores Explained

### Challenge 1: Response Time Prediction
- **Our NRMSE: 0.4680**
- **Interpretation:** Fair to good prediction
- **Context:** Only 420 training samples from 20 subjects
- **Comparison to naive baseline:**
  - Predicting mean: NRMSE = 1.0
  - Our model: NRMSE = 0.47 (53% better!)

### Challenge 2: Externalizing Factor Prediction
- **Our NRMSE: 0.0808**
- **Interpretation:** Excellent prediction
- **Context:** 2,315 training samples from 12 subjects
- **Comparison to naive baseline:**
  - Predicting mean: NRMSE = 1.0
  - Our model: NRMSE = 0.08 (92% better!)

### Overall Weighted Score
- **Formula:** 0.3 × Challenge1 + 0.7 × Challenge2
- **Our Score:** 0.3 × 0.4680 + 0.7 × 0.0808 = **0.1970**
- **Interpretation:** Strong overall performance

---

## Important Clarification

### There is NO "Baseline of 0.5"

In our earlier documentation, we incorrectly stated:
> "outperforming the competition baseline of 0.5"

**This was WRONG!**

- **0.5 is NOT a competition baseline**
- **0.5 was our internal target** (a reasonable threshold for "good" NRMSE)
- **We won't know the actual competition baseline** until we see the leaderboard

### What "Baseline" Actually Means

A "baseline" in competitions typically refers to:
1. **Naive baseline:** Predict the mean → NRMSE = 1.0
2. **Simple baseline:** Linear regression, simple CNN
3. **Starter kit baseline:** If organizers provide one

**The competition organizers have NOT published an official baseline score.**

---

## How to Interpret Our Results

### Challenge 1 (NRMSE 0.47)

**What we know:**
- ✅ Much better than predicting mean (1.0)
- ✅ Cross-validation shows it's stable (not overfitting)
- ✅ 53% improvement from baseline model (NRMSE 0.99 → 0.47)

**What we DON'T know:**
- ❓ How it compares to other teams
- ❓ What the test set performance will be
- ❓ What score wins the competition

**Realistic expectations:**
- **Best case:** Top 10-20% (if test set similar to validation)
- **Expected:** Middle of pack (some domain shift likely)
- **Worst case:** Lower half (if test set very different)

### Challenge 2 (NRMSE 0.08)

**What we know:**
- ✅ Excellent performance (< 0.1)
- ✅ Near-perfect correlation (0.997)
- ✅ Much more data than Challenge 1 (2,315 vs 420 samples)

**What we DON'T know:**
- ❓ If other teams have similar scores
- ❓ How generalizable this is

**Realistic expectations:**
- **Best case:** Top tier performance
- **Expected:** Competitive score
- **Worst case:** Still decent given quality metrics

---

## Comparison to Literature

### EEG Response Time Prediction (Challenge 1)

**Limited published work:**
- Most EEG-RT studies focus on classification (fast/slow), not regression
- Published NRMSE values rare
- Our 0.47 is competitive but hard to compare directly

### EEG Clinical Factor Prediction (Challenge 2)

**Comparison to related work:**
- Brain age prediction: typical MAE 2-5 years → NRMSE 0.1-0.3
- Clinical scores: typical NRMSE 0.3-0.6
- **Our 0.08 is excellent** compared to published work

---

## What Success Looks Like

### Competitive Success
- **Target:** Top 50% of teams (realistic goal)
- **Stretch:** Top 25% (would be great!)
- **Dream:** Top 10% (would require luck + strong test set match)

### Technical Success (Already Achieved!)
- ✅ Models trained and validated
- ✅ Better than naive baseline
- ✅ Reproducible results
- ✅ Validated through cross-validation
- ✅ Code submitted with proper format

### Learning Success
- ✅ Learned about EEG processing
- ✅ Implemented data augmentation
- ✅ Validated approaches properly
- ✅ Documented methodology

---

## Key Takeaways

1. **NRMSE is relative** - there's no universal "good" score
2. **Our scores are solid** - 0.47 and 0.08 show working models
3. **Competition position unknown** - need leaderboard to assess
4. **We're prepared to iterate** - 18 days to improve based on feedback

---

## Bottom Line

**Our scores (0.47 and 0.08) represent:**
- ✅ Functional, trained models
- ✅ Better than naive baselines
- ✅ Validated and reproducible
- ❓ Unknown competitive position (will know after submission!)

**The real test:** How we rank on the competition leaderboard!

---

*See also:*
- *Methods: docs/methods/METHODS_DOCUMENT.md*
- *Validation: docs/VALIDATION_SUMMARY_MASTER.md*
- *Next Steps: docs/NEXT_STEPS_ANALYSIS.md*
