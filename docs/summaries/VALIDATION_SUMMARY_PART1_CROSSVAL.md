## Validation Summary Part 1: Cross-Validation Results (Challenge 1)

**Date:** October 15, 2025
**Experiment:** 5-Fold Cross-Validation (Response Time Prediction, CCD Task)

---

### Purpose
To assess the robustness and generalization of the baseline model for Challenge 1 by evaluating its performance across multiple data splits using 5-fold cross-validation.

---

### Method
- **Dataset:** CCD Task (20 subjects, 420 segments)
- **Model:** Baseline ResponseTimeCNN (no augmentation)
- **Metric:** Normalized Root Mean Squared Error (NRMSE)
- **Splits:** 5 folds (336 train, 84 validation per fold)
- **Device:** CPU

---

### Results

| Fold | NRMSE  | Target (<0.5) |
|------|--------|---------------|
| 1    | 1.2900 | ❌            |
| 2    | 0.9855 | ❌            |
| 3    | 1.0086 | ❌            |
| 4    | 0.9507 | ❌            |
| 5    | 1.0305 | ❌            |

**Statistics:**
- **Mean NRMSE:** 1.0530
- **Std NRMSE:** 0.1214
- **Min NRMSE:** 0.9507
- **Max NRMSE:** 1.2900

---

### Interpretation
- All folds are above the competition target (0.5), indicating the baseline model is not sufficient for robust generalization.
- There is moderate variability across folds (std 0.12), suggesting some sensitivity to data splits.
- These results motivated the development of improved models with data augmentation and better architectures.

---

### Next Steps
- Compare with improved model and ensemble results (see Part 2).
- Use these results as a baseline for future model improvements.

---

*See also: `results/challenge1_crossval.txt` for raw output.*
