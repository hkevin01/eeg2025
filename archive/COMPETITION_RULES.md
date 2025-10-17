# 📋 EEG2025 Competition - Complete Rules & Guidelines
**Competition:** NeurIPS 2025 EEG Foundation Challenge  
**Source:** https://eeg2025.github.io/rules/  
**Last Updated:** October 17, 2025

---

## 🎯 COMPETITION STRUCTURE

### Challenge Format
```
Type:        Regression tasks (supervised learning)
Challenges:  2 separate challenges with combined scoring
Platform:    Codabench (https://www.codabench.org/competitions/4287/)
Timeline:    June 2025 - November 2, 2025
Venue:       NeurIPS 2025 Competition Track
```

### Challenge 1: Cross-Task Transfer Learning
```
Objective:   Predict response time from EEG signals
Task:        Contrast Change Detection (CCD)
Input:       EEG data (129 channels, 100Hz, variable length)
Output:      Response time (continuous value, seconds)
Metric:      NRMSE (Normalized Root Mean Square Error)
Weight:      30% of overall competition score
Pretraining: Suggested to use passive tasks (RS, SuS, MW)
```

### Challenge 2: Externalizing Factor Prediction
```
Objective:   Predict externalizing behavior factor
Task:        Subject-invariant representation learning
Input:       Resting State EEG (129 channels, 100Hz)
Output:      Externalizing factor score (continuous value)
Metric:      NRMSE (Normalized Root Mean Square Error)
Weight:      70% of overall competition score
Note:        Originally included 4 factors, reduced to 1 (externalizing)
```

---

## 📊 EVALUATION METRICS

### NRMSE (Normalized Root Mean Square Error)
```python
NRMSE = RMSE / std(y_true)

Where:
├─ RMSE = sqrt(mean((y_pred - y_true)^2))
├─ std(y_true) = standard deviation of ground truth values
├─ Lower is better (perfect score = 0.0)
└─ Baseline: Using mean prediction (NRMSE ≈ 1.0)

Properties:
├─ Scale-independent (normalized by std)
├─ Interpretable: NRMSE = 1.0 means same error as naive baseline
├─ NRMSE < 1.0 means better than baseline
└─ NRMSE > 1.0 means worse than baseline
```

### Overall Competition Score
```python
Overall_Score = 0.30 × NRMSE_Challenge1 + 0.70 × NRMSE_Challenge2

Rationale for weights:
├─ Challenge 2 (70%): More data, more general, clinical importance
├─ Challenge 1 (30%): Smaller dataset, more specific task
└─ Both challenges must perform well to win
```

### Ranking
```
Primary:    Overall score (lower is better)
Tiebreaker: Earlier submission timestamp
Leaderboard: Public leaderboard updated automatically
Final:       Private test set (R12) used for final ranking
```

---

## 📦 SUBMISSION REQUIREMENTS

### Required Files
```
submission.zip must contain:
├─ submission.py          # Main submission script (REQUIRED)
├─ weights_*.pt           # Model weights (any number, any names)
├─ methods.pdf            # 2-page methods document (OPTIONAL)
└─ Any additional files   # Libraries, configs, etc. (optional)

Size Limits:
├─ Total submission: < 2 GB (recommended < 500 MB)
├─ Individual files: No specific limit
└─ Execution time: 24 hours max per submission
```

### submission.py Requirements
```python
# Must contain these two functions:

def challenge1(X):
    """
    Predict response times for Challenge 1.
    
    Args:
        X: List of EEG arrays, each shape (n_channels, n_timepoints)
           - n_channels = 129
           - n_timepoints varies per trial
    
    Returns:
        predictions: np.ndarray of shape (n_trials,)
                    Response times in seconds
    """
    pass

def challenge2(X):
    """
    Predict externalizing factors for Challenge 2.
    
    Args:
        X: List of EEG arrays, each shape (n_channels, n_timepoints)
           - n_channels = 129
           - n_timepoints = fixed length windows
    
    Returns:
        predictions: np.ndarray of shape (n_samples,)
                    Externalizing factor scores
    """
    pass

# Important:
├─ Functions must handle variable-length inputs
├─ Must work in Codabench execution environment
├─ Can import standard libraries (numpy, torch, sklearn, etc.)
├─ Can load weights from ./weights/ or specified path
└─ Must return numpy arrays with correct shape
```

### Path Resolution
```python
# Recommended approach for loading weights:
from pathlib import Path

def resolve_path(name="model_file_name"):
    """Handle different execution environments"""
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(f"Could not find {name}")

# Usage:
weights_path = resolve_path("weights_challenge_1.pt")
model.load_state_dict(torch.load(weights_path))
```

---

## 🗓️ COMPETITION TIMELINE

### Key Dates
```
Phase 1: Development Phase
├─ Start:       June 2025
├─ Data Release: Training data (R1-R5) available
├─ Validation:  Participants can validate locally
└─ Submissions: Unlimited submissions during this phase

Phase 2: Evaluation Phase
├─ Start:       September 2025
├─ Test Data:   R12 (hidden test set) used for evaluation
├─ Submissions: Limited submissions (check platform)
└─ Leaderboard: Public leaderboard available

Phase 3: Final Evaluation
├─ Deadline:    November 2, 2025 (23:59 UTC)
├─ Freeze:      Submissions close
├─ Evaluation:  Final private test set evaluation
└─ Results:     Announced at NeurIPS 2025

Phase 4: Awards & Presentation
├─ NeurIPS:     December 2025 (exact dates TBD)
├─ Winners:     Top teams present at NeurIPS
├─ Awards:      Cash prizes + travel support
└─ Publication: Competition report paper
```

---

## 📚 DATA USAGE RULES

### Allowed Data
```
✅ HBN-EEG dataset (provided by competition)
   ├─ Training: R1, R2, R3, R4, R5
   ├─ Test: R12 (hidden)
   └─ All tasks: CCD, RS, SuS, MW, SL, SyS

✅ External EEG datasets (for pretraining)
   ├─ Publicly available EEG datasets
   ├─ Must be documented in methods
   └─ Must not include HBN participants

✅ Pretrained models
   ├─ Foundation models (if publicly available)
   ├─ Self-supervised pretraining
   └─ Transfer learning from other domains
```

### Prohibited Data
```
❌ Test set data (R12) - obviously prohibited
❌ HBN participants' data outside competition dataset
❌ Private datasets not publicly available
❌ Manual labeling of test data
❌ Leaking information from test set
```

### Data Splits
```
Training Strategy (Recommended):
├─ Use R1-R5 for training/validation
├─ Implement cross-validation across releases
├─ Test generalization across different releases
└─ Avoid overfitting to specific release distribution

Competition Evaluation:
├─ Models evaluated on R12 (hidden test set)
├─ R12 contains data from different subjects
├─ Distribution may differ from R1-R5
└─ Generalization is key to success
```

---

## 👥 TEAM RULES

### Team Composition
```
✅ Individual participants allowed
✅ Teams allowed (max size: check platform)
✅ Team mergers: Allowed before deadline
✅ Multiple teams: One person can be in multiple teams
```

### Code Sharing
```
✅ Share code/ideas in forums (encouraged)
✅ Open source submissions after competition (encouraged)
✅ Collaborate with others
⚠️  Must clearly attribute external code/methods
❌ Plagiarism not allowed
```

### Submission Rules
```
✅ Multiple submissions allowed during development
✅ Select best submission before deadline
⚠️  Limited submissions in final evaluation period
❌ No submissions after deadline
```

---

## 🏆 AWARDS & PRIZES

### Prize Categories
```
1st Place:
├─ Cash prize: Check competition website
├─ NeurIPS presentation opportunity
├─ Travel support to NeurIPS 2025
└─ Co-authorship on competition paper

2nd Place:
├─ Cash prize: Check competition website
├─ NeurIPS poster presentation
└─ Co-authorship on competition paper

3rd Place:
├─ Cash prize: Check competition website
└─ Recognition in competition paper

Special Awards:
├─ Diversity & Inclusion Award
├─ Best Novel Method Award
├─ Most Efficient Model Award
└─ Check website for details
```

### Winner Requirements
```
Mandatory for Prize Winners:
├─ [ ] Valid submission by deadline
├─ [ ] Reproducible code
├─ [ ] Detailed methods document (2-page+)
├─ [ ] Code release (open source preferred)
├─ [ ] Fact sheet / model card
└─ [ ] Presentation at NeurIPS 2025 (top teams)

Timeline:
├─ Winners announced: After competition close
├─ Code review: Within 2 weeks of announcement
├─ Methods paper: Within 1 month
└─ NeurIPS presentation: December 2025
```

---

## 📝 METHODS DOCUMENTATION

### Required for Winners
```
Methods Document Must Include:
├─ Model architecture description
├─ Training procedure
├─ Hyperparameters
├─ Data preprocessing
├─ Augmentation strategies
├─ External data used (if any)
├─ Computational requirements
├─ Code repository link
└─ Reproducibility instructions

Format:
├─ PDF format
├─ 2 pages minimum (no maximum for winners)
├─ LaTeX or similar professional format
├─ Figures and tables allowed
└─ References to prior work
```

### Recommended Content
```
1. Introduction
   └─ Brief overview of approach

2. Architecture
   ├─ Model diagrams
   ├─ Parameter counts
   └─ Design rationale

3. Training
   ├─ Optimization details
   ├─ Data splits
   ├─ Augmentation
   └─ Hyperparameters

4. Results
   ├─ Validation scores
   ├─ Ablation studies
   └─ Error analysis

5. Discussion
   ├─ What worked
   ├─ What didn't work
   └─ Future directions
```

---

## ⚖️ FAIR PLAY & ETHICS

### Code of Conduct
```
✅ Respect other participants
✅ Follow competition rules
✅ Report bugs/issues to organizers
✅ Help improve competition
⚠️  Use computational resources responsibly
❌ No cheating (manual labeling, test set leakage)
❌ No harassment or discrimination
❌ No attempts to hack/exploit platform
```

### Intellectual Property
```
Your Submission:
├─ You retain all rights to your code
├─ You grant organizers right to evaluate
├─ You decide on open source vs proprietary
└─ Winners encouraged to open source

Competition Data:
├─ HBN dataset: Follow HBN data usage agreement
├─ Competition format: Free to use in publications
├─ Starter code: MIT License (free to use/modify)
└─ Credit organizers if using competition setup
```

### Privacy & Data Protection
```
✅ Anonymized EEG data (no personal identifiers)
✅ Follow HBN data usage policies
✅ Do not attempt to re-identify subjects
⚠️  Computational environment is sandboxed
❌ No data extraction from test set
```

---

## 🔧 TECHNICAL REQUIREMENTS

### Execution Environment
```
Platform: Codabench
├─ OS: Linux
├─ Python: 3.9+
├─ Libraries: NumPy, PyTorch, TensorFlow, scikit-learn
├─ GPU: May or may not be available
└─ Memory: Limited (design accordingly)

Recommendations:
├─ Test on CPU (GPU not guaranteed)
├─ Optimize memory usage
├─ Handle large datasets efficiently
└─ Use checkpointing for large models
```

### Dependencies
```
Allowed:
✅ Standard Python libraries
✅ PyTorch, TensorFlow, JAX
✅ scikit-learn, pandas, NumPy
✅ MNE-Python (EEG processing)
✅ Any pip-installable package

Installation:
├─ Include requirements.txt
├─ Or use Docker (if supported)
└─ Organizers install dependencies

Restrictions:
⚠️  No internet access during evaluation
⚠️  No external API calls
❌ No proprietary/licensed software
```

---

## �� SUPPORT & RESOURCES

### Getting Help
```
Discord:    https://discord.gg/8jd7nVKwsc
Forum:      Competition platform forum
Email:      Check competition website
FAQ:        https://eeg2025.github.io/faq/
```

### Resources
```
Starter Kit:    https://eeg2025.github.io/baseline/
GitHub:         https://github.com/eeg2025/startkit
Documentation:  https://eeg2025.github.io/data/
Paper:          https://arxiv.org/abs/2506.19141
```

---

## ✅ PRE-SUBMISSION CHECKLIST

### Before Submitting
```
Code:
├─ [ ] submission.py implements challenge1() and challenge2()
├─ [ ] Functions return correct array shapes
├─ [ ] Model weights load correctly
├─ [ ] Tested locally on sample data
├─ [ ] No hardcoded paths (use resolve_path())
├─ [ ] No dependencies on local files
└─ [ ] Code is reproducible

Package:
├─ [ ] All files in .zip archive
├─ [ ] File sizes reasonable (< 500 MB)
├─ [ ] No corrupted files
├─ [ ] Test unzip works correctly
└─ [ ] Folder structure correct

Documentation:
├─ [ ] Methods document included (optional but recommended)
├─ [ ] README with instructions (optional)
├─ [ ] requirements.txt for dependencies
└─ [ ] Code comments for clarity

Testing:
├─ [ ] Ran locally without errors
├─ [ ] Validated output format
├─ [ ] Checked NRMSE on validation set
└─ [ ] Ready for Codabench evaluation
```

---

**Document Created:** October 17, 2025  
**Competition URL:** https://eeg2025.github.io/  
**Codabench:** https://www.codabench.org/competitions/4287/  
**Deadline:** November 2, 2025, 23:59 UTC  
**Status:** Competition Active - 16 days remaining

🏆 **Good luck and may the best model win!**
