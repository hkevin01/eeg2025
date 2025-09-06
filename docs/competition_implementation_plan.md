# Competition Tasks Implementation Plan

## 📊 Competition Overview Analysis

Based on the visual, the HBN-EEG Competition consists of two main challenges:

### Challenge 1: Transferability across subjects and tasks


- **Source Task**: Sustained Attention (SuS) - Passive task
- **Target Task**: Contrast Change Detection (CCD) - Active task
- **Goal**: Transfer learning function `f_train(X_test, P_test) → Y_test`
- **Metrics**: Response time prediction and success classification


### Challenge 2: Decoding psychopathology factors

- **Input**: Multi-task EEG data (X, Y)
- **Output**: P-factor scores (psychopathology dimensions)
- **Goal**: Function `f_θ(X, Y) → P`
- **Metrics**: Correlation with clinical assessments

## 🎯 Current Implementation Status

### ✅ Already Implemented Components

1. **HBN Dataset Integration** (`src/dataio/hbn_dataset.py`)
   - Official splits and labels loading
   - Challenge-compliant windowing (2.0s windows, 0.5 overlap)
   - Task-specific data filtering (SuS, CCD, RS, MW, SL, SyS)
   - CBCL psychopathology label extraction

2. **Foundation Model Architecture** (`src/models/advanced_foundation_model.py`)
   - Task-aware backbone with adapters
   - Multi-adversary domain adaptation
   - Cross-task transfer capabilities

3. **Cross-Task Transfer** (`src/training/train_cross_task.py`)
   - SuS → CCD transfer training
   - FiLM adapters for task conditioning
   - Official metrics computation

4. **Domain Adaptation** (`src/utils/domain_adaptation.py`)
   - DANN implementation
   - Subject-level invariance
   - Multi-domain adaptation

### ⚠️ Gaps and Required Enhancements

## 🚧 Implementation Plan

### Phase 1: Challenge 1 - Cross-Task Transfer Enhancement


#### Step 1.1: Enhance Transfer Learning Pipeline

```bash
# Files to modify/create:
- src/training/challenge1_trainer.py (NEW)
- src/models/challenge1_model.py (NEW)
- scripts/train_challenge1.py (NEW)

```

**Key Features to Implement:**

- Multi-stage transfer learning (SSL → SuS pretraining → CCD fine-tuning)
- Progressive unfreezing strategy

- Advanced domain alignment techniques
- Official challenge metric optimization

#### Step 1.2: Response Time Prediction Enhancement

```bash
# Files to enhance:

- src/models/heads/regression.py
- src/models/losses/regression_losses.py
```

**Enhancements:**


- Specialized RT prediction head with temporal modeling
- Correlation-aware loss functions
- Subject-specific calibration layers

#### Step 1.3: Success Classification Optimization


```bash
# Files to enhance:
- src/models/heads/classification.py
- src/training/metrics/challenge_metrics.py
```

**Enhancements:**


- Balanced accuracy optimization
- Confidence calibration
- Threshold optimization for binary classification

### Phase 2: Challenge 2 - Psychopathology Prediction

#### Step 2.1: Multi-Task Architecture for P-Factors


```bash
# Files to create:
- src/training/challenge2_trainer.py (NEW)
- src/models/challenge2_model.py (NEW)

- src/models/heads/psychopathology.py (ENHANCE)
```

**Key Components:**

- Multi-output regression for 4 CBCL factors
- Task-invariant feature extraction

- Clinical score normalization

#### Step 2.2: Clinical Data Integration

```bash

# Files to enhance:
- src/dataio/clinical_labels.py (NEW)
- src/dataio/hbn_dataset.py (ENHANCE)
```

**Features:**


- CBCL score preprocessing
- Age and demographic factor normalization
- Missing data handling strategies

#### Step 2.3: Cross-Subject Generalization

```bash

# Files to enhance:
- src/utils/domain_adaptation.py
- src/models/invariance/subject_invariance.py (NEW)
```

**Techniques:**


- Subject-specific batch normalization
- Population-level statistics alignment
- Adversarial subject classification

### Phase 3: Competition-Specific Optimizations


#### Step 3.1: Official Metrics Implementation

```bash
# Files to create:
- src/evaluation/official_metrics.py (ENHANCE)
- src/evaluation/competition_evaluator.py (NEW)

```

**Metrics to Implement:**

- Challenge 1: Pearson correlation for RT, balanced accuracy for success
- Challenge 2: Pearson correlation for each P-factor dimension
- Cross-validation strategies matching official splits

#### Step 3.2: Submission Pipeline

```bash
# Files to create:
- scripts/generate_submission.py (NEW)
- src/utils/submission_formatter.py (NEW)
```

**Features:**

- Model ensembling strategies
- Prediction post-processing
- Official format compliance

## 📝 Detailed Implementation Steps

### Step 1: Challenge 1 - Enhanced Cross-Task Transfer

Let me create the enhanced Challenge 1 implementation:
