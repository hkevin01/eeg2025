# Scripts Folder Reorganization Plan

## Current Structure Analysis

### Root Level Scripts (48 files)
**Training Scripts:**
- train_challenge1_attention.py → Already symlinked to training/challenge1/
- train_challenge2_multi_release.py → Should move to training/challenge2/
- train_*.py (various) → Move to training/

**Monitoring Scripts:**
- monitor_*.sh → Already in monitoring/ subdirectory
- watch_*.sh → Move to monitoring/

**Launch Scripts:**
- launch_*.sh → Create launchers/ subdirectory
- restart_*.sh → Move to launchers/
- run_*.sh → Move to launchers/
- start_*.sh → Move to launchers/

**Testing/Validation Scripts:**
- test_*.py → Move to testing/
- validate_*.py → Move to validation/

**Utility Scripts:**
- check_*.sh/py → Move to utilities/
- quick_*.sh → Move to utilities/
- organize_*.py → Move to utilities/

**Analysis Scripts:**
- analyze_*.py → Create analysis/ subdirectory

### Proposed Final Structure
```
scripts/
├── README.md (navigation guide)
├── analysis/ (data analysis, results visualization)
├── data/ (data processing, extraction)
├── deprecated/ (old scripts to keep for reference)
├── features/ (feature engineering)
├── gpu_tests/ (GPU testing)
├── inference/ (model inference, predictions)
├── launchers/ (NEW - shell scripts to start training/jobs)
├── models/ (model definitions, architecture tests)
├── monitoring/ (training monitoring, logging)
├── testing/ (unit tests, integration tests)
├── training/
│   ├── challenge1/ (Challenge 1 specific)
│   ├── challenge2/ (Challenge 2 specific)
│   └── common/ (shared training utilities)
├── utilities/ (helper scripts, checks)
└── validation/ (validation scripts)
```

## Action Plan

### Step 1: Create Missing Subdirectories
```bash
mkdir -p analysis launchers training/common
```

### Step 2: Move Scripts to Appropriate Locations
**Launchers (Shell Scripts):**
- launch_training.sh → launchers/
- restart_training*.sh → launchers/
- run_overnight_training.sh → launchers/
- start_independent_training.sh → launchers/

**Training Scripts:**
- train_challenge2_multi_release.py → training/challenge2/
- Other train_*.py → training/common/

**Monitoring:**
- watch_p300.sh → monitoring/
- status.sh → utilities/ (quick status check)
- quick_status.sh → utilities/

**Utilities:**
- check_training_status.sh → utilities/
- organize_project.py → utilities/
- organize_files.sh → utilities/

**Analysis:**
- Any analyze_*.py → analysis/

**Testing:**
- Any test_*.py → testing/

### Step 3: Update Symlinks
- Update symlinks to point to new locations
- Create new symlinks for commonly used scripts in root

### Step 4: Create README.md
- Document all subdirectories
- Provide quick reference guide
- Include common commands

### Step 5: Clean Up
- Remove duplicate scripts
- Archive truly deprecated scripts
- Update any hard-coded paths

## Benefits
✅ Logical organization
✅ Easy to find scripts
✅ Reduced root clutter
✅ Better maintainability
✅ Clearer project structure
