# EEG2025 CI/CD Pipeline

Automated validation and packaging system for NeurIPS 2025 EEG Foundation Challenge submissions.

## Overview

This CI/CD pipeline ensures every submission:
- ✅ Contains all required files
- ✅ Has correct file formats
- ✅ Can load model weights
- ✅ Passes inference tests
- ✅ Meets size requirements
- ✅ Is properly packaged as ZIP

## Quick Start

### Run Full Pipeline

```bash
./scripts/ci_cd/run_cicd.sh submissions/phase1_v14
```

This will:
1. Validate the submission directory
2. Package it as a ZIP file
3. Generate validation report

### Custom Output Name

```bash
./scripts/ci_cd/run_cicd.sh submissions/phase1_v14 phase1_v14_final
```

## Individual Tools

### 1. Validation Only

```bash
python3 scripts/ci_cd/validate_submission.py submissions/phase1_v14
```

**Checks:**
- Directory structure
- Required files (submission.py, weights_challenge_1.pt, weights_challenge_2.pt)
- submission.py format
- Model weight loading
- Inference testing
- File sizes

**Output:**
- Console report with detailed results
- `validation_report.json` in submission directory

### 2. Packaging Only

```bash
python3 scripts/ci_cd/package_submission.py submissions/phase1_v14
```

**Creates:**
- `phase1_v14.zip` in parent directory
- Includes only files from submission directory

## Submission Directory Structure

Required structure for validation:

```
submissions/phase1_v14/
├── submission.py            # Model definitions
├── weights_challenge_1.pt   # Challenge 1 weights
└── weights_challenge_2.pt   # Challenge 2 weights
```

## Validation Report

The `validation_report.json` contains:

```json
{
  "submission_dir": "submissions/phase1_v14",
  "validation_passed": true,
  "errors": [],
  "warnings": [],
  "results": {}
}
```

## Exit Codes

- `0`: Success
- `1`: Validation failed or packaging error

## Integration with Workflow

### Before Every Submission

```bash
# 1. Create submission directory
mkdir -p submissions/phase1_v15

# 2. Copy latest weights
cp checkpoints/c1_best.pt submissions/phase1_v15/weights_challenge_1.pt
cp checkpoints/c2_best.pt submissions/phase1_v15/weights_challenge_2.pt

# 3. Copy submission.py
cp submissions/phase1_v14/submission.py submissions/phase1_v15/

# 4. Run CI/CD
./scripts/ci_cd/run_cicd.sh submissions/phase1_v15

# 5. Upload resulting ZIP
```

### GitHub Actions Integration (Future)

```yaml
name: Validate Submission
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate submission
        run: ./scripts/ci_cd/run_cicd.sh submissions/latest
```

## Troubleshooting

### Validation Fails

Check `validation_report.json` for specific errors:

```bash
cat submissions/phase1_v14/validation_report.json
```

### Common Issues

**Missing files:**
```
❌ Missing required file: weights_challenge_2.pt
```
Solution: Ensure both C1 and C2 weights are present.

**Inference fails:**
```
❌ Inference failed: RuntimeError: size mismatch
```
Solution: Check model architecture matches weights.

**Size exceeded:**
```
❌ Submission exceeds size limit: 5.2 MB > 5 MB
```
Solution: Optimize model or use model compression.

## Requirements

- Python 3.8+
- PyTorch
- Standard library only (no external dependencies)

## File Descriptions

- `validate_submission.py`: Comprehensive validation checks
- `package_submission.py`: ZIP creation utility
- `run_cicd.sh`: Main orchestration script
- `README.md`: This documentation

## Development

To add new validation checks:

1. Edit `validate_submission.py`
2. Add new check method
3. Add to `run_all_checks()` list
4. Update tests

## License

Part of the EEG2025 Foundation Model project.
