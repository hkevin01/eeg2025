#!/bin/bash

echo "🧹 Organizing Root Directory"
echo "=" | head -c 80 | tr -d '\n'; echo

# Create necessary subdirectories
mkdir -p docs/status
mkdir -p docs/analysis
mkdir -p submissions/versions
mkdir -p submissions/scripts
mkdir -p tests/validation
mkdir -p scripts/testing

# Move status/training documents
echo "📋 Moving status documents..."
mv -v TRAINING_STATUS_CURRENT.md docs/status/ 2>/dev/null
mv -v TRAINING_STATUS.md docs/status/ 2>/dev/null
mv -v TRAINING_STOPPED_ANALYSIS.md docs/status/ 2>/dev/null
mv -v CHALLENGE1_IMPROVEMENT_ACTIVE.md docs/status/ 2>/dev/null

# Move analysis documents
echo "📊 Moving analysis documents..."
mv -v VSCODE_CRASH_ANALYSIS.md docs/analysis/ 2>/dev/null
mv -v SUBMISSION_FIX_REPORT.md docs/analysis/ 2>/dev/null
mv -v SUBMISSION_V5_ANALYSIS.md docs/analysis/ 2>/dev/null
mv -v gpu_issue_explanation.md docs/analysis/ 2>/dev/null
mv -v BEFORE_AFTER_COMPARISON.md docs/analysis/ 2>/dev/null
mv -v EXTENSION_CLEANUP_COMPLETE.md docs/analysis/ 2>/dev/null

# Move ROCm/GPU documents
echo "🎮 Moving GPU/ROCm documents..."
mv -v ROCM_GPU_SOLUTION_COMPLETE.md docs/analysis/ 2>/dev/null
mv -v ROCM_SOLUTION_FINAL_STATUS.md docs/analysis/ 2>/dev/null
mv -v FINAL_ROCM_SOLUTION_PLAN.md docs/analysis/ 2>/dev/null
mv -v FINAL_STATUS_REALISTIC.md docs/analysis/ 2>/dev/null

# Move submission-related documents
echo "📦 Moving submission documents..."
mv -v SUBMISSION_READY_V4.md docs/submissions/ 2>/dev/null || mkdir -p docs/submissions && mv -v SUBMISSION_READY_V4.md docs/submissions/ 2>/dev/null
mv -v SUBMISSION_UPLOAD_CHECKLIST.md docs/submissions/ 2>/dev/null
mv -v CHECKLIST_COMPLETE.md docs/submissions/ 2>/dev/null
mv -v CACHED_DATA_INFO.md docs/ 2>/dev/null

# Move submission zip files
echo "📮 Moving submission packages..."
mv -v submission_sam_fixed_v3.zip submissions/versions/ 2>/dev/null
mv -v submission_sam_fixed_v4.zip submissions/versions/ 2>/dev/null
# Keep v5 in root for now (latest)

# Move submission Python files
echo "🐍 Moving submission scripts..."
mv -v submission_sam_fixed_v3.py submissions/versions/ 2>/dev/null
mv -v submission_correct.py submissions/scripts/ 2>/dev/null
# Keep submission.py in root (working file)

# Move test scripts
echo "�� Moving test scripts..."
mv -v test_conv_*.py tests/validation/ 2>/dev/null
mv -v test_cpu_conv.py tests/validation/ 2>/dev/null
mv -v test_simple_conv.py tests/validation/ 2>/dev/null
mv -v test_submission_syntax.py tests/validation/ 2>/dev/null

# Move evaluation scripts
echo "📈 Moving evaluation scripts..."
mv -v evaluate_existing_model.py scripts/testing/ 2>/dev/null
mv -v evaluate_simple.py scripts/testing/ 2>/dev/null

# Move GPU validation scripts
echo "🔍 Moving GPU validation scripts..."
mv -v final_gpu_validation.py scripts/testing/ 2>/dev/null
mv -v quick_gpu_status.py scripts/testing/ 2>/dev/null

echo ""
echo "=" | head -c 80 | tr -d '\n'; echo
echo "✅ Organization complete!"
echo ""
echo "📂 New structure:"
echo "  docs/status/          - Training and status reports"
echo "  docs/analysis/        - Analysis and investigation docs"
echo "  docs/submissions/     - Submission checklists and guides"
echo "  submissions/versions/ - Old submission packages"
echo "  submissions/scripts/  - Submission helper scripts"
echo "  tests/validation/     - Validation test scripts"
echo "  scripts/testing/      - Testing and evaluation scripts"

