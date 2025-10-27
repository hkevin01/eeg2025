#!/bin/bash

# Root Directory Cleanup Script
# Organizes files from root into appropriate subfolders

set -e

echo "========================================"
echo "ROOT DIRECTORY CLEANUP"
echo "========================================"

# Create destination directories if they don't exist
mkdir -p docs/status_reports
mkdir -p docs/analysis
mkdir -p scripts/submission_builders
mkdir -p submissions/v9_ensemble
mkdir -p submissions/v8_tcn
mkdir -p submissions/v7_sam
mkdir -p submissions/old
mkdir -p weights/ensemble
mkdir -p weights/single_models
mkdir -p weights/challenge_specific

echo ""
echo "📁 Moving status reports to docs/status_reports/..."
mv -v CURRENT_STATUS_OCT26_6PM.md docs/status_reports/ 2>/dev/null || true
mv -v ENSEMBLE_TRAINING_CHECKLIST.md docs/status_reports/ 2>/dev/null || true
mv -v ENSEMBLE_TRAINING_COMPLETE.md docs/status_reports/ 2>/dev/null || true
mv -v FINAL_SUBMISSION_CHECKLIST.md docs/status_reports/ 2>/dev/null || true
mv -v TRAINING_READY_OCT26.md docs/status_reports/ 2>/dev/null || true
mv -v SUBMISSION_V7_READY.md docs/status_reports/ 2>/dev/null || true
mv -v SUBMISSION_V8_TCN_READY.md docs/status_reports/ 2>/dev/null || true

echo ""
echo "📊 Moving analysis documents to docs/analysis/..."
mv -v NEXT_SUBMISSION_PLAN.md docs/analysis/ 2>/dev/null || true
mv -v PATH_B_ENSEMBLE_ANALYSIS.md docs/analysis/ 2>/dev/null || true

echo ""
echo "🔨 Moving submission builder scripts to scripts/submission_builders/..."
mv -v create_ensemble_submission.py scripts/submission_builders/ 2>/dev/null || true
mv -v create_ensemble_weights_fixed.py scripts/submission_builders/ 2>/dev/null || true
mv -v create_ensemble_weights.py scripts/submission_builders/ 2>/dev/null || true
mv -v create_single_best_submission.py scripts/submission_builders/ 2>/dev/null || true

echo ""
echo "📝 Moving v9 ensemble files to submissions/v9_ensemble/..."
mv -v submission_v9_ensemble_final.zip submissions/v9_ensemble/ 2>/dev/null || true
mv -v submission_v9_single_best.zip submissions/v9_ensemble/ 2>/dev/null || true
mv -v submission_v9_ensemble.py submissions/v9_ensemble/ 2>/dev/null || true
mv -v submission_v9_ensemble_simple.py submissions/v9_ensemble/ 2>/dev/null || true
mv -v submission.py submissions/v9_ensemble/ 2>/dev/null || true
mv -v submission_single.py submissions/v9_ensemble/ 2>/dev/null || true

echo ""
echo "📝 Moving v8 TCN files to submissions/v8_tcn/..."
mv -v submission_tcn_v8.zip submissions/v8_tcn/ 2>/dev/null || true
mv -v submission_v8_tcn.py submissions/v8_tcn/ 2>/dev/null || true

echo ""
echo "📝 Moving v7 SAM files to submissions/v7_sam/..."
mv -v submission_sam_fixed_v5.zip submissions/v7_sam/ 2>/dev/null || true
mv -v submission_sam_fixed_v6.zip submissions/v7_sam/ 2>/dev/null || true
mv -v submission_sam_fixed_v7.zip submissions/v7_sam/ 2>/dev/null || true

echo ""
echo "📝 Moving old submission scripts to submissions/old/..."
mv -v submission_v6_correct.py submissions/old/ 2>/dev/null || true
mv -v submission_v6_robust.py submissions/old/ 2>/dev/null || true
mv -v submission_v7_class_format.py submissions/old/ 2>/dev/null || true

echo ""
echo "⚖️  Moving ensemble weights to weights/ensemble/..."
mv -v weights_c1_ensemble.pt weights/ensemble/ 2>/dev/null || true
mv -v weights_challenge_1.pt weights/ensemble/ 2>/dev/null || true

echo ""
echo "⚖️  Moving single model weights to weights/single_models/..."
mv -v weights_c1_compact.pt weights/single_models/ 2>/dev/null || true
mv -v weights_c1_tcn.pt weights/single_models/ 2>/dev/null || true
mv -v weights_challenge_1_single.pt weights/single_models/ 2>/dev/null || true

echo ""
echo "⚖️  Moving challenge-specific weights to weights/challenge_specific/..."
mv -v weights_challenge_1_sam.pt weights/challenge_specific/ 2>/dev/null || true
mv -v weights_challenge_2.pt weights/challenge_specific/ 2>/dev/null || true
mv -v weights_challenge_2_sam.pt weights/challenge_specific/ 2>/dev/null || true
mv -v weights_challenge_2_single.pt weights/challenge_specific/ 2>/dev/null || true

echo ""
echo "========================================"
echo "✅ CLEANUP COMPLETE!"
echo "========================================"

echo ""
echo "📂 New structure:"
echo "  docs/status_reports/      - Status and progress reports"
echo "  docs/analysis/            - Analysis documents"
echo "  scripts/submission_builders/ - Submission creation scripts"
echo "  submissions/v9_ensemble/  - v9 ensemble submissions"
echo "  submissions/v8_tcn/       - v8 TCN submissions"
echo "  submissions/v7_sam/       - v7 SAM submissions"
echo "  submissions/old/          - Old submission scripts"
echo "  weights/ensemble/         - Ensemble model weights"
echo "  weights/single_models/    - Single model weights"
echo "  weights/challenge_specific/ - Challenge-specific weights"

echo ""
echo "🧹 Root directory cleaned!"

