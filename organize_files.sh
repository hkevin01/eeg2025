#!/bin/bash

echo "🗂️  Organizing project files..."

# Move old/completed documentation to archive
mv COMPETITION_RESULTS_ANALYSIS.md docs/archive/
mv CRITICAL_FIXES_LOG.md docs/archive/
mv CURRENT_STATUS_SUMMARY.md docs/archive/
mv FINAL_COMPLETION_SUMMARY.md docs/archive/
mv FINAL_STATUS_AND_TODO.md docs/archive/
mv FINAL_SUBMISSION_READY.md docs/archive/
mv GIT_SYNC_FIXED.md docs/archive/
mv IMPLEMENTATION_SUMMARY.md docs/archive/
mv MULTI_RELEASE_TRAINING_PLAN.md docs/archive/
mv READY_FOR_TRAINING.md docs/archive/
mv SUBMISSION_CHECKLIST.md docs/archive/
mv SUBMISSION_READY_STATUS.md docs/archive/
mv SUBMISSION_VERIFICATION.md docs/archive/
mv TODO_AND_STATUS.md docs/archive/
mv TRAINING_CRASH_FIX.md docs/archive/
mv TRAINING_IN_PROGRESS.md docs/archive/
mv TRAINING_STATUS.md docs/archive/
mv TRAINING_STATUS_UPDATE.md docs/archive/
mv UPLOAD_STEPS.md docs/archive/

# Keep current status docs in docs/
mv CRITICAL_ISSUE_VALIDATION.md docs/
mv FINAL_STATUS_FIXED.md docs/

# Move active plans to docs/plans
mv PHASE2_TASK_SPECIFIC_PLAN.md docs/plans/

# Move old weights to archive
mv weights_challenge_1.pt weights/archive/
mv weights_challenge_2.pt weights/archive/

# Keep current weights in weights/
mv weights_challenge_1_multi_release.pt weights/ 2>/dev/null || true
mv weights_challenge_2_multi_release.pt weights/ 2>/dev/null || true

echo "✅ Files organized!"
echo ""
echo "📁 Structure:"
echo "   docs/archive/     - Old/completed documentation"
echo "   docs/plans/       - Future implementation plans"
echo "   docs/             - Current status documents"
echo "   weights/          - Current model weights"
echo "   weights/archive/  - Old model weights"
echo "   ROOT/             - Active files (TODO.md, README.md, etc.)"

