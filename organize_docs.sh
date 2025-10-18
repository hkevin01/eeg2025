#!/bin/bash
# Organize all markdown files into docs/ structure

echo "🧹 Organizing documentation files..."

# Status/Progress files
echo "📊 Moving status files..."
mv CHALLENGE2_TRAINING_STATUS.md docs/status/ 2>/dev/null
mv COMPETITION_TRAINING_STATUS.md docs/status/ 2>/dev/null
mv REAL_DATA_TRAINING_STATUS.md docs/status/ 2>/dev/null
mv TCN_TRAINING_COMPLETE.md docs/status/ 2>/dev/null
mv TRAINING_UPDATE.md docs/status/ 2>/dev/null
mv GPU_TRAINING_STATUS.md docs/status/ 2>/dev/null
mv TRAINING_STATUS*.md docs/status/ 2>/dev/null
mv *STATUS*.md docs/status/ 2>/dev/null
mv PROGRESS*.md docs/status/ 2>/dev/null
mv PHASE*.md docs/status/ 2>/dev/null

# Plans and todos
echo "📋 Moving planning files..."
mv TODO*.md docs/plans/ 2>/dev/null
mv *PLAN*.md docs/plans/ 2>/dev/null
mv *CHECKLIST*.md docs/plans/ 2>/dev/null
mv ACTION*.md docs/plans/ 2>/dev/null
mv ROADMAP*.md docs/plans/ 2>/dev/null
mv NEXT_STEPS.md docs/plans/ 2>/dev/null
mv IMPROVEMENT*.md docs/plans/ 2>/dev/null

# Submission related
echo "📦 Moving submission files..."
mv SUBMISSION*.md docs/submissions/ 2>/dev/null
mv *SUBMISSION*.md docs/submissions/ 2>/dev/null

# Methods and technical docs
echo "🔬 Moving methods files..."
mv METHODS*.md docs/methods/ 2>/dev/null
mv METHOD*.md docs/methods/ 2>/dev/null
mv *METHODS*.md docs/methods/ 2>/dev/null
mv COMPETITION_ANALYSIS.md docs/methods/ 2>/dev/null
mv ANSWERS_TO_YOUR_QUESTIONS.md docs/methods/ 2>/dev/null

# Archive old/completed files
echo "📁 Moving archive files..."
mv *COMPLETE*.md docs/archive/ 2>/dev/null
mv FINAL*.md docs/archive/ 2>/dev/null
mv *SUMMARY*.md docs/archive/ 2>/dev/null
mv MISSION_ACCOMPLISHED.txt docs/archive/ 2>/dev/null
mv SUCCESS.txt docs/archive/ 2>/dev/null
mv *ANALYSIS*.md docs/archive/ 2>/dev/null
mv IMPLEMENTATION*.md docs/archive/ 2>/dev/null
mv ORGANIZATION*.md docs/archive/ 2>/dev/null
mv EXTRACTION*.md docs/archive/ 2>/dev/null
mv OVERNIGHT*.md docs/archive/ 2>/dev/null
mv FRESH*.md docs/archive/ 2>/dev/null
mv INTEGRATED*.md docs/archive/ 2>/dev/null
mv SPARSE*.md docs/archive/ 2>/dev/null
mv ADVANCED*.md docs/archive/ 2>/dev/null
mv ROCM*.md docs/archive/ 2>/dev/null
mv GITIGNORE*.md docs/archive/ 2>/dev/null
mv GPU_USAGE_GUIDE.md docs/archive/ 2>/dev/null
mv VECTORIZED*.md docs/archive/ 2>/dev/null
mv P300*.md docs/archive/ 2>/dev/null
mv METADATA*.md docs/archive/ 2>/dev/null
mv TRAINING_FIX*.md docs/archive/ 2>/dev/null
mv SCORE*.md docs/archive/ 2>/dev/null

# Keep these in root
echo "✅ Keeping essential files in root..."
# README.md stays in root
# requirements*.txt stays in root
# MEMORY_BANK_COMPLETE.md stays in root (important for quick reference)

echo ""
echo "✅ Organization complete!"
echo ""
echo "📁 New structure:"
echo "  docs/status/      - Current training status"
echo "  docs/plans/       - TODO lists and plans"
echo "  docs/submissions/ - Submission documentation"
echo "  docs/methods/     - Technical methods"
echo "  docs/archive/     - Completed/historical docs"
echo ""
echo "Kept in root:"
ls -1 *.md *.txt 2>/dev/null | head -10

