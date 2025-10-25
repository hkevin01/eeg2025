#!/bin/bash

echo "🧹 Cleaning up root directory..."
echo ""

# Create documentation folder if it doesn't exist
mkdir -p docs/session_reports
mkdir -p docs/status_reports
mkdir -p docs/guides
mkdir -p scripts/training
mkdir -p scripts/monitoring
mkdir -p weights/challenge1
mkdir -p weights/challenge2
mkdir -p archive/old_training_scripts

# Move session reports to docs
echo "📄 Moving session reports..."
mv -v SESSION_COMPLETE_OCT24_FINAL.md docs/session_reports/ 2>/dev/null || true
mv -v SESSION_COMPLETE_OCT24.md docs/session_reports/ 2>/dev/null || true
mv -v FINAL_SUMMARY_OCT24.md docs/session_reports/ 2>/dev/null || true
mv -v FINAL_STATUS_OCT23.md docs/session_reports/ 2>/dev/null || true

# Move status reports to docs
echo "📊 Moving status reports..."
mv -v STATUS_OCT23_COMPLETE.md docs/status_reports/ 2>/dev/null || true
mv -v STATUS_UPDATED_OCT23.md docs/status_reports/ 2>/dev/null || true

# Move training documentation to docs
echo "📚 Moving training documentation..."
mv -v CHALLENGE1_IMPROVED_TRAINING.md docs/ 2>/dev/null || true
mv -v CHALLENGE1_IMPROVEMENT_PLAN.md docs/ 2>/dev/null || true
mv -v CHALLENGE1_TRAINING_ACTIVE.md docs/ 2>/dev/null || true
mv -v CHALLENGE1_TRAINING_COMPLETE.md docs/ 2>/dev/null || true
mv -v CHALLENGE2_TODO.md docs/ 2>/dev/null || true
mv -v TRAINING_PROGRESS_C1.md docs/ 2>/dev/null || true

# Move TODO files to docs
echo "✅ Moving TODO files..."
mv -v TODO_CHALLENGE1.md docs/ 2>/dev/null || true
mv -v TODO_SUBMISSION.md docs/ 2>/dev/null || true

# Move submission guides to docs
echo "📤 Moving submission documentation..."
mv -v SUBMISSION_PACKAGE_READY.md docs/guides/ 2>/dev/null || true
mv -v UPLOAD_CHECKLIST.md docs/guides/ 2>/dev/null || true
mv -v QUICK_REFERENCE.md docs/guides/ 2>/dev/null || true
mv -v READY_FOR_SUBMISSION.md docs/guides/ 2>/dev/null || true
mv -v GITIGNORE_UPDATE_COMPLETE.md docs/ 2>/dev/null || true

# Move training scripts to scripts
echo "🔧 Moving training scripts..."
mv -v train_challenge1_enhanced.py scripts/training/ 2>/dev/null || true
mv -v train_challenge1_simple.py scripts/training/ 2>/dev/null || true
mv -v train_challenge1_working.py scripts/training/ 2>/dev/null || true
mv -v train_challenge2_enhanced.py scripts/training/ 2>/dev/null || true
mv -v train_challenge2_hdf5_overnight.py scripts/training/ 2>/dev/null || true
mv -v train_challenge2_tonight.py scripts/training/ 2>/dev/null || true
mv -v train_universal.py scripts/training/ 2>/dev/null || true

# Keep train_challenge1_improved.py in root as it's the main one
echo "✨ Keeping train_challenge1_improved.py in root (main training script)"

# Move monitoring scripts
echo "📡 Moving monitoring scripts..."
mv -v monitor_c1_improved.sh scripts/monitoring/ 2>/dev/null || true
mv -v monitor_c1.sh scripts/monitoring/ 2>/dev/null || true
mv -v check_and_start_c1.sh scripts/monitoring/ 2>/dev/null || true
mv -v check_challenge1.sh scripts/monitoring/ 2>/dev/null || true
mv -v check_training.sh scripts/monitoring/ 2>/dev/null || true
mv -v start_challenge1.sh scripts/monitoring/ 2>/dev/null || true

# Move weights files to weights subdirectories
echo "⚖️  Moving weights files..."
mv -v weights_challenge_1_improved.pt weights/challenge1/ 2>/dev/null || true
mv -v weights_challenge_1.pt weights/challenge1/ 2>/dev/null || true
mv -v weights_challenge_2.pt weights/challenge2/ 2>/dev/null || true
mv -v weights_challenge_2_existing.pt weights/challenge2/ 2>/dev/null || true

# Keep submission files in root (they're needed for submission)
echo "📦 Keeping submission files in root:"
echo "  - submission.py"
echo "  - submission_improved.py"
echo "  - submission_eeg2025.zip"
echo "  - submission_final/"

echo ""
echo "✅ Root directory cleanup complete!"
echo ""
echo "📂 New structure:"
tree -L 2 -d docs/ scripts/ weights/ 2>/dev/null || ls -la docs/ scripts/ weights/

