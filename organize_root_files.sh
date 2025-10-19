#!/bin/bash

# Script to organize root directory files into proper structure

echo "🗂️  Organizing root directory files..."
echo ""

# Create archive subdirectories
mkdir -p archive/scripts/{monitoring,training,testing}
mkdir -p archive/docs/{status_reports,sessions,overnight,submission}
mkdir -p scripts/monitoring
mkdir -p scripts/training

# Files to keep in root (essential)
KEEP_FILES=(
    "README.md"
    "setup.py"
    "submission.py"
    "requirements.txt"
    "requirements-dev.txt"
    "pyproject.toml"
    "Makefile"
    "LICENSE"
)

# Current monitoring scripts (keep these active)
ACTIVE_MONITORING=(
    "watchdog_challenge2.sh"
    "manage_watchdog.sh"
    "monitor_challenge2.sh"
    "quick_training_status.sh"
)

# Current training scripts (keep these active)
ACTIVE_TRAINING=(
    "train_challenge2_correct.py"
)

# Current documentation (keep these active)
ACTIVE_DOCS=(
    "CHALLENGE2_TRAINING_STATUS.md"
    "WATCHDOG_QUICK_REFERENCE.md"
)

echo "📦 Moving old scripts to archive/scripts/..."

# Old monitoring scripts
for file in monitor_training.sh monitor_hybrid_training.sh check_training_simple.sh; do
    if [ -f "$file" ]; then
        mv "$file" archive/scripts/monitoring/
        echo "  ✓ $file → archive/scripts/monitoring/"
    fi
done

# Old training scripts
for file in train_and_validate_all.py train_attention_*.py train_challenge2_quick.py \
            train_in_tmux.sh train_safe_tmux.sh train_challenge2_tmux.sh; do
    if [ -f "$file" ]; then
        mv "$file" archive/scripts/training/
        echo "  ✓ $file → archive/scripts/training/"
    fi
done

# Old test scripts
for file in test_*.py evaluate_on_releases.py quick_model_test.py; do
    if [ -f "$file" ] && [[ ! " ${ACTIVE_TRAINING[@]} " =~ " ${file} " ]]; then
        # Skip test_submission_verbose.py and other active test files
        if [[ "$file" != "test_submission_verbose.py" ]]; then
            mv "$file" archive/scripts/testing/
            echo "  ✓ $file → archive/scripts/testing/"
        fi
    fi
done

# Other old Python files
for file in models_with_attention.py submission_tta.py test_hybrid_model.py; do
    if [ -f "$file" ]; then
        mv "$file" archive/scripts/
        echo "  ✓ $file → archive/scripts/"
    fi
done

echo ""
echo "📄 Moving old documentation to archive/docs/..."

# Status reports
for file in *STATUS*.md VALIDATION_RESULTS*.md MODELS_READY*.md; do
    if [ -f "$file" ] && [[ ! " ${ACTIVE_DOCS[@]} " =~ " ${file} " ]]; then
        mv "$file" archive/docs/status_reports/
        echo "  ✓ $file → archive/docs/status_reports/"
    fi
done

# Session documents
for file in SESSION*.md START_TRAINING_NOW.md TODO_NEXT_SESSION.md \
            QUICK_START*.md README_SESSION*.md; do
    if [ -f "$file" ]; then
        mv "$file" archive/docs/sessions/
        echo "  ✓ $file → archive/docs/sessions/"
    fi
done

# Overnight training docs
for file in OVERNIGHT*.md READY_TO_TRAIN_OVERNIGHT.md; do
    if [ -f "$file" ]; then
        mv "$file" archive/docs/overnight/
        echo "  ✓ $file → archive/docs/overnight/"
    fi
done

# Submission docs
for file in SUBMISSION_READY*.md; do
    if [ -f "$file" ]; then
        mv "$file" archive/docs/submission/
        echo "  ✓ $file → archive/docs/submission/"
    fi
done

# Other old docs
for file in CHANGES_SUMMARY.md DOCS_UPDATED*.md EEGNEX_ROCM_STRATEGY.md \
            HYBRID*.md QUICK_*.md TASK_SPECIFIC*.md TEAM_MEETING*.md \
            TRAINING_COMMANDS.md UPDATE_OCT18*.md README_FOR_MEETING.md; do
    if [ -f "$file" ] && [[ ! " ${ACTIVE_DOCS[@]} " =~ " ${file} " ]]; then
        # Skip QUICK_REFERENCE docs that might be active
        if [[ "$file" != "WATCHDOG_QUICK_REFERENCE.md" ]]; then
            mv "$file" archive/docs/
            echo "  ✓ $file → archive/docs/"
        fi
    fi
done

echo ""
echo "📁 Moving active scripts to scripts/ directory..."

# Move active monitoring scripts
for file in "${ACTIVE_MONITORING[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" scripts/monitoring/
        echo "  ✓ $file → scripts/monitoring/ (keeping root copy for now)"
    fi
done

# Move active training script
if [ -f "train_challenge2_correct.py" ]; then
    cp "train_challenge2_correct.py" scripts/training/
    echo "  ✓ train_challenge2_correct.py → scripts/training/ (keeping root copy for now)"
fi

echo ""
echo "✅ Organization complete!"
echo ""
echo "📊 Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Root directory cleaned - only essential files remain"
echo ""
echo "Archive structure:"
echo "  archive/scripts/monitoring/    - Old monitoring scripts"
echo "  archive/scripts/training/      - Old training scripts"
echo "  archive/scripts/testing/       - Old test scripts"
echo "  archive/docs/status_reports/   - Status report documents"
echo "  archive/docs/sessions/         - Session documents"
echo "  archive/docs/overnight/        - Overnight training docs"
echo "  archive/docs/submission/       - Submission docs"
echo "  archive/docs/                  - Other old documentation"
echo ""
echo "Active scripts copied to:"
echo "  scripts/monitoring/            - Current monitoring scripts"
echo "  scripts/training/              - Current training scripts"
echo ""
echo "Files remaining in root:"
ls -1 *.md *.py *.sh 2>/dev/null | head -20
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
