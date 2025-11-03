#!/bin/bash
# CI/CD Pipeline for EEG Submission Validation and Packaging

set -e

SUBMISSION_DIR=$1
OUTPUT_NAME=${2:-submission}

if [ -z "$SUBMISSION_DIR" ]; then
    echo "Usage: $0 <submission_dir> [output_name]"
    exit 1
fi

echo "============================================================"
echo "🔍 EEG2025 Submission CI/CD Pipeline"
echo "============================================================"
echo "Submission: $SUBMISSION_DIR"
echo "Output: ${OUTPUT_NAME}.zip"
echo ""

# 1. Check structure
echo "📋 Step 1: Validating structure..."
if [ ! -f "$SUBMISSION_DIR/submission.py" ]; then
    echo "❌ Missing submission.py"
    exit 1
fi
if [ ! -f "$SUBMISSION_DIR/weights_challenge_1.pt" ]; then
    echo "❌ Missing weights_challenge_1.pt"
    exit 1
fi
if [ ! -f "$SUBMISSION_DIR/weights_challenge_2.pt" ]; then
    echo "❌ Missing weights_challenge_2.pt"
    exit 1
fi
echo "✅ All required files present"

# 2. Check file sizes
echo ""
echo "📦 Step 2: Checking file sizes..."
c1_size=$(stat -f%z "$SUBMISSION_DIR/weights_challenge_1.pt" 2>/dev/null || stat -c%s "$SUBMISSION_DIR/weights_challenge_1.pt")
c2_size=$(stat -f%z "$SUBMISSION_DIR/weights_challenge_2.pt" 2>/dev/null || stat -c%s "$SUBMISSION_DIR/weights_challenge_2.pt")
c1_mb=$((c1_size / 1024 / 1024))
c2_mb=$((c2_size / 1024 / 1024))

echo "  C1 weights: ${c1_mb} MB"
echo "  C2 weights: ${c2_mb} MB"

if [ $c1_mb -gt 100 ]; then
    echo "⚠️  Warning: C1 weights > 100MB"
fi
if [ $c2_mb -gt 100 ]; then
    echo "⚠️  Warning: C2 weights > 100MB"
fi
echo "✅ File sizes reasonable"

# 3. Test import
echo ""
echo "🧪 Step 3: Testing Python import..."
cd "$SUBMISSION_DIR"
python3 << 'EOFTEST'
import sys
try:
    import submission
    print("✅ submission.py imports successfully")
    
    # Check for required classes/methods
    if hasattr(submission, 'Submission'):
        print("✅ Submission class found")
    else:
        print("⚠️  No Submission class (may use different format)")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
EOFTEST
cd - > /dev/null

# 4. Create package
echo ""
echo "📦 Step 4: Creating submission package..."
PACKAGE_DIR="submissions_packaged"
mkdir -p "$PACKAGE_DIR"
cd $(dirname "$SUBMISSION_DIR")
zip -q -r "../${PACKAGE_DIR}/${OUTPUT_NAME}.zip" $(basename "$SUBMISSION_DIR")
cd - > /dev/null

ZIP_SIZE=$(stat -f%z "${PACKAGE_DIR}/${OUTPUT_NAME}.zip" 2>/dev/null || stat -c%s "${PACKAGE_DIR}/${OUTPUT_NAME}.zip")
ZIP_MB=$((ZIP_SIZE / 1024 / 1024))

echo "✅ Package created: ${PACKAGE_DIR}/${OUTPUT_NAME}.zip (${ZIP_MB} MB)"

# 5. Summary
echo ""
echo "============================================================"
echo "✅ CI/CD Pipeline PASSED"
echo "============================================================"
echo "📦 Submission package ready:"
echo "   Location: ${PACKAGE_DIR}/${OUTPUT_NAME}.zip"
echo "   Size: ${ZIP_MB} MB"
echo ""
echo "🚀 Ready to upload to competition platform!"
echo "============================================================"
