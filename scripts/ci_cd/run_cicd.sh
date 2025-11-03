#!/bin/bash
# CI/CD Pipeline for EEG2025 Submissions
# Validates and packages submissions with full checks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <submission_directory> [output_name]"
    echo "Example: $0 submissions/phase1_v14"
    echo "         $0 submissions/phase1_v14 phase1_v14_final"
    exit 1
fi

SUBMISSION_DIR=$1
OUTPUT_NAME=${2:-$(basename $SUBMISSION_DIR)}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

print_header "🧪 EEG2025 CI/CD Pipeline"
echo "�� Submission: $SUBMISSION_DIR"
echo "📦 Output: $OUTPUT_NAME"
echo ""

# Step 1: Validate submission
print_header "Step 1: Validation"
python3 "$SCRIPT_DIR/validate_submission.py" "$SUBMISSION_DIR"
VALIDATION_EXIT=$?

if [ $VALIDATION_EXIT -ne 0 ]; then
    print_error "Validation failed! Please fix errors before packaging."
    exit 1
fi

print_success "Validation passed!"
echo ""

# Step 2: Package submission
print_header "Step 2: Packaging"
python3 "$SCRIPT_DIR/package_submission.py" "$SUBMISSION_DIR" "$OUTPUT_NAME"
PACKAGE_EXIT=$?

if [ $PACKAGE_EXIT -ne 0 ]; then
    print_error "Packaging failed!"
    exit 1
fi

print_success "Packaging complete!"
echo ""

# Step 3: Final summary
print_header "🎉 CI/CD Pipeline Complete!"
echo ""
echo "📋 Summary:"
echo "  ✅ Validation: PASSED"
echo "  ✅ Packaging: COMPLETE"
echo "  📦 Output: $SUBMISSION_DIR/../$OUTPUT_NAME.zip"
echo ""
print_success "Submission is ready for upload to competition platform!"
echo ""

exit 0
