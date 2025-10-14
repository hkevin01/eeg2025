#!/bin/bash
# Quick start script for downloading HBN-EEG dataset
# Part 2: Automated Data Acquisition

set -e  # Exit on error

echo "======================================================================="
echo "HBN-EEG Dataset Download Script"
echo "======================================================================="
echo

# Check for required tools
command -v aws >/dev/null 2>&1 || { 
    echo "❌ AWS CLI not installed. Install with: pip install awscli"
    exit 1
}

# Get data path from user or use default
read -p "Enter data directory path [/data/hbn-eeg]: " DATA_PATH
DATA_PATH=${DATA_PATH:-/data/hbn-eeg}

# Create directory
echo "Creating data directory: $DATA_PATH"
mkdir -p "$DATA_PATH"

# Ask for download mode
echo
echo "Download Options:"
echo "1) Quick Test (10 subjects, ~5GB) - Recommended for testing"
echo "2) Full Dataset (1500+ subjects, ~500GB) - For competition"
echo "3) Medium Set (100 subjects, ~50GB) - For development"
echo
read -p "Select option [1-3]: " OPTION

case $OPTION in
    1)
        echo "📦 Downloading test subset (10 subjects)..."
        
        # Download essential files first
        echo "Downloading metadata files..."
        aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG_BIDS/participants.tsv \
            "$DATA_PATH/participants.tsv" --no-sign-request
        aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG_BIDS/dataset_description.json \
            "$DATA_PATH/dataset_description.json" --no-sign-request
        
        # Download sample subjects
        SUBJECTS=(
            "NDARAA075AMK" "NDARAA948VFH" "NDARAB457VF4" "NDARAB582UM4" "NDARAC286UE8"
            "NDARAD121MJN" "NDARAD481FXF" "NDARAD533VF5" "NDARAD744EGJ" "NDARAE003GGV"
        )
        
        for sub_id in "${SUBJECTS[@]}"; do
            echo "Downloading sub-${sub_id}..."
            aws s3 sync "s3://fcp-indi/data/Projects/HBN/EEG_BIDS/sub-${sub_id}" \
                "$DATA_PATH/sub-${sub_id}" \
                --no-sign-request \
                --region us-east-1 \
                --quiet
        done
        
        echo "✅ Test subset downloaded successfully!"
        ;;
        
    2)
        echo "📦 Downloading full dataset (this will take several hours)..."
        aws s3 sync s3://fcp-indi/data/Projects/HBN/EEG_BIDS \
            "$DATA_PATH" \
            --no-sign-request \
            --region us-east-1
        
        echo "✅ Full dataset downloaded successfully!"
        ;;
        
    3)
        echo "📦 Downloading medium set (100 subjects)..."
        
        # Download metadata
        aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG_BIDS/participants.tsv \
            "$DATA_PATH/participants.tsv" --no-sign-request
        aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG_BIDS/dataset_description.json \
            "$DATA_PATH/dataset_description.json" --no-sign-request
        
        # Get list of first 100 subjects
        aws s3 ls s3://fcp-indi/data/Projects/HBN/EEG_BIDS/ --no-sign-request | \
            grep "sub-" | head -100 | while read -r line; do
                sub_dir=$(echo $line | awk '{print $2}' | tr -d '/')
                echo "Downloading ${sub_dir}..."
                aws s3 sync "s3://fcp-indi/data/Projects/HBN/EEG_BIDS/${sub_dir}" \
                    "$DATA_PATH/${sub_dir}" \
                    --no-sign-request \
                    --region us-east-1 \
                    --quiet
        done
        
        echo "✅ Medium set downloaded successfully!"
        ;;
        
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

# Set environment variable
echo
echo "Setting HBN_DATA_PATH environment variable..."
export HBN_DATA_PATH="$DATA_PATH"
echo "export HBN_DATA_PATH=\"$DATA_PATH\"" >> ~/.bashrc

# Verify download
echo
echo "Verifying download..."
SUBJECT_COUNT=$(find "$DATA_PATH" -maxdepth 1 -type d -name "sub-*" | wc -l)
EEG_COUNT=$(find "$DATA_PATH" -name "*_eeg.edf" -o -name "*_eeg.bdf" | wc -l)

echo "✅ Found $SUBJECT_COUNT subjects"
echo "✅ Found $EEG_COUNT EEG files"

# Create configuration file
echo
echo "Creating configuration file..."
cat > "$DATA_PATH/download_info.txt" << EOFINFO
HBN-EEG Dataset Download Information
=====================================
Download Date: $(date)
Download Option: $OPTION
Data Path: $DATA_PATH
Subjects: $SUBJECT_COUNT
EEG Files: $EEG_COUNT

Next Steps:
1. Run validation: python tests/test_data_acquisition.py
2. Update config: Set data_path in configs/enhanced.yaml
3. Test data loading: python scripts/dry_run.py
EOFINFO

cat "$DATA_PATH/download_info.txt"

echo
echo "======================================================================="
echo "✅ Download Complete!"
echo "======================================================================="
echo
echo "Next steps:"
echo "1. Run validation tests:"
echo "   cd /home/kevin/Projects/eeg2025"
echo "   export HBN_DATA_PATH=\"$DATA_PATH\""
echo "   python tests/test_data_acquisition.py"
echo
echo "2. Update your configuration:"
echo "   sed -i \"s|/path/to/hbn/data|$DATA_PATH|g\" configs/enhanced.yaml"
echo
echo "3. Test data loading:"
echo "   python scripts/dry_run.py --data_path $DATA_PATH"
echo
