#!/bin/bash

# Top 3 Challenge - SSL Training Pipeline
# This script starts self-supervised pre-training followed by fine-tuning

set -e  # Exit on error

echo "================================================================================"
echo "🚀 Top 3 Challenge - Self-Supervised Learning Pipeline"
echo "================================================================================"
echo ""
echo "Current Position: #65 (1.00613 overall)"
echo "Target: Top 3 (< 0.975 overall)"
echo "Strategy: Hybrid Supervised + Unsupervised Learning"
echo ""
echo "================================================================================"
echo ""

# Check data exists
if [ ! -d "data/processed_tuab_challenge_1" ]; then
    echo "❌ ERROR: data/processed_tuab_challenge_1 not found!"
    echo "Please ensure Challenge 1 data is available."
    exit 1
fi

# Count data files
n_files=$(ls -1 data/processed_tuab_challenge_1/*.h5 2>/dev/null | wc -l)
echo "📦 Found $n_files H5 data files"

# Create checkpoints directory
mkdir -p checkpoints

# Check GPU availability
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ GPU available"
    gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('No GPU')")
    echo "   Using: $gpu_name"
else
    echo "⚠️  WARNING: No GPU found, training will be SLOW"
    echo "   Consider using a GPU for faster training"
fi

echo ""
echo "================================================================================"
echo "PHASE 1: Self-Supervised Pre-training (SimCLR)"
echo "================================================================================"
echo "Expected time: 4-6 hours"
echo "Output: Pre-trained encoder with strong EEG representations"
echo ""

read -p "Start SSL pre-training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting SSL pre-training..."
    python3 ssl_pretrain_c1.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ SSL pre-training complete!"
        echo "   Checkpoint saved to: checkpoints/ssl_encoder_final.pt"
        echo ""
    else
        echo ""
        echo "❌ SSL pre-training failed!"
        exit 1
    fi
else
    echo "⏭️  Skipping SSL pre-training"
    
    # Check if pre-trained encoder exists
    if [ ! -f "checkpoints/ssl_encoder_final.pt" ]; then
        echo "❌ ERROR: No pre-trained encoder found!"
        echo "Cannot proceed to fine-tuning without SSL pre-training."
        exit 1
    fi
fi

echo ""
echo "================================================================================"
echo "PHASE 2: Progressive Fine-tuning"
echo "================================================================================"
echo "Expected time: 2-3 hours"
echo "Output: Fine-tuned model for age prediction"
echo ""

read -p "Start progressive fine-tuning? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "�� Starting progressive fine-tuning..."
    python3 finetune_ssl_c1.py --phase all
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Fine-tuning complete!"
        echo "   Final model saved to: checkpoints/ssl_finetuned_final.pt"
        echo ""
    else
        echo ""
        echo "❌ Fine-tuning failed!"
        exit 1
    fi
else
    echo "⏭️  Skipping fine-tuning"
fi

echo ""
echo "================================================================================"
echo "✅ SSL Training Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Check training logs for Val NRMSE"
echo "2. Compare with V8 baseline (Val NRMSE: 0.160418)"
echo "3. If better, create submission with ssl_finetuned_final.pt"
echo "4. Submit to competition and check test scores"
echo ""
echo "Target: Val NRMSE < 0.135 to potentially beat top teams"
echo ""
