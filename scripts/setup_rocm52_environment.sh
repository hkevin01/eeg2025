#!/bin/bash
set -e

echo "🚀 Setting up optimal ROCm 5.2 environment for gfx1030"
echo "=" * 55

# Create working directory
mkdir -p /home/kevin/rocm52_setup
cd /home/kevin/rocm52_setup

echo "📋 Step 1: Create clean Python 3.10 environment"
# ROCm 5.2 works best with Python 3.10
python3.10 -m venv venv_rocm52
source venv_rocm52/bin/activate

echo "📦 Step 2: Install PyTorch 1.13.1+rocm5.2 (known working combination)"
# This is the confirmed working combination from our research
pip install --upgrade pip
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 --index-url https://download.pytorch.org/whl/rocm5.2

echo "🔧 Step 3: Install compatible NumPy and other dependencies"
# Fix NumPy compatibility issues
pip install "numpy<2.0" scipy scikit-learn matplotlib pandas

echo "🧪 Step 4: Install ML/EEG specific packages"
pip install mne einops timm tensorboard wandb

echo "📊 Step 5: Test basic functionality"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    # Test basic operations
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = x @ y
    print(f'Basic GPU ops: Working')
else:
    print('GPU not available')
"

echo "✅ Environment setup complete!"
echo "📁 Location: /home/kevin/rocm52_setup/venv_rocm52"
echo "🔄 To activate: source /home/kevin/rocm52_setup/venv_rocm52/bin/activate"
