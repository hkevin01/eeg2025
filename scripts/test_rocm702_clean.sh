#!/bin/bash
# Completely isolated ROCm 7.0.2 test

# Clear ALL SDK environment
for var in $(env | grep -i rocm | cut -d= -f1); do
    unset $var
done
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset PATH

# Set minimal clean environment
export PATH=/opt/rocm-7.0.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export ROCM_PATH=/opt/rocm-7.0.2
export LD_LIBRARY_PATH=/opt/rocm-7.0.2/lib
export PYTORCH_ROCM_ARCH=gfx1030
export HIP_VISIBLE_DEVICES=0

# Activate venv
cd /home/kevin/Projects/eeg2025
source venv_rocm702/bin/activate

echo "=== ROCm 7.0.2 Clean Environment Test ==="
echo "ROCM_PATH: $ROCM_PATH"
echo "Python: $(which python)"
echo ""

python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ Device: {torch.cuda.get_device_name(0)}')
    print(f'✅ Architecture: {torch.cuda.get_device_properties(0).gcnArchName}')
    print(f'✅ Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"
