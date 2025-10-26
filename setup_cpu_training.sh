#!/bin/bash
# Setup script for stable CPU training

echo "🧠 EEG2025 - Setting up CPU training environment"
echo "="*70

# Activate virtual environment (PyTorch works great on CPU too!)
source venv_pytorch28_rocm70/bin/activate

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Optimize CPU performance
export OMP_NUM_THREADS=12  # Use all Ryzen 5 3600 threads
export MKL_NUM_THREADS=12

echo "✅ Environment configured for CPU training"
echo ""
echo "📊 System Info:"
python3 << 'PYEOF'
import torch
print(f"   PyTorch: {torch.__version__}")
print(f"   CPU Threads: {torch.get_num_threads()}")
print(f"   Device: CPU (stable and reliable)")
PYEOF

echo ""
echo "🎯 Ready for EEG model training!"
echo "   • Stable and reliable"
echo "   • All PyTorch features work"
echo "   • Perfect for model development"
echo ""
echo "Usage: source setup_cpu_training.sh"
echo "       python your_training_script.py"
