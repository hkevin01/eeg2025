#!/bin/bash

echo "==================================================================="
echo "🔍 GPU Architecture Verification Script"
echo "==================================================================="
echo ""

echo "📊 Method 1: rocminfo (ISA Detection - AUTHORITATIVE)"
echo "-------------------------------------------------------------------"
ISA=$(rocminfo | grep "Name:.*gfx" | head -1 | awk '{print $2}')
echo "GPU ISA: $ISA"
echo ""

echo "📊 Method 2: rocm-smi (Hardware Detection)"
echo "-------------------------------------------------------------------"
rocm-smi --showproductname | grep -A 3 "GPU\[0\]"
echo ""

echo "📊 Method 3: PyTorch Detection (ROCm SDK 6.1.2)"
echo "-------------------------------------------------------------------"
if [ -f "/opt/rocm_sdk_612/bin/python3" ]; then
    /opt/rocm_sdk_612/bin/python3 -c "import torch; print('gcnArchName:', torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else 'GPU not available')"
else
    echo "ROCm SDK 6.1.2 not found"
fi
echo ""

echo "📊 Method 4: PyTorch Detection (venv_rocm622)"
echo "-------------------------------------------------------------------"
if [ -d "/home/kevin/Projects/eeg2025/venv_rocm622" ]; then
    cd /home/kevin/Projects/eeg2025
    bash -c 'unset PYTHONPATH; unset LD_LIBRARY_PATH; source venv_rocm622/bin/activate; python -c "import torch; print(\"gcnArchName:\", torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else \"GPU not available\")"'
else
    echo "venv_rocm622 not found"
fi
echo ""

echo "==================================================================="
echo "✅ CONCLUSION"
echo "==================================================================="
echo "The AUTHORITATIVE value is from rocminfo ISA: $ISA"
echo ""
echo "⚠️  IMPORTANT:"
echo "   - Hardware may report as 'Navi 10' or 'gfx1010'"
echo "   - But the ISA (what PyTorch uses) is: $ISA"
echo "   - Use $ISA in PYTORCH_ROCM_ARCH and ROCm SDK builds"
echo "==================================================================="
