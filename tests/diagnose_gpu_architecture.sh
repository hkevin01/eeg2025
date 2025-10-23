#!/bin/bash
echo "════════════════════════════════════════════════════════════════"
echo "  COMPREHENSIVE GPU ARCHITECTURE DIAGNOSIS"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "1. ROCm SMI Output:"
echo "-------------------------------------------------------------------"
rocm-smi --showproductname 2>/dev/null | grep -A 3 "GPU\[0\]"
echo ""

echo "2. ROCmInfo GFX Version:"
echo "-------------------------------------------------------------------"
rocminfo 2>/dev/null | grep -E "Name:|gfx" | head -6
echo ""

echo "3. GPU Device ID:"
echo "-------------------------------------------------------------------"
lspci | grep -i "VGA\|Display\|3D"
echo ""

echo "4. ROCm Agent Enumerator:"
echo "-------------------------------------------------------------------"
/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null || echo "Not available"
echo ""

echo "5. PyTorch Detection:"
echo "-------------------------------------------------------------------"
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'gcnArchName: {props.gcnArchName}')
else:
    print('CUDA/ROCm not available')
" 2>/dev/null
echo ""

echo "6. HSA Environment Variables:"
echo "-------------------------------------------------------------------"
echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-Not set}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-Not set}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-Not set}"
echo ""

echo "7. Checking /opt/rocm_sdk_612:"
echo "-------------------------------------------------------------------"
if [ -d "/opt/rocm_sdk_612" ]; then
    echo "✅ Custom SDK exists at /opt/rocm_sdk_612"
    ls -lh /opt/rocm_sdk_612/ | head -10
    echo ""
    if [ -f "/opt/rocm_sdk_612/bin/rocm_agent_enumerator" ]; then
        echo "SDK Agent Enumerator:"
        /opt/rocm_sdk_612/bin/rocm_agent_enumerator 2>/dev/null || echo "Failed"
    fi
else
    echo "❌ Custom SDK not found at /opt/rocm_sdk_612"
fi
echo ""

echo "8. Current ROCm Installation:"
echo "-------------------------------------------------------------------"
ls -lh /opt/rocm* 2>/dev/null | head -5
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  CONCLUSION"
echo "════════════════════════════════════════════════════════════════"

# Determine actual architecture
ACTUAL_GFX=$(rocminfo 2>/dev/null | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')
echo "Actual GPU Architecture: ${ACTUAL_GFX}"

if [ "$ACTUAL_GFX" = "gfx1010" ]; then
    echo "✅ GPU is gfx1010 (Navi 10 - RX 5600 XT)"
    echo ""
    echo "RECOMMENDATION:"
    echo "  - gfx1010 is well-supported in ROCm 6.2"
    echo "  - The memory aperture violation is likely a software issue"
    echo "  - Custom SDK may help, but needs proper environment setup"
elif [ "$ACTUAL_GFX" = "gfx1030" ]; then
    echo "⚠️  GPU is gfx1030 (Navi 21/22)"
    echo ""
    echo "RECOMMENDATION:"
    echo "  - gfx1030 requires HSA_OVERRIDE_GFX_VERSION=10.3.0"
    echo "  - Custom SDK compilation is REQUIRED for stable operation"
    echo "  - Consider using CPU for competition deadline"
else
    echo "❓ Unknown architecture: $ACTUAL_GFX"
fi

echo "════════════════════════════════════════════════════════════════"
