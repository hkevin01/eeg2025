#!/bin/bash
echo "════════════════════════════════════════════════════════════════"
echo "  GPU ARCHITECTURE DETECTION TEST"
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "=== 1. ROCm System Management Interface ==="
rocm-smi --showproductname | grep -E "GPU|GFX|Card"

echo ""
echo "=== 2. Detailed GPU Info from rocminfo ==="
rocminfo | grep -A 2 "Marketing Name.*Radeon"
rocminfo | grep "Name:.*gfx"

echo ""
echo "=== 3. HSA Agent Info ==="
rocminfo | grep -E "Name:|Marketing Name:|Uuid:" | head -20

echo ""
echo "=== 4. PyTorch Detection ==="
python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU detected by PyTorch: {device_name}")
    props = torch.cuda.get_device_properties(0)
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
PYEOF

echo ""
echo "=== 5. Environment Variables ==="
echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-Not set}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-Not set}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-Not set}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ANALYSIS"
echo "════════════════════════════════════════════════════════════════"

# Determine actual architecture
ACTUAL_ARCH=$(rocminfo | grep "Name:.*gfx" | head -1 | awk '{print $2}')
echo "Actual GPU Architecture: $ACTUAL_ARCH"

if [[ "$ACTUAL_ARCH" == "gfx1010" ]]; then
    echo "✅ GPU is gfx1010 (Navi 10 - Radeon RX 5600 XT)"
    echo ""
    echo "This is NOT gfx1030!"
    echo "gfx1010 has different memory architecture than gfx1030"
elif [[ "$ACTUAL_ARCH" == "gfx1030" ]]; then
    echo "⚠️  GPU is gfx1030 (Navi 21/22)"
else
    echo "❓ GPU architecture: $ACTUAL_ARCH"
fi

if [[ -n "$HSA_OVERRIDE_GFX_VERSION" ]]; then
    echo ""
    echo "⚠️  WARNING: HSA_OVERRIDE_GFX_VERSION is set to $HSA_OVERRIDE_GFX_VERSION"
    echo "   This may cause compatibility issues if it doesn't match actual hardware"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
