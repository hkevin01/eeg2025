#!/bin/bash
echo "════════════════════════════════════════════════════════════════"
echo "  CHECKING REAL GPU ARCHITECTURE (No Overrides)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Temporarily unset override
export HSA_OVERRIDE_GFX_VERSION=""

echo "1. Hardware Detection (lspci):"
echo "-------------------------------------------------------------------"
lspci -vnn | grep -A 10 "VGA\|Display" | grep -E "Subsystem|Kernel driver"
echo ""

echo "2. ROCm Device ID:"
echo "-------------------------------------------------------------------"
rocminfo | grep -A 20 "Marketing Name.*5600" | grep -E "Name:|Chip ID:|ASIC"
echo ""

echo "3. AMD GPU PRO Info (if available):"
echo "-------------------------------------------------------------------"
if [ -f "/sys/class/drm/card0/device/product_name" ]; then
    cat /sys/class/drm/card0/device/product_name
fi
if [ -f "/sys/class/drm/card1/device/product_name" ]; then
    cat /sys/class/drm/card1/device/product_name
fi
echo ""

echo "4. Chip ID to Architecture Mapping:"
echo "-------------------------------------------------------------------"
CHIP_ID=$(rocminfo | grep "Chip ID" | grep -A 5 "5600" | grep "Chip ID" | awk '{print $3}' | tr -d '()')
echo "Chip ID: $CHIP_ID"
echo ""

case "$CHIP_ID" in
    "0x731f")
        echo "✅ Chip 0x731f = Navi 10 = gfx1010"
        echo "   This is RX 5600/5700 series (RDNA 1.0)"
        CORRECT_ARCH="gfx1010"
        ;;
    "0x73*")
        echo "✅ Chip 0x73xx = Navi 2x = gfx1030"  
        echo "   This is RX 6600/6700 series (RDNA 2.0)"
        CORRECT_ARCH="gfx1030"
        ;;
    *)
        echo "❓ Unknown chip ID: $CHIP_ID"
        CORRECT_ARCH="unknown"
        ;;
esac
echo ""

echo "5. ROCm Reported Architecture:"
echo "-------------------------------------------------------------------"
ROCM_ARCH=$(rocminfo | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')
echo "ROCm reports: $ROCM_ARCH"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  ANALYSIS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Hardware Chip ID:     $CHIP_ID"
echo "Correct Architecture: $CORRECT_ARCH"
echo "ROCm Reports:         $ROCM_ARCH"
echo ""

if [ "$CORRECT_ARCH" != "$ROCM_ARCH" ]; then
    echo "⚠️  MISMATCH DETECTED!"
    echo ""
    echo "Your hardware is actually $CORRECT_ARCH, but ROCm reports $ROCM_ARCH"
    echo ""
    echo "POSSIBLE CAUSES:"
    echo "  1. HSA_OVERRIDE_GFX_VERSION environment variable is set incorrectly"
    echo "  2. ROCm is misidentifying the GPU"
    echo "  3. Custom SDK was built with wrong target"
    echo ""
    echo "SOLUTION:"
    if [ "$CORRECT_ARCH" = "gfx1010" ]; then
        echo "  ✅ Your GPU (gfx1010) is NATIVELY SUPPORTED in ROCm 6.2"
        echo "  ✅ You DO NOT need HSA_OVERRIDE_GFX_VERSION"
        echo "  ✅ You DO NOT need a custom SDK"
        echo ""
        echo "  ACTION REQUIRED:"
        echo "    1. Unset HSA_OVERRIDE_GFX_VERSION"
        echo "    2. Use system ROCm (/opt/rocm)"
        echo "    3. GPU training should work!"
    else
        echo "  ⚠️  Your GPU ($CORRECT_ARCH) needs custom SDK"
        echo "  ⚠️  Rebuild SDK with correct -DAMDGPU_TARGETS=$CORRECT_ARCH"
    fi
else
    echo "✅ ROCm correctly identifies your GPU as $CORRECT_ARCH"
fi

echo "════════════════════════════════════════════════════════════════"
