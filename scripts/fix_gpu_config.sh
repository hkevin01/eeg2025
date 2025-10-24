#!/bin/bash

echo "════════════════════════════════════════════════════════════════"
echo "  GPU CONFIGURATION FIX SCRIPT"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This script will fix the GPU misconfiguration by removing the"
echo "incorrect HSA_OVERRIDE_GFX_VERSION=10.3.0 setting from .bashrc"
echo ""
echo "Your GPU: AMD Radeon RX 5600 XT (gfx1010)"
echo "Current:  Forced to gfx1030 (WRONG!)"
echo "Fix:      Remove override to use native gfx1010 support"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Backup .bashrc
echo "1. Creating backup of ~/.bashrc..."
cp ~/.bashrc ~/.bashrc.backup_$(date +%Y%m%d_%H%M%S)
echo "   ✅ Backup created"
echo ""

# Show current problematic lines
echo "2. Current problematic lines in ~/.bashrc:"
echo "-------------------------------------------------------------------"
grep -n "HSA_OVERRIDE_GFX_VERSION\|HIP_VISIBLE_DEVICES" ~/.bashrc
echo "-------------------------------------------------------------------"
echo ""

# Ask for confirmation
read -p "Do you want to FIX this by commenting out these lines? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "3. Fixing ~/.bashrc..."
    
    # Comment out the problematic lines
    sed -i 's/^export HSA_OVERRIDE_GFX_VERSION=10.3.0/# export HSA_OVERRIDE_GFX_VERSION=10.3.0  # COMMENTED OUT - gfx1010 does not need this!/' ~/.bashrc
    sed -i 's/^export HIP_VISIBLE_DEVICES=0/# export HIP_VISIBLE_DEVICES=0  # COMMENTED OUT - not needed/' ~/.bashrc
    
    echo "   ✅ Fixed ~/.bashrc"
    echo ""
    
    echo "4. Verifying fix..."
    echo "-------------------------------------------------------------------"
    grep -n "HSA_OVERRIDE_GFX_VERSION\|HIP_VISIBLE_DEVICES" ~/.bashrc
    echo "-------------------------------------------------------------------"
    echo ""
    
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ FIX APPLIED!"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "NEXT STEPS:"
    echo ""
    echo "1. Open a NEW terminal (or run: source ~/.bashrc)"
    echo "2. Verify: echo \$HSA_OVERRIDE_GFX_VERSION  (should be empty)"
    echo "3. Test GPU: python3 tests/test_rocm_eegnex_gpu.py"
    echo "4. Start training: python3 scripts/training/train_challenge2_fast.py"
    echo ""
    echo "Your GPU should now work correctly with native gfx1010 support!"
    echo "════════════════════════════════════════════════════════════════"
else
    echo ""
    echo "❌ Fix cancelled. No changes made."
    echo ""
    echo "To fix manually:"
    echo "  1. Edit ~/.bashrc"
    echo "  2. Comment out or delete lines with HSA_OVERRIDE_GFX_VERSION"
    echo "  3. Save and run: source ~/.bashrc"
fi

