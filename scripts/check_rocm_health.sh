#!/bin/bash
# ROCm Health Check Script
# Comprehensive diagnostic for AMD GPU + ROCm + PyTorch setup

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          🔍 ROCm Health Check & Diagnostic Tool             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

check_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  SYSTEM INFORMATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Kernel: $(uname -r)"
echo "OS: $(lsb_release -d | cut -f2)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  GPU DETECTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if lspci | grep -i "VGA.*AMD" > /dev/null 2>&1; then
    check_pass "AMD GPU detected"
    lspci | grep -i "VGA.*AMD"
else
    check_fail "No AMD GPU found"
    exit 1
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  ROCm INSTALLATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v rocm-smi &> /dev/null; then
    check_pass "rocm-smi found"
    rocm-smi --showproductname 2>&1 | head -5
else
    check_fail "rocm-smi not found (ROCm not installed?)"
fi
echo ""

if command -v rocminfo &> /dev/null; then
    check_pass "rocminfo found"
    echo "Running rocminfo check..."
    if rocminfo 2>&1 | grep -q "HSA_STATUS_ERROR"; then
        check_fail "rocminfo reports HSA errors"
        echo "Error output:"
        rocminfo 2>&1 | grep "HSA_STATUS_ERROR"
    else
        check_pass "rocminfo runs without errors"
    fi
else
    check_warn "rocminfo not found"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  ENVIRONMENT VARIABLES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

vars=(
    "PYTORCH_ROCM_ARCH"
    "HSA_OVERRIDE_GFX_VERSION"
    "HIP_VISIBLE_DEVICES"
    "HSAKMT_DEBUG_LEVEL"
    "PYTORCH_NO_HIP_MEMORY_CACHING"
    "HIPBLAS_WORKSPACE_CONFIG"
)

for var in "${vars[@]}"; do
    value="${!var}"
    if [ -z "$value" ]; then
        check_warn "$var not set"
    else
        check_pass "$var=$value"
    fi
done
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  PYTORCH ROCm SUPPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'PYEOF'
import sys
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    if torch.version.hip:
        print(f"✅ ROCm/HIP version: {torch.version.hip}")
    else:
        print(f"❌ PyTorch not built with ROCm support")
        sys.exit(1)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA/ROCm device available")
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        print(f"   Device capability: {torch.cuda.get_device_capability(0)}")
    else:
        print(f"❌ No CUDA/ROCm device available to PyTorch")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error checking PyTorch: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    check_pass "PyTorch ROCm check completed"
else
    check_fail "PyTorch ROCm check failed"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  PCIE ATOMICS SUPPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if lspci -vv 2>/dev/null | grep -i "atomicop" > /dev/null 2>&1; then
    check_pass "PCIe AtomicOp support detected"
else
    check_warn "PCIe AtomicOp support not detected (may cause issues)"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7️⃣  IOMMU STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if dmesg 2>/dev/null | grep -i "iommu.*enabled" > /dev/null 2>&1; then
    check_pass "IOMMU enabled"
    dmesg | grep -i iommu | tail -3
else
    check_warn "IOMMU not enabled (consider enabling for better address translation)"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8️⃣  SIMPLE TENSOR TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'PYEOF'
import torch
import sys

print("Attempting basic GPU tensor operations...")

try:
    # Small tensor test
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = x @ y
    result = z.cpu().sum().item()
    print(f"✅ Small tensor test PASS (result: {result:.4f})")
    
    # Medium tensor test
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = x @ y
    result = z.cpu().sum().item()
    print(f"✅ Medium tensor test PASS (result: {result:.4f})")
    
    print("✅ All tensor tests completed successfully")
    
except RuntimeError as e:
    error_msg = str(e)
    print(f"❌ Tensor test FAILED: {error_msg}")
    if "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION" in error_msg:
        print("⚠️  This is the known ROCm memory aperture violation!")
        print("   See docs/rocm_troubleshooting.md for mitigation strategies")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    check_pass "Tensor operations successful"
else
    check_fail "Tensor operations failed (see above)"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "9️⃣  RECOMMENDATIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "📚 For detailed troubleshooting, see:"
echo "   docs/rocm_troubleshooting.md"
echo ""
echo "🔧 Recommended environment for RX 5600 XT:"
echo "   export PYTORCH_ROCM_ARCH=gfx1010"
echo "   export HSA_OVERRIDE_GFX_VERSION=10.1.0"
echo "   export HIPBLAS_WORKSPACE_CONFIG=:0:0"
echo ""
echo "🚀 Training recommendations:"
echo "   - Start with batch_size=8, num_workers=1"
echo "   - Disable AMP if issues persist"
echo "   - Use CPU fallback mode (already implemented)"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Health Check Complete                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
