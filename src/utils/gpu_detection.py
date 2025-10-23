"""
GPU Compatibility Detection for EEGNeX Training
===============================================
Detects AMD gfx1030 (RX 5600 XT) which has ROCm issues with EEGNeX spatial convolution.
"""

import os
import torch


def is_problematic_amd_gpu() -> tuple[bool, str]:
    """Detect AMD gfx1030 (RX 5600 XT) on ROCm which is known to crash with EEGNeX spatial conv.

    Returns (is_problematic, reason)
    """
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        name = torch.cuda.get_device_name(0)
        # Heuristics: RX 5600 XT, gfx1030 strings, or env overrides indicating gfx1030
        env_arch = os.environ.get("PYTORCH_ROCM_ARCH", "")
        hsa_override = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
        
        is_amd = any(x in name.lower() for x in ["amd", "radeon", "rx", "5600"]) or "gfx" in env_arch.lower() or "gfx" in hsa_override.lower()
        is_gfx1030 = ("gfx1030" in env_arch.lower()) or ("10.3.0" in hsa_override) or ("5600" in name.lower())
        
        if is_amd and is_gfx1030:
            return True, f"Detected AMD GPU '{name}' with ROCm arch/env ({env_arch or hsa_override}), known to crash with EEGNeX"
        
        # Additional conservative safeguard: RDNA2 gaming GPUs often report 'gfx10'
        if is_amd and ("gfx10" in env_arch.lower() or "10." in hsa_override):
            return True, f"Detected AMD RDNA GPU ('{name}', arch={env_arch or hsa_override}), EEGNeX may crash on ROCm"
    except Exception as _:
        pass
    
    return False, ""


def apply_gfx1030_safeguard(prefer_gpu: bool, force_unsafe: bool = False) -> bool:
    """Apply gfx1030 safeguard to prevent EEGNeX crashes.
    
    Args:
        prefer_gpu: Original GPU preference
        force_unsafe: Override safeguard (--force-gpu-unsafe)
        
    Returns:
        Updated GPU preference (False if safeguard triggered)
    """
    if not prefer_gpu:
        return prefer_gpu
    
    problem_amd, amd_reason = is_problematic_amd_gpu()
    if problem_amd and not force_unsafe:
        print(f"⚠️  GPU disabled due to known ROCm issue on this AMD GPU: {amd_reason}")
        print("   Use --force-gpu-unsafe to override (may crash with memory aperture violation).")
        return False
    
    return prefer_gpu


if __name__ == "__main__":
    print("GPU Compatibility Detection")
    print("=" * 40)
    
    flag, reason = is_problematic_amd_gpu()
    print(f"Problematic AMD detected: {flag}")
    if flag:
        print(f"Reason: {reason}")
    
    # Test safeguard
    print("\nTesting safeguard:")
    result = apply_gfx1030_safeguard(True, False)
    print(f"GPU preference after safeguard: {result}")
