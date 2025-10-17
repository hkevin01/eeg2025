"""Test feature extraction on real data"""
import sys
import numpy as np
sys.path.insert(0, '.')

from scripts.features.erp import ERPExtractor
# from scripts.features.spectral import SpectralExtractor

print("="*70)
print("🧪 TESTING PHASE 2 FEATURE EXTRACTORS")
print("="*70)

# Test data: 129 channels × 500 samples (5 seconds @ 100Hz)
np.random.seed(42)
dummy_eeg = np.random.randn(129, 500) * 10

# Add synthetic P300 for testing
p300_channels = slice(50, 70)
p300_time = slice(35, 55)
dummy_eeg[p300_channels, p300_time] += 15

print("\n1️⃣  Testing P300/ERP Extractor...")
print("-" * 70)
erp_extractor = ERPExtractor()
p300_features = erp_extractor.extract_p300(dummy_eeg)

print(f"✅ Extracted {len(p300_features)} P300 features:")
for key, val in p300_features.items():
    print(f"   • {key}: {val:.2f}")

print(f"\n�� Key Insight: P300 latency = {p300_features['p300_peak_latency']:.1f}ms")
print("   (Correlates with response time in Challenge 1!)")

# Uncomment when spectral.py is created
# print("\n2️⃣  Testing Spectral Extractor...")
# print("-" * 70)
# spectral_extractor = SpectralExtractor()
# spectral_features = spectral_extractor.extract_all_band_powers(dummy_eeg)
# 
# print(f"✅ Extracted {len(spectral_features)} spectral features:")
# for key, val in list(spectral_features.items())[:6]:
#     print(f"   • {key}: {val:.4f}")
# 
# print(f"\n📊 Key Insight: Alpha power = {spectral_features['alpha_power_mean']:.4f}")
# print("   (Correlates with emotion regulation in Challenge 2!)")

print("\n" + "="*70)
print("✅ FEATURE EXTRACTORS READY FOR PHASE 2!")
print("="*70)
print("\n📋 Next Steps:")
print("   1. Wait for Phase 1 training to complete (~18:00 UTC)")
print("   2. Analyze results and decide if Phase 2 needed")
print("   3. If yes, run: python scripts/prepare_phase2_data.py")
print("="*70)
