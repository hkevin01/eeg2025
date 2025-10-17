#!/usr/bin/env python3
"""
Test to understand the numpy arange issue
"""
import numpy as np

print("Testing np.arange edge cases...")

# Test 1: Normal case
print("\n1. Normal case (0, 10, 1):")
try:
    result = np.arange(0, 10, 1)
    print(f"   ✅ Works: {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Stride 0
print("\n2. Stride 0 (0, 10, 0):")
try:
    result = np.arange(0, 10, 0)
    print(f"   ✅ Works: {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Negative stride with wrong direction
print("\n3. Negative stride wrong direction (0, 10, -1):")
try:
    result = np.arange(0, 10, -1)
    print(f"   ✅ Works: {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Float stride causing precision issues
print("\n4. Float stride (0.0, 10.0, 0.1):")
try:
    result = np.arange(0.0, 10.0, 0.1)
    print(f"   ✅ Works: length {len(result)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 5: Very small float stride
print("\n5. Very small float stride (0.0, 1.0, 0.001):")
try:
    result = np.arange(0.0, 1.0, 0.001)
    print(f"   ✅ Works: length {len(result)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 6: Zero-length range
print("\n6. Zero-length range (5.0, 5.0, 1.0):")
try:
    result = np.arange(5.0, 5.0, 1.0)
    print(f"   ✅ Works: {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 7: Infinity
print("\n7. Infinite stride (0.0, 10.0, float('inf')):")
try:
    result = np.arange(0.0, 10.0, float('inf'))
    print(f"   ✅ Works: {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 8: NaN
print("\n8. NaN stride (0.0, 10.0, float('nan')):")
try:
    result = np.arange(0.0, 10.0, float('nan'))
    print(f"   ✅ Works: {result}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
