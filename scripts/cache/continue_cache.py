#!/usr/bin/env python3
"""Continue cache creation for remaining releases"""

import sys
sys.path.append('.')

from create_challenge2_cache import create_cache_for_release

print("Starting cache creation for R3, R4, R5...")
print("=" * 80)

releases = ['R3', 'R4', 'R5']

for release in releases:
    print(f"\n{'='*80}")
    print(f"Processing {release}")
    print(f"{'='*80}\n")
    
    try:
        create_cache_for_release(release, is_validation=(release == 'R5'))
        print(f"\n✅ {release} complete!")
    except Exception as e:
        print(f"\n❌ {release} failed: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*80)
print("✅ Cache creation complete!")
print("="*80)
