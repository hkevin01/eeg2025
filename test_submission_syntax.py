#!/usr/bin/env python3
"""
Syntax validation test for submission_sam_fixed_v3.py
Tests file without actually running braindecode imports
"""

import ast
import sys

def validate_submission(filename):
    """Validate Python syntax and check for problematic imports"""
    
    print("\n" + "=" * 70)
    print(f"VALIDATING: {filename}")
    print("=" * 70)
    
    # Read file
    with open(filename, 'r') as f:
        content = f.read()
    
    # 1. Check syntax
    print("\n1. Checking Python syntax...")
    try:
        ast.parse(content)
        print("   ✅ Valid Python syntax")
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    
    # 2. Check for problematic imports
    print("\n2. Checking for problematic imports...")
    problematic = {
        'import pip': 'Deprecated pip introspection',
        'pip.get_installed_distributions': 'Removed in pip 10.0+',
        'pip._internal': 'Private API, unstable',
    }
    
    found_issues = []
    for pattern, reason in problematic.items():
        if pattern in content:
            found_issues.append((pattern, reason))
            print(f"   ❌ Found: {pattern} - {reason}")
    
    if not found_issues:
        print("   ✅ No problematic imports found")
    
    # 3. Check for required components
    print("\n3. Checking required components...")
    required = {
        'class Submission': 'Main submission class',
        'def challenge_1': 'Challenge 1 method',
        'def challenge_2': 'Challenge 2 method',
        'def resolve_path': 'Path resolution function',
        'from braindecode.models import EEGNeX': 'EEGNeX import',
    }
    
    all_present = True
    for component, description in required.items():
        if component in content:
            print(f"   ✅ Found: {description}")
        else:
            print(f"   ❌ Missing: {description}")
            all_present = False
    
    # 4. Count lines
    print("\n4. File statistics...")
    lines = content.split('\n')
    print(f"   Lines: {len(lines)}")
    print(f"   Size: {len(content)} bytes")
    
    # 5. Check weight file references
    print("\n5. Checking weight file references...")
    weight_files = [
        'weights_challenge_1_sam.pt',
        'weights_challenge_2_sam.pt',
    ]
    
    for wf in weight_files:
        if wf in content:
            print(f"   ✅ References: {wf}")
        else:
            print(f"   ⚠️  Missing reference: {wf}")
    
    # Summary
    print("\n" + "=" * 70)
    if not found_issues and all_present:
        print("RESULT: ✅ VALIDATION PASSED")
        print("=" * 70)
        return True
    else:
        print("RESULT: ❌ VALIDATION FAILED")
        print("=" * 70)
        return False

if __name__ == '__main__':
    result = validate_submission('submission_sam_fixed_v3.py')
    sys.exit(0 if result else 1)
