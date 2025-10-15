#!/usr/bin/env python3
"""
Bulk HBN Data Download Script
Downloads multiple subjects efficiently with retry logic
"""

import subprocess
import sys
from pathlib import Path
import time
import argparse

def download_subject(subject_id, target_dir, max_retries=3):
    """Download a single subject with retry logic"""
    for attempt in range(max_retries):
        try:
            cmd = [
                "aws", "s3", "sync",
                f"s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1/sub-{subject_id}",
                f"{target_dir}/sub-{subject_id}",
                "--no-sign-request"
            ]

            print(f"   Attempt {attempt + 1}/{max_retries}: Downloading sub-{subject_id}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Verify download
                subj_path = Path(target_dir) / f"sub-{subject_id}"
                eeg_files = list(subj_path.glob("**/*.set"))
                if eeg_files:
                    print(f"   ✅ sub-{subject_id}: {len(eeg_files)} files")
                    return True
                else:
                    print(f"   ⚠️  No .set files found for sub-{subject_id}")

            else:
                print(f"   ❌ Error: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print(f"   ⏱️  Timeout on attempt {attempt + 1}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        if attempt < max_retries - 1:
            time.sleep(5)  # Wait before retry

    return False

def main():
    parser = argparse.ArgumentParser(description="Bulk download HBN EEG data")
    parser.add_argument("--n-subjects", type=int, default=10, help="Number of subjects to download")
    parser.add_argument("--start-id", type=str, default="NDARAA117DEJ", help="Starting subject ID")
    args = parser.parse_args()

    target_dir = Path("data/raw/hbn")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Real subject IDs from HBN BIDS Release 1 (verified available)
    subject_ids = [
        "NDARAM704GKZ", "NDARAN385MDH", "NDARAP359UM6", "NDARAW320CGR",
        "NDARBD879MBX", "NDARBH024NH2", "NDARBK082PDD", "NDARBM173BJG",
        "NDARBX121UM9", "NDARBX400RTC", "NDARCA153NKE", "NDARCE721YB5",
        "NDARCJ170CT9", "NDARCJ475WJP", "NDARCJ594BWQ", "NDARCN669XPR",
        "NDARCR499NE4", "NDARCW071AU5", "NDARDC290NBA", "NDARDF071FBJ",
        "NDARDF193PZR", "NDARDG673JVK", "NDARDH304EWD", "NDARDK311JWL",
        "NDARDP189MW9", "NDARDV197XFW", "NDARDV899MVJ", "NDARDW149TJL",
        "NDARDX061ZME", "NDARDZ314LU8"  # 30 new subjects
    ]

    print("🚀 HBN Bulk Download")
    print("="*60)
    print(f"Target: {args.n_subjects} subjects")
    print(f"Output: {target_dir}")
    print()

    successful = 0
    failed = []

    for i, subj_id in enumerate(subject_ids[:args.n_subjects]):
        print(f"\n[{i+1}/{args.n_subjects}] Processing sub-{subj_id}")

        if download_subject(subj_id, target_dir):
            successful += 1
        else:
            failed.append(subj_id)

    print()
    print("="*60)
    print("📊 Download Summary")
    print()
    print(f"✅ Successful: {successful}/{args.n_subjects}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        print(f"   {', '.join(failed)}")
    print()

    return 0 if successful > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
