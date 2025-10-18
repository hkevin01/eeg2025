#!/usr/bin/env python3
"""
Train and Validate Both Challenges with Improvements
=====================================================
Trains Challenge 1 & 2 with:
- Stimulus-aligned windows
- R4 training data (33% more)
- L1 + L2 + Dropout regularization (Elastic Net)

Then compares new scores vs baseline.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

def print_section(text):
    print(f"\n{BOLD}{YELLOW}{text}{RESET}")
    print(f"{YELLOW}{'-'*len(text)}{RESET}")

def run_command(cmd, description, log_file=None):
    """Run a command and capture output"""
    print(f"\n{GREEN}▶ {description}{RESET}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream output to both terminal and log file
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                
                process.wait()
                returncode = process.returncode
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            returncode = result.returncode
        
        elapsed = time.time() - start_time
        print(f"{GREEN}✓ Completed in {elapsed:.1f}s{RESET}")
        
        return returncode == 0
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"{RED}✗ Failed after {elapsed:.1f}s{RESET}")
        print(f"Error: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print(e.stderr)
        return False

def extract_scores(log_file):
    """Extract final scores from training log"""
    scores = {'train_nrmse': None, 'val_nrmse': None}
    
    if not log_file.exists():
        return scores
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Look for best validation score
    for line in reversed(lines):
        if 'Best Val NRMSE' in line or 'best_nrmse' in line.lower():
            try:
                # Extract number after "NRMSE:"
                parts = line.split(':')
                if len(parts) > 1:
                    score = float(parts[-1].strip().split()[0])
                    scores['val_nrmse'] = score
                    break
            except:
                continue
    
    # Look for final training NRMSE
    for line in reversed(lines):
        if 'Train NRMSE' in line:
            try:
                parts = line.split(':')
                if len(parts) > 1:
                    score = float(parts[1].strip().split()[0])
                    scores['train_nrmse'] = score
                    break
            except:
                continue
    
    return scores

def main():
    print_header("🚀 TRAIN & VALIDATE WITH IMPROVEMENTS")
    
    print(f"{BOLD}Improvements applied:{RESET}")
    print("  ✓ Stimulus-aligned windows (not trial-aligned)")
    print("  ✓ R1-R4 training data (33% more subjects)")
    print("  ✓ L1 + L2 + Dropout regularization (Elastic Net)")
    print("  ✓ Enhanced model architecture")
    
    # Create logs directory
    log_dir = Path("logs/training_comparison")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =========================================================================
    # CHALLENGE 1: Response Time Prediction
    # =========================================================================
    print_header("🎯 CHALLENGE 1: Response Time Prediction")
    
    challenge1_log = log_dir / f"challenge1_improved_{timestamp}.log"
    
    print_section("Training Challenge 1 with improvements...")
    
    success_c1 = run_command(
        ['python3', 'scripts/training/challenge1/train_challenge1_multi_release.py'],
        "Training Challenge 1",
        log_file=challenge1_log
    )
    
    if success_c1:
        print(f"\n{GREEN}✓ Challenge 1 training complete!{RESET}")
        print(f"Log saved to: {challenge1_log}")
        
        # Extract scores
        scores_c1 = extract_scores(challenge1_log)
        print(f"\n{BOLD}Challenge 1 Results:{RESET}")
        if scores_c1['train_nrmse']:
            print(f"  Train NRMSE: {scores_c1['train_nrmse']:.4f}")
        if scores_c1['val_nrmse']:
            print(f"  Val NRMSE:   {scores_c1['val_nrmse']:.4f}")
    else:
        print(f"\n{RED}✗ Challenge 1 training failed!{RESET}")
        print("Check log for details")
    
    # =========================================================================
    # CHALLENGE 2: Externalizing Behavior Prediction
    # =========================================================================
    print_header("🎯 CHALLENGE 2: Externalizing Behavior")
    
    challenge2_log = log_dir / f"challenge2_improved_{timestamp}.log"
    
    print_section("Training Challenge 2 with improvements...")
    
    success_c2 = run_command(
        ['python3', 'scripts/training/challenge2/train_challenge2_multi_release.py'],
        "Training Challenge 2",
        log_file=challenge2_log
    )
    
    if success_c2:
        print(f"\n{GREEN}✓ Challenge 2 training complete!{RESET}")
        print(f"Log saved to: {challenge2_log}")
        
        # Extract scores
        scores_c2 = extract_scores(challenge2_log)
        print(f"\n{BOLD}Challenge 2 Results:{RESET}")
        if scores_c2['train_nrmse']:
            print(f"  Train NRMSE: {scores_c2['train_nrmse']:.4f}")
        if scores_c2['val_nrmse']:
            print(f"  Val NRMSE:   {scores_c2['val_nrmse']:.4f}")
    else:
        print(f"\n{RED}✗ Challenge 2 training failed!{RESET}")
        print("Check log for details")
    
    # =========================================================================
    # COMPARISON WITH BASELINE
    # =========================================================================
    print_header("📊 COMPARISON WITH BASELINE")
    
    # Baseline scores (from previous runs)
    baseline = {
        'challenge1': {'val_nrmse': 1.00, 'description': 'Trial-aligned, R1-R2 only'},
        'challenge2': {'val_nrmse': 1.46, 'description': 'Trial-aligned, R1-R2 only'}
    }
    
    print(f"{BOLD}Challenge 1: Response Time{RESET}")
    print(f"  Baseline:  {baseline['challenge1']['val_nrmse']:.4f} NRMSE")
    print(f"             ({baseline['challenge1']['description']})")
    if success_c1 and scores_c1['val_nrmse']:
        improvement_c1 = ((baseline['challenge1']['val_nrmse'] - scores_c1['val_nrmse']) 
                          / baseline['challenge1']['val_nrmse'] * 100)
        print(f"  Improved:  {scores_c1['val_nrmse']:.4f} NRMSE")
        print(f"             (Stimulus-aligned, R1-R4, Elastic Net)")
        if improvement_c1 > 0:
            print(f"  {GREEN}↓ {improvement_c1:.1f}% improvement! 🎉{RESET}")
        else:
            print(f"  {RED}↑ {-improvement_c1:.1f}% worse{RESET}")
    
    print(f"\n{BOLD}Challenge 2: Externalizing Behavior{RESET}")
    print(f"  Baseline:  {baseline['challenge2']['val_nrmse']:.4f} NRMSE")
    print(f"             ({baseline['challenge2']['description']})")
    if success_c2 and scores_c2['val_nrmse']:
        improvement_c2 = ((baseline['challenge2']['val_nrmse'] - scores_c2['val_nrmse']) 
                          / baseline['challenge2']['val_nrmse'] * 100)
        print(f"  Improved:  {scores_c2['val_nrmse']:.4f} NRMSE")
        print(f"             (Stimulus-aligned, R1-R4, Elastic Net)")
        if improvement_c2 > 0:
            print(f"  {GREEN}↓ {improvement_c2:.1f}% improvement! 🎉{RESET}")
        else:
            print(f"  {RED}↑ {-improvement_c2:.1f}% worse{RESET}")
    
    # Combined score
    if success_c1 and success_c2 and scores_c1['val_nrmse'] and scores_c2['val_nrmse']:
        combined_baseline = (baseline['challenge1']['val_nrmse'] + 
                            baseline['challenge2']['val_nrmse']) / 2
        combined_improved = (scores_c1['val_nrmse'] + scores_c2['val_nrmse']) / 2
        combined_improvement = ((combined_baseline - combined_improved) 
                               / combined_baseline * 100)
        
        print(f"\n{BOLD}Combined Score (Average):{RESET}")
        print(f"  Baseline:  {combined_baseline:.4f} NRMSE")
        print(f"  Improved:  {combined_improved:.4f} NRMSE")
        if combined_improvement > 0:
            print(f"  {GREEN}↓ {combined_improvement:.1f}% overall improvement! 🏆{RESET}")
        else:
            print(f"  {RED}↑ {-combined_improvement:.1f}% worse overall{RESET}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("✨ TRAINING SESSION COMPLETE")
    
    print(f"{BOLD}Model weights saved:{RESET}")
    if success_c1:
        print(f"  ✓ weights_challenge_1_multi_release.pt")
    if success_c2:
        print(f"  ✓ weights_challenge_2_multi_release.pt")
    
    print(f"\n{BOLD}Training logs:{RESET}")
    print(f"  {challenge1_log}")
    print(f"  {challenge2_log}")
    
    print(f"\n{BOLD}Next steps:{RESET}")
    print("  1. Review training logs for any issues")
    print("  2. Check overfitting (train vs val NRMSE gap)")
    print("  3. If good, create submission.zip and upload")
    print("  4. If not good, tune regularization hyperparameters")
    
    print(f"\n{BOLD}To create submission:{RESET}")
    print(f"  python submission.py")
    
    return success_c1 and success_c2

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
