"""
Comprehensive Submission Verification Script
Tests EVERYTHING before uploading to competition
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import importlib.util
import traceback
import zipfile
import tempfile
import shutil

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

class SubmissionVerifier:
    def __init__(self, submission_path):
        self.submission_path = Path(submission_path)
        self.errors = []
        self.warnings = []
        self.tests_passed = 0
        self.tests_failed = 0
        
    def add_error(self, msg):
        self.errors.append(msg)
        self.tests_failed += 1
        print_error(msg)
        
    def add_warning(self, msg):
        self.warnings.append(msg)
        print_warning(msg)
        
    def add_success(self, msg):
        self.tests_passed += 1
        print_success(msg)
    
    def verify_zip_structure(self):
        """Verify ZIP file has correct structure"""
        print_header("Step 1: Verify ZIP Structure")
        
        if not self.submission_path.exists():
            self.add_error(f"Submission file not found: {self.submission_path}")
            return False
        
        if not self.submission_path.suffix == '.zip':
            self.add_error(f"Submission must be a .zip file, got: {self.submission_path.suffix}")
            return False
        
        self.add_success(f"Found submission: {self.submission_path.name}")
        
        try:
            with zipfile.ZipFile(self.submission_path, 'r') as zf:
                files = zf.namelist()
                print_info(f"Files in ZIP: {len(files)}")
                for f in files:
                    print(f"  - {f}")
                
                # Check required files
                required = ['submission.py', 'weights_challenge_1.pt', 'weights_challenge_2.pt']
                for req in required:
                    if req in files:
                        self.add_success(f"Found {req}")
                    else:
                        self.add_error(f"Missing required file: {req}")
                
                # Check for extra files
                extra = [f for f in files if f not in required and not f.startswith('__MACOSX')]
                if extra:
                    self.add_warning(f"Extra files in ZIP: {extra}")
                
                return True
                
        except Exception as e:
            self.add_error(f"Failed to read ZIP file: {e}")
            return False
    
    def extract_and_load_submission(self):
        """Extract ZIP and load submission module"""
        print_header("Step 2: Extract and Load Submission")
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix='submission_verify_')
        print_info(f"Extracting to: {self.temp_dir}")
        
        try:
            with zipfile.ZipFile(self.submission_path, 'r') as zf:
                zf.extractall(self.temp_dir)
            self.add_success("Extracted submission files")
            
            # Load submission.py as module
            submission_py = Path(self.temp_dir) / 'submission.py'
            spec = importlib.util.spec_from_file_location("submission", submission_py)
            self.submission_module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, self.temp_dir)
            spec.loader.exec_module(self.submission_module)
            
            self.add_success("Loaded submission.py module")
            return True
            
        except Exception as e:
            self.add_error(f"Failed to load submission: {e}")
            traceback.print_exc()
            return False
    
    def verify_class_exists(self):
        """Verify Submission class exists"""
        print_header("Step 3: Verify Submission Class")
        
        if not hasattr(self.submission_module, 'Submission'):
            self.add_error("submission.py must have a 'Submission' class")
            return False
        
        self.add_success("Found Submission class")
        return True
    
    def verify_init_signature(self):
        """Verify __init__ signature"""
        print_header("Step 4: Verify __init__ Signature")
        
        Submission = self.submission_module.Submission
        
        # Check __init__ accepts SFREQ and DEVICE
        import inspect
        sig = inspect.signature(Submission.__init__)
        params = list(sig.parameters.keys())
        
        print_info(f"__init__ parameters: {params}")
        
        if 'SFREQ' not in params:
            self.add_error("__init__ must accept SFREQ parameter")
        else:
            self.add_success("Has SFREQ parameter")
        
        if 'DEVICE' not in params:
            self.add_error("__init__ must accept DEVICE parameter")
        else:
            self.add_success("Has DEVICE parameter")
        
        return 'SFREQ' in params and 'DEVICE' in params
    
    def verify_required_methods(self):
        """Verify all required methods exist"""
        print_header("Step 5: Verify Required Methods")
        
        Submission = self.submission_module.Submission
        
        required_methods = [
            'get_model_challenge_1',
            'get_model_challenge_2',
            'challenge_1',
            'challenge_2'
        ]
        
        all_found = True
        for method_name in required_methods:
            if hasattr(Submission, method_name):
                self.add_success(f"Found method: {method_name}")
            else:
                self.add_error(f"Missing required method: {method_name}")
                all_found = False
        
        # Check for wrong methods
        wrong_methods = ['challenge1', 'challenge2']
        for method_name in wrong_methods:
            if hasattr(Submission, method_name):
                self.add_warning(f"Found {method_name} - should be challenge_{method_name[-1]} with underscore!")
        
        return all_found
    
    def verify_method_signatures(self):
        """Verify method signatures"""
        print_header("Step 6: Verify Method Signatures")
        
        Submission = self.submission_module.Submission
        import inspect
        
        # Check challenge_1(self, X)
        if hasattr(Submission, 'challenge_1'):
            sig = inspect.signature(Submission.challenge_1)
            params = list(sig.parameters.keys())
            if len(params) == 2 and params[1] == 'X':
                self.add_success("challenge_1(self, X) signature correct")
            else:
                self.add_error(f"challenge_1 should be (self, X), got: {params}")
        
        # Check challenge_2(self, X)
        if hasattr(Submission, 'challenge_2'):
            sig = inspect.signature(Submission.challenge_2)
            params = list(sig.parameters.keys())
            if len(params) == 2 and params[1] == 'X':
                self.add_success("challenge_2(self, X) signature correct")
            else:
                self.add_error(f"challenge_2 should be (self, X), got: {params}")
        
        return True
    
    def test_instantiation(self):
        """Test instantiation with different DEVICE types"""
        print_header("Step 7: Test Instantiation")
        
        Submission = self.submission_module.Submission
        
        # Test with string device
        try:
            submission1 = Submission(SFREQ=100, DEVICE='cpu')
            self.add_success("Instantiated with DEVICE='cpu' (string)")
            
            # Check device was converted
            if hasattr(submission1, 'device'):
                if isinstance(submission1.device, torch.device):
                    self.add_success("DEVICE string converted to torch.device")
                else:
                    self.add_error(f"device should be torch.device, got {type(submission1.device)}")
        except Exception as e:
            self.add_error(f"Failed to instantiate with DEVICE='cpu': {e}")
            traceback.print_exc()
            return False
        
        # Test with torch.device
        try:
            submission2 = Submission(SFREQ=100, DEVICE=torch.device('cpu'))
            self.add_success("Instantiated with DEVICE=torch.device('cpu')")
        except Exception as e:
            self.add_error(f"Failed with torch.device: {e}")
        
        self.submission_instance = submission1
        return True
    
    def test_model_loading(self):
        """Test model loading"""
        print_header("Step 8: Test Model Loading")
        
        # Test Challenge 1 model
        try:
            print_info("Loading Challenge 1 model...")
            model1 = self.submission_instance.get_model_challenge_1()
            
            if model1 is None:
                self.add_error("get_model_challenge_1() returned None")
            elif isinstance(model1, nn.Module):
                self.add_success("Challenge 1 model is nn.Module")
                
                # Check model is in eval mode
                if model1.training:
                    self.add_warning("Model is in training mode - should be eval mode")
                else:
                    self.add_success("Model is in eval mode")
                
                # Check device
                device = next(model1.parameters()).device
                print_info(f"Model 1 on device: {device}")
                
            else:
                self.add_error(f"get_model_challenge_1() should return nn.Module, got {type(model1)}")
                
        except Exception as e:
            self.add_error(f"Failed to load Challenge 1 model: {e}")
            traceback.print_exc()
            return False
        
        # Test Challenge 2 model
        try:
            print_info("Loading Challenge 2 model...")
            model2 = self.submission_instance.get_model_challenge_2()
            
            if model2 is None:
                self.add_error("get_model_challenge_2() returned None")
            elif isinstance(model2, nn.Module):
                self.add_success("Challenge 2 model is nn.Module")
                
                if model2.training:
                    self.add_warning("Model is in training mode - should be eval mode")
                else:
                    self.add_success("Model is in eval mode")
                
            else:
                self.add_error(f"get_model_challenge_2() should return nn.Module, got {type(model2)}")
                
        except Exception as e:
            self.add_error(f"Failed to load Challenge 2 model: {e}")
            traceback.print_exc()
            return False
        
        return True
    
    def test_predictions(self):
        """Test making predictions"""
        print_header("Step 9: Test Predictions")
        
        test_cases = [
            ("Single sample", torch.randn(1, 129, 200), (1,)),
            ("Small batch", torch.randn(4, 129, 200), (4,)),
            ("Medium batch", torch.randn(16, 129, 200), (16,)),
            ("Large batch", torch.randn(32, 129, 200), (32,)),
        ]
        
        for test_name, X, expected_shape in test_cases:
            print_info(f"Testing: {test_name} - input shape {X.shape}")
            
            # Test Challenge 1
            try:
                pred1 = self.submission_instance.challenge_1(X)
                
                # Check type
                if not isinstance(pred1, torch.Tensor):
                    self.add_error(f"challenge_1 must return torch.Tensor, got {type(pred1)}")
                    continue
                
                # Check shape
                if pred1.shape != expected_shape:
                    self.add_error(f"challenge_1 output shape wrong: expected {expected_shape}, got {pred1.shape}")
                    continue
                
                # Check for NaN/Inf
                if torch.isnan(pred1).any():
                    self.add_error(f"challenge_1 returned NaN values!")
                    continue
                if torch.isinf(pred1).any():
                    self.add_error(f"challenge_1 returned Inf values!")
                    continue
                
                self.add_success(f"Challenge 1 - {test_name}: {X.shape} → {pred1.shape} ✓")
                
            except Exception as e:
                self.add_error(f"Challenge 1 failed on {test_name}: {e}")
                traceback.print_exc()
            
            # Test Challenge 2
            try:
                pred2 = self.submission_instance.challenge_2(X)
                
                if not isinstance(pred2, torch.Tensor):
                    self.add_error(f"challenge_2 must return torch.Tensor, got {type(pred2)}")
                    continue
                
                if pred2.shape != expected_shape:
                    self.add_error(f"challenge_2 output shape wrong: expected {expected_shape}, got {pred2.shape}")
                    continue
                
                if torch.isnan(pred2).any():
                    self.add_error(f"challenge_2 returned NaN values!")
                    continue
                if torch.isinf(pred2).any():
                    self.add_error(f"challenge_2 returned Inf values!")
                    continue
                
                self.add_success(f"Challenge 2 - {test_name}: {X.shape} → {pred2.shape} ✓")
                
            except Exception as e:
                self.add_error(f"Challenge 2 failed on {test_name}: {e}")
                traceback.print_exc()
        
        return True
    
    def test_determinism(self):
        """Test that predictions are deterministic"""
        print_header("Step 10: Test Determinism")
        
        X = torch.randn(4, 129, 200)
        
        # Challenge 1
        pred1_a = self.submission_instance.challenge_1(X)
        pred1_b = self.submission_instance.challenge_1(X)
        
        if torch.allclose(pred1_a, pred1_b):
            self.add_success("Challenge 1 predictions are deterministic")
        else:
            self.add_warning("Challenge 1 predictions not deterministic (may have dropout enabled)")
        
        # Challenge 2
        pred2_a = self.submission_instance.challenge_2(X)
        pred2_b = self.submission_instance.challenge_2(X)
        
        if torch.allclose(pred2_a, pred2_b):
            self.add_success("Challenge 2 predictions are deterministic")
        else:
            self.add_warning("Challenge 2 predictions not deterministic (may have dropout enabled)")
        
        return True
    
    def print_summary(self):
        """Print final summary"""
        print_header("Verification Summary")
        
        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        print(f"  {Colors.GREEN}✅ Tests Passed: {self.tests_passed}{Colors.END}")
        print(f"  {Colors.RED}❌ Tests Failed: {self.tests_failed}{Colors.END}")
        print(f"  {Colors.YELLOW}⚠️  Warnings: {len(self.warnings)}{Colors.END}")
        
        if self.errors:
            print(f"\n{Colors.BOLD}{Colors.RED}ERRORS:{Colors.END}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}WARNINGS:{Colors.END}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        
        if self.tests_failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ SUBMISSION VERIFIED - READY TO UPLOAD!{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ SUBMISSION HAS ERRORS - DO NOT UPLOAD!{Colors.END}")
            return False
    
    def cleanup(self):
        """Cleanup temp directory"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print_info(f"Cleaned up temp directory")
    
    def run_all_tests(self):
        """Run all verification tests"""
        try:
            if not self.verify_zip_structure():
                return False
            
            if not self.extract_and_load_submission():
                return False
            
            if not self.verify_class_exists():
                return False
            
            if not self.verify_init_signature():
                return False
            
            if not self.verify_required_methods():
                return False
            
            self.verify_method_signatures()
            
            if not self.test_instantiation():
                return False
            
            if not self.test_model_loading():
                return False
            
            if not self.test_predictions():
                return False
            
            self.test_determinism()
            
            return self.print_summary()
            
        finally:
            self.cleanup()

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_submission.py <submission.zip>")
        sys.exit(1)
    
    submission_path = sys.argv[1]
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SUBMISSION VERIFICATION SCRIPT" + " " * 22 + "║")
    print("║" + " " * 10 + "Comprehensive Pre-Upload Testing" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print(Colors.END)
    
    verifier = SubmissionVerifier(submission_path)
    success = verifier.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
