#!/usr/bin/env python3
"""
Enhanced GPU System Test
=========================

Comprehensive test of the enhanced GPU optimization system.
Tests all components:
- Enhanced GPU optimizer with platform detection
- Enhanced neural network layers
- Performance profiling and benchmarking
- Memory management
- Platform-specific optimizations
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import enhanced modules
from gpu.enhanced_gpu_optimizer import get_enhanced_optimizer
from models.enhanced_gpu_layers import (
    EnhancedLinear, 
    EnhancedMultiHeadAttention,
    EnhancedTransformerLayer,
    EnhancedEEGFoundationModel,
    create_enhanced_eeg_model
)

class ComprehensiveGPUTest:
    """Comprehensive test suite for enhanced GPU system"""
    
    def __init__(self):
        self.gpu_opt = get_enhanced_optimizer()
        self.device = self.gpu_opt.get_optimal_device("general")
        print(f"🧪 Enhanced GPU Test Suite")
        print(f"Platform: {self.gpu_opt.platform}")
        print(f"Device: {self.device}")
        print(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("="*80)
        
    def test_enhanced_linear(self):
        """Test enhanced linear layer"""
        print("🔧 Testing Enhanced Linear Layer...")
        
        # Create test data
        batch_size = 32
        input_size = 256
        output_size = 128
        
        x = torch.randn(batch_size, input_size)
        x = self.gpu_opt.optimize_tensor_for_operation(x, "linear")
        
        # Test layer
        layer = EnhancedLinear(input_size, output_size, use_enhanced_ops=True)
        layer = layer.to(self.device)
        
        # Forward pass
        start_time = time.time()
        with self.gpu_opt.memory_management("linear"):
            output = layer(x)
        
        execution_time = time.time() - start_time
        
        # Validate
        assert output.shape == (batch_size, output_size), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN values in output"
        
        print(f"   ✅ Enhanced Linear: {output.shape}, {execution_time:.4f}s")
        return True
        
    def test_enhanced_attention(self):
        """Test enhanced multi-head attention"""
        print("🔧 Testing Enhanced Multi-Head Attention...")
        
        # Parameters
        batch_size = 16
        seq_len = 128
        d_model = 256
        n_heads = 8
        
        # Create test data
        x = torch.randn(batch_size, seq_len, d_model)
        x = self.gpu_opt.optimize_tensor_for_operation(x, "attention")
        
        # Create attention layer
        attention = EnhancedMultiHeadAttention(d_model, n_heads, use_enhanced_ops=True)
        attention = attention.to(self.device)
        
        # Forward pass
        start_time = time.time()
        with self.gpu_opt.memory_management("attention"):
            output = attention(x, x, x)
        
        execution_time = time.time() - start_time
        
        # Validate
        assert output.shape == (batch_size, seq_len, d_model), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN values in output"
        
        print(f"   ✅ Enhanced Attention: {output.shape}, {execution_time:.4f}s")
        return True
        
    def test_enhanced_transformer(self):
        """Test enhanced transformer layer"""
        print("🔧 Testing Enhanced Transformer Layer...")
        
        # Parameters
        batch_size = 8
        seq_len = 64
        d_model = 128
        n_heads = 4
        
        # Create test data
        x = torch.randn(batch_size, seq_len, d_model)
        x = self.gpu_opt.optimize_tensor_for_operation(x, "transformer")
        
        # Create transformer layer
        transformer = EnhancedTransformerLayer(d_model, n_heads, use_enhanced_ops=True)
        transformer = transformer.to(self.device)
        
        # Forward pass
        start_time = time.time()
        with self.gpu_opt.memory_management("transformer"):
            output = transformer(x)
        
        execution_time = time.time() - start_time
        
        # Validate
        assert output.shape == (batch_size, seq_len, d_model), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN values in output"
        
        print(f"   ✅ Enhanced Transformer: {output.shape}, {execution_time:.4f}s")
        return True
        
    def test_enhanced_eeg_model(self):
        """Test complete enhanced EEG model"""
        print("🔧 Testing Enhanced EEG Foundation Model...")
        
        # Parameters
        batch_size = 4
        n_channels = 129
        sequence_length = 1000
        n_classes = 1
        
        # Create test EEG data
        x = torch.randn(batch_size, n_channels, sequence_length)
        x = self.gpu_opt.optimize_tensor_for_operation(x, "transformer")
        
        # Create model
        model = create_enhanced_eeg_model(
            n_channels=n_channels,
            num_classes=n_classes,
            d_model=128,
            n_heads=8,
            n_layers=4,
            use_enhanced_ops=True
        )
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {total_params:,}")
        
        # Forward pass
        start_time = time.time()
        with self.gpu_opt.memory_management("training"):
            output = model(x)
        
        execution_time = time.time() - start_time
        
        # Validate
        assert output.shape == (batch_size, n_classes), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN values in output"
        
        print(f"   ✅ Enhanced EEG Model: {output.shape}, {execution_time:.4f}s")
        return True
        
    def test_profiling_system(self):
        """Test performance profiling system"""
        print("🔧 Testing Performance Profiling System...")
        
        # Create test operation
        def test_operation():
            x = torch.randn(100, 100)
            x = self.gpu_opt.optimize_tensor_for_operation(x, "general")
            y = torch.mm(x, x.T)
            return y
            
        # Profile operation
        result = self.gpu_opt.profiler.profile_operation("test_mm", test_operation)
        
        # Get statistics
        stats = self.gpu_opt.get_performance_stats()
        
        # Validate
        assert "test_mm" in stats["operation_times"], "Operation not profiled"
        assert isinstance(stats["platform_info"], dict), "Platform info missing"
        
        print(f"   ✅ Profiling System: {len(stats['operation_times'])} operations tracked")
        return True
        
    def test_memory_management(self):
        """Test memory management system"""
        print("🔧 Testing Memory Management System...")
        
        initial_memory = 0
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        # Test memory management context
        with self.gpu_opt.memory_management("testing"):
            # Create large tensor
            large_tensor = torch.randn(1000, 1000)
            large_tensor = self.gpu_opt.optimize_tensor_for_operation(large_tensor, "general")
            
            # Some computation
            result = torch.mm(large_tensor, large_tensor.T)
            
        # Memory should be cleaned up
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            print(f"   Memory usage: {initial_memory/1024**2:.1f}MB → {final_memory/1024**2:.1f}MB")
        
        print(f"   ✅ Memory Management: Context manager working")
        return True
        
    def test_batch_size_optimization(self):
        """Test dynamic batch size optimization"""
        print("🔧 Testing Batch Size Optimization...")
        
        base_batch_size = 16
        optimized_batch_size = self.gpu_opt.optimize_batch_size(base_batch_size)
        
        print(f"   Base batch size: {base_batch_size}")
        print(f"   Optimized batch size: {optimized_batch_size}")
        
        # Validate
        assert optimized_batch_size > 0, "Invalid batch size"
        assert isinstance(optimized_batch_size, int), "Batch size must be integer"
        
        print(f"   ✅ Batch Size Optimization: {base_batch_size} → {optimized_batch_size}")
        return True
        
    def test_operation_routing(self):
        """Test operation-specific device routing"""
        print("🔧 Testing Operation Routing...")
        
        operations = ["general", "linear", "attention", "transformer", "fft", "conv"]
        routing_results = {}
        
        for op in operations:
            device = self.gpu_opt.get_optimal_device(op)
            routing_results[op] = device
            
        # Validate AMD FFT safety
        if self.gpu_opt.platform == "amd":
            assert "cpu" in str(routing_results["fft"]).lower(), "AMD FFT should route to CPU for safety"
            
        print(f"   Operation routing:")
        for op, device in routing_results.items():
            print(f"     {op}: {device}")
            
        print(f"   ✅ Operation Routing: {len(operations)} operations configured")
        return True
        
    def test_platform_optimizations(self):
        """Test platform-specific optimizations"""
        print("🔧 Testing Platform-Specific Optimizations...")
        
        # Test tensor optimization
        test_tensor = torch.randn(64, 64)
        operations = ["general", "linear", "attention", "fft"]
        
        for op in operations:
            optimized_tensor = self.gpu_opt.optimize_tensor_for_operation(test_tensor, op)
            print(f"   {op}: {test_tensor.device} → {optimized_tensor.device}")
            
        # Test platform-specific settings
        if self.gpu_opt.platform == "nvidia" and hasattr(torch.backends.cuda, 'matmul'):
            print(f"   NVIDIA optimizations enabled")
        elif self.gpu_opt.platform == "amd":
            print(f"   AMD conservative optimizations enabled")
            
        print(f"   ✅ Platform Optimizations: {self.gpu_opt.platform} specific settings applied")
        return True
        
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite"""
        print("🏁 Running Benchmark Suite...")
        
        # Benchmark parameters
        operations = [
            ("Matrix Multiplication", lambda: torch.mm(torch.randn(512, 512).to(self.device), 
                                                      torch.randn(512, 512).to(self.device))),
            ("Linear Layer", lambda: nn.Linear(256, 128).to(self.device)(torch.randn(64, 256).to(self.device))),
            ("Attention", lambda: nn.MultiheadAttention(128, 8).to(self.device)(
                torch.randn(32, 64, 128).to(self.device), 
                torch.randn(32, 64, 128).to(self.device), 
                torch.randn(32, 64, 128).to(self.device))[0])
        ]
        
        benchmark_results = {}
        
        for name, operation in operations:
            # Warm up
            for _ in range(3):
                _ = operation()
                
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                _ = operation()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start_time)
                
            avg_time = np.mean(times)
            std_time = np.std(times)
            benchmark_results[name] = {'mean': avg_time, 'std': std_time}
            
            print(f"   {name}: {avg_time:.4f}±{std_time:.4f}s")
            
        print(f"   ✅ Benchmark Suite: {len(operations)} operations benchmarked")
        return benchmark_results
        
    def run_all_tests(self):
        """Run all tests in the suite"""
        print("🚀 Starting Enhanced GPU System Test Suite")
        print("="*80)
        
        tests = [
            ("Enhanced Linear Layer", self.test_enhanced_linear),
            ("Enhanced Attention", self.test_enhanced_attention),
            ("Enhanced Transformer", self.test_enhanced_transformer),
            ("Enhanced EEG Model", self.test_enhanced_eeg_model),
            ("Profiling System", self.test_profiling_system),
            ("Memory Management", self.test_memory_management),
            ("Batch Size Optimization", self.test_batch_size_optimization),
            ("Operation Routing", self.test_operation_routing),
            ("Platform Optimizations", self.test_platform_optimizations)
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                results[test_name] = "PASSED" if success else "FAILED"
                if success:
                    passed += 1
            except Exception as e:
                results[test_name] = f"ERROR: {str(e)}"
                print(f"   ❌ {test_name}: {str(e)}")
                
        print("\n" + "="*80)
        print("📊 TEST RESULTS")
        print("="*80)
        
        for test_name, result in results.items():
            status_emoji = "✅" if result == "PASSED" else "❌"
            print(f"{status_emoji} {test_name}: {result}")
            
        print(f"\n🏆 Tests passed: {passed}/{len(tests)}")
        
        # Run benchmarks if all tests pass
        if passed == len(tests):
            print("\n" + "="*80)
            benchmark_results = self.run_benchmark_suite()
            
            # Get final performance statistics
            final_stats = self.gpu_opt.get_performance_stats()
            print("\n📊 Final Performance Statistics:")
            print(f"Platform: {final_stats['platform_info']['platform']}")
            print(f"GPU Available: {final_stats['platform_info']['gpu_available']}")
            print(f"Operations Profiled: {len(final_stats['operation_times'])}")
            
        print("\n🎉 Enhanced GPU System Test Complete!")
        return results

def main():
    """Main test function"""
    tester = ComprehensiveGPUTest()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    failed_tests = [k for k, v in results.items() if v != "PASSED"]
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} tests failed")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
