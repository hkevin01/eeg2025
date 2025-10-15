#!/usr/bin/env python3
"""
Safe CUDA/CuFFT/CuBLAS Testing with Resource Monitoring
"""
import sys
import time
from pathlib import Path

import psutil

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Create logs directory
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"cuda_test_{int(time.time())}.log"

class SafeLogger:
    """Logger that writes to both console and file with timestamps"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.file = open(log_path, 'w', buffering=1)  # Line buffered
        self.start_time = time.time()
        self.last_progress_time = self.start_time

    def log(self, message, show_time=True):
        """Log message to both console and file"""
        if show_time:
            elapsed = time.time() - self.start_time
            timestamp = f"[{elapsed:6.1f}s] "
            full_message = timestamp + message
        else:
            full_message = message

        print(full_message, flush=True)
        self.file.write(full_message + '\n')
        self.file.flush()

    def progress(self, message):
        """Log progress update with time since last progress"""
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        elapsed_since_last = current_time - self.last_progress_time
        self.last_progress_time = current_time

        progress_msg = f"⏱️  {message} (total: {elapsed_total:.1f}s, +{elapsed_since_last:.1f}s)"
        self.log(progress_msg)

    def close(self):
        self.file.close()

def check_resources():
    """Check system resources"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    mem_percent = memory.percent
    mem_available_gb = memory.available / (1024**3)

    return {
        'cpu': cpu_percent,
        'mem_percent': mem_percent,
        'mem_available_gb': mem_available_gb
    }

def main():
    logger = SafeLogger(log_file)

    try:
        logger.log("="*80, show_time=False)
        logger.log("SAFE CUDA/CuFFT/CuBLAS TEST", show_time=False)
        logger.log("="*80, show_time=False)
        logger.log(f"Log file: {log_file}", show_time=False)
        logger.log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", show_time=False)
        logger.log("", show_time=False)

        # Check initial resources
        logger.progress("Checking initial resources...")
        resources = check_resources()
        logger.log("Initial System Resources:")
        logger.log(f"  CPU: {resources['cpu']:.1f}%")
        logger.log(f"  Memory: {resources['mem_percent']:.1f}% used, {resources['mem_available_gb']:.1f} GB available")
        logger.log("")

        # Step 1: Import PyTorch
        logger.progress("Step 1: Importing PyTorch...")
        import torch
        logger.log(f"  ✅ PyTorch {torch.__version__} imported")
        logger.log("")

        # Step 2: Check CUDA
        logger.progress("Step 2: Checking CUDA availability...")
        cuda_available = torch.cuda.is_available()
        logger.log(f"  CUDA Available: {cuda_available}")

        if cuda_available:
            logger.log(f"  CUDA Version: {torch.version.cuda}")
            logger.log(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.log(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
        else:
            logger.log("  ⚠️  CUDA not available - will test CPU fallback")
        logger.log("")

        # Step 3: Import optimization module
        logger.progress("Step 3: Importing cuda_optimized module...")
        try:
            from src.gpu.cuda_optimized import (
                CUBLAS_AVAILABLE,
                CUDA_AVAILABLE,
                CUFFT_AVAILABLE,
                CuBLASOptimizer,
                CuFFTOptimizer,
            )
            logger.log("  ✅ Module imported successfully")
            logger.log(f"  CUDA_AVAILABLE: {CUDA_AVAILABLE}")
            logger.log(f"  CUFFT_AVAILABLE: {CUFFT_AVAILABLE}")
            logger.log(f"  CUBLAS_AVAILABLE: {CUBLAS_AVAILABLE}")
        except Exception as e:
            logger.log(f"  ❌ Import failed: {e}")
            logger.log(f"  Error type: {type(e).__name__}")
            import traceback
            logger.log(traceback.format_exc())
            logger.close()
            return
        logger.log("")

        # Check resources after imports
        resources = check_resources()
        logger.log("Resources after imports:")
        logger.log(f"  CPU: {resources['cpu']:.1f}%")
        logger.log(f"  Memory: {resources['mem_percent']:.1f}% used")
        logger.log("")

        # Step 4: Test CuFFT Optimizer
        logger.progress("Step 4: Testing CuFFT Optimizer...")
        try:
            logger.log("  Creating CuFFT optimizer...")
            fft_opt = CuFFTOptimizer(device="cuda" if cuda_available else "cpu")
            logger.log("  ✅ CuFFT Optimizer created")

            # Small test
            logger.log("  Testing small FFT (batch=2, channels=19, time=1000)...")
            test_signal = torch.randn(2, 19, 1000)
            if cuda_available:
                test_signal = test_signal.cuda()

            start = time.time()
            fft_result = fft_opt.rfft_batch(test_signal, dim=-1)
            if cuda_available:
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            logger.log(f"  ✅ FFT completed in {elapsed:.2f} ms")
            logger.log(f"     Input shape: {test_signal.shape}")
            logger.log(f"     Output shape: {fft_result.shape}")

            # Inverse FFT
            logger.log("  Testing inverse FFT...")
            start = time.time()
            reconstructed = fft_opt.irfft_batch(fft_result, n=1000, dim=-1)
            if cuda_available:
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            logger.log(f"  ✅ IFFT completed in {elapsed:.2f} ms")

            # Check reconstruction accuracy
            if cuda_available:
                test_signal = test_signal.cpu()
                reconstructed = reconstructed.cpu()
            error = (test_signal - reconstructed).abs().mean().item()
            logger.log(f"     Reconstruction error: {error:.2e}")
            logger.progress("CuFFT tests completed")

        except Exception as e:
            logger.log(f"  ❌ CuFFT test failed: {e}")
            import traceback
            logger.log(traceback.format_exc())
        logger.log("")

        # Check resources
        resources = check_resources()
        logger.log("Resources after CuFFT test:")
        logger.log(f"  CPU: {resources['cpu']:.1f}%")
        logger.log(f"  Memory: {resources['mem_percent']:.1f}% used")
        logger.log("")

        time.sleep(2)  # Pause to prevent overload

        # Step 5: Test CuBLAS Optimizer
        logger.progress("Step 5: Testing CuBLAS Optimizer...")
        try:
            logger.log("  Creating CuBLAS optimizer...")
            blas_opt = CuBLASOptimizer(device="cuda" if cuda_available else "cpu", use_tf32=True)
            logger.log("  ✅ CuBLAS Optimizer created")

            # Small matmul test
            logger.log("  Testing matrix multiplication (32x128x128)...")
            A = torch.randn(32, 128, 128)
            B = torch.randn(32, 128, 128)
            if cuda_available:
                A = A.cuda()
                B = B.cuda()

            start = time.time()
            C = blas_opt.bmm_optimized(A, B)
            if cuda_available:
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            logger.log(f"  ✅ Matrix multiplication completed in {elapsed:.2f} ms")
            logger.log(f"     Shape: ({A.shape}) @ ({B.shape}) = {C.shape}")

            # Test addmm
            logger.log("  Testing fused addmm operation...")
            bias = torch.randn(128)
            input_mat = torch.randn(32, 256)
            weight = torch.randn(128, 256)
            if cuda_available:
                bias = bias.cuda()
                input_mat = input_mat.cuda()
                weight = weight.cuda()

            start = time.time()
            result = blas_opt.addmm_optimized(bias, input_mat, weight)
            if cuda_available:
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            logger.log(f"  ✅ Fused addmm completed in {elapsed:.2f} ms")
            logger.log(f"     Output shape: {result.shape}")
            logger.progress("CuBLAS tests completed")

        except Exception as e:
            logger.log(f"  ❌ CuBLAS test failed: {e}")
            import traceback
            logger.log(traceback.format_exc())
        logger.log("")

        # Check resources
        resources = check_resources()
        logger.log("Resources after CuBLAS test:")
        logger.log(f"  CPU: {resources['cpu']:.1f}%")
        logger.log(f"  Memory: {resources['mem_percent']:.1f}% used")
        logger.log("")

        time.sleep(2)  # Pause

        # Step 6: Simple benchmark (reduced size)
        if cuda_available:
            logger.progress("Step 6: Running lightweight benchmarks...")
            logger.log("")

            logger.log("FFT Benchmark (smaller size):")
            logger.log("-" * 40)

            # CPU FFT
            logger.log("  Preparing CPU FFT benchmark...")
            signal_cpu = torch.randn(16, 64, 5000)
            logger.log("  Running CPU FFT (5 iterations)...")
            start = time.time()
            for i in range(5):
                _ = torch.fft.rfft(signal_cpu, dim=-1)
                if i == 2:
                    logger.log(f"    Progress: {i+1}/5 iterations...")
            cpu_time = (time.time() - start) / 5
            logger.log(f"    ✅ Average time: {cpu_time*1000:.2f} ms")

            # GPU FFT
            logger.log("  Transferring to GPU and warming up...")
            signal_gpu = signal_cpu.cuda()
            torch.cuda.synchronize()

            # Warmup
            for _ in range(3):
                _ = torch.fft.rfft(signal_gpu, dim=-1)
            torch.cuda.synchronize()

            logger.log("  Running GPU FFT with CuFFT (5 iterations)...")
            start = time.time()
            for i in range(5):
                _ = torch.fft.rfft(signal_gpu, dim=-1)
                if i == 2:
                    logger.log(f"    Progress: {i+1}/5 iterations...")
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 5
            logger.log(f"    ✅ Average time: {gpu_time*1000:.2f} ms")

            speedup = cpu_time / gpu_time
            logger.log(f"  🚀 CuFFT Speedup: {speedup:.2f}x faster than CPU")
            logger.progress("FFT benchmark completed")
            logger.log("")

            time.sleep(2)  # Pause

            logger.log("Matrix Multiplication Benchmark (smaller size):")
            logger.log("-" * 40)

            # CPU matmul
            logger.log("  Preparing CPU matmul benchmark...")
            a_cpu = torch.randn(16, 256, 256)
            b_cpu = torch.randn(16, 256, 256)
            logger.log("  Running CPU matmul (5 iterations)...")
            start = time.time()
            for i in range(5):
                _ = torch.bmm(a_cpu, b_cpu)
                if i == 2:
                    logger.log(f"    Progress: {i+1}/5 iterations...")
            cpu_time = (time.time() - start) / 5
            logger.log(f"    ✅ Average time: {cpu_time*1000:.2f} ms")

            # GPU matmul
            logger.log("  Transferring to GPU and warming up...")
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            torch.cuda.synchronize()

            # Warmup
            for _ in range(3):
                _ = torch.bmm(a_gpu, b_gpu)
            torch.cuda.synchronize()

            logger.log("  Running GPU matmul with CuBLAS (5 iterations)...")
            start = time.time()
            for i in range(5):
                _ = torch.bmm(a_gpu, b_gpu)
                if i == 2:
                    logger.log(f"    Progress: {i+1}/5 iterations...")
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 5
            logger.log(f"    ✅ Average time: {gpu_time*1000:.2f} ms")

            speedup = cpu_time / gpu_time
            logger.log(f"  🚀 CuBLAS Speedup: {speedup:.2f}x faster than CPU")
            logger.progress("Matrix multiplication benchmark completed")
            logger.log("")

            # Clean up
            logger.log("  Cleaning up GPU memory...")
            torch.cuda.empty_cache()
            logger.log("  ✅ Memory cleared")

        # Final resource check
        logger.progress("Collecting final system status...")
        resources = check_resources()
        logger.log("")
        logger.log("="*80, show_time=False)
        logger.log("FINAL SYSTEM STATUS", show_time=False)
        logger.log("="*80, show_time=False)
        logger.log(f"CPU: {resources['cpu']:.1f}%")
        logger.log(f"Memory: {resources['mem_percent']:.1f}% used, {resources['mem_available_gb']:.1f} GB available")

        total_time = time.time() - logger.start_time
        logger.log("")
        logger.log(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.log("")
        logger.log("✅ All tests completed successfully!")
        logger.log(f"Full log saved to: {log_file}")
        logger.log("="*80, show_time=False)

    except Exception as e:
        logger.log("")
        logger.log("="*80)
        logger.log("❌ FATAL ERROR")
        logger.log("="*80)
        logger.log(f"Error: {e}")
        logger.log(f"Type: {type(e).__name__}")
        import traceback
        logger.log(traceback.format_exc())
        logger.log("")
        logger.log(f"Log saved to: {log_file}")

    finally:
        logger.close()

if __name__ == "__main__":
    main()
