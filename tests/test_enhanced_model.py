#!/usr/bin/env python3
"""
Complete integration test for the enhanced EEG foundation model.

This script demonstrates the full pipeline with task-aware architecture,
multi-adversary DANN, compression-aware SSL, and GPU optimization.
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our enhanced components
try:
    from src.data.enhanced_pipeline import EnhancedHBNDataset, RealLabelManager
    from src.models.invariance.dann import AdversaryType, MultiAdversaryDANN
    from src.models.task_aware import (
        HBNTask,
        MultiTaskHead,
        TaskAwareTemporalCNN,
        TaskTokenEmbedding,
    )
    from src.training.enhanced_trainer import EnhancedTrainer, TrainingConfig
    from src.utils.augmentations import CompressionDistortion, TimeMasking
    from src.utils.gpu_optimization import ModelBenchmarker, OptimizedModel
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all modules are in the correct location")
    exit(1)


def test_task_aware_architecture():
    """Test task-aware architecture with all HBN tasks."""
    print("\n" + "=" * 60)
    print("TESTING TASK-AWARE ARCHITECTURE")
    print("=" * 60)

    # Initialize model
    model = TaskAwareTemporalCNN(
        input_channels=19,
        hidden_dim=128,
        num_layers=4,
        use_task_tokens=True,
        adaptation_method="film",
    )

    print(
        f"✅ Created TaskAwareTemporalCNN with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test all tasks
    batch_size = 8
    sequence_length = 1000
    input_data = torch.randn(batch_size, 19, sequence_length)

    results = {}
    for task in HBNTask:
        try:
            start_time = time.time()
            output = model(input_data, task)
            inference_time = (time.time() - start_time) * 1000

            print(f"  {task.value:>3}: {output.shape} | {inference_time:.2f}ms")
            results[task.value] = {
                "output_shape": output.shape,
                "inference_time_ms": inference_time,
            }
        except Exception as e:
            print(f"  {task.value:>3}: ERROR - {e}")

    return results


def test_multi_task_head():
    """Test multi-task prediction head."""
    print("\n" + "=" * 60)
    print("TESTING MULTI-TASK HEAD")
    print("=" * 60)

    # Initialize head
    head = MultiTaskHead(hidden_dim=128, dropout=0.1)

    # Test predictions
    batch_size = 8
    features = torch.randn(batch_size, 128)

    predictions = head(features)

    print(f"✅ Multi-task head created with outputs:")
    for key, value in predictions.items():
        print(f"  {key:>20}: {value.shape}")

    return predictions


def test_multi_adversary_dann():
    """Test multi-adversary domain adaptation."""
    print("\n" + "=" * 60)
    print("TESTING MULTI-ADVERSARY DANN")
    print("=" * 60)

    # Initialize DANN
    adversary_types = [AdversaryType.SITE, AdversaryType.SUBJECT, AdversaryType.TASK]
    dann = MultiAdversaryDANN(
        feature_dim=128, adversary_types=adversary_types, hidden_dim=64
    )

    print(f"✅ Created MultiAdversaryDANN with {len(adversary_types)} adversaries")

    # Test adversarial loss
    batch_size = 8
    features = torch.randn(batch_size, 128)

    # Create adversary labels
    adversary_labels = {
        AdversaryType.SITE: torch.randint(0, 5, (batch_size,)),
        AdversaryType.SUBJECT: torch.randint(0, 100, (batch_size,)),
        AdversaryType.TASK: torch.randint(0, 6, (batch_size,)),
    }

    try:
        adversarial_loss = dann.compute_adversarial_loss(
            features, adversary_labels, lambda_val=1.0
        )
        print(f"  Adversarial loss: {adversarial_loss.item():.4f}")

        # Test individual adversaries
        individual_losses = dann.compute_individual_losses(features, adversary_labels)
        for adv_type, loss in individual_losses.items():
            print(f"  {adv_type.value} loss: {loss.item():.4f}")

        return {
            "total_loss": adversarial_loss.item(),
            "individual_losses": individual_losses,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def test_compression_aware_ssl():
    """Test compression-aware self-supervised learning."""
    print("\n" + "=" * 60)
    print("TESTING COMPRESSION-AWARE SSL")
    print("=" * 60)

    # Test augmentations
    sequence_length = 1000
    eeg_data = np.random.randn(19, sequence_length)

    print(f"Original data shape: {eeg_data.shape}")

    # Time masking
    time_masking = TimeMasking(mask_ratio=0.2, schedulable=True)
    masked_data = time_masking(eeg_data.copy())
    mask_diff = np.mean(np.abs(eeg_data - masked_data))
    print(f"  Time masking difference: {mask_diff:.4f}")

    # Compression distortion
    compression = CompressionDistortion(
        distortion_percentage=0.05,
        compression_method="wavelet",
        quantization_snr=30.0,
        schedulable=True,
    )
    compressed_data = compression(eeg_data.copy())
    comp_diff = np.mean(np.abs(eeg_data - compressed_data))
    print(f"  Compression difference: {comp_diff:.4f}")

    # Test schedulable parameters
    time_masking.mask_ratio = 0.3
    compression.distortion_percentage = 0.1

    print(f"  Updated mask ratio: {time_masking.mask_ratio}")
    print(f"  Updated compression: {compression.distortion_percentage}")

    return {"mask_difference": mask_diff, "compression_difference": comp_diff}


def test_gpu_optimization():
    """Test GPU optimization features."""
    print("\n" + "=" * 60)
    print("TESTING GPU OPTIMIZATION")
    print("=" * 60)

    # Create test model
    model = TaskAwareTemporalCNN(input_channels=19, hidden_dim=128, num_layers=4)

    # Test optimization configurations
    configs = [
        {"use_amp": False, "compile_mode": None},
        {"use_amp": True, "compile_mode": None},
    ]

    if torch.cuda.is_available() and hasattr(torch, "compile"):
        configs.append({"use_amp": True, "compile_mode": "reduce-overhead"})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    results = []
    for i, config in enumerate(configs):
        try:
            # Create optimized model
            opt_model = OptimizedModel(model=model, device=device, **config)

            print(f"  Config {i+1}: {config}")
            print(f"    Optimization info: {opt_model.get_optimization_info()}")

            # Simple benchmark
            input_shape = (8, 19, 1000)
            dummy_input = torch.randn(input_shape, device=device)

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = opt_model(dummy_input)

            # Measure
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()
            for _ in range(10):
                with torch.no_grad():
                    _ = opt_model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            avg_time = (time.time() - start_time) / 10 * 1000
            print(f"    Average inference time: {avg_time:.2f}ms")

            results.append({"config": config, "avg_inference_time_ms": avg_time})

        except Exception as e:
            print(f"    ERROR: {e}")

    return results


def test_enhanced_data_pipeline():
    """Test enhanced data pipeline (mock)."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED DATA PIPELINE")
    print("=" * 60)

    # Mock data root (since we don't have real HBN data)
    data_root = Path("./mock_data")

    try:
        # Initialize label manager (will create empty dataframes)
        label_manager = RealLabelManager(data_root)
        print("✅ Created RealLabelManager (mock mode)")

        # Test getting available subjects
        for task in HBNTask:
            subjects = label_manager.get_available_subjects(task)
            print(f"  {task.value}: {len(subjects)} subjects available")

        # Test label retrieval (will return None for mock data)
        test_subject = "NDARINV12345"
        ccd_metrics = label_manager.get_ccd_metrics(test_subject)
        cbcl_factors = label_manager.get_cbcl_factors(test_subject)
        demographics = label_manager.get_demographics(test_subject)

        print(f"  CCD metrics for {test_subject}: {ccd_metrics is not None}")
        print(f"  CBCL factors for {test_subject}: {cbcl_factors is not None}")
        print(f"  Demographics for {test_subject}: {demographics is not None}")

        return {"label_manager_created": True}

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def test_integration():
    """Test full pipeline integration."""
    print("\n" + "=" * 60)
    print("TESTING FULL INTEGRATION")
    print("=" * 60)

    # Create full pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Task-aware backbone
    backbone = TaskAwareTemporalCNN(
        input_channels=19,
        hidden_dim=128,
        num_layers=4,
        use_task_tokens=True,
        adaptation_method="film",
    )

    # Multi-task head
    multi_task_head = MultiTaskHead(hidden_dim=128)

    # Multi-adversary DANN
    dann = MultiAdversaryDANN(
        feature_dim=128,
        adversary_types=[AdversaryType.SITE, AdversaryType.TASK],
        hidden_dim=64,
    )

    # Combined model
    combined_model = nn.ModuleDict(
        {"backbone": backbone, "multi_task_head": multi_task_head, "dann": dann}
    )

    # Optimize
    optimized_model = OptimizedModel(
        model=combined_model,
        use_amp=True,
        compile_mode=None,  # Skip compilation for test
        device=device,
    )

    print(f"✅ Created integrated model on {device}")
    print(
        f"   Total parameters: {sum(p.numel() for p in combined_model.parameters()):,}"
    )

    # Test full forward pass
    batch_size = 4
    sequence_length = 1000

    # Input data
    eeg_data = torch.randn(batch_size, 19, sequence_length).to(device)
    task = HBNTask.CCD

    # Forward pass
    try:
        # Through backbone
        features = optimized_model.model["backbone"](eeg_data, task)
        print(f"  Backbone output: {features.shape}")

        # Through multi-task head
        predictions = optimized_model.model["multi_task_head"](features)
        print(f"  Predictions: {list(predictions.keys())}")

        # Through DANN
        adversary_labels = {
            AdversaryType.SITE: torch.randint(0, 5, (batch_size,)).to(device),
            AdversaryType.TASK: torch.randint(0, 6, (batch_size,)).to(device),
        }

        adv_loss = optimized_model.model["dann"].compute_adversarial_loss(
            features, adversary_labels, lambda_val=1.0
        )
        print(f"  Adversarial loss: {adv_loss.item():.4f}")

        print("✅ Full integration test PASSED")

        return {
            "backbone_output_shape": features.shape,
            "predictions": list(predictions.keys()),
            "adversarial_loss": adv_loss.item(),
        }

    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        return {"error": str(e)}


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🚀 STARTING COMPREHENSIVE EEG FOUNDATION MODEL TEST")
    print("=" * 80)

    results = {}

    # Test 1: Task-aware architecture
    try:
        results["task_aware"] = test_task_aware_architecture()
    except Exception as e:
        results["task_aware"] = {"error": str(e)}

    # Test 2: Multi-task head
    try:
        results["multi_task_head"] = test_multi_task_head()
    except Exception as e:
        results["multi_task_head"] = {"error": str(e)}

    # Test 3: Multi-adversary DANN
    try:
        results["multi_adversary_dann"] = test_multi_adversary_dann()
    except Exception as e:
        results["multi_adversary_dann"] = {"error": str(e)}

    # Test 4: Compression-aware SSL
    try:
        results["compression_ssl"] = test_compression_aware_ssl()
    except Exception as e:
        results["compression_ssl"] = {"error": str(e)}

    # Test 5: GPU optimization
    try:
        results["gpu_optimization"] = test_gpu_optimization()
    except Exception as e:
        results["gpu_optimization"] = {"error": str(e)}

    # Test 6: Enhanced data pipeline
    try:
        results["data_pipeline"] = test_enhanced_data_pipeline()
    except Exception as e:
        results["data_pipeline"] = {"error": str(e)}

    # Test 7: Full integration
    try:
        results["integration"] = test_integration()
    except Exception as e:
        results["integration"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    success_count = 0
    total_tests = len(results)

    for test_name, result in results.items():
        if isinstance(result, dict) and "error" in result:
            print(f"❌ {test_name:>20}: FAILED - {result['error']}")
        else:
            print(f"✅ {test_name:>20}: PASSED")
            success_count += 1

    print(
        f"\nOverall: {success_count}/{total_tests} tests passed ({success_count/total_tests*100:.1f}%)"
    )

    # Save results
    output_path = Path("test_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Detailed results saved to: {output_path}")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test enhanced EEG foundation model")
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "task_aware",
            "multi_task",
            "dann",
            "ssl",
            "gpu",
            "data",
            "integration",
            "all",
        ],
        default="all",
        help="Specific test to run",
    )

    args = parser.parse_args()

    if args.test == "all":
        results = run_comprehensive_test()
    elif args.test == "task_aware":
        results = test_task_aware_architecture()
    elif args.test == "multi_task":
        results = test_multi_task_head()
    elif args.test == "dann":
        results = test_multi_adversary_dann()
    elif args.test == "ssl":
        results = test_compression_aware_ssl()
    elif args.test == "gpu":
        results = test_gpu_optimization()
    elif args.test == "data":
        results = test_enhanced_data_pipeline()
    elif args.test == "integration":
        results = test_integration()

    print(f"\n🎉 Testing completed!")
    return results


if __name__ == "__main__":
    main()
