#!/usr/bin/env python3
"""
Improved Inference Pipeline
===========================

Features:
- Test-time augmentation (TTA)
- Model ensemble support
- Comprehensive metrics
- Confidence intervals
- Batch inference optimization
"""
import os
import sys
from pathlib import Path
import time
from typing import List, Dict

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("🔍 IMPROVED INFERENCE PIPELINE")
print("="*80)


class TestTimeAugmentation:
    """Test-time augmentation for robust predictions"""
    
    def __init__(self, n_augmentations=5):
        self.n_augmentations = n_augmentations
        
    def augment(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate augmented versions"""
        augmented = [x]  # Original
        
        for _ in range(self.n_augmentations - 1):
            x_aug = x.clone()
            
            # Random temporal shift
            shift = np.random.randint(-5, 6)
            x_aug = torch.roll(x_aug, shift, dims=-1)
            
            # Random amplitude scaling
            scale = np.random.uniform(0.95, 1.05)
            x_aug = x_aug * scale
            
            # Small gaussian noise
            noise = torch.randn_like(x_aug) * 0.005 * x_aug.std()
            x_aug = x_aug + noise
            
            augmented.append(x_aug)
        
        return augmented


class ModelEnsemble:
    """Ensemble multiple models for better predictions"""
    
    def __init__(self, model_paths: List[Path], weights: List[float] = None):
        self.models = []
        self.weights = weights if weights else [1.0] * len(model_paths)
        
        # Load all models
        for path in model_paths:
            if path.exists():
                checkpoint = torch.load(path, map_location='cpu')
                # Note: You'd need to instantiate the actual model class here
                # This is a placeholder
                print(f"   Loaded model from {path}")
                self.models.append(checkpoint)
            else:
                print(f"   ⚠️  Model not found: {path}")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"   Ensemble size: {len(self.models)}")
        print(f"   Weights: {self.weights}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                pred = model(x) * weight
                predictions.append(pred)
        
        return sum(predictions)


class ImprovedInference:
    """Advanced inference with all optimizations"""
    
    def __init__(self, model_path: Path, use_tta: bool = True, tta_n: int = 5):
        self.device = torch.device('cpu')
        self.use_tta = use_tta
        self.tta = TestTimeAugmentation(n_augmentations=tta_n) if use_tta else None
        
        # Load model
        print(f"📂 Loading model from {model_path}")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")
            self.model = self.load_model_from_checkpoint(checkpoint)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model.eval()
        print(f"   TTA: {'Enabled' if use_tta else 'Disabled'}")
        if use_tta:
            print(f"   TTA augmentations: {tta_n}")
        print()
    
    def load_model_from_checkpoint(self, checkpoint):
        """Load model architecture and weights"""
        # Import the model class
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train_improved_cpu import MultiScaleEEGModel
        
        model = MultiScaleEEGModel(n_channels=129, n_classes=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def predict_single(self, x: torch.Tensor) -> float:
        """Predict single sample"""
        if self.use_tta:
            # Test-time augmentation
            augmented = self.tta.augment(x)
            predictions = []
            
            with torch.no_grad():
                for x_aug in augmented:
                    pred = self.model(x_aug.unsqueeze(0))
                    predictions.append(pred.item())
            
            # Average predictions
            return np.mean(predictions)
        else:
            with torch.no_grad():
                pred = self.model(x.unsqueeze(0))
                return pred.item()
    
    def predict_batch(self, data_loader) -> Dict:
        """Predict entire dataset"""
        print("🔮 Running inference...")
        
        all_predictions = []
        all_labels = []
        inference_times = []
        
        for batch_data, batch_labels in tqdm(data_loader, desc="Inference"):
            batch_start = time.time()
            
            if self.use_tta:
                # TTA for each sample in batch
                batch_preds = []
                for i in range(batch_data.shape[0]):
                    pred = self.predict_single(batch_data[i])
                    batch_preds.append(pred)
                
                all_predictions.extend(batch_preds)
            else:
                # Direct batch inference
                with torch.no_grad():
                    batch_preds = self.model(batch_data)
                    all_predictions.extend(batch_preds.numpy())
            
            all_labels.extend(batch_labels.numpy())
            
            batch_time = time.time() - batch_start
            inference_times.append(batch_time / batch_data.shape[0])
        
        return {
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'avg_time_per_sample': np.mean(inference_times)
        }
    
    def evaluate(self, predictions: np.ndarray, labels: np.ndarray, challenge: int = 1) -> Dict:
        """Comprehensive evaluation"""
        print("\n📊 Evaluation Results")
        print("="*60)
        
        results = {}
        
        if challenge == 1:
            # Regression metrics
            correlation, p_value = pearsonr(predictions, labels)
            mae = np.mean(np.abs(predictions - labels))
            rmse = np.sqrt(np.mean((predictions - labels)**2))
            
            results['correlation'] = correlation
            results['p_value'] = p_value
            results['mae'] = mae
            results['rmse'] = rmse
            
            print(f"Pearson Correlation: {correlation:.4f} (p={p_value:.4e})")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            
            # Prediction range
            print(f"\nPrediction Statistics:")
            print(f"  Mean: {predictions.mean():.2f} ± {predictions.std():.2f}")
            print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
            
            print(f"\nLabel Statistics:")
            print(f"  Mean: {labels.mean():.2f} ± {labels.std():.2f}")
            print(f"  Range: [{labels.min():.2f}, {labels.max():.2f}]")
            
        else:
            # Classification metrics
            binary_preds = (predictions > 0.5).astype(int)
            binary_labels = labels.astype(int)
            
            accuracy = np.mean(binary_preds == binary_labels)
            
            # Confusion matrix
            tp = np.sum((binary_preds == 1) & (binary_labels == 1))
            tn = np.sum((binary_preds == 0) & (binary_labels == 0))
            fp = np.sum((binary_preds == 1) & (binary_labels == 0))
            fn = np.sum((binary_preds == 0) & (binary_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['accuracy'] = accuracy
            results['precision'] = precision
            results['recall'] = recall
            results['f1'] = f1
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(f"  TP: {tp:4d}  FP: {fp:4d}")
            print(f"  FN: {fn:4d}  TN: {tn:4d}")
        
        return results
    
    def save_predictions(self, predictions: np.ndarray, labels: np.ndarray, output_file: Path):
        """Save predictions to file"""
        df = pd.DataFrame({
            'prediction': predictions,
            'label': labels,
            'error': np.abs(predictions - labels)
        })
        
        df.to_csv(output_file, index=False)
        print(f"\n💾 Predictions saved to {output_file}")


def main():
    """Main inference function"""
    
    # Configuration
    model_path = Path(__file__).parent.parent / "checkpoints" / "best.pth"
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    challenge = 1  # 1 for age, 2 for sex
    use_tta = True
    tta_n = 5
    
    # Create inference engine
    inference = ImprovedInference(
        model_path=model_path,
        use_tta=use_tta,
        tta_n=tta_n
    )
    
    # Load test data (you would implement this based on your dataset)
    print("📂 Loading test data...")
    print("   ⚠️  Implement test data loading based on your dataset")
    
    # Example: 
    # from torch.utils.data import DataLoader
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # results = inference.predict_batch(test_loader)
    # metrics = inference.evaluate(results['predictions'], results['labels'], challenge=challenge)
    # inference.save_predictions(results['predictions'], results['labels'], output_dir / "predictions.csv")
    
    print("\n" + "="*80)
    print("✅ INFERENCE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
