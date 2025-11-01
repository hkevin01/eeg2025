"""
C1 Calibration - Linear Post-Processing for Variance Reduction
Fits y_cal = a*y_pred + b on validation set to minimize bias
Expected improvement: 1e-5 to 5e-5
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
from sklearn.linear_model import Ridge
import json

# Import model architecture
from train_c1_phase1_aggressive import EnhancedCompactCNN, load_c1_data


def load_ensemble_predictions(checkpoint_paths, X_val):
    """Load 5-seed ensemble and get predictions"""
    predictions = []
    
    for path in checkpoint_paths:
        # Load model
        model = EnhancedCompactCNN()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            X_batch = torch.FloatTensor(X_val)
            pred = model(X_batch).squeeze(-1).numpy()
            predictions.append(pred)
    
    # Ensemble mean
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred


def fit_calibration(y_pred, y_true, alpha=1.0):
    """
    Fit linear calibration: y_cal = a*y_pred + b
    Uses Ridge regression for stability
    
    Args:
        y_pred: Predicted values (N,)
        y_true: True values (N,)
        alpha: L2 regularization strength (default 1.0)
    
    Returns:
        (a, b): Calibration coefficients
    """
    # Reshape for sklearn
    X = y_pred.reshape(-1, 1)
    y = y_true.reshape(-1, 1)
    
    # Fit Ridge regression
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    
    a = model.coef_[0, 0]
    b = model.intercept_[0]
    
    return a, b


def apply_calibration(y_pred, a, b):
    """Apply calibration transform"""
    return a * y_pred + b


def main():
    print("="*80)
    print("C1 Calibration - Linear Post-Processing")
    print("="*80)
    print()
    
    # Load validation data
    print("Loading validation data...")
    X_train, y_train, X_val, y_val = load_c1_data()
    print(f"Validation: X {X_val.shape}, y {y_val.shape}")
    print()
    
    # Checkpoint paths
    seeds = [42, 123, 456, 789, 1337]
    checkpoint_paths = [f'checkpoints/c1_phase1_seed{seed}_ema_best.pt' 
                       for seed in seeds]
    
    # Get ensemble predictions on validation set
    print("Loading 5-seed ensemble and computing predictions...")
    y_pred = load_ensemble_predictions(checkpoint_paths, X_val)
    print(f"Ensemble predictions: {y_pred.shape}")
    print()
    
    # Convert y_val to numpy if needed
    if torch.is_tensor(y_val):
        y_val = y_val.numpy()
    
    # Compute baseline NRMSE (before calibration)
    baseline_mse = np.mean((y_pred - y_val) ** 2)
    baseline_nrmse = np.sqrt(baseline_mse)
    print(f"Baseline NRMSE (before calibration): {baseline_nrmse:.6f}")
    print()
    
    # Fit calibration with different alpha values
    alphas = [0.1, 0.5, 1.0, 5.0, 10.0]
    best_alpha = None
    best_nrmse = float('inf')
    best_a = None
    best_b = None
    
    print("Fitting calibration with different regularization strengths:")
    print("-" * 60)
    for alpha in alphas:
        a, b = fit_calibration(y_pred, y_val, alpha=alpha)
        y_cal = apply_calibration(y_pred, a, b)
        
        # Compute calibrated NRMSE
        cal_mse = np.mean((y_cal - y_val) ** 2)
        cal_nrmse = np.sqrt(cal_mse)
        improvement = baseline_nrmse - cal_nrmse
        
        print(f"Alpha {alpha:5.1f}: a={a:7.5f}, b={b:8.5f}, "
              f"NRMSE={cal_nrmse:.6f}, Δ={improvement:+.6f}")
        
        if cal_nrmse < best_nrmse:
            best_nrmse = cal_nrmse
            best_alpha = alpha
            best_a = a
            best_b = b
    
    print("-" * 60)
    print()
    
    # Report best calibration
    print("Best Calibration:")
    print(f"  Alpha: {best_alpha}")
    print(f"  a: {best_a:.6f}")
    print(f"  b: {best_b:.6f}")
    print(f"  NRMSE: {best_nrmse:.6f}")
    print(f"  Improvement: {baseline_nrmse - best_nrmse:+.6f}")
    print()
    
    # Save calibration parameters
    calibration_params = {
        'a': float(best_a),
        'b': float(best_b),
        'alpha': float(best_alpha),
        'baseline_nrmse': float(baseline_nrmse),
        'calibrated_nrmse': float(best_nrmse),
        'improvement': float(baseline_nrmse - best_nrmse)
    }
    
    with open('c1_calibration_params.json', 'w') as f:
        json.dump(calibration_params, f, indent=2)
    
    print(f"✅ Calibration parameters saved to c1_calibration_params.json")
    print()
    
    # Statistics
    print("="*80)
    print("Summary:")
    print(f"  Baseline NRMSE:    {baseline_nrmse:.6f}")
    print(f"  Calibrated NRMSE:  {best_nrmse:.6f}")
    print(f"  Improvement:       {baseline_nrmse - best_nrmse:+.6f}")
    print(f"  Relative:          {(baseline_nrmse - best_nrmse)/baseline_nrmse * 100:+.4f}%")
    print()
    
    # Prediction statistics
    print("Prediction Statistics:")
    print(f"  Original predictions:")
    print(f"    Mean: {y_pred.mean():.6f}, Std: {y_pred.std():.6f}")
    print(f"    Min: {y_pred.min():.6f}, Max: {y_pred.max():.6f}")
    print()
    y_cal_best = apply_calibration(y_pred, best_a, best_b)
    print(f"  Calibrated predictions:")
    print(f"    Mean: {y_cal_best.mean():.6f}, Std: {y_cal_best.std():.6f}")
    print(f"    Min: {y_cal_best.min():.6f}, Max: {y_cal_best.max():.6f}")
    print()
    print(f"  True values:")
    print(f"    Mean: {y_val.mean():.6f}, Std: {y_val.std():.6f}")
    print(f"    Min: {y_val.min():.6f}, Max: {y_val.max():.6f}")
    print("="*80)


if __name__ == "__main__":
    main()
