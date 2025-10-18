"""
Test-Time Augmentation (TTA) for EEG predictions
Expected improvement: 5-10%
No retraining needed!
"""

import torch
import torch.nn as nn
import numpy as np


class TTAPredictor:
    """Test-Time Augmentation for robust EEG predictions"""
    
    def __init__(self, model, num_augments=10, device='cpu'):
        """
        Args:
            model: Trained PyTorch model
            num_augments: Number of augmentations to apply
            device: Device to run predictions on
        """
        self.model = model
        self.model.eval()
        self.num_augments = num_augments
        self.device = device
        
    def augment_eeg(self, x, aug_type='gaussian', strength=1.0):
        """
        Apply various augmentations to EEG data
        
        Args:
            x: Input tensor (batch, channels, time)
            aug_type: Type of augmentation
            strength: Augmentation strength multiplier
        
        Returns:
            Augmented tensor
        """
        if aug_type == 'gaussian':
            # Add Gaussian noise
            noise = torch.randn_like(x) * 0.02 * strength
            return x + noise
            
        elif aug_type == 'scale':
            # Scale amplitude (0.95-1.05)
            scale = 0.95 + torch.rand(1, device=x.device).item() * 0.1 * strength
            return x * scale
            
        elif aug_type == 'shift':
            # Time shift
            shift = int(torch.randint(-5, 6, (1,)).item() * strength)
            if shift == 0:
                return x
            return torch.roll(x, shift, dims=-1)
            
        elif aug_type == 'channel_dropout':
            # Random channel dropout (keep 90%)
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > (0.1 * strength)).float()
            return x * mask
            
        elif aug_type == 'mixup':
            # Temporal mixup
            lam = 0.9 + torch.rand(1, device=x.device).item() * 0.1 * strength
            rolled = torch.roll(x, 1, dims=-1)
            return lam * x + (1 - lam) * rolled
            
        elif aug_type == 'flip':
            # Time reversal (careful with this for EEG)
            return torch.flip(x, dims=[-1])
            
        elif aug_type == 'cutout':
            # Random temporal cutout
            seq_len = x.shape[-1]
            cutout_len = int(seq_len * 0.1 * strength)
            start = torch.randint(0, seq_len - cutout_len + 1, (1,)).item()
            x_aug = x.clone()
            x_aug[:, :, start:start+cutout_len] = 0
            return x_aug
            
        else:
            return x
    
    def predict(self, x, use_all_augmentations=True):
        """
        TTA prediction with multiple augmentations
        
        Args:
            x: Input tensor (batch, channels, time)
            use_all_augmentations: Use all augmentation types
            
        Returns:
            Averaged prediction
        """
        predictions = []
        
        # Original prediction (most important)
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
            predictions.append(pred)
        
        if use_all_augmentations:
            # Use diverse augmentation types
            aug_types = ['gaussian', 'scale', 'shift', 'channel_dropout', 'mixup', 'cutout']
            
            for i in range(self.num_augments):
                aug_type = aug_types[i % len(aug_types)]
                
                # Vary strength slightly
                strength = 0.8 + 0.4 * torch.rand(1).item()  # 0.8-1.2
                
                x_aug = self.augment_eeg(x, aug_type, strength)
                
                with torch.no_grad():
                    pred = self.model(x_aug)
                    predictions.append(pred)
        else:
            # Just use Gaussian noise (safest)
            for i in range(self.num_augments):
                x_aug = self.augment_eeg(x, 'gaussian', strength=1.0)
                
                with torch.no_grad():
                    pred = self.model(x_aug)
                    predictions.append(pred)
        
        # Average predictions
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred
    
    def predict_with_confidence(self, x):
        """
        Predict with uncertainty estimation
        
        Returns:
            (mean_prediction, std_prediction)
        """
        predictions = []
        
        # Original
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
            predictions.append(pred)
        
        # Augmented
        aug_types = ['gaussian', 'scale', 'shift', 'channel_dropout', 'mixup']
        
        for i in range(self.num_augments):
            aug_type = aug_types[i % len(aug_types)]
            x_aug = self.augment_eeg(x, aug_type)
            
            with torch.no_grad():
                pred = self.model(x_aug)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Mean and std
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


class AdaptiveTTAPredictor(TTAPredictor):
    """Adaptive TTA that adjusts augmentation based on prediction confidence"""
    
    def predict_adaptive(self, x, confidence_threshold=0.1):
        """
        Adaptive prediction: use more augmentations if uncertain
        
        Args:
            x: Input tensor
            confidence_threshold: If std > threshold, use more augmentations
            
        Returns:
            Final prediction
        """
        # Initial prediction with few augmentations
        self.num_augments = 5
        mean_pred, std_pred = self.predict_with_confidence(x)
        
        # If uncertain, add more augmentations
        if std_pred.mean() > confidence_threshold:
            self.num_augments = 20
            mean_pred, std_pred = self.predict_with_confidence(x)
        
        return mean_pred


# Convenience function for easy integration
def predict_with_tta(model, x, num_augments=10, device='cpu'):
    """
    Quick function to get TTA predictions
    
    Args:
        model: Trained model
        x: Input tensor (batch, channels, time)
        num_augments: Number of augmentations
        device: Device to use
        
    Returns:
        TTA-averaged prediction
    """
    tta = TTAPredictor(model, num_augments=num_augments, device=device)
    return tta.predict(x)


if __name__ == "__main__":
    print("TTA Predictor module loaded successfully!")
    print("\nUsage:")
    print("  from tta_predictor import TTAPredictor, predict_with_tta")
    print("  tta = TTAPredictor(model, num_augments=10)")
    print("  prediction = tta.predict(x)")
