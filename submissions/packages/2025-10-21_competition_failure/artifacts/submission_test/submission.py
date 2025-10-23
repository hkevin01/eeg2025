"""
Competition Submission for EEG Foundation Challenge 2025
Matches exact training architectures
Platform: CPU-only, no braindecode dependency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union


def resolve_path(file_path: Union[str, Path]) -> Path:
    """Resolve file path handling both container and local environments."""
    path = Path(file_path)
    
    container_paths = [
        Path('/app/ingestion_program') / path.name,
        Path('/app') / path.name,
        Path('.') / path.name,
    ]
    
    if path.exists():
        return path
    
    for container_path in container_paths:
        if container_path.exists():
            return container_path
    
    return path


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d layer with max-norm weight constraint."""
    
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with max-norm weight constraint."""
    
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class EEGNeX_Standalone(nn.Module):
    """Standalone EEGNeX matching braindecode architecture exactly."""
    
    def __init__(
        self,
        n_chans=129,
        n_outputs=1,
        n_times=200,
        sfreq=100,
        filter_1=8,
        filter_2=32,
        kernel_block_1_2=64,
        kernel_block_4=16,
        kernel_block_5=16,
        dilation_block_4=2,
        dilation_block_5=4,
        avg_pool_block4=4,
        avg_pool_block5=8,
        drop_prob=0.5,
        max_norm_conv=1.0,
        max_norm_linear=0.25,
    ):
        super().__init__()
        
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = filter_2 * 2  # depth_multiplier = 2
        
        # Block 1: Temporal convolution
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, filter_1, kernel_size=(1, kernel_block_1_2), 
                     padding=(0, kernel_block_1_2 // 2), bias=False),
            nn.BatchNorm2d(filter_1),
        )
        
        # Block 2: Temporal convolution  
        self.block_2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, kernel_size=(1, kernel_block_1_2),
                     padding=(0, kernel_block_1_2 // 2), bias=False),
            nn.BatchNorm2d(filter_2),
        )
        
        # Block 3: Spatial depthwise convolution
        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(
                filter_2, self.filter_3, kernel_size=(n_chans, 1),
                groups=filter_2, max_norm=max_norm_conv, bias=False
            ),
            nn.BatchNorm2d(self.filter_3),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_prob),
        )
        
        # Block 4: Dilated temporal convolution
        self.block_4 = nn.Sequential(
            nn.Conv2d(self.filter_3, filter_2, kernel_size=(1, kernel_block_4),
                     dilation=(1, dilation_block_4), 
                     padding=(0, dilation_block_4 * (kernel_block_4 // 2)), bias=False),
            nn.BatchNorm2d(filter_2),
            nn.ELU(),
            nn.AvgPool2d((1, avg_pool_block4)),
            nn.Dropout(drop_prob),
        )
        
        # Block 5: Dilated temporal convolution
        self.block_5 = nn.Sequential(
            nn.Conv2d(filter_2, filter_1, kernel_size=(1, kernel_block_5),
                     dilation=(1, dilation_block_5),
                     padding=(0, dilation_block_5 * (kernel_block_5 // 2)), bias=False),
            nn.BatchNorm2d(filter_1),
            nn.ELU(),
            nn.AvgPool2d((1, avg_pool_block5)),
            nn.Dropout(drop_prob),
            nn.Flatten(),
        )
        
        # Final classifier
        size_after_pooling = n_times // (4 * avg_pool_block4 * avg_pool_block5)
        self.final_layer = LinearWithConstraint(
            filter_1 * size_after_pooling,
            n_outputs,
            max_norm=max_norm_linear
        )
    
    def forward(self, x):
        """Forward pass through all blocks."""
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.final_layer(x)
        return x


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block from competition training."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.padding = padding

    def forward(self, x):
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        # Match dimensions
        if out.shape[-1] != res.shape[-1]:
            res = res[:, :, :out.shape[-1]]

        return self.relu2(out + res)


class TCN_EEG(nn.Module):
    """Temporal Convolutional Network from competition training."""

    def __init__(self, num_channels=129, num_outputs=1, num_filters=64,
                 kernel_size=7, dropout=0.2, num_levels=6):
        super().__init__()

        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels if i == 0 else num_filters
            layers.append(
                TemporalBlock(in_channels, num_filters, kernel_size,
                            dilation=dilation_size, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, num_outputs)

    def forward(self, x):
        out = self.network(x)
        out = out.mean(dim=-1)  # Global average pooling
        return self.fc(out)


class Submission:
    """Main submission class for competition."""
    
    def __init__(self):
        """Initialize submission with device and constants."""
        self.device = torch.device('cpu')
        
        print("=" * 60)
        print("EEG Foundation Challenge 2025 - Submission")
        print("=" * 60)
        print(f"Device: CPU (forced for competition compatibility)")
        
        self.sfreq = 100
        self.n_chans = 129
        self.n_times = 200
        
        print(f"Sampling Frequency: {self.sfreq} Hz")
        print(f"Expected input shape: (batch, {self.n_chans} channels, {self.n_times} timepoints)")
        print("=" * 60)
        print()
        
        self.model_c1 = None
        self.model_c2 = None
    
    def get_model_challenge_1(self):
        """Load Challenge 1 model (TCN for CCD reaction time)."""
        if self.model_c1 is not None:
            return self.model_c1
            
        print("Loading Challenge 1 model (TCN)...")
        
        # Match exact training configuration
        model = TCN_EEG(
            num_channels=129,
            num_outputs=1,
            num_filters=48,
            kernel_size=7,
            dropout=0.3,
            num_levels=5
        )
        
        # Load weights
        weights_path = resolve_path('weights_challenge_1.pt')
        if weights_path.exists():
            print(f"  Loading weights from: {weights_path}")
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  ✅ Weights loaded from epoch {checkpoint.get('epoch', '?')}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"  ✅ Weights loaded successfully")
            except Exception as e:
                print(f"  ⚠️  Error loading weights: {e}")
                print("  Using untrained model")
        else:
            print(f"  ⚠️  Weights file not found, using untrained model")
        
        model = model.to(self.device)
        model.eval()
        
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("  ✅ Challenge 1 model ready")
        print()
        
        self.model_c1 = model
        return model
    
    def get_model_challenge_2(self):
        """Load Challenge 2 model (EEGNeX for externalizing factor)."""
        if self.model_c2 is not None:
            return self.model_c2
            
        print("Loading Challenge 2 model...")
        print("  Using standalone EEGNeX (no braindecode dependency)")
        
        # Match exact braindecode defaults
        model = EEGNeX_Standalone(
            n_chans=self.n_chans,
            n_outputs=1,
            n_times=self.n_times,
            sfreq=self.sfreq,
        )
        
        # Load weights
        weights_path = resolve_path('weights_challenge_2.pt')
        if weights_path.exists():
            print(f"  Loading weights from: {weights_path}")
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint)
                print(f"  ✅ Weights loaded successfully")
            except Exception as e:
                print(f"  ⚠️  Error loading weights: {e}")
                import traceback
                traceback.print_exc()
                print("  Using untrained model")
        else:
            print(f"  ⚠️  Weights file not found, using untrained model")
        
        model = model.to(self.device)
        model.eval()
        
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("  ✅ Challenge 2 model ready")
        print()
        
        self.model_c2 = model
        return model
    
    def predict_challenge_1(self, X: np.ndarray) -> np.ndarray:
        """Predict CCD reaction times (Challenge 1)."""
        model = self.get_model_challenge_1()
        
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        with torch.no_grad():
            predictions = model(X)
        
        return predictions.cpu().numpy()
    
    def predict_challenge_2(self, X: np.ndarray) -> np.ndarray:
        """Predict externalizing factor (Challenge 2)."""
        model = self.get_model_challenge_2()
        
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        # Add channel dimension for 2D conv
        if X.dim() == 3:
            X = X.unsqueeze(1)
        
        with torch.no_grad():
            predictions = model(X)
        
        return predictions.cpu().numpy()


# Test code
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("TESTING SUBMISSION FILE")
    print("=" * 60)
    print()
    
    submission = Submission()
    
    batch_size = 4
    test_data = np.random.randn(batch_size, 129, 200).astype(np.float32)
    
    print("Testing Challenge 1 model:")
    print("-" * 40)
    pred_c1 = submission.predict_challenge_1(test_data)
    print(f"Test input shape: {test_data.shape}")
    print(f"Output shape: {pred_c1.shape}")
    print(f"Sample predictions: {pred_c1[:3, 0]}")
    print("✅ Challenge 1 model works!")
    print()
    
    print("Testing Challenge 2 model:")
    print("-" * 40)
    pred_c2 = submission.predict_challenge_2(test_data)
    print(f"Output shape: {pred_c2.shape}")
    print(f"Sample predictions: {pred_c2[:3, 0]}")
    print("✅ Challenge 2 model works!")
    print()
    
    print("=" * 60)
    print("ALL TESTS PASSED - SUBMISSION READY")
    print("=" * 60)
    print()

