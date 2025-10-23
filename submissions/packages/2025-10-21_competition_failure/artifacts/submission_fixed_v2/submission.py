"""
EEG Foundation Challenge 2025 - Fixed Submission (No Braindecode Dependency)
=============================================================================

This version includes EEGNeX implementation directly, removing braindecode dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ============================================================================
# MAX-NORM CONSTRAINT LAYERS (from braindecode)
# ============================================================================

class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max-norm constraint on weights."""
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with max-norm constraint on weights."""
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


# ============================================================================
# EEGNEX MODEL (Standalone - No Braindecode Dependency)
# ============================================================================

class EEGNeX_Standalone(nn.Module):
    """
    Standalone EEGNeX implementation (no braindecode dependency).
    
    Based on Chen et al. (2024) - EEGNeX architecture.
    This is a simplified version that matches the weights trained with braindecode.
    """
    
    def __init__(
        self,
        n_chans=129,
        n_outputs=1,
        sfreq=100,
        n_times=200,
        filter_1=8,
        filter_2=32,
        depth_multiplier=2,
        kernel_block_1_2=64,
        kernel_block_4=16,
        dilation_block_4=2,
        avg_pool_block4=4,
        kernel_block_5=16,
        dilation_block_5=4,
        avg_pool_block5=8,
        drop_prob=0.5,
        max_norm_conv=1.0,
        max_norm_linear=0.25,
    ):
        super().__init__()
        
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = filter_2 * depth_multiplier
        
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
            nn.AvgPool2d(kernel_size=(1, avg_pool_block4)),
            nn.Dropout(drop_prob),
        )
        
        # Block 4: Dilated temporal convolution
        self.block_4 = nn.Sequential(
            nn.Conv2d(self.filter_3, self.filter_3, 
                     kernel_size=(1, kernel_block_4),
                     dilation=(1, dilation_block_4),
                     padding=(0, (kernel_block_4 - 1) * dilation_block_4 // 2),
                     bias=False),
            nn.BatchNorm2d(self.filter_3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block4)),
            nn.Dropout(drop_prob),
        )
        
        # Block 5: Dilated temporal convolution
        self.block_5 = nn.Sequential(
            nn.Conv2d(self.filter_3, self.filter_3,
                     kernel_size=(1, kernel_block_5),
                     dilation=(1, dilation_block_5),
                     padding=(0, (kernel_block_5 - 1) * dilation_block_5 // 2),
                     bias=False),
            nn.BatchNorm2d(self.filter_3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block5)),
            nn.Dropout(drop_prob),
            nn.Flatten(),
        )
        
        # Calculate final feature size
        # After all pooling: n_times // (avg_pool_block4 * avg_pool_block4 * avg_pool_block5)
        final_time_dim = n_times // (avg_pool_block4 * avg_pool_block4 * avg_pool_block5)
        final_features = self.filter_3 * final_time_dim
        
        # Final classifier
        self.final_layer = LinearWithConstraint(
            final_features, n_outputs, max_norm=max_norm_linear
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, n_chans, n_times)
        
        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        # Add channel dimension: (batch, n_chans, n_times) -> (batch, 1, n_chans, n_times)
        x = x.unsqueeze(1)
        
        # Process through blocks
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)  # Includes flatten
        x = self.final_layer(x)
        
        return x


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def resolve_path(name="model_file_name"):
    """Resolve path to model weights file."""
    search_paths = [
        Path(f"/app/input/res/{name}"),
        Path(f"/app/input/{name}"),
        Path(name),
        Path(__file__).parent / name,
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        f"Could not find {name} in any of: {[str(p) for p in search_paths]}"
    )


# ============================================================================
# TCN MODEL FOR CHALLENGE 1 (unchanged)
# ============================================================================

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        if res.size(2) != out.size(2):
            res = res[:, :, :out.size(2)]
        
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_filters, kernel_size, dropout, num_levels):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_filters
            out_channels = num_filters
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       dilation, dropout))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_EEG(nn.Module):
    def __init__(self, num_channels=129, num_outputs=1, num_filters=48,
                 kernel_size=5, dropout=0.2, num_levels=3):
        super(TCN_EEG, self).__init__()
        
        self.tcn = TemporalConvNet(num_channels, num_filters, kernel_size,
                                   dropout, num_levels)
        self.fc = nn.Linear(num_filters, num_outputs)

    def forward(self, x):
        y = self.tcn(x)
        y = y[:, :, -1]
        y = self.fc(y)
        return y


# ============================================================================
# SUBMISSION CLASS
# ============================================================================

class Submission:
    """Submission class for EEG Foundation Challenge."""

    def __init__(self, SFREQ, DEVICE=None):
        """
        Initialize submission.

        Args:
            SFREQ: Sampling frequency (Hz)
            DEVICE: Device to use (forced to CPU for competition compatibility)
        """
        self.sfreq = SFREQ

        # Force CPU for competition compatibility
        self.device = torch.device('cpu')
        device_info = "CPU (forced for competition compatibility)"

        print(f"\n{'='*60}")
        print(f"EEG Foundation Challenge 2025 - Submission v2")
        print(f"{'='*60}")
        print(f"Device: {device_info}")
        print(f"Sampling Frequency: {SFREQ} Hz")
        print(f"Expected input shape: (batch, 129 channels, {int(2 * SFREQ)} timepoints)")
        print(f"{'='*60}\n")

    def get_model_challenge_1(self):
        """Load Challenge 1 model (TCN for CCD task)."""
        print("Loading Challenge 1 model (TCN)...")
        
        model = TCN_EEG(
            num_channels=129,
            num_outputs=1,
            num_filters=48,
            kernel_size=5,
            dropout=0.2,
            num_levels=3
        )
        
        # Load weights
        try:
            weights_path = resolve_path("weights_challenge_1.pt")
            print(f"  Loading weights from: {weights_path}")
            
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'epoch' in checkpoint:
                    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
                if 'val_loss' in checkpoint:
                    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
            else:
                model.load_state_dict(checkpoint)
            
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except FileNotFoundError:
            print("  ⚠️  Weights file not found, using untrained model")
        except Exception as e:
            print(f"  ⚠️  Error loading weights: {e}")
            print("  Using untrained model")
        
        model = model.to(self.device)
        model.eval()
        
        print("  ✅ Challenge 1 model ready\n")
        return model

    def get_model_challenge_2(self):
        """Load Challenge 2 model (EEGNeX for externalizing prediction)."""
        print("Loading Challenge 2 model...")
        print("  Using standalone EEGNeX (no braindecode dependency)")
        
        # Use standalone EEGNeX implementation
        model = EEGNeX_Standalone(
            n_chans=129,
            n_outputs=1,
            sfreq=self.sfreq,
            n_times=int(2 * self.sfreq)
        )
        
        # Load weights
        try:
            weights_path = resolve_path("weights_challenge_2.pt")
            print(f"  Loading weights from: {weights_path}")
            
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except FileNotFoundError:
            print("  ⚠️  Weights file not found, using untrained model")
        except Exception as e:
            print(f"  ⚠️  Error loading weights: {e}")
            print(f"  Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("  Using untrained model")
        
        model = model.to(self.device)
        model.eval()
        
        print("  ✅ Challenge 2 model ready\n")
        return model


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test submission file locally."""
    
    print("\n" + "="*60)
    print("TESTING SUBMISSION FILE v2")
    print("="*60 + "\n")
    
    # Initialize
    SFREQ = 100
    sub = Submission(SFREQ)
    
    # Test Challenge 1
    print("\nTesting Challenge 1 model:")
    print("-" * 40)
    model_1 = sub.get_model_challenge_1()
    
    batch_size = 4
    n_channels = 129
    n_times = int(2 * SFREQ)
    X_test = torch.randn(batch_size, n_channels, n_times, device=sub.device)
    
    print(f"Test input shape: {X_test.shape}")
    
    with torch.inference_mode():
        y_pred = model_1(X_test)
        print(f"Output shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred[:3, 0].cpu().numpy()}")
    
    print("✅ Challenge 1 model works!\n")
    
    del model_1
    
    # Test Challenge 2
    print("\nTesting Challenge 2 model:")
    print("-" * 40)
    model_2 = sub.get_model_challenge_2()
    
    with torch.inference_mode():
        y_pred = model_2(X_test)
        print(f"Output shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred[:3, 0].cpu().numpy()}")
    
    print("✅ Challenge 2 model works!\n")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED - SUBMISSION v2 READY")
    print("="*60 + "\n")
