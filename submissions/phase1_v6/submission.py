"""
EEG Foundation Challenge 2025 - Phase 1 V6
Clean submission with correct TCN architecture
"""

import torch
import torch.nn as nn
from pathlib import Path


def resolve_path(filename):
    """Find file in competition or local environment"""
    search_paths = [
        f"/app/input/res/{filename}",
        f"/app/input/{filename}",
        filename,
        str(Path(__file__).parent / filename),
    ]
    for path in search_paths:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}")


class TemporalBlock(nn.Module):
    """Temporal block for TCN with BatchNorm."""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

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
    """TCN for EEG regression - Challenge 1 model"""
    def __init__(self, num_channels=129, num_outputs=1, num_filters=48,
                 kernel_size=7, num_levels=5, dropout=0.3):
        super(TCN_EEG, self).__init__()

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
        out = out.mean(dim=-1)
        out = self.fc(out)
        return out


class Submission:
    """EEG 2025 Competition Submission"""
    
    def __init__(self, SFREQ, DEVICE):
        # Device handling
        if isinstance(DEVICE, str):
            self.device = torch.device(DEVICE)
        elif isinstance(DEVICE, torch.device):
            self.device = DEVICE
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sfreq = SFREQ
        self.n_chans = 129
        self.n_times = int(2 * SFREQ)
        self.model_c1 = None
        self.model_c2 = None
    
    def get_model_challenge_1(self):
        """Load Challenge 1 model"""
        if self.model_c1 is not None:
            return self.model_c1
        
        model = TCN_EEG(num_channels=self.n_chans, num_outputs=1, 
                       num_filters=48, kernel_size=7, num_levels=5, dropout=0.3)
        
        weights_path = resolve_path('weights_challenge_1.pt')
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        self.model_c1 = model
        return model
    
    def get_model_challenge_2(self):
        """Load Challenge 2 model"""
        if self.model_c2 is not None:
            return self.model_c2
        
        from braindecode.models import EEGNeX
        
        model = EEGNeX(
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_outputs=1,
            sfreq=self.sfreq,
        )
        
        weights_path = resolve_path('weights_challenge_2.pt')
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        self.model_c2 = model
        return model
    
    def challenge_1(self, X):
        """Challenge 1: Predict response time from EEG
        
        Args:
            X (torch.Tensor): EEG data, shape [batch, n_chans, n_times]
        
        Returns:
            torch.Tensor: predictions, shape [batch,]
        """
        model = self.get_model_challenge_1()
        X = X.to(self.device)
        
        with torch.no_grad():
            predictions = model(X)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
        
        return predictions
    
    def challenge_2(self, X):
        """Challenge 2: Predict externalizing factor from EEG
        
        Args:
            X (torch.Tensor): EEG data, shape [batch, n_chans, n_times]
        
        Returns:
            torch.Tensor: predictions, shape [batch,]
        """
        model = self.get_model_challenge_2()
        X = X.to(self.device)
        
        with torch.no_grad():
            predictions = model(X)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
        
        return predictions
