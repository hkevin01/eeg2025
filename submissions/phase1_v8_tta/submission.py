"""
EEG Foundation Challenge 2025 - V8 + Test-Time Augmentation
Enhances V8 with TTA for potential small improvement
"""

import torch
import torch.nn as nn
import numpy as np
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

    raise FileNotFoundError(
        f"Could not find {filename} in: {', '.join(search_paths)}"
    )


class CompactResponseTimeCNN(nn.Module):
    """Compact CNN for response time prediction (Challenge 1)"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output


class Submission:
    """EEG 2025 Competition Submission - V8 + TTA"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        if isinstance(DEVICE, str):
            self.device = torch.device(DEVICE)
        else:
            self.device = DEVICE
        self.n_chans = 129
        self.n_times = 200
        self.model_c1 = None
        self.model_c2 = None
        
        # TTA config
        self.use_tta = True
        self.n_tta = 5  # Number of augmented predictions to average

    def augment_sample(self, x, aug_idx):
        """Apply augmentation for TTA"""
        x = x.clone()
        
        if aug_idx == 0:
            # Original, no augmentation
            return x
        elif aug_idx == 1:
            # Small time shift right
            shift = 3
            x = torch.roll(x, shift, dims=-1)
        elif aug_idx == 2:
            # Small time shift left
            shift = -3
            x = torch.roll(x, shift, dims=-1)
        elif aug_idx == 3:
            # Slight amplitude scaling up
            x = x * 1.02
        elif aug_idx == 4:
            # Slight amplitude scaling down
            x = x * 0.98
        
        return x

    def get_model_challenge_1(self):
        """Load Challenge 1 model"""
        if self.model_c1 is not None:
            return self.model_c1

        model = CompactResponseTimeCNN()
        weights_path = resolve_path('weights_challenge_1.pt')
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
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

    def challenge_1(self, eeg):
        """Challenge 1: Response time prediction with TTA"""
        model = self.get_model_challenge_1()
        
        eeg = torch.tensor(eeg, dtype=torch.float32, device=self.device)
        if eeg.ndim == 2:
            eeg = eeg.unsqueeze(0)
        
        if not self.use_tta:
            # Standard prediction
            with torch.no_grad():
                prediction = model(eeg).cpu().item()
            return prediction
        
        # TTA: Average predictions across augmented versions
        predictions = []
        with torch.no_grad():
            for aug_idx in range(self.n_tta):
                aug_eeg = self.augment_sample(eeg, aug_idx)
                pred = model(aug_eeg).cpu().item()
                predictions.append(pred)
        
        # Return mean of all predictions
        return np.mean(predictions)

    def challenge_2(self, eeg):
        """Challenge 2: Personality prediction"""
        model = self.get_model_challenge_2()
        
        eeg = torch.tensor(eeg, dtype=torch.float32, device=self.device)
        if eeg.ndim == 2:
            eeg = eeg.unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(eeg).cpu().item()
        
        return prediction
