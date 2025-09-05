"""
EEG Foundation Challenge 2025

Foundation models and compression for robust EEG analysis.
"""

__version__ = "0.1.0"
__author__ = "EEG Challenge Team"
__email__ = "team@eeg2025.org"

# Core imports for easy access
from src.dataio.bids_loader import HBNDataLoader
from src.models.backbones.temporal_cnn import TemporalCNN
from src.models.backbones.transformer_tiny import TransformerTiny

__all__ = [
    "HBNDataLoader",
    "TemporalCNN",
    "TransformerTiny",
]
