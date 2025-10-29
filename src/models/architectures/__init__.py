"""Collection of reusable EEG-specific architectures."""

from .eegnet import EEGNetBackbone
from .transformer_eeg import EEGTransformerRegressor

__all__ = [
	"EEGNetBackbone",
	"EEGTransformerRegressor",
]
