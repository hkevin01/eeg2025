"""Backbone architectures for feature extraction."""

from .temporal_cnn import TemporalCNN, create_temporal_cnn

__all__ = ["TemporalCNN", "create_temporal_cnn"]
