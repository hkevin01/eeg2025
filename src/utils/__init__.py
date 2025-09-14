"""
Utility functions for EEG Foundation Challenge 2025.

This package contains schedulers, augmentations, and other utility functions.
"""

from .augmentations import SSLViewPipeline
from .schedulers import ParameterScheduler

__all__ = [
    "ParameterScheduler",
    "SSLViewPipeline",
]
