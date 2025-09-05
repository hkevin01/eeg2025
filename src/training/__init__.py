"""
Training modules for EEG Foundation Challenge 2025.

This package contains training loops, SSL objectives, and cross-task transfer implementations.
"""

from .pretrain_ssl import SSLPretrainer
from .train_cross_task import CrossTaskTrainer

__all__ = [
    "SSLPretrainer",
    "CrossTaskTrainer",
]
