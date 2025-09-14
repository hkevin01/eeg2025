"""
Invariance modules for domain adaptation and transfer learning.

This package contains domain adversarial training, invariant risk minimization,
and other techniques for learning domain-invariant representations.
"""

from .dann import DANNModel, DomainAdversarialHead, GradientReversalLayer, GRLScheduler

__all__ = [
    "GradientReversalLayer",
    "DomainAdversarialHead",
    "DANNModel",
    "GRLScheduler",
]
