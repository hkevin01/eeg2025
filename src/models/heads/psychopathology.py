"""
Psychopathology Prediction Heads for Challenge 2
================================================

Specialized heads for CBCL factor prediction with clinical normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class ClinicalNormalizationLayer(nn.Module):
    """
    Clinical normalization layer for age and demographic adjustment.

    Adjusts predictions based on normative data for age and gender.
    """

    def __init__(
        self,
        num_factors: int,
        age_bins: int = 10,
        use_gender: bool = True,
        age_range: Tuple[float, float] = (5.0, 21.0)
    ):
        super().__init__()

        self.num_factors = num_factors
        self.age_bins = age_bins
        self.use_gender = use_gender
        self.age_range = age_range

        # Age-based normalization parameters
        self.age_means = nn.Parameter(torch.zeros(age_bins, num_factors))
        self.age_stds = nn.Parameter(torch.ones(age_bins, num_factors))

        # Gender-based adjustment if enabled
        if use_gender:
            self.gender_adjustments = nn.Parameter(torch.zeros(2, num_factors))  # Male=0, Female=1

        # Learnable normalization weights
        self.norm_weights = nn.Parameter(torch.ones(num_factors))

    def get_age_bin(self, age: torch.Tensor) -> torch.Tensor:
        """Convert continuous age to age bin indices."""
        min_age, max_age = self.age_range
        normalized_age = (age - min_age) / (max_age - min_age)
        normalized_age = torch.clamp(normalized_age, 0.0, 1.0)

        age_bins = (normalized_age * (self.age_bins - 1)).long()
        return age_bins

    def forward(
        self,
        predictions: torch.Tensor,
        age: torch.Tensor,
        gender: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply clinical normalization.

        Args:
            predictions: Raw predictions [batch_size, num_factors]
            age: Age values [batch_size]
            gender: Gender values [batch_size] (0=male, 1=female)

        Returns:
            Normalized predictions [batch_size, num_factors]
        """
        batch_size = predictions.size(0)

        # Get age bins
        age_bins = self.get_age_bin(age)

        # Age-based normalization
        age_mean = self.age_means[age_bins]  # [batch_size, num_factors]
        age_std = self.age_stds[age_bins]    # [batch_size, num_factors]

        # Z-score normalization
        normalized = (predictions - age_mean) / (age_std + 1e-8)

        # Gender adjustment if available
        if self.use_gender and gender is not None:
            gender_adj = self.gender_adjustments[gender]  # [batch_size, num_factors]
            normalized = normalized + gender_adj

        # Apply learnable weights
        normalized = normalized * self.norm_weights

        return normalized


class FactorCorrelationModule(nn.Module):
    """Module to enforce known correlations between CBCL factors."""

    def __init__(self, num_factors: int):
        super().__init__()

        self.num_factors = num_factors

        # Learnable correlation matrix (upper triangular)
        self.correlation_weights = nn.Parameter(torch.zeros(num_factors, num_factors))

        # Known clinical correlations (initialized based on literature)
        self.register_buffer('prior_correlations', self._get_prior_correlations())

    def _get_prior_correlations(self) -> torch.Tensor:
        """Get prior correlation matrix based on clinical knowledge."""
        # Example correlations between CBCL factors (simplified)
        correlations = torch.eye(4)  # Assume 4 factors

        # P-factor correlates with all others
        correlations[0, 1] = 0.7  # p-factor <-> internalizing
        correlations[0, 2] = 0.8  # p-factor <-> externalizing
        correlations[0, 3] = 0.6  # p-factor <-> attention

        # Internalizing and externalizing
        correlations[1, 2] = 0.4

        # Attention with others
        correlations[1, 3] = 0.3
        correlations[2, 3] = 0.5

        # Make symmetric
        correlations = correlations + correlations.T - torch.diag(correlations.diag())

        return correlations

    def forward(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute factor correlations and regularization.

        Args:
            predictions: Factor predictions [batch_size, num_factors]

        Returns:
            Dictionary with correlation info
        """
        # Compute empirical correlations
        centered_preds = predictions - predictions.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(centered_preds.T, centered_preds) / (predictions.size(0) - 1)

        # Compute correlation matrix
        std_devs = torch.sqrt(torch.diag(cov_matrix))
        corr_matrix = cov_matrix / (std_devs.unsqueeze(1) * std_devs.unsqueeze(0) + 1e-8)

        # Correlation regularization loss
        corr_loss = F.mse_loss(corr_matrix, self.prior_correlations)

        return {
            "correlation_matrix": corr_matrix,
            "correlation_loss": corr_loss,
            "predicted_correlations": corr_matrix
        }


class UncertaintyEstimationHead(nn.Module):
    """Head for uncertainty estimation in factor predictions."""

    def __init__(self, input_dim: int, num_factors: int):
        super().__init__()

        # Mean prediction
        self.mean_head = nn.Linear(input_dim, num_factors)

        # Variance prediction (log-variance for stability)
        self.log_var_head = nn.Linear(input_dim, num_factors)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict mean and uncertainty.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Dictionary with mean and variance predictions
        """
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        var = torch.exp(log_var)

        return {
            "mean": mean,
            "variance": var,
            "log_variance": log_var
        }

    def compute_nll_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss."""
        mean = predictions["mean"]
        var = predictions["variance"]

        # Negative log-likelihood
        nll = 0.5 * (torch.log(var) + (targets - mean).pow(2) / var)

        return nll.mean()


class PsychopathologyHead(nn.Module):
    """
    Comprehensive head for psychopathology factor prediction.

    Features:
    - Multi-output regression for CBCL factors
    - Uncertainty estimation
    - Factor correlation enforcement
    - Clinical normalization support
    """

    def __init__(
        self,
        input_dim: int,
        num_factors: int = 4,
        factor_names: List[str] = None,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        use_uncertainty: bool = True,
        use_correlation_loss: bool = True,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_factors = num_factors
        self.factor_names = factor_names or [f"factor_{i}" for i in range(num_factors)]
        self.use_uncertainty = use_uncertainty
        self.use_correlation_loss = use_correlation_loss

        # Shared feature extraction
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Prediction heads
        if use_uncertainty:
            self.uncertainty_head = UncertaintyEstimationHead(in_dim, num_factors)
        else:
            self.prediction_head = nn.Linear(in_dim, num_factors)

        # Factor correlation module
        if use_correlation_loss:
            self.correlation_module = FactorCorrelationModule(num_factors)

        # Factor-specific fine-tuning layers
        self.factor_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout // 2),
                nn.Linear(in_dim // 2, 1)
            )
            for _ in range(num_factors)
        ])

        # Attention mechanism for factor importance
        self.factor_attention = nn.Sequential(
            nn.Linear(in_dim, num_factors),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for psychopathology prediction.

        Args:
            x: Input features [batch_size, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and additional outputs
        """
        # Shared feature extraction
        shared_features = self.shared_layers(x)

        # Main predictions
        if self.use_uncertainty:
            uncertainty_outputs = self.uncertainty_head(shared_features)
            main_predictions = uncertainty_outputs["mean"]
            uncertainties = uncertainty_outputs["variance"]
        else:
            main_predictions = self.prediction_head(shared_features)
            uncertainties = None

        # Factor-specific predictions
        factor_predictions = []
        for factor_layer in self.factor_layers:
            factor_pred = factor_layer(shared_features)
            factor_predictions.append(factor_pred)

        factor_predictions = torch.cat(factor_predictions, dim=-1)

        # Attention-weighted combination
        attention_weights = self.factor_attention(shared_features)

        # Combine main and factor-specific predictions
        combined_predictions = (
            0.7 * main_predictions +
            0.3 * factor_predictions
        )

        outputs = {
            "predictions": combined_predictions,
            "main_predictions": main_predictions,
            "factor_predictions": factor_predictions
        }

        if uncertainties is not None:
            outputs["uncertainties"] = uncertainties

        if return_attention:
            outputs["attention_weights"] = attention_weights

        # Factor correlations
        if self.use_correlation_loss:
            correlation_outputs = self.correlation_module(combined_predictions)
            outputs.update(correlation_outputs)

        return outputs

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        correlation_loss: Optional[torch.Tensor] = None,
        loss_weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for psychopathology prediction.

        Args:
            predictions: Model predictions [batch_size, num_factors]
            targets: Target scores [batch_size, num_factors]
            uncertainties: Uncertainty estimates [batch_size, num_factors]
            correlation_loss: Factor correlation loss
            loss_weights: Weights for different loss components

        Returns:
            Dictionary of loss components
        """
        if loss_weights is None:
            loss_weights = {
                "mse": 1.0,
                "correlation": 0.1,
                "uncertainty": 0.1
            }

        losses = {}

        # Main MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        losses["mse_loss"] = mse_loss

        # Correlation-based loss
        pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
        target_centered = targets - targets.mean(dim=0, keepdim=True)

        correlation_loss_value = 0.0
        for i in range(self.num_factors):
            pred_factor = pred_centered[:, i]
            target_factor = target_centered[:, i]

            pred_std = torch.sqrt((pred_factor ** 2).mean() + 1e-8)
            target_std = torch.sqrt((target_factor ** 2).mean() + 1e-8)

            correlation = (pred_factor * target_factor).mean() / (pred_std * target_std + 1e-8)
            correlation_loss_value += (1 - correlation)

        losses["correlation_loss"] = correlation_loss_value / self.num_factors

        # Uncertainty loss if available
        if uncertainties is not None:
            uncertainty_loss = torch.mean(uncertainties)  # Regularize uncertainty
            losses["uncertainty_loss"] = uncertainty_loss

        # Factor correlation regularization
        if correlation_loss is not None:
            losses["factor_correlation_loss"] = correlation_loss

        # Total loss
        total_loss = (
            loss_weights["mse"] * losses["mse_loss"] +
            loss_weights["correlation"] * losses["correlation_loss"]
        )

        if "uncertainty_loss" in losses:
            total_loss += loss_weights["uncertainty"] * losses["uncertainty_loss"]

        if "factor_correlation_loss" in losses:
            total_loss += 0.05 * losses["factor_correlation_loss"]

        losses["total_loss"] = total_loss

        return losses


class MultiTaskPsychopathologyHead(nn.Module):
    """
    Multi-task head that handles different clinical assessments.

    Can predict CBCL factors, ADHD ratings, anxiety scores, etc.
    """

    def __init__(
        self,
        input_dim: int,
        task_configs: Dict[str, Dict],
        shared_dim: int = 256
    ):
        super().__init__()

        self.task_configs = task_configs

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()

        for task_name, config in task_configs.items():
            self.task_heads[task_name] = PsychopathologyHead(
                input_dim=shared_dim,
                **config
            )

    def forward(self, x: torch.Tensor, task_name: str) -> Dict[str, torch.Tensor]:
        """Forward pass for specific task."""
        shared_features = self.shared_layers(x)

        if task_name in self.task_heads:
            return self.task_heads[task_name](shared_features)
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def forward_all_tasks(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass for all tasks."""
        shared_features = self.shared_layers(x)

        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)

        return outputs
