"""
Correlation-based Mean Squared Error (CorrMSE) Loss for EEG Challenge 2025.

This module implements the CorrMSE loss which combines standard MSE with
Pearson correlation coefficient to optimize both accuracy and correlation
in regression tasks, particularly for reaction time (RT) prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CorrMSELoss(nn.Module):
    """
    Correlation-based Mean Squared Error Loss.

    Combines MSE loss with negative Pearson correlation to encourage both
    accurate predictions and high correlation with ground truth values.

    The loss is computed as:
        L = alpha * MSE(y_pred, y_true) - beta * PearsonCorr(y_pred, y_true)

    Where:
    - MSE encourages prediction accuracy
    - Negative correlation encourages high correlation (minimizing negative correlation maximizes positive correlation)
    - alpha and beta balance the two objectives
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-8,
        normalize_targets: bool = True
    ):
        """
        Initialize CorrMSE loss.

        Args:
            alpha: Weight for MSE component (accuracy)
            beta: Weight for correlation component (correlation)
            eps: Small epsilon for numerical stability
            normalize_targets: Whether to normalize targets for correlation computation
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.normalize_targets = normalize_targets

        # MSE loss for accuracy component
        self.mse_loss = nn.MSELoss()

    def pearson_correlation(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Pearson correlation coefficient between predictions and targets.

        Args:
            y_pred: Predicted values [batch_size]
            y_true: True values [batch_size]

        Returns:
            Pearson correlation coefficient (scalar)
        """
        # Ensure tensors are 1D
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # Remove NaN values if any
        valid_mask = ~(torch.isnan(y_pred) | torch.isnan(y_true))
        if valid_mask.sum() < 2:
            # Not enough valid samples for correlation
            return torch.tensor(0.0, device=y_pred.device)

        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]

        # Center the variables (subtract mean)
        y_pred_centered = y_pred - y_pred.mean()
        y_true_centered = y_true - y_true.mean()

        # Compute correlation
        numerator = (y_pred_centered * y_true_centered).sum()

        # Compute standard deviations
        pred_std = torch.sqrt((y_pred_centered ** 2).sum() + self.eps)
        true_std = torch.sqrt((y_true_centered ** 2).sum() + self.eps)

        denominator = pred_std * true_std + self.eps

        correlation = numerator / denominator

        # Clamp to valid correlation range [-1, 1]
        correlation = torch.clamp(correlation, -1.0, 1.0)

        return correlation

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CorrMSE loss.

        Args:
            y_pred: Predicted values [batch_size] or [batch_size, 1]
            y_true: True values [batch_size] or [batch_size, 1]

        Returns:
            CorrMSE loss (scalar)
        """
        # Ensure proper shape
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()
        if y_true.dim() > 1:
            y_true = y_true.squeeze()

        # Check for sufficient samples
        if y_pred.numel() == 0 or y_true.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # Normalize targets if requested (helps with correlation stability)
        if self.normalize_targets:
            y_true_mean = y_true.mean()
            y_true_std = y_true.std() + self.eps
            y_true_norm = (y_true - y_true_mean) / y_true_std

            # Also normalize predictions with same statistics for fair comparison
            y_pred_norm = (y_pred - y_true_mean) / y_true_std
        else:
            y_true_norm = y_true
            y_pred_norm = y_pred

        # Compute MSE component
        mse_component = self.mse_loss(y_pred_norm, y_true_norm)

        # Compute correlation component
        correlation = self.pearson_correlation(y_pred_norm, y_true_norm)

        # Combined loss (note: negative correlation to maximize positive correlation)
        loss = self.alpha * mse_component - self.beta * correlation

        return loss


class AdaptiveCorrMSELoss(nn.Module):
    """
    Adaptive Correlation-based MSE Loss with dynamic weight balancing.

    Automatically adjusts the balance between MSE and correlation components
    based on training progress and current performance.
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        eps: float = 1e-8,
        normalize_targets: bool = True,
        adapt_frequency: int = 100,
        alpha_range: Tuple[float, float] = (0.1, 2.0),
        beta_range: Tuple[float, float] = (0.1, 2.0)
    ):
        """
        Initialize Adaptive CorrMSE loss.

        Args:
            initial_alpha: Initial weight for MSE component
            initial_beta: Initial weight for correlation component
            eps: Small epsilon for numerical stability
            normalize_targets: Whether to normalize targets
            adapt_frequency: How often to adapt weights (in forward calls)
            alpha_range: Valid range for alpha parameter
            beta_range: Valid range for beta parameter
        """
        super().__init__()

        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.eps = eps
        self.normalize_targets = normalize_targets
        self.adapt_frequency = adapt_frequency
        self.alpha_range = alpha_range
        self.beta_range = beta_range

        # Current weights (learnable parameters)
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.beta = nn.Parameter(torch.tensor(initial_beta))

        # Base CorrMSE loss
        self.base_loss = CorrMSELoss(
            alpha=1.0,  # We'll handle weighting ourselves
            beta=1.0,
            eps=eps,
            normalize_targets=normalize_targets
        )

        # Adaptation tracking
        self.call_count = 0
        self.running_mse = 0.0
        self.running_corr = 0.0
        self.adaptation_momentum = 0.9

    def adapt_weights(self, mse_value: float, corr_value: float):
        """
        Adapt alpha and beta weights based on current performance.

        Args:
            mse_value: Current MSE value
            corr_value: Current correlation value
        """
        # Update running averages
        self.running_mse = self.adaptation_momentum * self.running_mse + (1 - self.adaptation_momentum) * mse_value
        self.running_corr = self.adaptation_momentum * self.running_corr + (1 - self.adaptation_momentum) * abs(corr_value)

        # Adaptive logic: if MSE is much larger than correlation, reduce alpha
        # If correlation is very low, increase beta

        if self.running_mse > 0 and self.running_corr > 0:
            # Scale factor based on relative magnitudes
            mse_scale = math.log(max(self.running_mse, 1e-6))
            corr_scale = math.log(max(self.running_corr, 1e-6))

            # Adjust alpha (MSE weight)
            if mse_scale > corr_scale:
                # MSE is dominating, reduce its weight
                new_alpha = self.alpha.data * 0.95
            else:
                # Correlation is dominating, increase MSE weight
                new_alpha = self.alpha.data * 1.05

            # Adjust beta (correlation weight)
            if self.running_corr < 0.3:  # Low correlation, increase beta
                new_beta = self.beta.data * 1.1
            elif self.running_corr > 0.8:  # High correlation, can reduce beta
                new_beta = self.beta.data * 0.95
            else:
                new_beta = self.beta.data

            # Clamp to valid ranges
            self.alpha.data = torch.clamp(
                torch.tensor(new_alpha),
                self.alpha_range[0],
                self.alpha_range[1]
            )
            self.beta.data = torch.clamp(
                torch.tensor(new_beta),
                self.beta_range[0],
                self.beta_range[1]
            )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive CorrMSE loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Adaptive CorrMSE loss
        """
        # Ensure proper shape
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()
        if y_true.dim() > 1:
            y_true = y_true.squeeze()

        # Compute individual components
        mse_component = F.mse_loss(y_pred, y_true)
        correlation = self.base_loss.pearson_correlation(y_pred, y_true)

        # Adaptive weight adjustment
        self.call_count += 1
        if self.call_count % self.adapt_frequency == 0:
            with torch.no_grad():
                self.adapt_weights(mse_component.item(), correlation.item())

        # Combined loss with current weights
        loss = self.alpha * mse_component - self.beta * correlation

        return loss


class RobustCorrMSELoss(nn.Module):
    """
    Robust Correlation-based MSE Loss with outlier handling.

    Includes mechanisms to handle outliers and extreme values that might
    destabilize correlation computation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-8,
        normalize_targets: bool = True,
        outlier_quantile: float = 0.95,
        use_huber: bool = False,
        huber_delta: float = 1.0
    ):
        """
        Initialize Robust CorrMSE loss.

        Args:
            alpha: Weight for MSE component
            beta: Weight for correlation component
            eps: Numerical stability epsilon
            normalize_targets: Whether to normalize targets
            outlier_quantile: Quantile threshold for outlier clipping
            use_huber: Whether to use Huber loss instead of MSE
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.normalize_targets = normalize_targets
        self.outlier_quantile = outlier_quantile
        self.use_huber = use_huber
        self.huber_delta = huber_delta

        # Base losses
        if use_huber:
            self.regression_loss = nn.HuberLoss(delta=huber_delta)
        else:
            self.regression_loss = nn.MSELoss()

    def clip_outliers(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Clip outliers based on quantile thresholds.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Clipped predictions and targets
        """
        # Compute quantile thresholds for true values
        q_low = torch.quantile(y_true, 1 - self.outlier_quantile)
        q_high = torch.quantile(y_true, self.outlier_quantile)

        # Create mask for non-outlier samples
        mask = (y_true >= q_low) & (y_true <= q_high)

        # Also clip predictions to similar range for stability
        pred_q_low = torch.quantile(y_pred, 1 - self.outlier_quantile)
        pred_q_high = torch.quantile(y_pred, self.outlier_quantile)

        pred_mask = (y_pred >= pred_q_low) & (y_pred <= pred_q_high)

        # Combine masks
        combined_mask = mask & pred_mask

        if combined_mask.sum() < 2:  # Need at least 2 samples
            return y_pred, y_true

        return y_pred[combined_mask], y_true[combined_mask]

    def spearman_correlation(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Spearman rank correlation (more robust to outliers).

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Spearman correlation coefficient
        """
        # Get ranks
        pred_ranks = torch.argsort(torch.argsort(y_pred)).float()
        true_ranks = torch.argsort(torch.argsort(y_true)).float()

        # Compute Pearson correlation on ranks
        pred_centered = pred_ranks - pred_ranks.mean()
        true_centered = true_ranks - true_ranks.mean()

        numerator = (pred_centered * true_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + self.eps)
        true_std = torch.sqrt((true_centered ** 2).sum() + self.eps)

        denominator = pred_std * true_std + self.eps
        correlation = numerator / denominator

        return torch.clamp(correlation, -1.0, 1.0)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute robust CorrMSE loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Robust CorrMSE loss
        """
        # Ensure proper shape
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()
        if y_true.dim() > 1:
            y_true = y_true.squeeze()

        if y_pred.numel() == 0 or y_true.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # Clip outliers for stability
        y_pred_clipped, y_true_clipped = self.clip_outliers(y_pred, y_true)

        # Normalize if requested
        if self.normalize_targets:
            y_true_mean = y_true_clipped.mean()
            y_true_std = y_true_clipped.std() + self.eps
            y_true_norm = (y_true_clipped - y_true_mean) / y_true_std
            y_pred_norm = (y_pred_clipped - y_true_mean) / y_true_std
        else:
            y_true_norm = y_true_clipped
            y_pred_norm = y_pred_clipped

        # Compute regression component
        regression_component = self.regression_loss(y_pred_norm, y_true_norm)

        # Compute correlation component (use Spearman for robustness)
        correlation = self.spearman_correlation(y_pred_norm, y_true_norm)

        # Combined loss
        loss = self.alpha * regression_component - self.beta * correlation

        return loss


# Factory function for easy loss creation
def create_corr_mse_loss(
    loss_type: str = "standard",
    alpha: float = 1.0,
    beta: float = 1.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CorrMSE loss variants.

    Args:
        loss_type: Type of loss ("standard", "adaptive", "robust")
        alpha: Weight for MSE component
        beta: Weight for correlation component
        **kwargs: Additional arguments for specific loss types

    Returns:
        CorrMSE loss instance
    """
    if loss_type == "standard":
        return CorrMSELoss(alpha=alpha, beta=beta, **kwargs)
    elif loss_type == "adaptive":
        return AdaptiveCorrMSELoss(
            initial_alpha=alpha,
            initial_beta=beta,
            **kwargs
        )
    elif loss_type == "robust":
        return RobustCorrMSELoss(alpha=alpha, beta=beta, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
