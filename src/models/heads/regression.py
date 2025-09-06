"""
Enhanced Regression Heads for Challenge 1
==========================================

Specialized regression heads with temporal modeling for response time prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence modeling."""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention."""
        if x.dim() == 2:
            # Add sequence dimension if needed
            x = x.unsqueeze(1)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)

        # Residual connection and layer norm
        out = self.layer_norm(x + attn_out)

        # Remove sequence dimension if added
        if out.size(1) == 1:
            out = out.squeeze(1)

        return out


class TemporalRegressionHead(nn.Module):
    """
    Enhanced regression head with temporal modeling for RT prediction.

    Features:
    - Temporal attention for sequence modeling
    - Multi-scale feature aggregation
    - Uncertainty estimation
    - Subject-specific calibration
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        use_temporal_attention: bool = True,
        temporal_window: int = 5,
        use_uncertainty: bool = True,
        use_calibration: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.use_temporal_attention = use_temporal_attention
        self.use_uncertainty = use_uncertainty
        self.use_calibration = use_calibration

        # Temporal modeling
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(input_dim)

        # Feature projection layers
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Prediction heads
        if use_uncertainty:
            # Predict both mean and variance
            self.mean_head = nn.Linear(in_dim, 1)
            self.var_head = nn.Sequential(
                nn.Linear(in_dim, 1),
                nn.Softplus()  # Ensure positive variance
            )
        else:
            self.pred_head = nn.Linear(in_dim, 1)

        # Subject-specific calibration layers
        if use_calibration:
            self.calibration_scale = nn.Parameter(torch.ones(1))
            self.calibration_bias = nn.Parameter(torch.zeros(1))

        # Multi-scale aggregation
        self.scale_weights = nn.Parameter(torch.ones(len(hidden_dims)))

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for RT prediction.

        Args:
            x: Input features [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
            subject_ids: Subject IDs for calibration [batch_size]

        Returns:
            RT predictions [batch_size, 1] or [batch_size, 2] if uncertainty
        """
        # Temporal attention if enabled
        if self.use_temporal_attention:
            x = self.temporal_attention(x)

        # Feature extraction with multi-scale aggregation
        features = x
        scale_features = []

        for i, layer in enumerate(self.feature_layers):
            if isinstance(layer, nn.Linear):
                features = layer(features)
                if i // 4 < len(self.scale_weights):  # Every 4 layers (Linear + BN + ReLU + Dropout)
                    scale_features.append(features * self.scale_weights[i // 4])
            else:
                features = layer(features)

        # Aggregate multi-scale features
        if scale_features:
            aggregated_features = torch.stack(scale_features, dim=0).mean(dim=0)
        else:
            aggregated_features = features

        # Prediction
        if self.use_uncertainty:
            mean_pred = self.mean_head(aggregated_features)
            var_pred = self.var_head(aggregated_features)
            predictions = torch.cat([mean_pred, var_pred], dim=-1)
        else:
            predictions = self.pred_head(aggregated_features)

        # Subject-specific calibration
        if self.use_calibration:
            if self.use_uncertainty:
                predictions[:, 0:1] = predictions[:, 0:1] * self.calibration_scale + self.calibration_bias
            else:
                predictions = predictions * self.calibration_scale + self.calibration_bias

        return predictions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute regression loss with uncertainty.

        Args:
            predictions: Model predictions [batch_size, 1] or [batch_size, 2]
            targets: Target RT values [batch_size, 1]
            mask: Valid sample mask [batch_size]

        Returns:
            Loss value
        """
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        if self.use_uncertainty and predictions.size(-1) == 2:
            # Uncertainty-aware loss (negative log-likelihood)
            mean_pred = predictions[:, 0:1]
            var_pred = predictions[:, 1:2]

            # Compute negative log-likelihood
            loss = 0.5 * (torch.log(var_pred) + (targets - mean_pred).pow(2) / var_pred)
            return loss.mean()
        else:
            # Standard MSE loss
            if predictions.size(-1) == 2:
                predictions = predictions[:, 0:1]
            return F.mse_loss(predictions, targets)


class AdaptivePooling(nn.Module):
    """Adaptive pooling with learnable attention weights."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive pooling."""
        if x.dim() == 3:  # [batch_size, seq_len, feature_dim]
            weights = self.attention(x)  # [batch_size, seq_len, 1]
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1)
        else:
            pooled = x
        return pooled


class MultiTaskRegressionHead(nn.Module):
    """
    Multi-task regression head for multiple RT-related predictions.

    Can predict:
    - Mean response time
    - Response time variability
    - Reaction time components (motor, cognitive)
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int = 3,  # mean_rt, rt_var, rt_components
        shared_dim: int = 256,
        task_specific_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_tasks = num_tasks

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, task_specific_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_specific_dim, 1)
            )
            for _ in range(num_tasks)
        ])

        # Task weights for adaptive combination
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task regression."""

        # Shared features
        shared_features = self.shared_layers(x)

        # Task-specific predictions
        predictions = {}
        for i, head in enumerate(self.task_heads):
            task_pred = head(shared_features)
            predictions[f"task_{i}"] = task_pred

        # Weighted combination for primary prediction
        all_preds = torch.stack([predictions[f"task_{i}"] for i in range(self.num_tasks)], dim=-1)
        weights = F.softmax(self.task_weights, dim=0)
        combined_pred = (all_preds * weights).sum(dim=-1, keepdim=True)

        predictions["combined"] = combined_pred
        predictions["task_weights"] = weights

        return predictions


class CovariateAwareRegressionHead(nn.Module):
    """
    Regression head that incorporates demographic covariates.

    Useful for accounting for age, gender, and other factors in RT prediction.
    """

    def __init__(
        self,
        eeg_input_dim: int,
        covariate_dim: int = 5,  # age, gender, handedness, etc.
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        # EEG feature processing
        self.eeg_processor = nn.Sequential(
            nn.Linear(eeg_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Covariate processing
        self.covariate_processor = nn.Sequential(
            nn.Linear(covariate_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        eeg_features: torch.Tensor,
        covariates: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with covariates."""

        # Process EEG features
        eeg_processed = self.eeg_processor(eeg_features)

        # Process covariates
        cov_processed = self.covariate_processor(covariates)

        # Fuse features
        fused_features = torch.cat([eeg_processed, cov_processed], dim=-1)

        # Final prediction
        prediction = self.fusion_layer(fused_features)

        return prediction
