"""
Temporal Regression Head
========================

Specialized head for temporal regression tasks in EEG analysis.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalRegressionHead(nn.Module):
    """
    Head for temporal regression tasks with uncertainty estimation.

    Predicts continuous values over time with confidence estimates.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.1,
        use_uncertainty: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_uncertainty = use_uncertainty

        # Choose activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "swish":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()

        # Build regression network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    act_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.regression_net = nn.Sequential(*layers)

        # Output layers
        self.mean_head = nn.Linear(prev_dim, output_dim)

        if use_uncertainty:
            self.log_var_head = nn.Linear(prev_dim, output_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through regression head.

        Args:
            x: Input features of shape [batch_size, seq_len, input_dim]
            return_features: Whether to return intermediate features

        Returns:
            mean: Predicted means of shape [batch_size, seq_len, output_dim]
            log_var: Log variances (if use_uncertainty=True)
            features: Intermediate features (if return_features=True)
        """
        batch_size, seq_len, _ = x.shape

        # Reshape for processing
        x = x.reshape(-1, self.input_dim)  # [batch_size * seq_len, input_dim]

        # Extract features
        features = self.regression_net(x)  # [batch_size * seq_len, hidden_dim]

        # Predict mean
        mean = self.mean_head(features)  # [batch_size * seq_len, output_dim]

        # Predict uncertainty if enabled
        log_var = None
        if self.use_uncertainty:
            log_var = self.log_var_head(features)

        # Reshape back to sequence format
        mean = mean.reshape(batch_size, seq_len, self.output_dim)
        if log_var is not None:
            log_var = log_var.reshape(batch_size, seq_len, self.output_dim)

        if return_features:
            features = features.reshape(batch_size, seq_len, -1)
            return mean, log_var, features

        return mean, log_var, None

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute regression loss with optional uncertainty weighting.

        Args:
            predictions: Predicted values
            targets: Target values
            log_var: Log variances for uncertainty weighting
            mask: Mask for valid timesteps

        Returns:
            Loss value
        """
        if log_var is not None:
            # Uncertainty-weighted loss
            var = torch.exp(log_var)
            loss = 0.5 * ((predictions - targets) ** 2 / var + log_var)
        else:
            # Standard MSE loss
            loss = F.mse_loss(predictions, targets, reduction="none")

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss


class CalibratedClassificationHead(nn.Module):
    """
    Classification head with temperature scaling for calibration.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.1,
        use_temperature_scaling: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.use_temperature_scaling = use_temperature_scaling

        # Build classification network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.classifier = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, n_classes)

        # Temperature parameter for calibration
        if use_temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through classification head.

        Args:
            x: Input features of shape [batch_size, seq_len, input_dim]

        Returns:
            logits: Raw logits
            calibrated_logits: Temperature-scaled logits
        """
        batch_size, seq_len, _ = x.shape

        # Global average pooling over sequence
        x = x.mean(dim=1)  # [batch_size, input_dim]

        # Classification
        features = self.classifier(x)
        logits = self.output_layer(features)

        # Apply temperature scaling
        if self.use_temperature_scaling:
            calibrated_logits = logits / self.temperature
        else:
            calibrated_logits = logits

        return logits, calibrated_logits

    def compute_calibration_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for calibration.

        Args:
            logits: Predicted logits
            targets: Target class indices

        Returns:
            Loss value
        """
        return F.cross_entropy(logits, targets)


class PsychopathologyHead(nn.Module):
    """
    Multi-task head for psychopathology prediction.
    Predicts multiple disorder probabilities simultaneously.
    """

    def __init__(
        self,
        input_dim: int,
        disorders: list = ["adhd", "asd", "anxiety", "depression"],
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.1,
        use_multi_scale: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.disorders = disorders
        self.n_disorders = len(disorders)
        self.use_multi_scale = use_multi_scale

        # Shared feature extractor
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Disorder-specific heads
        self.disorder_heads = nn.ModuleDict()
        for disorder in disorders:
            self.disorder_heads[disorder] = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1], 1),
                nn.Sigmoid(),
            )

        # Multi-scale processing if enabled
        if use_multi_scale:
            self.scale_layers = nn.ModuleList(
                [
                    nn.Conv1d(input_dim, input_dim, kernel_size=k, padding=k // 2)
                    for k in [3, 7, 15]  # Different temporal scales
                ]
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through psychopathology head.

        Args:
            x: Input features of shape [batch_size, seq_len, input_dim]

        Returns:
            Dictionary mapping disorder names to probabilities
        """
        batch_size, seq_len, _ = x.shape

        # Multi-scale processing
        if self.use_multi_scale:
            x_transpose = x.transpose(1, 2)  # [batch, input_dim, seq_len]

            scale_features = []
            for scale_layer in self.scale_layers:
                scale_feat = scale_layer(x_transpose)
                scale_feat = scale_feat.transpose(
                    1, 2
                )  # Back to [batch, seq_len, input_dim]
                scale_features.append(scale_feat.mean(dim=1))  # Global average pooling

            # Combine multi-scale features
            x = torch.stack(scale_features, dim=1).mean(
                dim=1
            )  # [batch_size, input_dim]
        else:
            # Simple global average pooling
            x = x.mean(dim=1)  # [batch_size, input_dim]

        # Shared feature extraction
        shared_features = self.shared_net(x)

        # Disorder-specific predictions
        predictions = {}
        for disorder in self.disorders:
            predictions[disorder] = self.disorder_heads[disorder](
                shared_features
            ).squeeze(-1)

        return predictions

    def compute_loss(
        self, predictions: dict, targets: dict, weights: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Compute multi-task loss for psychopathology prediction.

        Args:
            predictions: Dictionary of disorder predictions
            targets: Dictionary of disorder targets
            weights: Optional weights for different disorders

        Returns:
            Combined loss value
        """
        total_loss = 0.0

        for disorder in self.disorders:
            if disorder in targets:
                disorder_loss = F.binary_cross_entropy(
                    predictions[disorder], targets[disorder].float()
                )

                if weights and disorder in weights:
                    disorder_loss *= weights[disorder]

                total_loss += disorder_loss

        return total_loss


def create_temporal_regression_head(config: dict) -> TemporalRegressionHead:
    """Factory function to create temporal regression head from config."""
    head_config = config.get("heads", {}).get("temporal_regression", {})

    return TemporalRegressionHead(
        input_dim=config.get("d_model", 768),
        output_dim=head_config.get("output_dim", 128),
        hidden_dims=head_config.get("hidden_dims", [512, 256]),
        dropout=config.get("dropout", 0.1),
        use_uncertainty=head_config.get("use_uncertainty", True),
    )


def create_classification_head(config: dict) -> CalibratedClassificationHead:
    """Factory function to create classification head from config."""
    head_config = config.get("heads", {}).get("classification", {})

    return CalibratedClassificationHead(
        input_dim=config.get("d_model", 768),
        n_classes=head_config.get("n_classes", 2),
        hidden_dims=head_config.get("hidden_dims", [512, 256]),
        dropout=config.get("dropout", 0.1),
        use_temperature_scaling=head_config.get("use_temperature_scaling", True),
    )


def create_psychopathology_head(config: dict) -> PsychopathologyHead:
    """Factory function to create psychopathology head from config."""
    head_config = config.get("heads", {}).get("psychopathology", {})

    return PsychopathologyHead(
        input_dim=config.get("d_model", 768),
        disorders=head_config.get(
            "disorders", ["adhd", "asd", "anxiety", "depression"]
        ),
        hidden_dims=head_config.get("hidden_dims", [512, 256, 128]),
        dropout=config.get("dropout", 0.1),
        use_multi_scale=head_config.get("use_multi_scale", True),
    )
