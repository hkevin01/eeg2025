"""
SSL Loss functions for EEG Foundation Challenge 2025.

This module implements various self-supervised learning objectives including
masked reconstruction, contrastive learning, and predictive coding.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedReconstructionLoss(nn.Module):
    """
    Masked reconstruction loss for self-supervised learning.

    Computes reconstruction loss only on masked time steps to encourage
    temporal representation learning in EEG signals.
    """

    def __init__(self, loss_type: str = "mse", reduction: str = "mean"):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self, reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked reconstruction loss.

        Args:
            reconstructed: Reconstructed signal [batch_size, n_channels, seq_len]
            target: Original signal [batch_size, n_channels, seq_len]
            mask: Boolean mask [batch_size, seq_len] where True = masked

        Returns:
            Scalar loss value
        """
        # Compute element-wise loss
        loss = self.loss_fn(reconstructed, target)  # [batch_size, n_channels, seq_len]

        # Apply mask - only compute loss on masked timesteps
        # Expand mask to match loss dimensions
        mask_expanded = mask.unsqueeze(1).expand_as(
            loss
        )  # [batch_size, n_channels, seq_len]

        # Apply mask
        masked_loss = loss * mask_expanded

        # Reduce loss
        if self.reduction == "mean":
            # Average over masked elements only
            num_masked = mask_expanded.sum()
            if num_masked > 0:
                return masked_loss.sum() / num_masked
            else:
                return torch.tensor(0.0, device=loss.device)
        elif self.reduction == "sum":
            return masked_loss.sum()
        else:
            return masked_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.

    Implements InfoNCE loss to bring positive pairs closer and push negative pairs apart.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two views.

        Args:
            z1: Projections from view 1 [batch_size, emb_dim]
            z2: Projections from view 2 [batch_size, emb_dim]

        Returns:
            Scalar contrastive loss
        """
        batch_size = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity matrix
        # Positive pairs: (z1[i], z2[i])
        # Negative pairs: (z1[i], z2[j]) where i != j

        # Similarity between z1 and z2
        sim_matrix = (
            torch.matmul(z1, z2.T) / self.temperature
        )  # [batch_size, batch_size]

        # Create labels - positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=z1.device)

        # Compute cross-entropy loss
        loss_1to2 = F.cross_entropy(sim_matrix, labels)
        loss_2to1 = F.cross_entropy(sim_matrix.T, labels)

        # Symmetric loss
        loss = (loss_1to2 + loss_2to1) / 2.0

        return loss


class PredictiveResidualLoss(nn.Module):
    """
    Predictive residual loss for temporal representation learning.

    Encourages the model to predict future representations from past context.
    """

    def __init__(self, loss_type: str = "mse", prediction_steps: int = 1):
        super().__init__()
        self.loss_type = loss_type
        self.prediction_steps = prediction_steps

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive residual loss.

        Args:
            predicted: Predicted features [batch_size, seq_len, d_model]
            target: Target features [batch_size, seq_len, d_model]

        Returns:
            Scalar loss value
        """
        # Shift targets by prediction_steps for future prediction
        if self.prediction_steps > 0:
            # Predict future: predicted[t] should match target[t + prediction_steps]
            pred_truncated = predicted[:, : -self.prediction_steps, :]
            target_shifted = target[:, self.prediction_steps :, :]
        else:
            # Autoregressive: predict current from past
            pred_truncated = predicted
            target_shifted = target

        # Compute loss
        loss = self.loss_fn(pred_truncated, target_shifted)

        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for smooth representations.

    Encourages temporal smoothness in learned representations.
    """

    def __init__(self, loss_type: str = "mse", smoothness_weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.smoothness_weight = smoothness_weight

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Args:
            features: Feature sequence [batch_size, seq_len, d_model]

        Returns:
            Scalar temporal consistency loss
        """
        # Compute temporal differences
        diff = features[:, 1:, :] - features[:, :-1, :]

        # Compute smoothness loss (penalize large changes)
        smoothness_loss = self.loss_fn(diff, torch.zeros_like(diff))

        return self.smoothness_weight * smoothness_loss


class VICRegLoss(nn.Module):
    """
    VICReg loss for self-supervised learning.

    Implements Variance-Invariance-Covariance Regularization.
    """

    def __init__(
        self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute VICReg loss.

        Args:
            z1: Representations from view 1 [batch_size, dim]
            z2: Representations from view 2 [batch_size, dim]

        Returns:
            Total loss and loss components dictionary
        """
        batch_size, dim = z1.shape

        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss (prevent collapse)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-6)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-6)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss (decorrelate features)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)

        # Off-diagonal elements should be zero
        cov_loss = (
            cov_z1.fill_diagonal_(0).pow_(2).sum() / dim
            + cov_z2.fill_diagonal_(0).pow_(2).sum() / dim
        )

        # Total loss
        total_loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        loss_dict = {
            "sim_loss": sim_loss.item(),
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
        }

        return total_loss, loss_dict


class CombinedSSLLoss(nn.Module):
    """
    Combined SSL loss with multiple objectives.

    Combines different SSL objectives with learnable or fixed weights.
    """

    def __init__(
        self,
        objectives: list,
        weights: Optional[dict] = None,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.objectives = objectives

        # Initialize loss functions
        self.losses = nn.ModuleDict()
        for obj in objectives:
            if obj == "masked":
                self.losses[obj] = MaskedReconstructionLoss()
            elif obj == "contrastive":
                self.losses[obj] = ContrastiveLoss()
            elif obj == "predictive_residual":
                self.losses[obj] = PredictiveResidualLoss()
            elif obj == "temporal_consistency":
                self.losses[obj] = TemporalConsistencyLoss()
            elif obj == "vicreg":
                self.losses[obj] = VICRegLoss()

        # Initialize weights
        if weights is None:
            weights = {obj: 1.0 for obj in objectives}

        if learnable_weights:
            self.weights = nn.ParameterDict(
                {
                    obj: nn.Parameter(torch.tensor(weights.get(obj, 1.0)))
                    for obj in objectives
                }
            )
        else:
            self.register_buffer(
                "weights", torch.tensor([weights.get(obj, 1.0) for obj in objectives])
            )
            self.weight_dict = {obj: weights.get(obj, 1.0) for obj in objectives}

    def forward(self, outputs: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined SSL loss.

        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary

        Returns:
            Total loss and individual loss components
        """
        total_loss = 0
        loss_dict = {}

        for obj in self.objectives:
            if obj in self.losses:
                # Get weight
                if hasattr(self, "weight_dict"):
                    weight = self.weight_dict[obj]
                else:
                    weight = self.weights[obj]

                # Compute objective-specific loss
                if obj == "masked":
                    loss = self.losses[obj](
                        outputs["reconstructed"], targets["original"], targets["mask"]
                    )
                elif obj in ["contrastive", "vicreg"]:
                    if obj == "vicreg":
                        loss, vicreg_dict = self.losses[obj](
                            outputs["projections1"], outputs["projections2"]
                        )
                        loss_dict.update(vicreg_dict)
                    else:
                        loss = self.losses[obj](
                            outputs["projections1"], outputs["projections2"]
                        )
                elif obj == "predictive_residual":
                    loss = self.losses[obj](outputs["predicted"], outputs["features"])
                elif obj == "temporal_consistency":
                    loss = self.losses[obj](outputs["features"])

                loss_dict[f"{obj}_loss"] = loss.item()
                total_loss += weight * loss

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
