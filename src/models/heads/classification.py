"""
Enhanced Classification Heads for Challenge 1
==============================================

Specialized classification heads with calibration and confidence estimation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def set_temperature(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Tune the temperature parameter using validation data.

        Args:
            logits: Model logits [batch_size, num_classes]
            labels: True labels [batch_size]
        """

        def nll_criterion(temperature):
            scaled_logits = logits / temperature
            loss = F.cross_entropy(scaled_logits, labels)
            return loss.item()

        # Optimize temperature using grid search
        temperatures = torch.linspace(0.1, 10.0, 100)
        best_temp = 1.0
        best_loss = float("inf")

        for temp in temperatures:
            loss = nll_criterion(temp)
            if loss < best_loss:
                best_loss = loss
                best_temp = temp.item()

        self.temperature.data.fill_(best_temp)


class CalibratedClassificationHead(nn.Module):
    """
    Enhanced classification head with probability calibration.

    Features:
    - Temperature scaling calibration
    - Confidence estimation
    - Threshold optimization for balanced accuracy
    - Focal loss for class imbalance
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        use_calibration: bool = True,
        calibration_bins: int = 15,
        use_confidence_estimation: bool = True,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_calibration = use_calibration
        self.use_confidence_estimation = use_confidence_estimation
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Feature extraction layers
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Linear(in_dim, num_classes)

        # Confidence estimation
        if use_confidence_estimation:
            self.confidence_head = nn.Sequential(
                nn.Linear(in_dim, hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1] // 2, 1),
                nn.Sigmoid(),
            )

        # Temperature scaling for calibration
        if use_calibration:
            self.temperature_scaling = TemperatureScaling()

        # Learnable threshold for balanced accuracy optimization
        self.decision_threshold = nn.Parameter(torch.tensor(0.5))

        # Track calibration statistics
        self.register_buffer(
            "bin_boundaries", torch.linspace(0, 1, calibration_bins + 1)
        )
        self.register_buffer("bin_lowers", self.bin_boundaries[:-1])
        self.register_buffer("bin_uppers", self.bin_boundaries[1:])

    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.

        Args:
            x: Input features [batch_size, input_dim]
            return_confidence: Whether to return confidence scores
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing logits, probabilities, and optional outputs
        """
        # Feature extraction
        features = self.feature_layers(x)

        # Classification logits
        logits = self.classifier(features)

        # Apply temperature scaling if calibrated
        if self.use_calibration and hasattr(self.temperature_scaling, "temperature"):
            calibrated_logits = self.temperature_scaling(logits)
        else:
            calibrated_logits = logits

        # Probabilities
        probabilities = F.softmax(calibrated_logits, dim=-1)

        outputs = {
            "logits": logits,
            "calibrated_logits": calibrated_logits,
            "probabilities": probabilities,
        }

        # Confidence estimation
        if self.use_confidence_estimation and return_confidence:
            confidence = self.confidence_head(features)
            outputs["confidence"] = confidence

        # Intermediate features
        if return_features:
            outputs["features"] = features

        return outputs

    def compute_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.

        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            alpha: Class weights [num_classes]

        Returns:
            Focal loss value
        """
        if alpha is None:
            alpha = (
                torch.ones(self.num_classes, device=logits.device) * self.focal_alpha
            )

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        # Apply class weights
        alpha_t = alpha[targets]

        # Focal weight
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma

        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()

    def calibrate_probabilities(
        self, val_logits: torch.Tensor, val_labels: torch.Tensor
    ):
        """
        Calibrate probabilities using validation data.

        Args:
            val_logits: Validation logits [num_samples, num_classes]
            val_labels: Validation labels [num_samples]
        """
        if self.use_calibration:
            self.temperature_scaling.set_temperature(val_logits, val_labels)

    def optimize_threshold(
        self, val_probabilities: torch.Tensor, val_labels: torch.Tensor
    ) -> float:
        """
        Optimize decision threshold for balanced accuracy.

        Args:
            val_probabilities: Validation probabilities [num_samples, num_classes]
            val_labels: Validation labels [num_samples]

        Returns:
            Optimal threshold value
        """
        if self.num_classes != 2:
            return 0.5

        # Get positive class probabilities
        pos_probs = val_probabilities[:, 1].detach().cpu().numpy()
        labels = val_labels.detach().cpu().numpy()

        # Grid search for optimal threshold
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_balanced_acc = 0.0

        for threshold in thresholds:
            predictions = (pos_probs >= threshold).astype(int)

            # Compute balanced accuracy
            tp = np.sum((predictions == 1) & (labels == 1))
            tn = np.sum((predictions == 0) & (labels == 0))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2

            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                best_threshold = threshold

        # Update parameter
        self.decision_threshold.data.fill_(best_threshold)

        return best_threshold

    def predict_with_threshold(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using optimized threshold.

        Args:
            probabilities: Model probabilities [batch_size, num_classes]

        Returns:
            Binary predictions [batch_size]
        """
        if self.num_classes == 2:
            pos_probs = probabilities[:, 1]
            predictions = (pos_probs >= self.decision_threshold).long()
        else:
            predictions = torch.argmax(probabilities, dim=-1)

        return predictions

    def compute_calibration_error(
        self, probabilities: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE) and other calibration metrics.

        Args:
            probabilities: Model probabilities [batch_size, num_classes]
            labels: True labels [batch_size]

        Returns:
            Dictionary of calibration metrics
        """
        if self.num_classes == 2:
            # Binary classification
            confidences = torch.max(probabilities, dim=1)[0]
            predictions = torch.argmax(probabilities, dim=1)
            accuracies = predictions.eq(labels)
        else:
            # Multi-class classification
            confidences, predictions = torch.max(probabilities, dim=1)
            accuracies = predictions.eq(labels)

        # Convert to numpy for easier processing
        confidences = confidences.detach().cpu().numpy()
        accuracies = accuracies.detach().cpu().numpy()

        # Compute ECE
        ece = 0.0
        mce = 0.0  # Maximum Calibration Error

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower.item()) & (
                confidences <= bin_upper.item()
            )
            prop_in_bin = in_bin.sum() / len(confidences)

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                # MCE update
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "average_confidence": np.mean(confidences),
            "accuracy": np.mean(accuracies),
        }


class EnsembleClassificationHead(nn.Module):
    """
    Ensemble classification head with multiple diverse classifiers.

    Combines predictions from multiple heads for improved robustness.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        num_heads: int = 3,
        head_configs: Optional[List[Dict]] = None,
        ensemble_method: str = "voting",  # "voting", "weighted", "stacking"
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method

        # Default head configurations
        if head_configs is None:
            head_configs = [
                {"hidden_dims": [512, 256], "dropout": 0.3},
                {"hidden_dims": [768, 384, 192], "dropout": 0.2},
                {"hidden_dims": [256, 128], "dropout": 0.4},
            ]

        # Create ensemble heads
        self.heads = nn.ModuleList(
            [
                CalibratedClassificationHead(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    use_calibration=True,
                    **config,
                )
                for config in head_configs[:num_heads]
            ]
        )

        # Ensemble weights
        if ensemble_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(num_heads))
        elif ensemble_method == "stacking":
            self.meta_classifier = nn.Linear(num_heads * num_classes, num_classes)

    def forward(
        self, x: torch.Tensor, return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.

        Args:
            x: Input features [batch_size, input_dim]
            return_individual: Whether to return individual head outputs

        Returns:
            Dictionary containing ensemble predictions
        """
        # Get predictions from all heads
        head_outputs = []
        head_probabilities = []

        for head in self.heads:
            output = head(x)
            head_outputs.append(output)
            head_probabilities.append(output["probabilities"])

        # Ensemble combination
        if self.ensemble_method == "voting":
            # Simple averaging
            ensemble_probs = torch.stack(head_probabilities, dim=0).mean(dim=0)

        elif self.ensemble_method == "weighted":
            # Weighted averaging
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_probs = torch.stack(
                [weight * probs for weight, probs in zip(weights, head_probabilities)],
                dim=0,
            )
            ensemble_probs = weighted_probs.sum(dim=0)

        elif self.ensemble_method == "stacking":
            # Meta-learning approach
            stacked_probs = torch.cat(head_probabilities, dim=-1)
            ensemble_logits = self.meta_classifier(stacked_probs)
            ensemble_probs = F.softmax(ensemble_logits, dim=-1)

        outputs = {
            "probabilities": ensemble_probs,
            "logits": torch.log(ensemble_probs + 1e-8),  # Convert back to logits
        }

        if return_individual:
            outputs["individual_outputs"] = head_outputs
            outputs["individual_probabilities"] = head_probabilities

        return outputs

    def compute_diversity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss to encourage different head behaviors.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Diversity loss value
        """
        head_probabilities = []

        for head in self.heads:
            output = head(x)
            head_probabilities.append(output["probabilities"])

        # Compute pairwise KL divergence between heads
        diversity_loss = 0.0
        num_pairs = 0

        for i in range(len(head_probabilities)):
            for j in range(i + 1, len(head_probabilities)):
                kl_div = F.kl_div(
                    torch.log(head_probabilities[i] + 1e-8),
                    head_probabilities[j],
                    reduction="batchmean",
                )
                diversity_loss += kl_div
                num_pairs += 1

        return -diversity_loss / num_pairs  # Negative to encourage diversity
