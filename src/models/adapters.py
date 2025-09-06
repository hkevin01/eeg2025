"""
Task-Aware Multi-Task Architecture
=================================

Task tokens and lightweight adapters for multi-task EEG foundation models.
Supports FiLM (Feature-wise Linear Modulation) and LoRA (Low-Rank Adaptation)
adapters conditioned on task embeddings.

Key Features:
- Task token embeddings for different EEG paradigms (RS, SuS, MW, CCD, SL, SyS)
- FiLM adapters for feature modulation
- LoRA adapters for efficient parameter adaptation
- Task-conditioned attention mechanisms
- Minimal parameter overhead while maintaining task-specific adaptability
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# Task definitions for HBN-EEG paradigms
TASK_NAMES = {
    'RS': 0,    # Resting State
    'SuS': 1,   # Sustained Attention
    'MW': 2,    # Mind Wandering
    'CCD': 3,   # Cognitive Control Demand
    'SL': 4,    # Statistical Learning
    'SyS': 5,   # Symbolic System
}

TASK_ID_TO_NAME = {v: k for k, v in TASK_NAMES.items()}


@dataclass
class TaskAdapterConfig:
    """Configuration for task adapters."""
    adapter_type: str = "film"  # "film", "lora", "both"
    task_emb_dim: int = 64
    hidden_dim: int = 128
    lora_rank: int = 16
    lora_alpha: float = 16.0
    dropout: float = 0.1
    use_task_attention: bool = False
    freeze_backbone: bool = False
    adapter_layers: List[int] = None  # Which layers to adapt (None = all)


class TaskTokenEmbedding(nn.Module):
    """
    Learnable task token embeddings for different EEG paradigms.

    Creates dense embeddings for each task type that can be used to
    condition the model architecture or processing pipeline.

    Args:
        num_tasks: Number of different tasks/paradigms
        emb_dim: Embedding dimension
        learnable: Whether embeddings are learnable or fixed
        init_std: Standard deviation for initialization
    """

    def __init__(
        self,
        num_tasks: int = 6,
        emb_dim: int = 64,
        learnable: bool = True,
        init_std: float = 0.02
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.emb_dim = emb_dim

        # Create task embeddings
        self.task_embeddings = nn.Parameter(
            torch.randn(num_tasks, emb_dim) * init_std,
            requires_grad=learnable
        )

        # Optional task names for interpretability
        self.register_buffer('task_names', torch.arange(num_tasks))

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Get task embeddings for given task IDs.

        Args:
            task_ids: Task IDs (B,) or (B, seq_len)

        Returns:
            Task embeddings (B, emb_dim) or (B, seq_len, emb_dim)
        """
        return F.embedding(task_ids, self.task_embeddings)

    def get_task_similarity(self) -> torch.Tensor:
        """Get pairwise cosine similarity between tasks."""
        norm_embs = F.normalize(self.task_embeddings, p=2, dim=1)
        return torch.mm(norm_embs, norm_embs.t())


class FiLMAdapter(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) adapter.

    Applies task-conditioned affine transformations to feature maps:
    output = gamma * features + beta

    where gamma and beta are predicted from task embeddings.

    Args:
        feature_dim: Dimension of features to modulate
        task_emb_dim: Dimension of task embeddings
        hidden_dim: Hidden dimension for gamma/beta prediction
        dropout: Dropout rate
    """

    def __init__(
        self,
        feature_dim: int,
        task_emb_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Network to predict gamma and beta from task embedding
        self.film_net = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * feature_dim)  # gamma and beta
        )

        # Initialize to identity transformation
        with torch.no_grad():
            self.film_net[-1].weight.zero_()
            self.film_net[-1].bias.zero_()
            # Set gamma to 1, beta to 0
            self.film_net[-1].bias[:feature_dim].fill_(1.0)

    def forward(self, features: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to features.

        Args:
            features: Input features (..., feature_dim)
            task_emb: Task embedding (B, task_emb_dim)

        Returns:
            Modulated features (..., feature_dim)
        """
        # Get gamma and beta
        film_params = self.film_net(task_emb)  # (B, 2 * feature_dim)
        gamma, beta = film_params.chunk(2, dim=-1)  # Each (B, feature_dim)

        # Expand dimensions to match features
        original_shape = features.shape
        if len(original_shape) > 2:
            # Reshape for broadcasting
            batch_size = gamma.shape[0]
            gamma = gamma.view(batch_size, *([1] * (len(original_shape) - 2)), self.feature_dim)
            beta = beta.view(batch_size, *([1] * (len(original_shape) - 2)), self.feature_dim)

        return gamma * features + beta


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) adapter for efficient fine-tuning.

    Adds low-rank decomposition to linear layers:
    W' = W + (B @ A) * scaling

    where A and B are low-rank matrices, and W is frozen.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        rank: Rank of decomposition
        alpha: Scaling factor
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Adapted output (..., out_features)
        """
        # Low-rank adaptation: x @ A^T @ B^T * scaling
        result = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return result


class TaskAdaptiveLinear(nn.Module):
    """
    Task-adaptive linear layer with optional FiLM and LoRA adapters.

    Combines a base linear layer with task-specific adaptations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        task_emb_dim: int,
        config: TaskAdapterConfig,
        bias: bool = True
    ):
        super().__init__()
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)
        self.config = config

        # Freeze base layer if specified
        if config.freeze_backbone:
            for param in self.base_linear.parameters():
                param.requires_grad = False

        # Add adapters based on configuration
        self.film_adapter = None
        self.lora_adapter = None

        if config.adapter_type in ["film", "both"]:
            self.film_adapter = FiLMAdapter(
                feature_dim=out_features,
                task_emb_dim=task_emb_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout
            )

        if config.adapter_type in ["lora", "both"]:
            self.lora_adapter = LoRAAdapter(
                in_features=in_features,
                out_features=out_features,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.dropout
            )

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with task adaptation.

        Args:
            x: Input tensor (..., in_features)
            task_emb: Task embedding (B, task_emb_dim)

        Returns:
            Adapted output (..., out_features)
        """
        # Base transformation
        output = self.base_linear(x)

        # Add LoRA adaptation
        if self.lora_adapter is not None:
            output = output + self.lora_adapter(x)

        # Apply FiLM modulation
        if self.film_adapter is not None:
            output = self.film_adapter(output, task_emb)

        return output


class TaskConditionedAttention(nn.Module):
    """
    Task-conditioned attention mechanism.

    Uses task embeddings to modulate attention patterns,
    allowing the model to focus on task-relevant features.
    """

    def __init__(
        self,
        hidden_dim: int,
        task_emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Standard attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Task conditioning
        self.task_to_bias = nn.Linear(task_emb_dim, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        task_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Task-conditioned attention forward pass.

        Args:
            x: Input tensor (B, seq_len, hidden_dim)
            task_emb: Task embedding (B, task_emb_dim)
            mask: Optional attention mask (B, seq_len) or (B, seq_len, seq_len)

        Returns:
            Attended output (B, seq_len, hidden_dim)
        """
        B, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Task-conditioned bias
        task_bias = self.task_to_bias(task_emb)  # (B, num_heads)
        task_bias = task_bias.unsqueeze(-1).unsqueeze(-1)  # (B, num_heads, 1, 1)
        attn_scores = attn_scores + task_bias

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output


class TaskAwareBackbone(nn.Module):
    """
    Task-aware backbone that wraps existing models with task adaptation.

    This wrapper adds task tokens and adapters to any existing backbone
    architecture while maintaining compatibility.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: TaskAdapterConfig,
        backbone_output_dim: int,
        num_tasks: int = 6
    ):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.backbone_output_dim = backbone_output_dim

        # Task token embeddings
        self.task_tokens = TaskTokenEmbedding(
            num_tasks=num_tasks,
            emb_dim=config.task_emb_dim
        )

        # Optional task-conditioned layers
        self.task_projection = None
        if config.use_task_attention:
            self.task_projection = TaskConditionedAttention(
                hidden_dim=backbone_output_dim,
                task_emb_dim=config.task_emb_dim,
                dropout=config.dropout
            )

        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with task conditioning.

        Args:
            x: Input tensor (B, C, T) for EEG data
            task_ids: Task IDs (B,)
            return_features: Whether to return intermediate features

        Returns:
            Output tensor, optionally with features
        """
        # Get task embeddings
        task_emb = self.task_tokens(task_ids)  # (B, task_emb_dim)

        # Forward through backbone
        features = self.backbone(x)  # (B, backbone_output_dim) or (B, seq_len, backbone_output_dim)

        # Apply task conditioning if configured
        if self.task_projection is not None:
            if features.dim() == 2:
                # Add sequence dimension for attention
                features = features.unsqueeze(1)  # (B, 1, backbone_output_dim)
                features = self.task_projection(features, task_emb)
                features = features.squeeze(1)  # (B, backbone_output_dim)
            else:
                features = self.task_projection(features, task_emb)

        if return_features:
            return features, task_emb
        return features

    def get_task_similarities(self) -> torch.Tensor:
        """Get task similarity matrix for analysis."""
        return self.task_tokens.get_task_similarity()


class TaskSpecificHead(nn.Module):
    """
    Task-specific prediction head with adapters.

    Can be used for different tasks like regression (RT prediction)
    or classification (success prediction).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_emb_dim: int,
        config: TaskAdapterConfig,
        head_type: str = "regression"  # "regression" or "classification"
    ):
        super().__init__()
        self.head_type = head_type
        self.config = config

        # Build head layers with optional adaptation
        if config.adapter_type == "none":
            # Simple linear head
            self.head = nn.Linear(input_dim, output_dim)
        else:
            # Adaptive head
            hidden_dim = config.hidden_dim
            self.head = nn.Sequential(
                TaskAdaptiveLinear(
                    in_features=input_dim,
                    out_features=hidden_dim,
                    task_emb_dim=task_emb_dim,
                    config=config
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                TaskAdaptiveLinear(
                    in_features=hidden_dim,
                    out_features=output_dim,
                    task_emb_dim=task_emb_dim,
                    config=config
                )
            )

        # Output activation
        if head_type == "classification":
            self.output_activation = nn.Sigmoid()  # For binary classification
        else:
            self.output_activation = nn.Identity()

    def forward(self, features: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through task-specific head.

        Args:
            features: Input features (B, input_dim)
            task_emb: Task embedding (B, task_emb_dim)

        Returns:
            Predictions (B, output_dim)
        """
        if isinstance(self.head, nn.Linear):
            # Simple linear case
            output = self.head(features)
        else:
            # Adaptive case - pass task embeddings through each layer
            output = features
            for layer in self.head:
                if isinstance(layer, TaskAdaptiveLinear):
                    output = layer(output, task_emb)
                else:
                    output = layer(output)

        return self.output_activation(output)


def create_task_aware_model(
    backbone: nn.Module,
    backbone_output_dim: int,
    config: TaskAdapterConfig,
    num_tasks: int = 6,
    output_dims: Optional[Dict[str, int]] = None
) -> Tuple[TaskAwareBackbone, Dict[str, TaskSpecificHead]]:
    """
    Create a complete task-aware model with backbone and heads.

    Args:
        backbone: Base backbone model
        backbone_output_dim: Output dimension of backbone
        config: Task adapter configuration
        num_tasks: Number of tasks
        output_dims: Output dimensions for different heads

    Returns:
        Task-aware backbone and dictionary of task-specific heads
    """
    if output_dims is None:
        output_dims = {
            "regression": 1,      # RT prediction
            "classification": 1,  # Success prediction
            "psychopathology": 4  # CBCL factors
        }

    # Create task-aware backbone
    task_backbone = TaskAwareBackbone(
        backbone=backbone,
        config=config,
        backbone_output_dim=backbone_output_dim,
        num_tasks=num_tasks
    )

    # Create task-specific heads
    heads = {}
    for head_name, output_dim in output_dims.items():
        head_type = "classification" if "classification" in head_name else "regression"
        heads[head_name] = TaskSpecificHead(
            input_dim=backbone_output_dim,
            output_dim=output_dim,
            task_emb_dim=config.task_emb_dim,
            config=config,
            head_type=head_type
        )

    return task_backbone, heads


# Utility functions for task management
def get_task_id(task_name: str) -> int:
    """Get task ID from task name."""
    return TASK_NAMES.get(task_name.upper(), 0)


def get_task_name(task_id: int) -> str:
    """Get task name from task ID."""
    return TASK_ID_TO_NAME.get(task_id, "UNKNOWN")


def create_task_batch(task_names: List[str], batch_size: int) -> torch.Tensor:
    """Create task ID tensor for a batch."""
    task_ids = [get_task_id(name) for name in task_names]
    return torch.tensor(task_ids * (batch_size // len(task_ids) + 1))[:batch_size]


# Example usage and testing
if __name__ == "__main__":
    # Test task tokens
    print("Testing task token embeddings...")
    task_tokens = TaskTokenEmbedding(num_tasks=6, emb_dim=64)
    task_ids = torch.tensor([0, 1, 2, 3, 4, 5])  # All tasks
    embeddings = task_tokens(task_ids)
    print(f"Task embeddings shape: {embeddings.shape}")

    # Test task similarity
    similarity = task_tokens.get_task_similarity()
    print(f"Task similarity matrix shape: {similarity.shape}")

    # Test FiLM adapter
    print("\nTesting FiLM adapter...")
    film = FiLMAdapter(feature_dim=256, task_emb_dim=64)
    features = torch.randn(32, 256)
    task_emb = torch.randn(32, 64)
    modulated = film(features, task_emb)
    print(f"FiLM output shape: {modulated.shape}")

    # Test LoRA adapter
    print("\nTesting LoRA adapter...")
    lora = LoRAAdapter(in_features=256, out_features=256, rank=16)
    adapted = lora(features)
    print(f"LoRA output shape: {adapted.shape}")

    # Test task-adaptive linear layer
    print("\nTesting task-adaptive linear layer...")
    config = TaskAdapterConfig(adapter_type="both", task_emb_dim=64)
    adaptive_linear = TaskAdaptiveLinear(256, 128, 64, config)
    output = adaptive_linear(features, task_emb)
    print(f"Adaptive linear output shape: {output.shape}")

    print("\nAll tests passed!")
