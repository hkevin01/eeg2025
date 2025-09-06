"""
Task-Aware Adapters
===================

Implements FiLM and LoRA adapters for task-specific conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class TaskTokenEmbedding(nn.Module):
    """Learnable task token embeddings."""

    def __init__(self, num_tasks: int, d_model: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.d_model = d_model

        self.task_embeddings = nn.Embedding(num_tasks, d_model)

        # Initialize with small random values
        nn.init.normal_(self.task_embeddings.weight, std=0.02)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Get task embeddings.

        Args:
            task_ids: Task IDs of shape [batch_size]

        Returns:
            Task embeddings of shape [batch_size, d_model]
        """
        return self.task_embeddings(task_ids)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies affine transformation conditioned on task.
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()

        self.d_model = d_model

        # Networks to predict scale and shift
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid()  # Ensure positive scaling
        )

        self.shift_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: Input features of shape [batch_size, seq_len, d_model]
            task_embedding: Task embeddings of shape [batch_size, d_model]

        Returns:
            Modulated features of same shape as input
        """
        # Compute scale and shift parameters
        scale = self.scale_net(task_embedding)  # [batch_size, d_model]
        shift = self.shift_net(task_embedding)  # [batch_size, d_model]

        # Expand for broadcasting
        scale = scale.unsqueeze(1)  # [batch_size, 1, d_model]
        shift = shift.unsqueeze(1)  # [batch_size, 1, d_model]

        # Apply FiLM: x' = scale * x + shift
        return scale * x + shift


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) adapter.
    Applies low-rank updates to linear transformations.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Linear(d_model, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input features of shape [batch_size, seq_len, d_model]
            task_embedding: Task embeddings (used for conditioning)

        Returns:
            Adapted features of same shape as input
        """
        # Apply low-rank transformation
        lora_out = self.lora_B(self.lora_A(x))
        lora_out = self.dropout(lora_out)

        # Scale and add to original features
        return x + lora_out * self.scale


class TaskAwareAdapter(nn.Module):
    """
    Combined task-aware adapter using both FiLM and LoRA.
    """

    def __init__(
        self,
        d_model: int,
        num_tasks: int,
        adapter_type: str = "both",  # "film", "lora", "both"
        film_hidden_dim: int = 256,
        lora_rank: int = 16,
        lora_alpha: float = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.adapter_type = adapter_type
        self.d_model = d_model

        # Task embeddings
        self.task_embeddings = TaskTokenEmbedding(num_tasks, d_model)

        # Adapters
        if adapter_type in ["film", "both"]:
            self.film_layer = FiLMLayer(d_model, film_hidden_dim)

        if adapter_type in ["lora", "both"]:
            self.lora_adapter = LoRAAdapter(d_model, lora_rank, lora_alpha, dropout)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply task-aware adaptation.

        Args:
            x: Input features of shape [batch_size, seq_len, d_model]
            task_ids: Task IDs of shape [batch_size]

        Returns:
            Adapted features of same shape as input
        """
        # Get task embeddings
        task_emb = self.task_embeddings(task_ids)

        # Apply adapters
        adapted_x = x

        if self.adapter_type in ["film", "both"]:
            adapted_x = self.film_layer(adapted_x, task_emb)

        if self.adapter_type in ["lora", "both"]:
            adapted_x = self.lora_adapter(adapted_x, task_emb)

        # Apply normalization
        adapted_x = self.norm(adapted_x)

        return adapted_x


class AdapterStack(nn.Module):
    """
    Stack of task-aware adapters for multi-level adaptation.
    """

    def __init__(
        self,
        d_model: int,
        num_tasks: int,
        num_layers: int = 3,
        **adapter_kwargs
    ):
        super().__init__()

        self.adapters = nn.ModuleList([
            TaskAwareAdapter(d_model, num_tasks, **adapter_kwargs)
            for _ in range(num_layers)
        ])

        # Residual connections
        self.use_residual = True

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply stack of adapters.

        Args:
            x: Input features
            task_ids: Task IDs

        Returns:
            Adapted features
        """
        for adapter in self.adapters:
            if self.use_residual:
                x = x + adapter(x, task_ids)
            else:
                x = adapter(x, task_ids)

        return x


class HBNTaskAdapter(nn.Module):
    """
    Specialized adapter for HBN dataset tasks.
    Maps HBN task names to task IDs.
    """

    HBN_TASKS = {
        'RS': 0,    # Resting State
        'SuS': 1,   # Sustained Attention
        'MW': 2,    # Mind Wandering
        'CCD': 3,   # Continuous Performance
        'SL': 4,    # Statistical Learning
        'SyS': 5    # Syllable Processing
    }

    def __init__(
        self,
        d_model: int,
        **adapter_kwargs
    ):
        super().__init__()

        self.task_to_id = self.HBN_TASKS
        self.num_tasks = len(self.HBN_TASKS)

        self.adapter = TaskAwareAdapter(
            d_model=d_model,
            num_tasks=self.num_tasks,
            **adapter_kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        task_names: Optional[list] = None,
        task_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply HBN-specific task adaptation.

        Args:
            x: Input features
            task_names: List of task names (e.g., ['RS', 'SuS'])
            task_ids: Task IDs (alternative to task_names)

        Returns:
            Adapted features
        """
        if task_ids is None:
            if task_names is None:
                raise ValueError("Either task_names or task_ids must be provided")

            # Convert task names to IDs
            task_ids = torch.tensor([
                self.task_to_id.get(name, 0) for name in task_names
            ], device=x.device)

        return self.adapter(x, task_ids)


def create_task_adapter(config: Dict[str, Any]) -> TaskAwareAdapter:
    """
    Factory function to create task adapter from config.

    Args:
        config: Configuration dictionary

    Returns:
        Task adapter instance
    """
    adapter_config = config.get('adapter_configs', {})

    return TaskAwareAdapter(
        d_model=config['d_model'],
        num_tasks=config.get('num_tasks', 6),
        adapter_type=adapter_config.get('type', 'both'),
        film_hidden_dim=adapter_config.get('film_hidden_dim', 256),
        lora_rank=adapter_config.get('lora_rank', 16),
        lora_alpha=adapter_config.get('lora_alpha', 32),
        dropout=adapter_config.get('dropout', 0.1)
    )
