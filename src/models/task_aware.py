"""
Task-aware multi-task architecture with task tokens and lightweight adapters.

This module implements task tokens, FiLM adapters, and LoRA adapters to enable
efficient task-conditioned feature modulation while preserving a single foundation backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
from enum import Enum
import math


class HBNTask(Enum):
    """HBN-EEG task enumeration."""
    RS = "resting_state"      # Resting state
    SUS = "sustained_attention"  # Sustained attention
    MW = "mind_wandering"     # Mind wandering
    CCD = "cognitive_control"  # Cognitive control
    SL = "social_learning"    # Social learning
    SYS = "systems"          # Systems task


class TaskTokenEmbedding(nn.Module):
    """
    Learnable task token embeddings for task-aware processing.

    Each task gets a unique embedding that can be added to temporal sequences
    or used to condition adapter layers.
    """

    def __init__(self,
                 embed_dim: int,
                 tasks: List[HBNTask] = None,
                 dropout: float = 0.1):
        """
        Initialize task token embeddings.

        Args:
            embed_dim: Embedding dimension
            tasks: List of tasks (defaults to all HBN tasks)
            dropout: Dropout rate
        """
        super().__init__()

        if tasks is None:
            tasks = list(HBNTask)

        self.tasks = tasks
        self.task_to_idx = {task: idx for idx, task in enumerate(tasks)}
        self.num_tasks = len(tasks)

        # Learnable task embeddings
        self.task_embeddings = nn.Embedding(self.num_tasks, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize with small random values
        nn.init.normal_(self.task_embeddings.weight, std=0.02)

    def forward(self,
                task_ids: Union[torch.Tensor, List[HBNTask], List[str]],
                batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Get task token embeddings.

        Args:
            task_ids: Task identifiers (tensor of indices, list of HBNTask, or list of strings)
            batch_size: Batch size (inferred if None)

        Returns:
            Task embeddings [batch_size, embed_dim]
        """
        if isinstance(task_ids, (list, tuple)):
            if isinstance(task_ids[0], str):
                # Convert string names to HBNTask
                task_ids = [HBNTask(name) for name in task_ids]

            if isinstance(task_ids[0], HBNTask):
                # Convert HBNTask to indices
                indices = [self.task_to_idx[task] for task in task_ids]
                task_indices = torch.tensor(indices, device=self.task_embeddings.weight.device)
            else:
                task_indices = torch.tensor(task_ids, device=self.task_embeddings.weight.device)
        else:
            task_indices = task_ids

        # Handle single task broadcast to batch
        if task_indices.dim() == 0:
            if batch_size is None:
                raise ValueError("batch_size must be provided for scalar task_ids")
            task_indices = task_indices.unsqueeze(0).expand(batch_size)

        embeddings = self.task_embeddings(task_indices)
        return self.dropout(embeddings)

    def get_task_index(self, task: Union[HBNTask, str]) -> int:
        """Get index for a specific task."""
        if isinstance(task, str):
            task = HBNTask(task)
        return self.task_to_idx[task]


class FiLMAdapter(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) adapter for task conditioning.

    Applies affine transformation γ * x + β where γ and β are predicted
    from task embeddings.
    """

    def __init__(self,
                 feature_dim: int,
                 task_embed_dim: int,
                 hidden_dim: Optional[int] = None,
                 activation: str = 'relu'):
        """
        Initialize FiLM adapter.

        Args:
            feature_dim: Input feature dimension
            task_embed_dim: Task embedding dimension
            hidden_dim: Hidden dimension (defaults to task_embed_dim)
            activation: Activation function
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = task_embed_dim

        self.feature_dim = feature_dim

        # Task conditioning network
        self.task_net = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_dim),
            getattr(nn, activation.title())() if hasattr(nn, activation.title()) else nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim)  # γ and β
        )

        # Initialize to identity transformation
        with torch.no_grad():
            self.task_net[-1].weight.zero_()
            self.task_net[-1].bias.zero_()
            # Set γ to 1, β to 0
            self.task_net[-1].bias[:feature_dim] = 1.0

    def forward(self, x: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: Input features [..., feature_dim]
            task_embed: Task embeddings [batch_size, task_embed_dim]

        Returns:
            Modulated features [..., feature_dim]
        """
        # Get γ and β from task embedding
        film_params = self.task_net(task_embed)  # [batch_size, 2 * feature_dim]
        gamma, beta = film_params.chunk(2, dim=-1)  # Each [batch_size, feature_dim]

        # Broadcast to match input shape
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)

        # Apply FiLM: γ * x + β
        return gamma * x + beta


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for efficient task-specific fine-tuning.

    Adds low-rank matrices A and B such that W + α * A @ B replaces W,
    where α is learned per task.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        """
        Initialize LoRA adapter.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Low-rank dimension
            alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()

        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize A with Gaussian, B with zeros (so initial adaptation is zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input features

        Returns:
            Low-rank adaptation term
        """
        return self.lora_B(self.dropout(self.lora_A(x))) * self.alpha


class TaskConditionedLinear(nn.Module):
    """
    Linear layer with optional task-specific LoRA adapters.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 num_tasks: int = 6,
                 lora_rank: int = 8,
                 lora_alpha: float = 1.0,
                 use_lora: bool = False):
        """
        Initialize task-conditioned linear layer.

        Args:
            in_features: Input features
            out_features: Output features
            bias: Whether to use bias
            num_tasks: Number of tasks
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            use_lora: Whether to use LoRA adapters
        """
        super().__init__()

        # Base linear layer
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_lora = use_lora

        # Task-specific LoRA adapters
        if use_lora:
            self.lora_adapters = nn.ModuleList([
                LoRAAdapter(in_features, out_features, lora_rank, lora_alpha)
                for _ in range(num_tasks)
            ])

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with optional task adaptation.

        Args:
            x: Input tensor
            task_id: Task identifier for LoRA selection

        Returns:
            Output tensor
        """
        # Base transformation
        output = self.base_linear(x)

        # Add task-specific adaptation
        if self.use_lora and task_id is not None:
            output = output + self.lora_adapters[task_id](x)

        return output


class TaskAwareBlock(nn.Module):
    """
    Transformer-like block with task conditioning via FiLM.
    """

    def __init__(self,
                 dim: int,
                 task_embed_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_film: bool = True):
        """
        Initialize task-aware block.

        Args:
            dim: Feature dimension
            task_embed_dim: Task embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            use_film: Whether to use FiLM conditioning
        """
        super().__init__()

        self.dim = dim
        self.use_film = use_film

        # Multi-head attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

        # FiLM conditioning
        if use_film:
            self.film1 = FiLMAdapter(dim, task_embed_dim)
            self.film2 = FiLMAdapter(dim, task_embed_dim)

    def forward(self,
                x: torch.Tensor,
                task_embed: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with task conditioning.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            task_embed: Task embeddings [batch_size, task_embed_dim]
            mask: Attention mask

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        if self.use_film and task_embed is not None:
            x_norm = self.film1(x_norm, task_embed)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out

        # MLP with residual
        x_norm = self.norm2(x)
        if self.use_film and task_embed is not None:
            x_norm = self.film2(x_norm, task_embed)

        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class TaskAwareTemporalCNN(nn.Module):
    """
    Enhanced TemporalCNN with task conditioning capabilities.
    """

    def __init__(self,
                 input_channels: int = 19,
                 temporal_kernel_size: int = 25,
                 num_layers: int = 5,
                 hidden_channels: List[int] = None,
                 dropout: float = 0.3,
                 task_embed_dim: int = 64,
                 use_film: bool = True,
                 use_lora: bool = False,
                 lora_rank: int = 8):
        """
        Initialize task-aware TemporalCNN.

        Args:
            input_channels: Number of EEG channels
            temporal_kernel_size: 1D convolution kernel size
            num_layers: Number of CNN layers
            hidden_channels: Hidden channel dimensions
            dropout: Dropout rate
            task_embed_dim: Task embedding dimension
            use_film: Use FiLM conditioning
            use_lora: Use LoRA adapters
            lora_rank: LoRA rank
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256, 512][:num_layers]

        self.use_film = use_film
        self.use_lora = use_lora
        self.task_embed_dim = task_embed_dim

        # Task token embedding
        self.task_embedder = TaskTokenEmbedding(task_embed_dim)

        # Build CNN layers
        self.layers = nn.ModuleList()
        self.film_adapters = nn.ModuleList() if use_film else None

        in_channels = input_channels
        for i, out_channels in enumerate(hidden_channels):
            # Depthwise separable convolution
            layer = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, temporal_kernel_size,
                         padding=temporal_kernel_size//2, groups=in_channels),
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)

            # FiLM adapter for this layer
            if use_film:
                self.film_adapters.append(FiLMAdapter(out_channels, task_embed_dim))

            in_channels = out_channels

        # Global pooling and output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = hidden_channels[-1]

    def forward(self,
                x: torch.Tensor,
                task_id: Union[torch.Tensor, HBNTask, str, None] = None) -> torch.Tensor:
        """
        Forward pass with task conditioning.

        Args:
            x: Input EEG data [batch_size, channels, time]
            task_id: Task identifier

        Returns:
            Feature representation [batch_size, output_dim]
        """
        batch_size = x.size(0)

        # Get task embedding
        task_embed = None
        if task_id is not None and (self.use_film or self.use_lora):
            task_embed = self.task_embedder(task_id, batch_size)

        # Forward through CNN layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply FiLM conditioning
            if self.use_film and task_embed is not None:
                # Reshape for FiLM: [B, C, T] -> [B, T, C]
                x_film = x.transpose(1, 2)
                x_film = self.film_adapters[i](x_film, task_embed)
                x = x_film.transpose(1, 2)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # [batch_size, output_dim]

        return x

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head with task-specific adapters.
    """

    def __init__(self,
                 input_dim: int,
                 task_configs: Dict[str, Dict],
                 task_embed_dim: int = 64,
                 use_film: bool = True,
                 shared_hidden_dim: int = 256):
        """
        Initialize multi-task head.

        Args:
            input_dim: Input feature dimension
            task_configs: Task-specific configurations
            task_embed_dim: Task embedding dimension
            use_film: Use FiLM conditioning
            shared_hidden_dim: Shared hidden dimension
        """
        super().__init__()

        self.task_configs = task_configs
        self.use_film = use_film

        # Shared feature processing
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(shared_hidden_dim, shared_hidden_dim)
        )

        # FiLM conditioning for shared features
        if use_film:
            self.film_adapter = FiLMAdapter(shared_hidden_dim, task_embed_dim)

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            output_dim = config['output_dim']
            head_type = config.get('type', 'regression')

            if head_type == 'regression':
                head = nn.Sequential(
                    nn.Linear(shared_hidden_dim, output_dim)
                )
            elif head_type == 'classification':
                head = nn.Sequential(
                    nn.Linear(shared_hidden_dim, output_dim),
                    nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=-1)
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")

            self.task_heads[task_name] = head

    def forward(self,
                features: torch.Tensor,
                task_embed: torch.Tensor,
                task_name: str) -> torch.Tensor:
        """
        Forward pass for specific task.

        Args:
            features: Input features [batch_size, input_dim]
            task_embed: Task embedding [batch_size, task_embed_dim]
            task_name: Target task name

        Returns:
            Task-specific predictions
        """
        # Shared processing
        shared_features = self.shared_net(features)

        # Task conditioning
        if self.use_film:
            shared_features = self.film_adapter(shared_features, task_embed)

        # Task-specific prediction
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")

        return self.task_heads[task_name](shared_features)


# Utility functions
def create_hbn_task_configs():
    """Create standard HBN task configurations."""
    return {
        'ccd_rt': {
            'output_dim': 1,
            'type': 'regression',
            'loss': 'corr_mse',
            'metrics': ['pearson_r', 'rmse']
        },
        'ccd_success': {
            'output_dim': 1,
            'type': 'classification',
            'loss': 'bce',
            'metrics': ['auroc', 'auprc', 'balanced_accuracy']
        },
        'cbcl_p_factor': {
            'output_dim': 1,
            'type': 'regression',
            'loss': 'mse',
            'metrics': ['pearson_r']
        },
        'cbcl_internalizing': {
            'output_dim': 1,
            'type': 'regression',
            'loss': 'mse',
            'metrics': ['pearson_r']
        },
        'cbcl_externalizing': {
            'output_dim': 1,
            'type': 'regression',
            'loss': 'mse',
            'metrics': ['pearson_r']
        },
        'cbcl_attention': {
            'output_dim': 1,
            'type': 'regression',
            'loss': 'mse',
            'metrics': ['pearson_r']
        }
    }


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test task token embedding
    task_embedder = TaskTokenEmbedding(embed_dim=64)
    task_embed = task_embedder([HBNTask.SUS, HBNTask.CCD])
    print(f"Task embeddings shape: {task_embed.shape}")

    # Test task-aware CNN
    model = TaskAwareTemporalCNN(
        input_channels=19,
        num_layers=4,
        task_embed_dim=64,
        use_film=True
    ).to(device)

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 19, 1000).to(device)
    task_ids = [HBNTask.SUS] * batch_size

    features = model(x, task_ids)
    print(f"Output features shape: {features.shape}")

    # Test multi-task head
    task_configs = create_hbn_task_configs()
    head = MultiTaskHead(
        input_dim=model.get_output_dim(),
        task_configs=task_configs,
        task_embed_dim=64
    ).to(device)

    task_embed = task_embedder(task_ids)
    ccd_rt_pred = head(features, task_embed, 'ccd_rt')
    print(f"CCD RT predictions shape: {ccd_rt_pred.shape}")

    print("✅ Task-aware architecture test completed successfully!")
