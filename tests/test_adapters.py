"""
Unit tests for Task-Aware Adapters (FiLM + LoRA).
"""

import pytest
import torch
import torch.nn as nn
from src.models.adapters import (
    TaskTokenEmbedding,
    FiLMLayer,
    LoRAAdapter,
    AdapterStack,
    TaskAwareTransformer
)


class TestTaskTokenEmbedding:
    """Test task token embedding implementation."""

    def test_initialization(self):
        """Test proper initialization."""
        task_embedding = TaskTokenEmbedding(
            num_tasks=6,
            embedding_dim=128
        )

        assert task_embedding.num_tasks == 6
        assert task_embedding.embedding_dim == 128
        assert task_embedding.embeddings.num_embeddings == 6
        assert task_embedding.embeddings.embedding_dim == 128

    def test_forward_pass(self):
        """Test forward pass shapes."""
        task_embedding = TaskTokenEmbedding(6, 128)

        task_ids = torch.tensor([0, 1, 2, 1, 0, 3])  # (batch_size,)
        embeddings = task_embedding(task_ids)

        assert embeddings.shape == (6, 128)

    def test_different_tasks_different_embeddings(self):
        """Test that different tasks produce different embeddings."""
        task_embedding = TaskTokenEmbedding(6, 128)

        task_id_0 = torch.tensor([0])
        task_id_1 = torch.tensor([1])

        emb_0 = task_embedding(task_id_0)
        emb_1 = task_embedding(task_id_1)

        # Should not be equal (with very high probability)
        assert not torch.allclose(emb_0, emb_1, atol=1e-6)


class TestFiLMLayer:
    """Test FiLM (Feature-wise Linear Modulation) implementation."""

    def test_initialization(self):
        """Test proper initialization."""
        film = FiLMLayer(
            feature_dim=256,
            condition_dim=128
        )

        assert film.feature_dim == 256
        assert film.condition_dim == 128

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        film = FiLMLayer(feature_dim=256, condition_dim=128)

        features = torch.randn(32, 256, 1000)  # (B, C, T)
        condition = torch.randn(32, 128)       # (B, condition_dim)

        modulated = film(features, condition)

        assert modulated.shape == features.shape

    def test_conditioning_changes_output(self):
        """Test that different conditions produce different outputs."""
        torch.manual_seed(42)
        film = FiLMLayer(feature_dim=64, condition_dim=32)

        features = torch.randn(16, 64, 100)
        condition_1 = torch.randn(16, 32)
        condition_2 = torch.randn(16, 32)

        output_1 = film(features, condition_1)
        output_2 = film(features, condition_2)

        # Outputs should be different
        assert not torch.allclose(output_1, output_2, atol=1e-4)

    def test_film_parameters_shape(self):
        """Test that gamma and beta have correct shapes."""
        film = FiLMLayer(feature_dim=256, condition_dim=128)

        condition = torch.randn(32, 128)
        gamma, beta = film.compute_film_params(condition)

        assert gamma.shape == (32, 256, 1)
        assert beta.shape == (32, 256, 1)


class TestLoRAAdapter:
    """Test LoRA (Low-Rank Adaptation) implementation."""

    def test_initialization(self):
        """Test proper initialization."""
        base_layer = nn.Linear(256, 512)
        lora = LoRAAdapter(base_layer, rank=16, alpha=32)

        assert lora.rank == 16
        assert lora.alpha == 32
        assert lora.scaling == 32 / 16  # alpha / rank
        assert lora.lora_A.shape == (16, 256)
        assert lora.lora_B.shape == (512, 16)

    def test_forward_pass_shapes(self):
        """Test forward pass shapes match base layer."""
        base_layer = nn.Linear(256, 512)
        lora = LoRAAdapter(base_layer, rank=16)

        x = torch.randn(32, 256)
        output = lora(x)

        assert output.shape == (32, 512)

    def test_adaptation_changes_output(self):
        """Test that LoRA adaptation changes output."""
        torch.manual_seed(42)
        base_layer = nn.Linear(64, 128)
        lora = LoRAAdapter(base_layer, rank=8)

        x = torch.randn(16, 64)

        # Output with base layer only
        base_output = base_layer(x)

        # Output with LoRA adaptation
        lora_output = lora(x)

        # Should be different (LoRA adds to base output)
        assert not torch.allclose(base_output, lora_output, atol=1e-6)

    def test_zero_initialization(self):
        """Test LoRA starts with minimal impact."""
        base_layer = nn.Linear(64, 128)

        # Initialize LoRA with near-zero values
        lora = LoRAAdapter(base_layer, rank=8)
        nn.init.zeros_(lora.lora_A.weight)
        nn.init.zeros_(lora.lora_B.weight)

        x = torch.randn(16, 64)
        base_output = base_layer(x)
        lora_output = lora(x)

        # Should be very close when LoRA weights are zero
        assert torch.allclose(base_output, lora_output, atol=1e-6)


class TestAdapterStack:
    """Test complete adapter stack integration."""

    def test_initialization(self):
        """Test adapter stack initialization."""
        adapter = AdapterStack(
            feature_dim=256,
            task_embedding_dim=128,
            num_tasks=6,
            use_film=True,
            use_lora=True
        )

        assert adapter.feature_dim == 256
        assert adapter.num_tasks == 6
        assert adapter.use_film
        assert adapter.use_lora

    def test_forward_with_task_conditioning(self):
        """Test forward pass with task conditioning."""
        adapter = AdapterStack(
            feature_dim=256,
            task_embedding_dim=128,
            num_tasks=6,
            use_film=True,
            use_lora=False
        )

        features = torch.randn(32, 256, 1000)  # (B, C, T)
        task_ids = torch.randint(0, 6, (32,))   # (B,)

        adapted_features = adapter(features, task_ids)

        assert adapted_features.shape == features.shape

    def test_task_conditioning_changes_output_distribution(self):
        """Test that different tasks produce different output distributions."""
        torch.manual_seed(42)
        adapter = AdapterStack(
            feature_dim=128,
            task_embedding_dim=64,
            num_tasks=6,
            use_film=True
        )

        features = torch.randn(32, 128, 500)

        # Test two different tasks
        task_ids_1 = torch.zeros(32, dtype=torch.long)  # All task 0
        task_ids_2 = torch.ones(32, dtype=torch.long)   # All task 1

        output_1 = adapter(features, task_ids_1)
        output_2 = adapter(features, task_ids_2)

        # Check that outputs are different
        assert not torch.allclose(output_1, output_2, atol=1e-4)

        # Check that distributions have different statistics
        mean_1 = output_1.mean()
        mean_2 = output_2.mean()
        std_1 = output_1.std()
        std_2 = output_2.std()

        # Means should be different (with high probability)
        assert abs(mean_1 - mean_2) > 1e-3

    def test_film_only_mode(self):
        """Test adapter with FiLM only."""
        adapter = AdapterStack(
            feature_dim=256,
            task_embedding_dim=128,
            num_tasks=6,
            use_film=True,
            use_lora=False
        )

        features = torch.randn(16, 256, 200)
        task_ids = torch.randint(0, 6, (16,))

        output = adapter(features, task_ids)
        assert output.shape == features.shape

    def test_lora_only_mode(self):
        """Test adapter with LoRA only."""
        adapter = AdapterStack(
            feature_dim=256,
            task_embedding_dim=128,
            num_tasks=6,
            use_film=False,
            use_lora=True,
            base_layer=nn.Linear(256, 256)
        )

        features = torch.randn(16, 256, 200)
        task_ids = torch.randint(0, 6, (16,))

        output = adapter(features, task_ids)
        assert output.shape == features.shape


class TestTaskAwareTransformer:
    """Test task-aware transformer integration."""

    def test_initialization(self):
        """Test task-aware transformer initialization."""
        model = TaskAwareTransformer(
            d_model=256,
            num_heads=8,
            num_layers=4,
            num_tasks=6
        )

        assert model.d_model == 256
        assert model.num_tasks == 6
        assert len(model.layers) == 4

    def test_forward_pass_eeg_data(self):
        """Test forward pass on EEG-like data."""
        model = TaskAwareTransformer(
            d_model=128,
            num_heads=8,
            num_layers=2,
            num_tasks=6
        )

        # Simulate EEG data: (batch, channels, time)
        eeg_data = torch.randn(16, 128, 1000)
        task_ids = torch.randint(0, 6, (16,))

        output = model(eeg_data, task_ids)

        # Should preserve spatial dimensions, adapt temporal
        assert output.shape[0] == 16  # batch
        assert output.shape[1] == 128  # channels preserved
        # Temporal dimension may change due to processing

    def test_different_tasks_different_representations(self):
        """Test that different tasks produce different representations."""
        torch.manual_seed(42)
        model = TaskAwareTransformer(
            d_model=64,
            num_heads=4,
            num_layers=2,
            num_tasks=6
        )

        eeg_data = torch.randn(8, 64, 500)

        # Same input, different tasks
        task_0 = torch.zeros(8, dtype=torch.long)
        task_1 = torch.ones(8, dtype=torch.long)

        repr_0 = model(eeg_data, task_0)
        repr_1 = model(eeg_data, task_1)

        # Representations should be different
        assert not torch.allclose(repr_0, repr_1, atol=1e-4)


class TestHBNTaskIntegration:
    """Test integration with HBN dataset tasks."""

    def test_hbn_six_tasks(self):
        """Test adapter with 6 HBN tasks."""
        # HBN tasks: RS, SuS, MW, CCD, SL, SyS
        adapter = AdapterStack(
            feature_dim=768,
            task_embedding_dim=128,
            num_tasks=6,  # 6 HBN paradigms
            use_film=True,
            use_lora=True
        )

        # Simulate batch with different tasks
        features = torch.randn(32, 768, 2000)  # 2s at 1000Hz
        task_ids = torch.randint(0, 6, (32,))   # Random task assignment

        adapted = adapter(features, task_ids)

        assert adapted.shape == features.shape

        # Test each task individually
        for task_id in range(6):
            task_batch = torch.full((8,), task_id, dtype=torch.long)
            features_batch = torch.randn(8, 768, 2000)

            output = adapter(features_batch, task_batch)
            assert output.shape == features_batch.shape

    def test_task_specific_statistics(self):
        """Test that each task produces different output statistics."""
        torch.manual_seed(42)
        adapter = AdapterStack(
            feature_dim=128,
            task_embedding_dim=64,
            num_tasks=6
        )

        features = torch.randn(100, 128, 1000)  # Large batch for statistics

        task_means = []
        task_stds = []

        for task_id in range(6):
            task_ids = torch.full((100,), task_id, dtype=torch.long)
            output = adapter(features, task_ids)

            task_means.append(output.mean().item())
            task_stds.append(output.std().item())

        # Check that tasks produce different statistics
        assert len(set([round(m, 3) for m in task_means])) > 1  # Different means
        assert len(set([round(s, 3) for s in task_stds])) > 1   # Different stds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
