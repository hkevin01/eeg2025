#!/usr/bin/env python3
"""
Feature Visualization for EEG Models
=====================================
Visualize what the models are learning using gradient-based saliency maps.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80, flush=True)
print("🔍 FEATURE VISUALIZATION - EEG MODELS", flush=True)
print("="*80, flush=True)

# Import models
import sys
sys.path.insert(0, str(Path.cwd()))
from submission import Submission, ResponseTimeCNN, ExternalizingCNN

def compute_saliency(model, input_data, target_output=None):
    """Compute saliency map using gradients"""
    model.eval()
    input_data.requires_grad = True

    # Forward pass
    output = model(input_data)

    # Use first sample output for gradient
    if target_output is None:
        target_output = output[0].sum()  # Sum to get scalar for backward

    # Backward pass
    model.zero_grad()
    target_output.backward()

    # Get gradients
    saliency = input_data.grad.abs()

    return saliency, output

def visualize_channel_importance(saliency, channels_to_show=10):
    """Visualize which EEG channels are most important"""
    # Average saliency across time for each channel
    # Use first sample if batch
    if saliency.dim() == 3:
        saliency_single = saliency[0]
    else:
        saliency_single = saliency

    channel_importance = saliency_single.mean(dim=1).numpy()

    # Get top channels
    top_indices = np.argsort(channel_importance)[-channels_to_show:][::-1]
    top_values = channel_importance[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(channels_to_show), top_values)
    ax.set_yticks(range(channels_to_show))
    ax.set_yticklabels([f"Channel {i+1}" for i in top_indices])
    ax.set_xlabel('Average Importance')
    ax.set_title('Top 10 Most Important EEG Channels')
    ax.invert_yaxis()

    return fig, top_indices

def visualize_temporal_importance(saliency, sample_idx=0):
    """Visualize when in time the model pays attention"""
    # Average across channels
    temporal_importance = saliency[sample_idx].mean(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(temporal_importance)
    ax.set_xlabel('Time (samples @ 100Hz)')
    ax.set_ylabel('Importance')
    ax.set_title('Temporal Attention Pattern')
    ax.grid(True, alpha=0.3)

    return fig

def visualize_saliency_heatmap(saliency, sample_idx=0, channels_to_show=30):
    """Visualize full spatiotemporal saliency heatmap"""
    saliency_map = saliency[sample_idx, :channels_to_show, :].numpy()

    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(saliency_map, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Time (samples @ 100Hz)')
    ax.set_ylabel('EEG Channel')
    ax.set_title('Saliency Heatmap (Channel × Time)')
    plt.colorbar(im, ax=ax, label='Importance')

    return fig

def main():
    """Generate feature visualizations"""
    print("\n📦 Loading models...", flush=True)

    SFREQ = 100
    DEVICE = torch.device('cpu')

    # Create test data
    print("🔢 Creating test data...", flush=True)
    X_test = torch.randn(2, 129, 200)  # 2 samples, 129 channels, 200 samples

    results_dir = Path("results/visualizations")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Challenge 1: Response Time
    print("\n🎯 Challenge 1: Response Time Visualization", flush=True)
    try:
        model1 = ResponseTimeCNN()
        checkpoint = torch.load('weights_challenge_1.pt', map_location='cpu', weights_only=True)
        model1.load_state_dict(checkpoint)

        print("   Computing saliency...", end=' ', flush=True)
        saliency1, output1 = compute_saliency(model1, X_test)
        print("✓", flush=True)

        print("   Generating visualizations...", end=' ', flush=True)

        # Channel importance
        fig1, top_channels1 = visualize_channel_importance(saliency1)
        fig1.savefig(results_dir / 'c1_channel_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # Temporal importance
        fig2 = visualize_temporal_importance(saliency1)
        fig2.savefig(results_dir / 'c1_temporal_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)

        # Heatmap
        fig3 = visualize_saliency_heatmap(saliency1)
        fig3.savefig(results_dir / 'c1_saliency_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)

        print("✓", flush=True)
        print(f"   Prediction: {output1[0].item():.2f}s", flush=True)
        print(f"   Top channels: {top_channels1[:5] + 1}", flush=True)

    except Exception as e:
        print(f"⚠️  C1 visualization failed: {e}", flush=True)

    # Challenge 2: Externalizing Factor
    print("\n�� Challenge 2: Externalizing Factor Visualization", flush=True)
    try:
        model2 = ExternalizingCNN()
        checkpoint = torch.load('weights_challenge_2.pt', map_location='cpu', weights_only=True)
        model2.load_state_dict(checkpoint)

        print("   Computing saliency...", end=' ', flush=True)
        saliency2, output2 = compute_saliency(model2, X_test)
        print("✓", flush=True)

        print("   Generating visualizations...", end=' ', flush=True)

        # Channel importance
        fig4, top_channels2 = visualize_channel_importance(saliency2)
        fig4.savefig(results_dir / 'c2_channel_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)

        # Temporal importance
        fig5 = visualize_temporal_importance(saliency2)
        fig5.savefig(results_dir / 'c2_temporal_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig5)

        # Heatmap
        fig6 = visualize_saliency_heatmap(saliency2)
        fig6.savefig(results_dir / 'c2_saliency_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig6)

        print("✓", flush=True)
        print(f"   Prediction: {output2[0].item():.3f}", flush=True)
        print(f"   Top channels: {top_channels2[:5] + 1}", flush=True)

    except Exception as e:
        print(f"⚠️  C2 visualization failed: {e}", flush=True)

    print("\n" + "="*80, flush=True)
    print("📁 Visualizations saved to: results/visualizations/", flush=True)
    print("="*80, flush=True)

    print("\nGenerated files:", flush=True)
    for f in sorted(results_dir.glob("*.png")):
        print(f"   - {f.name}", flush=True)

    print("\n✅ FEATURE VISUALIZATION COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
