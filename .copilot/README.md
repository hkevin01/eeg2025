# EEG Foundation Challenge Copilot Configuration

This folder contains configuration files and prompts to enhance GitHub Copilot's assistance with this EEG machine learning project.

## Custom Prompts

The project involves:
- EEG signal processing and analysis
- Self-supervised learning on brain signals
- Cross-task transfer learning
- Psychopathology prediction from neural data
- Real-time compression algorithms
- PyTorch/Lightning model architectures

## Code Context

When working on this project, consider:
- EEG data is typically sampled at 250-1000 Hz
- Common preprocessing: filtering (0.1-40 Hz), re-referencing, artifact removal
- Models should handle variable-length sequences
- Subject-invariant features are crucial for generalization
- Compression should preserve clinically relevant information
- Follow BIDS standard for neuroimaging data organization

## Best Practices

- Use type hints for all functions
- Document signal processing parameters
- Include shape comments for tensor operations
- Test with multiple sampling rates
- Validate preprocessing pipeline thoroughly
- Ensure reproducibility with fixed seeds
