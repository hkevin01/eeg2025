#!/usr/bin/env python3
"""
Minimal GPU test: load model, run single forward pass, log everything, timeout safeguard.
"""
import os
import sys
import time
import signal
from pathlib import Path
import torch
import torch.nn as nn
import logging

# Set ROCm environment variables (OpenNLP-GPU style)
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['HIP_PATH'] = '/opt/rocm'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'

# Setup logging
logfile = Path(__file__).parent / f"gpu_minimal_test_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Timeout safeguard (60s)
def handler(signum, frame):
    logging.error("Timeout reached! Exiting.")
    sys.exit(1)
signal.signal(signal.SIGALRM, handler)
signal.alarm(60)

class FoundationTransformer(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=64, n_heads=4, n_layers=2, n_classes=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

def main():
    logging.info("==== Minimal GPU Test: Model Load + Forward Pass ====")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        device = torch.device('cpu')
        logging.warning("No GPU detected, using CPU!")

    # Model
    model = FoundationTransformer().to(device)
    logging.info("Model loaded to device.")

    # Forward pass
    try:
        test_input = torch.randn(2, 129, 1000).to(device)
        logging.info("Test input created.")
        output = model(test_input)
        logging.info(f"Forward pass successful. Output shape: {output.shape}")
        logging.info("Test PASSED.")
    except Exception as e:
        logging.error(f"Forward pass FAILED: {e}")
        sys.exit(2)

    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    signal.alarm(0)  # Cancel timeout
    logging.info("==== Minimal GPU Test Complete ====")

if __name__ == "__main__":
    main()
