# Interactive EEG GPU Demo

This directory contains the interactive demonstration interface for our GPU-accelerated EEG processing pipeline.

## Quick Start

### Option 1: Full Docker Stack (Recommended)

```bash
# Start complete demo environment
./scripts/demo.sh start

# View at: http://localhost:8080/demo/
```

### Option 2: Development Server

```bash
# Install dependencies
pip install fastapi uvicorn[standard] scipy

# Start development server
./scripts/demo.sh dev

# View at: http://localhost:8000/web/demo.html
```

## Demo Features

### Real-Time GPU Processing

- **Triton Fused Kernels**: Toggle bandpass+notch+CAR preprocessing
- **RMSNorm**: Fast temporal normalization
- **CuPy Compression**: Perceptual quantization augmentation

### Live Visualization

- **Time Series**: Real-time EEG signal display
- **Power Spectrum**: Frequency domain analysis
- **Performance Metrics**: Latency tracking (preproc, quant, forward)

### Robustness Testing

- **Channel Dropout**: Simulate montage variations
- **Compression Stress**: Test deployment robustness
- **Scenario Presets**: Clean, noisy, dropout, robustness mix

### Mobile Responsive

- Works on phones, tablets, and desktops
- Touch-friendly controls
- Adaptive layout

## Architecture

```text
Demo Stack:
├── Frontend (web/demo.html)
│   ├── Real-time charts (Canvas API)
│   ├── GPU controls
│   └── Performance visualization
├── Backend (backend/demo_server.py)
│   ├── FastAPI server
│   ├── GPU inference pipeline
│   └── Streaming processor
└── Infrastructure
    ├── Docker containers
    ├── Nginx proxy
    └── Health monitoring
```

## Performance Targets

- **Preprocessing**: <1ms (Triton fused kernels)
- **Quantization**: <0.5ms (CuPy acceleration)
- **Forward Pass**: <2ms (compiled transformer)
- **Total Latency**: <5ms (real-time capable)

## Configuration

The demo automatically detects available GPU features and enables them when possible:

- **CUDA**: PyTorch GPU support
- **Triton**: Fused preprocessing kernels
- **CuPy**: Compression augmentation
- **Compilation**: torch.compile optimization

## Troubleshooting

### Backend Issues

```bash
# Check server status
./scripts/demo.sh status

# View logs
./scripts/demo.sh logs

# Restart services
./scripts/demo.sh restart
```

### Common Problems

1. **"Backend Offline"**: Start server with `./scripts/demo.sh start` or `./scripts/demo.sh dev`
2. **GPU Features Disabled**: Install GPU libraries: `pip install triton cupy-cuda12x`
3. **CORS Errors**: Use provided demo URLs, not file:// protocol
4. **Port Conflicts**: Stop existing services or change ports in configuration

## Development

### Adding New Features

1. Backend API endpoints: `backend/demo_server.py`
2. Frontend controls: `web/demo.html`
3. GPU kernels: `src/gpu/triton/` or `src/gpu/cupy/`

### Testing

```bash
# Test API health
curl http://localhost:8000/health

# Test inference endpoint
curl -X POST http://localhost:8000/infer_once \
  -H "Content-Type: application/json" \
  -d '{"use_fused_preproc": true, "use_rmsnorm": true}'

# Test demo data
curl http://localhost:8000/demo_data
```

### Performance Profiling

```bash
# Start with profiling
python scripts/launch_demo.py --reload --profile

# View GPU utilization
nvidia-smi -l 1

# Monitor memory usage
htop
```
