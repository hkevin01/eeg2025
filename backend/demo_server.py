# -*- coding: utf-8 -*-
"""
Interactive GPU Demo Server
==========================

FastAPI backend for the EEG GPU demonstration interface.
Wraps GPU-accelerated streaming inference with real-time controls.
"""
from __future__ import annotations
import asyncio
import time
from typing import Dict, Any, Optional, List
import os
import sys

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    # Import GPU components with fallbacks
    from src.gpu.triton.fir_iir_fused import fused_bandpass_notch_car
    from src.gpu.triton.rmsnorm import rmsnorm_time
    from src.gpu.cupy.perceptual_quant import perceptual_quantize_torch
    from scripts.stream_infer_demo import ModelStub
    from scipy.signal import butter, iirnotch, sos2tf
    GPU_AVAILABLE = True
except ImportError as e:
    print(f"GPU components not available: {e}")
    GPU_AVAILABLE = False

app = FastAPI(
    title="EEG GPU Demo Server",
    description="Interactive demonstration of GPU-accelerated EEG processing",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (demo HTML)
web_dir = os.path.join(project_root, "web")
if os.path.exists(web_dir):
    app.mount("/web", StaticFiles(directory=web_dir), name="web")

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def initialize_model():
    """Initialize the model stub for demonstration."""
    global model
    if GPU_AVAILABLE:
        try:
            model = ModelStub(in_ch=128).to(device).eval()
            # Try to compile for optimization
            model = torch.compile(model, mode="max-autotune")
            print(f"Model initialized on {device} with compilation")
        except Exception as e:
            print(f"Model compilation failed, using standard model: {e}")
            model = ModelStub(in_ch=128).to(device).eval()
    else:
        # CPU fallback
        model = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 8)
        ).to(device).eval()

def make_biquad_coeffs(sfreq: float, bp_lo: float, bp_hi: float, notch: float, Q: float = 30.0, device="cuda"):
    """Create biquad filter coefficients for GPU processing."""
    if not GPU_AVAILABLE:
        return None, None, None
    
    try:
        # Design filters
        sos_bp = butter(4, [bp_lo/(sfreq/2), bp_hi/(sfreq/2)], btype="bandpass", output="sos")
        b1, a1 = sos2tf(sos_bp[0:1, :])
        b2, a2 = sos2tf(sos_bp[1:2, :])
        b_nt, a_nt = iirnotch(notch/(sfreq/2), Q=Q)
        
        def pack(b, a):
            b = b / a[0]
            a = a / a[0]
            return torch.tensor([b[0], b[1], b[2], a[1], a[2]], dtype=torch.float32, device=device)
        
        return pack(b1, a1), pack(b2, a2), pack(b_nt, a_nt)
    except Exception as e:
        print(f"Filter coefficient creation failed: {e}")
        return None, None, None

class InferenceConfig(BaseModel):
    """Configuration for inference request."""
    channels: int = 128
    sfreq: int = 500
    window_s: float = 2.0
    stride_s: float = 0.5
    use_fused_preproc: bool = True
    bandpass: List[float] = [0.1, 40.0]
    notch: float = 60.0
    use_rmsnorm: bool = True
    use_perceptual_quant: bool = False
    snr_db: float = 30.0
    dann_subject: bool = True
    dann_site: bool = False
    montage_aug: bool = False
    channel_drop_prob: float = 0.0
    simulate: bool = True

@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup."""
    initialize_model()
    print(f"Demo server started with device: {device}")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "gpu_components": GPU_AVAILABLE,
        "model_ready": model is not None
    }

@app.post("/infer_once")
async def infer_once(cfg: InferenceConfig):
    """
    Run single inference with current configuration.
    
    Returns latency metrics and model outputs for demonstration.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        C = cfg.channels
        T = int(cfg.sfreq * cfg.window_s)
        B = 32
        
        # Generate simulated EEG data
        x = torch.randn(B, C, T, device=device, dtype=torch.float32) * 0.5
        
        # Optional montage/channel dropout simulation
        if cfg.channel_drop_prob > 0:
            mask = (torch.rand(B, C, 1, device=device) > cfg.channel_drop_prob).float()
            x = x * mask
        
        total_start = time.perf_counter()
        
        # GPU Preprocessing with timing
        fused_ms = 0.0
        if cfg.use_fused_preproc and GPU_AVAILABLE:
            bp1, bp2, nt = make_biquad_coeffs(
                cfg.sfreq, cfg.bandpass[0], cfg.bandpass[1], cfg.notch, device=str(device)
            )
            if bp1 is not None:
                preproc_start = time.perf_counter()
                x = fused_bandpass_notch_car(x, bp1, bp2, nt)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fused_ms = (time.perf_counter() - preproc_start) * 1000.0
        
        # RMSNorm timing
        if cfg.use_rmsnorm and GPU_AVAILABLE:
            x = rmsnorm_time(x)
        
        # Perceptual quantization timing
        quant_ms = 0.0
        if cfg.use_perceptual_quant and GPU_AVAILABLE and device.type == "cuda":
            quant_start = time.perf_counter()
            x = perceptual_quantize_torch(x, snr_db=cfg.snr_db)
            torch.cuda.synchronize()
            quant_ms = (time.perf_counter() - quant_start) * 1000.0
        
        # Forward pass timing
        forward_start = time.perf_counter()
        with torch.no_grad():
            if hasattr(torch.cuda, 'amp') and device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    if hasattr(model, '__call__') and 'kv_cache' in model.__code__.co_varnames:
                        out, _ = model(x, kv_cache=None, use_cache=False)
                    else:
                        out = model(x)
            else:
                if hasattr(model, '__call__') and hasattr(model, 'forward'):
                    try:
                        out, _ = model(x, kv_cache=None, use_cache=False)
                    except:
                        out = model(x)
                else:
                    out = model(x)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - forward_start) * 1000.0
        
        total_ms = (time.perf_counter() - total_start) * 1000.0
        
        # Generate demo outputs (simulated EEG analysis results)
        if out.dim() == 2:
            # Response time (normalized and scaled to realistic range)
            rt = (torch.tanh(out[:, 0]).mean().item() + 1.0) * 300.0 + 200.0  # 200-800ms range
            
            # Success probability
            succ = torch.sigmoid(out[:, 1] if out.shape[1] > 1 else out[:, 0]).mean().item()
            
            # CBCL factors (normalized to [-1, 1] range)
            if out.shape[1] >= 4:
                cbcl = torch.tanh(out.mean(dim=0)[:4]).tolist()
            else:
                # Pad with random values for demo
                base_cbcl = torch.tanh(out.mean(dim=0)).tolist()
                cbcl = (base_cbcl + [0.0, 0.0, 0.0, 0.0])[:4]
        else:
            # Fallback for different output shapes
            out_flat = out.flatten()
            rt = (torch.tanh(out_flat[0]).item() + 1.0) * 300.0 + 200.0
            succ = torch.sigmoid(out_flat[1] if len(out_flat) > 1 else out_flat[0]).item()
            cbcl = torch.tanh(out_flat[:4] if len(out_flat) >= 4 else torch.cat([out_flat, torch.zeros(4-len(out_flat))])).tolist()
        
        return {
            "latency_ms": {
                "preproc": fused_ms,
                "quant": quant_ms,
                "forward": fwd_ms,
                "total": total_ms
            },
            "outputs": {
                "response_time_ms": float(rt),
                "success_prob": float(succ),
                "cbcl": {
                    "p_factor": float(cbcl[0]),
                    "internalizing": float(cbcl[1]),
                    "externalizing": float(cbcl[2]),
                    "attention": float(cbcl[3])
                }
            },
            "config": cfg.dict(),
            "gpu_features": {
                "fused_preprocessing": cfg.use_fused_preproc and GPU_AVAILABLE,
                "rmsnorm": cfg.use_rmsnorm and GPU_AVAILABLE,
                "perceptual_quant": cfg.use_perceptual_quant and GPU_AVAILABLE,
                "device": str(device)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/demo_data")
async def get_demo_data():
    """Get sample EEG data for visualization."""
    # Generate sample EEG-like data for frontend visualization
    sfreq = 500
    duration = 2.0
    t = np.linspace(0, duration, int(sfreq * duration))
    
    # Simulate EEG with multiple frequency components
    alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    beta = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
    noise = 0.2 * np.random.randn(len(t))
    
    eeg_sample = alpha + beta + noise
    
    # Simple frequency spectrum (toy implementation)
    freqs = np.fft.fftfreq(len(t), 1/sfreq)[:len(t)//2]
    fft_vals = np.abs(np.fft.fft(eeg_sample))[:len(t)//2]
    
    return {
        "time_series": eeg_sample.tolist()[:512],  # Subsample for demo
        "frequencies": freqs[:128].tolist(),
        "spectrum": fft_vals[:128].tolist(),
        "metadata": {
            "sfreq": sfreq,
            "duration": duration,
            "channels": 128
        }
    }

@app.get("/")
async def root():
    """Root endpoint with demo information."""
    return {
        "message": "EEG GPU Demo Server",
        "demo_url": "/web/demo.html",
        "health": "/health",
        "endpoints": {
            "infer_once": "POST /infer_once",
            "demo_data": "GET /demo_data",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "demo_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
