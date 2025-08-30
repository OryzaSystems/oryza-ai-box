#!/usr/bin/env python3
"""
AI Model Benchmarking Tool
"""

import argparse
import time
import torch
import numpy as np
from typing import Dict, Any

def benchmark_pytorch(device: str = "cpu", quick_test: bool = False) -> Dict[str, Any]:
    """Benchmark PyTorch operations"""
    print(f"🔥 Benchmarking PyTorch on {device}...")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Test matrix multiplication
    size = 1000 if not quick_test else 100
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    start_time = time.time()
    for _ in range(10 if not quick_test else 3):
        z = torch.mm(x, y)
        if device == "cuda":
            torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    ops_per_sec = (10 if not quick_test else 3) / duration
    
    result = {
        "device": device,
        "matrix_size": size,
        "operations": 10 if not quick_test else 3,
        "duration_seconds": duration,
        "ops_per_second": ops_per_sec,
        "status": "✅ PASS"
    }
    
    print(f"   Matrix multiplication ({size}x{size}): {ops_per_sec:.2f} ops/sec")
    return result

def benchmark_memory(device: str = "cpu") -> Dict[str, Any]:
    """Benchmark memory operations"""
    print(f"💾 Benchmarking memory on {device}...")
    
    if device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        
        result = {
            "device": device,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "memory_total_mb": memory_total,
            "memory_utilization_percent": (memory_allocated / memory_total) * 100,
            "status": "✅ PASS"
        }
        
        print(f"   GPU Memory: {memory_allocated:.1f}MB / {memory_total:.1f}MB ({result['memory_utilization_percent']:.1f}%)")
    else:
        import psutil
        memory = psutil.virtual_memory()
        
        result = {
            "device": "cpu",
            "memory_total_mb": memory.total / 1024**2,
            "memory_available_mb": memory.available / 1024**2,
            "memory_utilization_percent": memory.percent,
            "status": "✅ PASS"
        }
        
        print(f"   System Memory: {memory.used / 1024**2:.1f}MB / {memory.total / 1024**2:.1f}MB ({memory.percent:.1f}%)")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="AI Model Benchmarking Tool")
    parser.add_argument("--platform", choices=["cpu", "cuda", "auto"], default="auto",
                       help="Platform to benchmark")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick tests only")
    
    args = parser.parse_args()
    
    print("🚀 AI Box Model Benchmarking")
    print("=" * 40)
    
    # Determine device
    if args.platform == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.platform
    
    print(f"Target device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("-" * 40)
    
    # Run benchmarks
    results = {}
    
    try:
        results["pytorch"] = benchmark_pytorch(device, args.quick_test)
        results["memory"] = benchmark_memory(device)
        
        print("-" * 40)
        print("📊 Benchmark Summary:")
        print(f"   PyTorch Operations: {results['pytorch']['ops_per_second']:.2f} ops/sec")
        
        if device == "cuda":
            print(f"   GPU Memory Usage: {results['memory']['memory_utilization_percent']:.1f}%")
        else:
            print(f"   System Memory Usage: {results['memory']['memory_utilization_percent']:.1f}%")
        
        print("✅ All benchmarks completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
