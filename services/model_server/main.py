"""
Model Server Main Application
"""

from fastapi import FastAPI
import torch

app = FastAPI(
    title="Oryza AI Box Model Server",
    description="AI Model Inference Server",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Oryza AI Box Model Server", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "model-server",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
