"""
Data Manager Main Application
"""

from fastapi import FastAPI

app = FastAPI(
    title="Oryza AI Box Data Manager",
    description="Data Management and Analytics Service",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Oryza AI Box Data Manager", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "data-manager"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
