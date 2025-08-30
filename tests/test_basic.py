"""
Basic tests for Oryza AI Box
"""

import pytest
import torch
import cv2
import numpy as np
from fastapi.testclient import TestClient

def test_torch_installation():
    """Test PyTorch installation"""
    assert torch.__version__ is not None
    # Test basic tensor operations
    x = torch.randn(5, 3)
    y = torch.randn(3, 4)
    z = torch.mm(x, y)
    assert z.shape == (5, 4)

def test_opencv_installation():
    """Test OpenCV installation"""
    assert cv2.__version__ is not None
    # Test basic image operations
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 100)

def test_api_gateway():
    """Test API Gateway"""
    from services.api_gateway.main import app
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200
    assert "Oryza AI Box API Gateway" in response.json()["message"]
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_server():
    """Test Model Server"""
    from services.model_server.main import app
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200
    assert "Model Server" in response.json()["message"]
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_data_manager():
    """Test Data Manager"""
    from services.data_manager.main import app
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200
    assert "Data Manager" in response.json()["message"]
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_package_import():
    """Test package import"""
    import oryza_ai_box
    assert oryza_ai_box.__version__ == "1.0.0"
