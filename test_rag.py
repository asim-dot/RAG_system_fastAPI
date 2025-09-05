# test_rag.py - Simple tests for your RAG
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_root():
    """Test if API is running"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "RAG API is running"

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_invalid_file():
    """Test uploading non-PDF"""
    response = client.post("/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    # Should handle gracefully
    assert response.status_code in [400, 422]

# Run with: pytest test_rag.py