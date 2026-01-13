#!/usr/bin/env python3
"""
Test HTTP Server Functionality
Tests that the FastAPI HTTP server starts correctly and responds on port 8000.
"""

import asyncio
import sys
import time
import os
import subprocess
import requests
from pathlib import Path

def test_http_server():
    """Test that the HTTP server starts and responds correctly"""
    print("Testing HTTP Server on port 8000...")
    
    # Ensure data directory exists
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Prepare environment with DATABASE_URL
    env = os.environ.copy()
    env["DATABASE_URL"] = "sqlite+aiosqlite:///data/test.db"
    
    # Start the server process
    server_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=Path(__file__).parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    try:
        # Test 1: Health endpoint
        print("Testing /health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200, f"Health check failed with status {response.status_code}"
        health_data = response.json()
        assert health_data["status"] == "healthy", "Health status is not healthy"
        assert health_data["service"] == "mcp-db-server", "Service name mismatch"
        print("✓ /health endpoint working correctly")
        
        # Test 2: List tables endpoint
        print("Testing /mcp/list_tables endpoint...")
        response = requests.get("http://localhost:8000/mcp/list_tables", timeout=5)
        assert response.status_code == 200, f"List tables failed with status {response.status_code}"
        tables = response.json()
        assert isinstance(tables, list), "List tables should return a list"
        print("✓ /mcp/list_tables endpoint working correctly")
        
        # Test 3: API docs endpoint
        print("Testing /docs endpoint...")
        response = requests.get("http://localhost:8000/docs", timeout=5)
        assert response.status_code == 200, f"Docs failed with status {response.status_code}"
        assert "swagger-ui" in response.text.lower(), "Swagger UI not found in docs"
        print("✓ /docs endpoint working correctly")
        
        print("\n✅ All HTTP server tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
        
    finally:
        # Stop the server
        print("\nStopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop gracefully
            server_process.kill()
            server_process.wait()

if __name__ == "__main__":
    success = test_http_server()
    sys.exit(0 if success else 1)
