"""
Basic API Tests for Pix2Pix Service

Run with: pytest tests/test_api.py -v
"""

import requests
import base64
import io
import os
from PIL import Image
import pytest

# Configuration - can be overridden with environment variables
BASE_URL = os.environ.get('TEST_BASE_URL', 'http://localhost:8000')
TIMEOUT = int(os.environ.get('TEST_TIMEOUT', '60'))

print(f"Testing server: {BASE_URL}")
print(f"Request timeout: {TIMEOUT}s")


def create_test_image():
    """Create a test satellite-like image."""
    img = Image.new('RGB', (512, 512), color=(100, 150, 100))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def test_health_endpoint():
    """Test GET /health returns service status."""
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    print(f"✓ Health check passed: {data['status']}")


def test_root_endpoint():
    """Test GET / returns service info."""
    response = requests.get(f"{BASE_URL}/", timeout=10)
    assert response.status_code == 200
    
    data = response.json()
    assert "service" in data
    assert "endpoints" in data
    print(f"✓ Root endpoint passed: {data['service']}")


def test_generate_endpoint():
    """Test POST /generate creates map."""
    image_bytes = create_test_image()
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    
    response = requests.post(
        f"{BASE_URL}/generate",
        files=files,
        timeout=TIMEOUT
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "generated_image" in data
    print(f"✓ Generate endpoint passed")


def test_generate_enhanced_endpoint():
    """Test POST /generate-enhanced with Rekognition."""
    # Check if Rekognition is enabled first
    health = requests.get(f"{BASE_URL}/health").json()
    
    if not health.get("rekognition_enabled"):
        pytest.skip("Rekognition not enabled - skipping enhanced endpoint test")
    
    image_bytes = create_test_image()
    files = {'image': ('test.jpg', image_bytes, 'image/jpeg')}
    
    response = requests.post(
        f"{BASE_URL}/generate-enhanced",
        files=files,
        timeout=TIMEOUT
    )
    
    # If we get a 500 error, check if it's AWS credentials (expected)
    if response.status_code == 500:
        error_data = response.json()
        error_msg = error_data.get("message", "")
        
        # List of AWS-related error keywords
        aws_error_keywords = [
            "credentials", "rekognition", "authorization", 
            "access denied", "unable to locate", "signature",
            "aws", "iam", "authentication"
        ]
        
        # Check if error is AWS-related (skip test, not a failure)
        is_aws_error = any(keyword in error_msg.lower() for keyword in aws_error_keywords)
        
        if is_aws_error:
            pytest.skip(f"AWS credentials not configured (expected): {error_msg}")
        else:
            # Not an AWS error - this is a real failure
            pytest.fail(f"Unexpected server error: {error_msg}")
    
    # If we get here, response should be 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data["status"] in ["success", "filtered"]
    
    if data["status"] == "success":
        assert "terrain_type" in data
        assert "rekognition_analysis" in data
        print(f"✓ Enhanced endpoint passed: Terrain={data['terrain_type']}")
    else:
        print(f"✓ Enhanced endpoint passed: Image filtered - {data['reason']}")


if __name__ == "__main__":
    print("Run with: pytest tests/test_api.py -v")