"""
Tests for health check endpoint.
"""
from grappa import should


def test_health_check(test_client_blank_model):
    """Test health check endpoint"""
    response = test_client_blank_model.get("/health")
    response.status_code | should.equal(200)
    response.headers["content-type"] | should.equal("application/json")
    response.json() | should.equal({"status": "healthy"}) 