"""
Tests for authentication and authorization functionality.
"""
from grappa import should


def test_chat_completion_no_auth(test_client_blank_model):
    """Test chat completion endpoint without auth header"""
    response = test_client_blank_model.post(
        "/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]},
    )

    response.status_code | should.equal(401)
    error = response.json()["error"]
    error | should.have.keys("message", "type")
    error["type"] | should.equal("auth_error")
    error["message"] | should.equal("Authorization header is required") 