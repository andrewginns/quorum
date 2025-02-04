"""
Tests for parallel backend functionality and response aggregation.
"""
import pytest
from grappa import should
import httpx
from .conftest import MOCK_COMPLETION_RESPONSE, MOCK_COMPLETION_RESPONSE_2, MockResponse


@pytest.mark.asyncio
async def test_chat_completion_parallel_backends(test_client_parallel_backends, monkeypatch):
    """Test chat completion with parallel backend calls and response aggregation"""
    async def mock_post(*args, **kwargs):
        url = str(args[1])  # Get the actual URL from args[1], not args[0]
        if "test1.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_RESPONSE)
        elif "test2.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_RESPONSE_2)
        return MockResponse(500, {"error": {"message": "Unknown backend", "type": "backend_error"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Make request to chat completions endpoint
    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify successful response
    response.status_code | should.equal(200)
    data = response.json()

    # Verify response structure
    data | should.have.key("choices")
    data["choices"] | should.have.length(1)
    
    # Verify concatenated content
    expected_content = (
        "Hello there, how may I assist you today?\n-------------\n"
        "I'm the second assistant, ready to help!"
    )
    data["choices"][0]["message"]["content"] | should.equal(expected_content)
    
    # Verify combined usage statistics
    data["usage"] | should.have.keys("prompt_tokens", "completion_tokens", "total_tokens")
    data["usage"]["prompt_tokens"] | should.equal(19)  # 9 + 10
    data["usage"]["completion_tokens"] | should.equal(27)  # 12 + 15
    data["usage"]["total_tokens"] | should.equal(46)  # 21 + 25

    # Verify other response metadata
    data | should.have.key("created")
    data | should.have.key("model")
    data | should.have.key("object")
    data["object"] | should.equal("chat.completion")


@pytest.mark.asyncio
async def test_chat_completion_parallel_backends_partial_failure(test_client_parallel_backends, monkeypatch):
    """Test chat completion with parallel backend calls where one backend fails"""
    async def mock_post(*args, **kwargs):
        url = str(args[1])  # Get the actual URL from args[1], not args[0]
        if "test1.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_RESPONSE)
        elif "test2.example.com" in url:
            return MockResponse(500, {"error": {"message": "Backend error", "type": "backend_error"}})
        return MockResponse(500, {"error": {"message": "Unknown backend", "type": "backend_error"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Make request to chat completions endpoint
    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify successful response (should still work with one backend)
    response.status_code | should.equal(200)
    data = response.json()

    # Verify response has content from successful backend only
    data["choices"][0]["message"]["content"] | should.equal(
        "Hello there, how may I assist you today?"
    )
    
    # Verify usage statistics from successful backend only
    data["usage"]["prompt_tokens"] | should.equal(9)
    data["usage"]["completion_tokens"] | should.equal(12)
    data["usage"]["total_tokens"] | should.equal(21)


@pytest.mark.asyncio
async def test_chat_completion_parallel_backends_all_failure(test_client_parallel_backends, monkeypatch):
    """Test chat completion with parallel backend calls where all backends fail"""
    async def mock_post(*args, **kwargs):
        return MockResponse(500, {"error": {"message": "Backend error", "type": "backend_error"}})
        
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Make request to chat completions endpoint
    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify error response
    response.status_code | should.equal(500)
    error = response.json()["error"]
    error | should.have.keys("message", "type")
    error["type"] | should.equal("proxy_error")
    error["message"] | should.contain("All backends failed") 