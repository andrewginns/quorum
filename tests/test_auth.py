"""
Tests for authentication and authorization functionality.
"""

from grappa import should
import os
import pytest


def test_chat_completion_no_auth(test_client_blank_model, monkeypatch):
    """Test chat completion endpoint without auth header"""
    # Ensure OPENAI_API_KEY is not set in the environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = test_client_blank_model.post(
        "/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]},
    )

    response.status_code | should.equal(401)
    error = response.json()["error"]
    error | should.have.keys("message", "type")
    error["type"] | should.equal("auth_error")
    error["message"] | should.equal(
        "Authorization header is required and OPENAI_API_KEY environment variable is not set"
    )


def test_chat_completion_env_var_fallback(test_client_blank_model, monkeypatch):
    """Test chat completion endpoint with env var but no auth header"""
    # Set OPENAI_API_KEY in the environment
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-from-env")

    # Setup mock
    import quorum.oai_proxy
    from unittest.mock import patch
    import json

    # Store the auth header for verification
    captured_headers = []

    # Define a mock for httpx.AsyncClient.post
    async def mock_post(*args, **kwargs):
        # Capture the headers for verification
        headers = kwargs.get("headers", {})
        captured_headers.append(headers)

        # Return a successful response
        from tests.conftest import MockResponse

        return MockResponse(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from the mock!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
            headers={"content-type": "application/json"},
        )

    # Apply the mock
    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        # Make the request without an auth header
        response = test_client_blank_model.post(
            "/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        # Verify the request was successful
        response.status_code | should.equal(200)

        # Verify the response contains the expected content
        result = response.json()
        result["choices"][0]["message"]["content"] | should.equal(
            "Hello from the mock!"
        )

        # Verify the API key from env var was passed to the backend
        assert len(captured_headers) > 0, "No headers were captured"
        assert any(
            "authorization" in header.lower() for header in captured_headers[0]
        ), "No authorization header found"
        for key, value in captured_headers[0].items():
            if key.lower() == "authorization":
                value | should.equal("Bearer test-api-key-from-env")
