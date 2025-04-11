"""
Tests for basic chat completion functionality (non-streaming, single backend).
"""

import json
import pytest
from grappa import should
import httpx
import importlib
from fastapi.testclient import TestClient
from .conftest import MOCK_COMPLETION_RESPONSE, MockResponse


@pytest.mark.asyncio
async def test_chat_completion_no_model_error(test_client_blank_model):
    """Test chat completion request without model when config model is blank"""
    response = test_client_blank_model.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello!"}]},
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify error response
    response.status_code | should.equal(400)
    error = response.json()["error"]
    error | should.have.keys("message", "type")
    error["type"] | should.equal("invalid_request_error")
    error["message"] | should.equal(
        "Model must be specified when config.yaml model is blank"
    )


@pytest.mark.asyncio
async def test_chat_completion_with_model_override(test_client_with_model, monkeypatch):
    """Test that config model always overrides request model when config model is set"""

    async def mock_post(*args, **kwargs):
        # Verify request details
        request_body = json.loads(kwargs["content"])
        request_body | should.have.keys("model", "messages")
        request_body["model"] | should.equal(
            "gpt-4-test"
        )  # Should always use config model
        request_body["messages"] | should.have.length(1)
        request_body["messages"][0] | should.equal(
            {"role": "user", "content": "Hello!"}
        )

        return MockResponse(
            status_code=200,
            content=MOCK_COMPLETION_RESPONSE,
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Test with model in request - should still use config model
    response = test_client_with_model.post(
        "/chat/completions",
        json={
            "model": "gpt-4",  # This should be ignored in favor of config model
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify response
    response.status_code | should.equal(200)
    response.headers["content-type"] | should.equal("application/json")
    response_data = response.json()
    response_data | should.have.keys(
        "id", "object", "created", "model", "choices", "usage"
    )
    response_data["object"] | should.equal("chat.completion")

    # Test without model in request - should use config model
    response = test_client_with_model.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify response
    response.status_code | should.equal(200)
    response_data = response.json()
    response_data | should.have.keys(
        "id", "object", "created", "model", "choices", "usage"
    )


@pytest.mark.asyncio
async def test_chat_completion_with_request_model(test_client_blank_model, monkeypatch):
    """Test that request model is used when config model is blank"""

    async def mock_post(*args, **kwargs):
        # Verify request details
        request_body = json.loads(kwargs["content"])
        request_body | should.have.keys("model", "messages")
        request_body["model"] | should.equal("gpt-4")  # Should use request model
        request_body["messages"] | should.have.length(1)
        request_body["messages"][0] | should.equal(
            {"role": "user", "content": "Hello!"}
        )

        return MockResponse(
            status_code=200,
            content=MOCK_COMPLETION_RESPONSE,
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_blank_model.post(
        "/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify response
    response.status_code | should.equal(200)
    response.headers["content-type"] | should.equal("application/json")
    response_data = response.json()
    response_data | should.have.keys(
        "id", "object", "created", "model", "choices", "usage"
    )
    response_data["object"] | should.equal("chat.completion")


@pytest.mark.asyncio
async def test_chat_completion_content_length(test_client_blank_model, monkeypatch):
    """Test chat completion with content length verification"""

    async def mock_post(*args, **kwargs):
        # Verify the content matches content-length
        content = kwargs.get("content")
        headers = kwargs.get("headers", {})
        content_length = headers.get("content-length")

        if content_length:
            actual_length = (
                len(content)
                if isinstance(content, bytes)
                else len(str(content).encode())
            )
            actual_length | should.equal(int(content_length))

        return MockResponse(200, MOCK_COMPLETION_RESPONSE)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Test with a specific model and message
    body = {
        "model": "deepseek-r1:1.5bt",
        "messages": [{"role": "user", "content": "what AI are you"}],
    }

    # Calculate actual content length
    body_bytes = json.dumps(body).encode()
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer test-key",
        "content-length": str(len(body_bytes)),
    }

    response = test_client_blank_model.post(
        "/chat/completions", json=body, headers=headers
    )

    # Verify response
    response.status_code | should.equal(200)
    response_data = response.json()
    response_data | should.have.key("choices")


@pytest.mark.asyncio
async def test_direct_httpx_request(test_client_blank_model, monkeypatch):
    """Test direct httpx client request with proper content length handling"""

    async def mock_post(*args, **kwargs):
        # Verify content and headers are properly set
        content = kwargs.get("content")
        headers = kwargs.get("headers", {})

        # Content should be bytes
        content | should.be.a(bytes)

        # Content-Length should match actual content length
        content_length = headers.get("content-length")
        content_length | should.not_be.none
        len(content) | should.equal(int(content_length))

        # Verify the body can be parsed as JSON
        body = json.loads(content)
        body | should.have.keys("model", "messages")

        return MockResponse(200, MOCK_COMPLETION_RESPONSE)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Prepare request body and headers
    body = {
        "model": "deepseek-r1:1.5bt",
        "messages": [{"role": "user", "content": "what AI are you"}],
    }
    body_bytes = json.dumps(body).encode()

    headers = {
        "user-agent": "curl/8.7.1",
        "accept": "*/*",
        "content-type": "application/json",
        "authorization": "Bearer sk-0121bc7116008c1860c8d31a3923bf4117550132d8",
        "content-length": str(len(body_bytes)),
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/v1/chat/completions",
            content=body_bytes,
            headers=headers,
        )

    # Verify response
    response.status_code | should.equal(200)
    response_data = response.json()
    response_data | should.have.key("choices")


@pytest.mark.asyncio
async def test_default_config_fallback(monkeypatch, test_client_blank_model):
    """Test fallback to default config when config.yaml fails to load"""

    async def mock_post(self, url, **kwargs):
        # Verify request uses test URL from mock config
        url | should.contain("test.example.com")
        return MockResponse(200, MOCK_COMPLETION_RESPONSE)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_blank_model.post(
        "/chat/completions",
        json={
            "model": "gpt-4",  # Model required since default config has blank model
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    response.status_code | should.equal(200)


@pytest.mark.asyncio
async def test_multiple_backends_config(monkeypatch):
    """Test handling of multiple backends in config"""

    called_urls = []

    async def mock_post(self, url, **kwargs):
        called_urls.append(url)
        return MockResponse(200, MOCK_COMPLETION_RESPONSE)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Mock the config.yaml read to return our test config
    def mock_read_text(*args, **kwargs):
        return json.dumps(
            {
                "primary_backends": [
                    {"name": "backend1", "url": "http://backend1.test", "model": ""},
                    {"name": "backend2", "url": "http://backend2.test", "model": ""},
                ],
                "settings": {"timeout": 60},
            }
        )

    monkeypatch.setattr("pathlib.Path.read_text", mock_read_text)

    # Reload the module to use new config
    import quorum.oai_proxy

    importlib.reload(quorum.oai_proxy)

    # Create a new test client with the reloaded module
    test_client = TestClient(quorum.oai_proxy.app)

    response = test_client.post(
        "/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    response.status_code | should.equal(200)
    # Both backends are called in parallel for non-streaming requests
    len(called_urls) | should.equal(2)
    called_urls | should.contain("http://backend1.test/chat/completions")
    called_urls | should.contain("http://backend2.test/chat/completions")


@pytest.mark.asyncio
async def test_config_timeout_setting(monkeypatch, test_client_blank_model):
    """Test timeout setting from config is respected"""

    async def mock_post(*args, **kwargs):
        # Verify timeout from config is used
        kwargs["timeout"] | should.equal(30.0)
        return MockResponse(200, MOCK_COMPLETION_RESPONSE)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    # Override config to set custom timeout
    test_client_blank_model.app.state.config = {
        "primary_backends": [
            {"name": "default", "url": "https://api.openai.com/v1", "model": ""}
        ],
        "settings": {"timeout": 30},
    }

    response = test_client_blank_model.post(
        "/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    response.status_code | should.equal(200)
