import pytest
from fastapi.testclient import TestClient
import json
import httpx
import importlib
from pathlib import Path
from grappa import should

# Mock response payloads
MOCK_COMPLETION_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello there, how may I assist you today?",
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}

MOCK_STREAMING_CHUNKS = [
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
        ],
    },
]

# Mock configurations for different test scenarios
MOCK_CONFIG_BLANK_MODEL = {
    "primary_backends": [
        {"name": "LLM1", "url": "http://test.example.com/v1", "model": ""}
    ],
    "settings": {"timeout": 30},
}

MOCK_CONFIG_WITH_MODEL = {
    "primary_backends": [
        {"name": "LLM1", "url": "http://test.example.com/v1", "model": "gpt-4-test"}
    ],
    "settings": {"timeout": 30},
}


@pytest.fixture
def mock_config_blank_model(monkeypatch):
    """Mock config file with blank model"""

    def mock_read_text(*args, **kwargs):
        return json.dumps(MOCK_CONFIG_BLANK_MODEL)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import deliberato.oai_proxy

    importlib.reload(deliberato.oai_proxy)
    return MOCK_CONFIG_BLANK_MODEL


@pytest.fixture
def mock_config_with_model(monkeypatch):
    """Mock config file with model set"""

    def mock_read_text(*args, **kwargs):
        return json.dumps(MOCK_CONFIG_WITH_MODEL)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import deliberato.oai_proxy

    importlib.reload(deliberato.oai_proxy)
    return MOCK_CONFIG_WITH_MODEL


@pytest.fixture
def test_client_blank_model(mock_config_blank_model):
    """Create a test client with blank model config"""
    from deliberato.oai_proxy import app

    return TestClient(app)


@pytest.fixture
def test_client_with_model(mock_config_with_model):
    """Create a test client with model set in config"""
    from deliberato.oai_proxy import app

    return TestClient(app)


def test_load_config_blank_model(mock_config_blank_model):
    """Test configuration loading with blank model"""
    from deliberato.oai_proxy import load_config

    config = load_config()

    # Verify the exact structure and values from mock config
    config | should.have.length(2)  # primary_backends and settings only
    config["settings"]["timeout"] | should.equal(30)  # specific timeout value
    config["primary_backends"] | should.have.length(1)
    backend = config["primary_backends"][0]
    backend | should.have.keys("name", "url", "model")
    backend["name"] | should.equal("LLM1")
    backend["url"] | should.equal("http://test.example.com/v1")
    backend["model"] | should.equal("")


def test_load_config_with_model(mock_config_with_model):
    """Test configuration loading with model set"""
    from deliberato.oai_proxy import load_config

    config = load_config()

    # Verify the exact structure and values from mock config
    config | should.have.length(2)  # primary_backends and settings only
    config["settings"]["timeout"] | should.equal(30)  # specific timeout value
    config["primary_backends"] | should.have.length(1)
    backend = config["primary_backends"][0]
    backend | should.have.keys("name", "url", "model")
    backend["name"] | should.equal("LLM1")
    backend["url"] | should.equal("http://test.example.com/v1")
    backend["model"] | should.equal("gpt-4-test")


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
        "Model must be specified in request when config.json model is blank"
    )


@pytest.mark.asyncio
async def test_chat_completion_with_model_override(test_client_with_model, monkeypatch):
    """Test that config model overrides request model when config model is set"""

    async def mock_post(*args, **kwargs):
        # Verify request details
        request_body = json.loads(kwargs["content"])
        request_body | should.have.keys("model", "messages")
        request_body["model"] | should.equal("gpt-4-test")  # Should use config model
        request_body["messages"] | should.have.length(1)
        request_body["messages"][0] | should.equal(
            {"role": "user", "content": "Hello!"}
        )

        return type(
            "MockResponse",
            (),
            {
                "status_code": 200,
                "content": json.dumps(MOCK_COMPLETION_RESPONSE).encode(),
                "headers": {"content-type": "application/json"},
                "json": lambda self: MOCK_COMPLETION_RESPONSE,
            },
        )()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_with_model.post(
        "/chat/completions",
        json={
            "model": "gpt-4",  # This should be overridden by config
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

        return type(
            "MockResponse",
            (),
            {
                "status_code": 200,
                "content": json.dumps(MOCK_COMPLETION_RESPONSE).encode(),
                "headers": {"content-type": "application/json"},
                "json": lambda self: MOCK_COMPLETION_RESPONSE,
            },
        )()

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
async def test_chat_completion_streaming(test_client_blank_model, monkeypatch):
    """Test streaming chat completion request with chunk validation"""

    async def mock_aiter_bytes():
        for chunk in MOCK_STREAMING_CHUNKS:
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    class MockStreamingResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            async for chunk in mock_aiter_bytes():
                yield chunk

    async def mock_post(*args, **kwargs):
        # Verify streaming flag is set
        request_body = json.loads(kwargs["content"])
        request_body | should.have.key("stream")
        request_body["stream"] | should.be.true

        return MockStreamingResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_blank_model.post(
        "/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify streaming response format
    response.status_code | should.equal(200)
    response.headers["content-type"] | should.equal("text/event-stream")

    # Get all chunks and filter out empty lines
    chunks = [c for c in response.iter_lines() if c.strip()]

    # Verify we have the expected number of chunks
    chunks | should.have.length(4)  # 3 content chunks + [DONE]

    # Verify first chunk format (role)
    first_chunk = json.loads(chunks[0].replace("data: ", ""))
    first_chunk | should.have.keys("id", "object", "created", "model", "choices")
    first_chunk["object"] | should.equal("chat.completion.chunk")
    first_chunk["choices"][0]["delta"] | should.have.key("role")
    first_chunk["choices"][0]["delta"]["role"] | should.equal("assistant")

    # Verify content chunk format
    content_chunk = json.loads(chunks[1].replace("data: ", ""))
    content_chunk["choices"][0]["delta"] | should.have.key("content")

    # Verify final chunk format
    final_chunk = json.loads(chunks[2].replace("data: ", ""))
    final_chunk["choices"][0]["finish_reason"] | should.equal("stop")

    # Verify [DONE] marker
    chunks[3] | should.equal("data: [DONE]")


def test_health_check(test_client_blank_model):
    """Test health check endpoint"""
    response = test_client_blank_model.get("/health")
    response.status_code | should.equal(200)
    response.headers["content-type"] | should.equal("application/json")
    response.json() | should.equal({"status": "healthy"})
