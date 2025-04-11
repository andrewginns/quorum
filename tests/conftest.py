import pytest
from fastapi.testclient import TestClient
import json
import yaml
import importlib
from pathlib import Path

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

MOCK_COMPLETION_RESPONSE_2 = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I'm the second assistant, ready to help!",
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
}

MOCK_STREAMING_CHUNKS = [
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "parallel-proxy",
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
        "model": "parallel-proxy",
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
        "model": "parallel-proxy",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
        ],
    },
]

# Mock configurations
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

MOCK_CONFIG_MULTIPLE_BACKENDS = {
    "primary_backends": [
        {"name": "LLM1", "url": "http://test1.example.com/v1", "model": "gpt-4-1"},
        {"name": "LLM2", "url": "http://test2.example.com/v1", "model": "gpt-4-2"},
        {"name": "LLM3", "url": "http://test3.example.com/v1", "model": "gpt-4-3"},
    ],
    "settings": {"timeout": 30},
}

MOCK_CONFIG_PARALLEL_BACKENDS = {
    "primary_backends": [
        {"name": "LLM1", "url": "http://test1.example.com/v1", "model": "gpt-4-1"},
        {"name": "LLM2", "url": "http://test2.example.com/v1", "model": "gpt-4-2"},
    ],
    "iterations": {"aggregation": {"strategy": "concatenate"}},
    "strategy": {
        "concatenate": {
            "separator": "\n-------------\n",
            "hide_intermediate_think": True,
            "hide_final_think": False,
            "thinking_tags": ["think", "reason", "reasoning", "thought", "Thought"],
            "skip_final_aggregation": False,
        }
    },
    "settings": {"timeout": 30},
}

MOCK_CONFIG_SOME_INVALID_BACKENDS = {
    "primary_backends": [
        {"name": "LLM1", "url": "http://test1.example.com/v1", "model": "gpt-4-1"},
        {"name": "LLM2", "url": "", "model": "gpt-4-2"},  # Invalid backend
        {"name": "LLM3", "url": "http://test3.example.com/v1", "model": "gpt-4-3"},
    ],
    "settings": {"timeout": 30},
}

MOCK_COMPLETION_WITH_THINKING = {
    "id": "chatcmpl-789",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>Let me think about this... 2+2 is a basic arithmetic operation</think>The answer is 4.",
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 20, "total_tokens": 28},
}

MOCK_COMPLETION_WITH_MULTIPLE_THINKING = {
    "id": "chatcmpl-101112",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>First, let me consider this carefully</think>The answer is 4.<reason>This is because 2+2 has always equaled 4 in base-10 arithmetic</reason>",
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 25, "total_tokens": 33},
}


class MockResponse:
    """Base mock response class with proper async methods"""

    def __init__(self, status_code, content=None, headers=None):
        self.status_code = status_code
        self._content = content if content is not None else b""
        self.headers = headers or {"content-type": "application/json"}

    async def aread(self):
        if isinstance(self._content, (dict, list)):
            return json.dumps(self._content).encode()
        return (
            self._content
            if isinstance(self._content, bytes)
            else str(self._content).encode()
        )

    def json(self):
        """Synchronous json method to match httpx.Response behavior"""
        if isinstance(self._content, (dict, list)):
            return self._content
        content_str = (
            self._content.decode()
            if isinstance(self._content, bytes)
            else self._content
        )
        return json.loads(content_str)


class MockStreamingResponse:
    """Mock response for streaming requests"""

    def __init__(self):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self._chunks = MOCK_STREAMING_CHUNKS

    async def aread(self):
        # For non-streaming access to the content
        return b"".join(
            f"data: {json.dumps(chunk)}\n\n".encode() for chunk in self._chunks
        )

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        
    def __aiter__(self):
        self._iter_index = 0
        return self
        
    async def __anext__(self):
        if self._iter_index < len(self._chunks):
            chunk = self._chunks[self._iter_index]
            self._iter_index += 1
            return f"data: {json.dumps(chunk)}\n\n".encode()
        elif self._iter_index == len(self._chunks):
            self._iter_index += 1
            return b"data: [DONE]\n\n"
        else:
            raise StopAsyncIteration()

    def json(self):
        """Return the first chunk as a dict for non-streaming access"""
        return self._chunks[0] if self._chunks else {"stream": True}


# Shared fixtures
@pytest.fixture
def mock_config_blank_model(monkeypatch):
    """Mock config file with blank model"""

    def mock_read_text(*args, **kwargs):
        return yaml.dump(MOCK_CONFIG_BLANK_MODEL)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import quorum.oai_proxy

    importlib.reload(quorum.oai_proxy)
    return MOCK_CONFIG_BLANK_MODEL


@pytest.fixture
def mock_config_with_model(monkeypatch):
    """Mock config file with model set"""

    def mock_read_text(*args, **kwargs):
        return yaml.dump(MOCK_CONFIG_WITH_MODEL)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import quorum.oai_proxy

    importlib.reload(quorum.oai_proxy)
    return MOCK_CONFIG_WITH_MODEL


@pytest.fixture
def mock_config_multiple_backends(monkeypatch):
    """Mock config file with multiple backends"""

    def mock_read_text(*args, **kwargs):
        return yaml.dump(MOCK_CONFIG_MULTIPLE_BACKENDS)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import quorum.oai_proxy

    importlib.reload(quorum.oai_proxy)
    return MOCK_CONFIG_MULTIPLE_BACKENDS


@pytest.fixture
def mock_config_some_invalid_backends(monkeypatch):
    """Mock config file with some invalid backends"""

    def mock_read_text(*args, **kwargs):
        return yaml.dump(MOCK_CONFIG_SOME_INVALID_BACKENDS)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import quorum.oai_proxy

    importlib.reload(quorum.oai_proxy)
    return MOCK_CONFIG_SOME_INVALID_BACKENDS


@pytest.fixture
def mock_config_parallel_backends(monkeypatch):
    """Mock config file with parallel backends and aggregation settings"""

    def mock_read_text(*args, **kwargs):
        return yaml.dump(MOCK_CONFIG_PARALLEL_BACKENDS)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import quorum.oai_proxy

    importlib.reload(quorum.oai_proxy)
    return MOCK_CONFIG_PARALLEL_BACKENDS


@pytest.fixture
def test_client_blank_model(mock_config_blank_model):
    """Create a test client with blank model config"""
    from quorum.oai_proxy import app

    return TestClient(app)


@pytest.fixture
def test_client_with_model(mock_config_with_model):
    """Create a test client with model set in config"""
    from quorum.oai_proxy import app

    return TestClient(app)


@pytest.fixture
def test_client_multiple_backends(mock_config_multiple_backends):
    """Create a test client with multiple backends config"""
    from quorum.oai_proxy import app

    return TestClient(app)


@pytest.fixture
def test_client_some_invalid_backends(mock_config_some_invalid_backends):
    """Create a test client with some invalid backends config"""
    from quorum.oai_proxy import app

    return TestClient(app)


@pytest.fixture
def test_client_parallel_backends(mock_config_parallel_backends):
    """Create a test client with parallel backends config"""
    from quorum.oai_proxy import app

    return TestClient(app)
