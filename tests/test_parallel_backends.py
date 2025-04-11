"""
Tests for parallel backend functionality and response aggregation.
"""

import pytest
from grappa import should
import httpx
import json
from .conftest import (
    MOCK_COMPLETION_RESPONSE,
    MOCK_COMPLETION_RESPONSE_2,
    MOCK_COMPLETION_WITH_THINKING,
    MOCK_COMPLETION_WITH_MULTIPLE_THINKING,
    MockResponse,
)


@pytest.mark.asyncio
async def test_chat_completion_parallel_backends(
    test_client_parallel_backends, monkeypatch
):
    """Test chat completion with parallel backend calls and response aggregation"""

    async def mock_post(*args, **kwargs):
        url = str(args[1])  # Get the actual URL from args[1], not args[0]
        if "test1.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_RESPONSE)
        elif "test2.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_RESPONSE_2)
        return MockResponse(
            500, {"error": {"message": "Unknown backend", "type": "backend_error"}}
        )

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
    data["usage"] | should.have.keys(
        "prompt_tokens", "completion_tokens", "total_tokens"
    )
    data["usage"]["prompt_tokens"] | should.equal(19)  # 9 + 10
    data["usage"]["completion_tokens"] | should.equal(27)  # 12 + 15
    data["usage"]["total_tokens"] | should.equal(46)  # 21 + 25

    # Verify other response metadata
    data | should.have.key("created")
    data | should.have.key("model")
    data | should.have.key("object")
    data["object"] | should.equal("chat.completion")


@pytest.mark.asyncio
async def test_chat_completion_parallel_backends_partial_failure(
    test_client_parallel_backends, monkeypatch
):
    """Test chat completion with parallel backend calls where one backend fails"""

    async def mock_post(*args, **kwargs):
        url = str(args[1])  # Get the actual URL from args[1], not args[0]
        if "test1.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_RESPONSE)
        elif "test2.example.com" in url:
            return MockResponse(
                500, {"error": {"message": "Backend error", "type": "backend_error"}}
            )
        return MockResponse(
            500, {"error": {"message": "Unknown backend", "type": "backend_error"}}
        )

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
async def test_chat_completion_parallel_backends_all_failure(
    test_client_parallel_backends, monkeypatch
):
    """Test chat completion with parallel backend calls where all backends fail"""

    async def mock_post(*args, **kwargs):
        return MockResponse(
            500, {"error": {"message": "Backend error", "type": "backend_error"}}
        )

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


@pytest.mark.asyncio
async def test_strip_thinking_tags_non_streaming(
    test_client_parallel_backends, monkeypatch
):
    """Test that thinking tags and their content are stripped in non-streaming mode when configured"""

    async def mock_post(*args, **kwargs):
        url = str(args[1])
        if "test1.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_WITH_THINKING)
        elif "test2.example.com" in url:
            return MockResponse(200, MOCK_COMPLETION_WITH_MULTIPLE_THINKING)
        return MockResponse(500)

    # Create a mock config with tag stripping enabled
    mock_config = {
        "primary_backends": [
            {"name": "LLM1", "url": "http://test1.example.com/v1", "model": "gpt-4-1"},
            {"name": "LLM2", "url": "http://test2.example.com/v1", "model": "gpt-4-2"},
        ],
        "iterations": {"aggregation": {"strategy": "concatenate"}},
        "strategy": {
            "concatenate": {
                "separator": "\n-------------\n",
                "hide_intermediate_think": True,
                "hide_final_think": True,
                "thinking_tags": ["think", "reason", "reasoning", "thought"],
                "skip_final_aggregation": False,
            }
        },
        "settings": {"timeout": 30},
    }

    # Patch the config
    monkeypatch.setattr("quorum.oai_proxy.config", mock_config)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "What is 2+2?"}]},
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify successful response
    response.status_code | should.equal(200)
    data = response.json()

    # Verify content has thinking tags and their content stripped
    content = data["choices"][0]["message"]["content"]
    content | should.do_not.contain("<think>")
    content | should.do_not.contain("</think>")
    content | should.do_not.contain("<reason>")
    content | should.do_not.contain("</reason>")
    content | should.do_not.contain(
        "Let me think about this"
    )  # Thinking content should be stripped
    content | should.do_not.contain(
        "This is because 2+2 has always equaled 4"
    )  # Reasoning content should be stripped
    content | should.contain("The answer is 4")  # Main content preserved
    content | should.do_not.contain(
        "Let me explain why"
    )  # This was part of thinking content


@pytest.mark.asyncio
async def test_strip_thinking_tags_streaming(
    test_client_parallel_backends, monkeypatch
):
    """Test that thinking tags and their content are stripped in streaming mode when configured"""

    class MockStreamingThinkingResponse:
        def __init__(self, content_type="think"):
            self.status_code = 200
            self.headers = {"content-type": "text/event-stream"}
            self._content_type = content_type

        async def aiter_bytes(self):
            # Initial role event
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'

            # Thinking block - send it in smaller chunks to test streaming behavior
            yield f'data: {{"choices":[{{"delta":{{"content":"<{self._content_type}>"}}}}]}}\n\n'.encode()
            yield b'data: {"choices":[{"delta":{"content":"Let me think about this..."}}]}\n\n'
            yield f'data: {{"choices":[{{"delta":{{"content":"</{self._content_type}>"}}}}]}}\n\n'.encode()

            # Actual answer - send in chunks
            yield b'data: {"choices":[{"delta":{"content":"The answer "}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"is "}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"4"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"."}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

    async def mock_post(*args, **kwargs):
        url = str(args[1])
        if "test1.example.com" in url:
            return MockStreamingThinkingResponse("think")
        elif "test2.example.com" in url:
            return MockStreamingThinkingResponse("reason")
        return MockResponse(500)

    # Ensure config has tag stripping enabled
    mock_config = {
        "primary_backends": [
            {"name": "LLM1", "url": "http://test1.example.com/v1", "model": "gpt-4-1"},
            {"name": "LLM2", "url": "http://test2.example.com/v1", "model": "gpt-4-2"},
        ],
        "iterations": {"aggregation": {"strategy": "concatenate"}},
        "strategy": {
            "concatenate": {
                "separator": "\n-------------\n",
                "hide_intermediate_think": True,
                "hide_final_think": True,
                "thinking_tags": ["think", "reason", "reasoning", "thought"],
                "skip_final_aggregation": False,
            }
        },
        "settings": {"timeout": 30},
    }

    # Patch the config
    monkeypatch.setattr("quorum.oai_proxy.config", mock_config)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify streaming response format
    response.status_code | should.equal(200)
    response.headers["content-type"].split(";")[0] | should.equal("text/event-stream")

    # Collect all content chunks
    content = ""
    final_content = None
    for line in response.iter_lines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])  # Skip "data: " prefix
                if data == "[DONE]":
                    continue
                if "choices" in data and data["choices"][0].get("delta", {}).get(
                    "content"
                ):
                    content += data["choices"][0]["delta"]["content"]
                    # Check if this is the final aggregated content
                    if "chatcmpl-parallel-final" in data.get("id", ""):
                        final_content = data["choices"][0]["delta"]["content"]
            except json.JSONDecodeError:
                continue

    # The intermediate content should NOT have tags and their content (since hide_intermediate_think is True)
    content | should.do_not.contain("<think>")
    content | should.do_not.contain("</think>")
    content | should.do_not.contain("Let me think about this")

    # The final aggregated content should also have tags and their content stripped
    final_content | should.do_not.contain("<think>")
    final_content | should.do_not.contain("</think>")
    final_content | should.do_not.contain("<reason>")
    final_content | should.do_not.contain("</reason>")
    final_content | should.do_not.contain(
        "Let me think about this"
    )  # Thinking content should be stripped
    final_content | should.contain("The answer is 4")  # Main content preserved


@pytest.mark.asyncio
async def test_strip_thinking_tags_disabled(test_client_parallel_backends, monkeypatch):
    """Test that thinking tags are preserved when stripping is disabled"""

    # Create a mock config with disabled tag stripping
    mock_config = {
        "primary_backends": [
            {"name": "LLM1", "url": "http://test1.example.com/v1", "model": "gpt-4-1"},
        ],
        "iterations": {"aggregation": {"strategy": "concatenate"}},
        "strategy": {
            "concatenate": {
                "separator": "\n-------------\n",
                "hide_intermediate_think": False,
                "hide_final_think": False,
                "thinking_tags": ["think", "reason", "reasoning", "thought"],
            }
        },
        "settings": {"timeout": 30},
    }

    # Patch the config directly in the module
    monkeypatch.setattr("quorum.oai_proxy.config", mock_config)

    async def mock_post(*args, **kwargs):
        return MockResponse(200, MOCK_COMPLETION_WITH_THINKING)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "What is 2+2?"}]},
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify successful response
    response.status_code | should.equal(200)
    data = response.json()

    # Verify thinking tags are preserved
    content = data["choices"][0]["message"]["content"]
    content | should.contain("<think>")
    content | should.contain("</think>")
    content | should.contain("Let me think about this")  # Thinking content preserved


@pytest.mark.asyncio
async def test_config_loading(monkeypatch, tmp_path):
    """Test configuration loading functionality"""
    from quorum.oai_proxy import load_config
    import yaml

    # Create a temporary config file
    config_content = {
        "primary_backends": [
            {"name": "test", "url": "http://test.example.com/v1", "model": "gpt-4"},
        ],
        "settings": {"timeout": 45},
        "iterations": {"aggregation": {"strategy": "concatenate"}},
        "strategy": {
            "concatenate": {
                "separator": "\n---\n",
                "hide_intermediate_think": True,
                "hide_final_think": True,
                "thinking_tags": ["think", "reason"],
            }
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    # Mock the config path
    monkeypatch.setattr("pathlib.Path.read_text", lambda x: yaml.dump(config_content))

    # Test successful config loading
    loaded_config = load_config()
    loaded_config | should.have.key("primary_backends")
    loaded_config | should.have.key("settings")
    loaded_config["settings"]["timeout"] | should.equal(45)

    # Test default config when loading fails
    def mock_read_text_error(*args):
        raise FileNotFoundError("Config file not found")

    monkeypatch.setattr("pathlib.Path.read_text", mock_read_text_error)
    default_config = load_config()
    default_config | should.have.key("primary_backends")
    default_config | should.have.key("settings")
    default_config["settings"]["timeout"] | should.equal(60)  # Default timeout
