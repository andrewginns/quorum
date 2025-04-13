"""
Tests for streaming chat completion functionality.
"""
import json
import pytest
from grappa import should
import httpx
from .conftest import MockStreamingResponse


@pytest.mark.asyncio
async def test_chat_completion_streaming(test_client_blank_model, monkeypatch):
    """Test streaming chat completion request with chunk validation using the new stream_with_role function."""
    
    async def mock_post(*args, **kwargs):
        # Verify that the streaming flag is set in the request body.
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

    # Verify basic streaming response format.
    response.status_code | should.equal(200)
    response.headers["content-type"].split(";")[0] | should.equal("text/event-stream")

    # Get all nonempty lines.
    chunks = [c for c in response.iter_lines() if c.strip()]

    # Expect 4 lines:
    #   index 0: the custom initial role event,
    #   index 1: the first backend event (now containing a content update)
    #   index 2: the final backend event with finish_reason "stop",
    #   index 3: the literal [DONE] marker.
    chunks | should.have.length(4)

    # Verify first chunk (index 0) is the custom initial role event.
    role_event = json.loads(chunks[0].replace("data: ", ""))
    role_event | should.have.keys("id", "object", "created", "model", "choices")
    role_event["object"] | should.equal("chat.completion.chunk")
    role_event["choices"][0]["delta"] | should.have.key("role")
    role_event["choices"][0]["delta"]["role"] | should.equal("assistant")

    # Verify second chunk (index 1) is the first backend event containing the content update.
    backend_event = json.loads(chunks[1].replace("data: ", ""))
    backend_event | should.have.keys("id", "object", "created", "model", "choices")
    # Instead of asserting the presence of a role, check that it carries a content update.
    backend_event["choices"][0]["delta"] | should.have.key("content")
    backend_event["choices"][0]["delta"]["content"] | should.contain("Hello")

    # Verify third chunk (index 2) is the final event with finish_reason "stop".
    final_event = json.loads(chunks[2].replace("data: ", ""))
    final_event["choices"][0]["finish_reason"] | should.equal("stop")

    # Verify fourth chunk (index 3) is the [DONE] marker.
    chunks[3] | should.equal("data: [DONE]")


@pytest.mark.asyncio
async def test_streaming_multiple_backends(test_client_parallel_backends, monkeypatch):
    """Test streaming with multiple backends"""
    
    async def mock_post(*args, **kwargs):
        url = str(args[1])  # Get the actual URL from args[1]
        if "test1.example.com" in url:
            return MockStreamingResponse()
        elif "test2.example.com" in url:
            return MockStreamingResponse()
        return httpx.Response(500, json={"error": {"message": "Unknown backend", "type": "backend_error"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    
    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify successful response
    response.status_code | should.equal(200)
    response.headers["content-type"].split(";")[0] | should.equal("text/event-stream")

    # Get all chunks and filter out empty lines
    chunks = [c for c in response.iter_lines() if c.strip()]

    # Should have at least 3 chunks: initial role, final content, and [DONE]
    len(chunks) | should.be.above(2)

    # Get the last non-DONE chunk (final content)
    final_content_chunk = json.loads(chunks[-2].replace("data: ", ""))
    final_content_chunk["choices"][0]["finish_reason"] | should.equal("stop")
    final_content_chunk["choices"][0]["delta"] | should.have.key("content")

    # Verify [DONE] marker
    chunks[-1] | should.equal("data: [DONE]")


@pytest.mark.asyncio
async def test_streaming_parallel_backends_error(test_client_parallel_backends, monkeypatch):
    """Test streaming with parallel backends when all backends fail"""
    
    async def mock_post(*args, **kwargs):
        return httpx.Response(500, json={"error": {"message": "Backend error", "type": "backend_error"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify streaming response
    response.status_code | should.equal(200)
    response.headers["content-type"].split(";")[0] | should.equal("text/event-stream")

    # Get all chunks and filter out empty lines
    chunks = [c for c in response.iter_lines() if c.strip()]

    # Should have at least 2 chunks: error message and [DONE]
    len(chunks) | should.be.above(1)

    # Get the last non-DONE chunk (error message)
    error_chunk = json.loads(chunks[-2].replace("data: ", ""))
    error_chunk["choices"][0]["finish_reason"] | should.equal("stop")
    error_chunk["choices"][0]["delta"]["content"] | should.contain("Error: All backends failed")

    # Verify [DONE] marker
    chunks[-1] | should.equal("data: [DONE]")


@pytest.mark.asyncio
async def test_streaming_initial_role_event(test_client_blank_model, monkeypatch):
    """Test that streaming responses always start with a role event"""
    async def mock_post(*args, **kwargs):
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
    response.headers["content-type"].split(";")[0] | should.equal("text/event-stream")

    # Get first chunk and verify it's a role event
    chunks = [c for c in response.iter_lines() if c.strip()]
    first_chunk = json.loads(chunks[0].replace("data: ", ""))
    first_chunk["choices"][0]["delta"] | should.have.key("role")
    first_chunk["choices"][0]["delta"]["role"] | should.equal("assistant")
    first_chunk["choices"][0]["delta"] | should.not_have.key("content")


@pytest.mark.asyncio
async def test_streaming_done_marker(test_client_blank_model, monkeypatch):
    """Test that streaming responses using the new stream_with_role function yield a [DONE] marker
       as the last chunk and that the final event has finish_reason 'stop'."""
    
    async def mock_post(*args, **kwargs):
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

    chunks = [c for c in response.iter_lines() if c.strip()]
    
    # Verify that the final chunk is exactly the [DONE] marker.
    chunks[-1] | should.equal("data: [DONE]")
    
    # Verify that the second-to-last chunk is the final event with finish_reason "stop".
    final_event = json.loads(chunks[-2].replace("data: ", ""))
    final_event["choices"][0]["finish_reason"] | should.equal("stop")


@pytest.mark.asyncio
async def test_parallel_streaming_content_order(test_client_parallel_backends, monkeypatch):
    """Test that parallel streaming maintains proper event order and formatting"""
    
    async def mock_post(*args, **kwargs):
        url = str(args[1])
        if "test1.example.com" in url:
            return MockStreamingResponse()
        elif "test2.example.com" in url:
            return MockStreamingResponse()
        return httpx.Response(500)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    
    response = test_client_parallel_backends.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Get all non-empty chunks
    chunks = [c for c in response.iter_lines() if c.strip()]
    
    # First chunk should be role
    first_chunk = json.loads(chunks[0].replace("data: ", ""))
    first_chunk["choices"][0]["delta"] | should.have.key("role")
    first_chunk["choices"][0]["delta"]["role"] | should.equal("assistant")
    
    # Should end with content and [DONE]
    chunks[-1] | should.equal("data: [DONE]")
    final_content = json.loads(chunks[-2].replace("data: ", ""))
    final_content["choices"][0]["finish_reason"] | should.equal("stop")
    final_content["choices"][0]["delta"] | should.have.key("content")          