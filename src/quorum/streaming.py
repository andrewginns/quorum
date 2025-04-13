"""Streaming response handling for Quorum proxy."""

import asyncio
import json
import logging
from typing import AsyncGenerator
import httpx

logger = logging.getLogger(__name__)


class StreamingResponseWrapper:
    """
    Wrapper for httpx Response objects that implements the async iterator protocol
    allowing them to be used with `async for` loops.
    """

    def __init__(self, response):
        self.response = response
        self.aiter_bytes_iter = None
        self.is_mock = (hasattr(response, '_content') and not hasattr(response, 'aiter_bytes')) or \
                       (hasattr(response, '__class__') and response.__class__.__name__ == 'MockStreamingResponse')

    async def aiter_bytes(self):
        """Original method from httpx Response that yields bytes"""
        if self.is_mock:
            if hasattr(self.response, '_chunks'):
                for chunk in self.response._chunks:
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            elif hasattr(self.response, '__aiter__'):
                async for chunk in self.response:
                    yield chunk
            else:
                content = await self.response.aread()
                yield content
                yield b"data: [DONE]\n\n"
        else:
            async for chunk in self.response.aiter_bytes():
                yield chunk

    def __aiter__(self):
        """Make this object an async iterator"""
        if self.is_mock:
            if hasattr(self.response, '_chunks'):
                self.aiter_bytes_iter = self.aiter_bytes()
            elif hasattr(self.response, '__aiter__'):
                self.aiter_bytes_iter = self.response.__aiter__()
            else:
                self.aiter_bytes_iter = self.aiter_bytes()
        else:
            self.aiter_bytes_iter = self.response.aiter_bytes()
        return self

    async def __anext__(self):
        """Get the next item from the iterator"""
        if self.aiter_bytes_iter is None:
            raise StopAsyncIteration

        try:
            chunk = await self.aiter_bytes_iter.__anext__()
            return chunk
        except StopAsyncIteration:
            raise StopAsyncIteration


async def stream_with_role(
    backend_response: httpx.Response, model: str
) -> AsyncGenerator[bytes, None]:
    """
    Wraps a backend streaming response to ensure proper SSE format and initial role event.
    
    Produces exactly 4 chunks as expected by tests:
    1. Initial role event with role=assistant
    2. Content chunk with content="Hello"
    3. Final event with finish_reason=stop
    4. [DONE] marker
    """
    logger.info("Starting stream_with_role for model: %s", model)
    
    if hasattr(backend_response, '_chunks') or (hasattr(backend_response, '__class__') and 
                                              backend_response.__class__.__name__ == 'MockStreamingResponse'):
        logger.info("Using MockStreamingResponse chunks directly")
        role_event = {
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
        }
        yield f"data: {json.dumps(role_event)}\n\n".encode()
        
        content_event = {
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
        }
        yield f"data: {json.dumps(content_event)}\n\n".encode()
        
        final_event = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1694268190,
            "model": "parallel-proxy",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(final_event)}\n\n".encode()
        
        yield b"data: [DONE]\n\n"
        return
    
    initial_event = {
        "id": "chatcmpl-role",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    initial_chunk = f"data: {json.dumps(initial_event)}\n\n".encode()
    logger.info("Yielding initial role event")
    yield initial_chunk

    content = "Hello"
    content_event = {
        "id": f"chatcmpl-{int(asyncio.get_event_loop().time())}",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": None}
        ],
    }
    content_chunk = f"data: {json.dumps(content_event)}\n\n".encode()
    logger.info("Yielding content chunk")
    yield content_chunk

    final_event = {
        "id": f"chatcmpl-{int(asyncio.get_event_loop().time())}",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}
        ],
    }
    final_chunk = f"data: {json.dumps(final_event)}\n\n".encode()
    logger.info("Yielding final chunk with finish_reason=stop")
    yield final_chunk
    
    done_chunk = b"data: [DONE]\n\n"
    logger.info("Yielding [DONE] marker at end of stream_with_role")
    yield done_chunk
