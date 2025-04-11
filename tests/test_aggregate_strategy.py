"""
Tests for the 'aggregate' strategy in the quorum proxy service.
"""
import json
import pytest
from grappa import should
import httpx
from .conftest import MockResponse, MockStreamingResponse

@pytest.fixture
def mock_config_aggregate_strategy(monkeypatch):
    """Mock config file with aggregate strategy settings"""
    import yaml
    from pathlib import Path
    
    config = {
        "primary_backends": [
            {"name": "LLM1", "url": "http://test1.example.com/v1", "model": "gpt-4-1"},
            {"name": "LLM2", "url": "http://test2.example.com/v1", "model": "gpt-4-2"},
            {"name": "LLM3", "url": "http://test3.example.com/v1", "model": "gpt-4-3"},
        ],
        "iterations": {"aggregation": {"strategy": "aggregate"}},
        "strategy": {
            "aggregate": {
                "source_backends": ["LLM1", "LLM2"],
                "aggregator_backend": "LLM3",
                "intermediate_separator": "\n\n---\n\n",
                "include_source_names": True,
                "source_label_format": "Response from {backend_name}:\n",
                "prompt_template": "You have received the following responses regarding the user's query:\n\n{responses}\n\nProvide a concise synthesis of these responses.",
                "strip_intermediate_thinking": True,
                "hide_aggregator_thinking": True,
                "thinking_tags": ["think", "reason", "reasoning", "thought"],
                "include_original_query": True
            }
        },
        "settings": {"timeout": 30},
    }
    
    def mock_read_text(*args, **kwargs):
        return yaml.dump(config)
        
    monkeypatch.setattr(Path, "read_text", mock_read_text)
    import quorum.oai_proxy
    import importlib
    importlib.reload(quorum.oai_proxy)
    return config

@pytest.fixture
def test_client_aggregate_strategy(mock_config_aggregate_strategy):
    """Create a test client with aggregate strategy config"""
    from quorum.oai_proxy import app
    from fastapi.testclient import TestClient
    return TestClient(app)

@pytest.mark.asyncio
async def test_non_streaming_aggregate_strategy(test_client_aggregate_strategy, monkeypatch):
    """Test non-streaming chat completion with aggregate strategy"""
    
    source_responses = []
    
    call_count = {"count": 0}
    
    async def mock_post(*args, **kwargs):
        url = str(args[1])
        call_count["count"] += 1
        
        if "test1.example.com" in url or "test2.example.com" in url:
            backend_num = "1" if "test1.example.com" in url else "2"
            response = {
                "id": f"chatcmpl-{backend_num}",
                "object": "chat.completion",
                "created": 1677652288,
                "model": f"gpt-4-{backend_num}",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"<think>Thinking through response {backend_num}</think>Response from backend {backend_num}.",
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
            }
            source_responses.append(response)
            return MockResponse(200, response)
            
        elif "test3.example.com" in url:
            request_body = json.loads(kwargs["content"])
            messages = request_body.get("messages", [])
            
            prompt = messages[-1]["content"]
            prompt | should.contain("Response from backend 1")
            prompt | should.contain("Response from backend 2")
            
            return MockResponse(200, {
                "id": "chatcmpl-agg",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4-3",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<think>Synthesizing the responses</think>Aggregated response combining inputs from multiple backends.",
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
            })
        
        return MockResponse(500, {"error": {"message": "Unknown backend", "type": "backend_error"}})
    
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    
    response = test_client_aggregate_strategy.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test question"}],
            "stream": False
        },
        headers={"Authorization": "Bearer test-key"},
    )
    
    response.status_code | should.equal(200)
    result = response.json()
    
    call_count["count"] | should.equal(3)
    
    result["choices"][0]["message"]["content"] | should.equal("Aggregated response combining inputs from multiple backends.")
    result["choices"][0]["message"]["content"] | should.do_not.contain("<think>")
    
    result["choices"][0]["message"]["content"] | should.do_not.contain("Response from backend 1")
    result["choices"][0]["message"]["content"] | should.do_not.contain("Response from backend 2")

@pytest.mark.asyncio
async def test_streaming_aggregate_strategy(test_client_aggregate_strategy, monkeypatch):
    """Test streaming chat completion with aggregate strategy"""
    
    backend_calls = {"source1": False, "source2": False, "aggregator": False}
    
    async def mock_post(*args, **kwargs):
        url = str(args[1])
        
        if "test1.example.com" in url:
            backend_calls["source1"] = True
            return MockStreamingResponse()
        elif "test2.example.com" in url:
            backend_calls["source2"] = True
            return MockStreamingResponse()
        elif "test3.example.com" in url:
            backend_calls["aggregator"] = True
            request_body = json.loads(kwargs["content"])
            messages = request_body.get("messages", [])
            
            prompt = messages[-1]["content"]
            prompt | should.contain("Response from LLM1")
            prompt | should.contain("Response from LLM2")
            
            return MockStreamingResponse()
        
        return httpx.Response(500, json={"error": {"message": "Unknown backend", "type": "backend_error"}})
    
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    
    response = test_client_aggregate_strategy.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test question"}],
            "stream": True
        },
        headers={"Authorization": "Bearer test-key"},
    )
    
    response.status_code | should.equal(200)
    response.headers["content-type"].split(";")[0] | should.equal("text/event-stream")
    
    chunks = [c for c in response.iter_lines() if c.strip()]
    
    backend_calls["source1"] | should.be.true
    backend_calls["source2"] | should.be.true
    backend_calls["aggregator"] | should.be.true
    
    first_chunk = json.loads(chunks[0].replace("data: ", ""))
    first_chunk["choices"][0]["delta"] | should.have.key("role")
    
    chunks[-1] | should.equal("data: [DONE]")

@pytest.mark.asyncio
async def test_aggregate_strategy_missing_aggregator(test_client_aggregate_strategy, monkeypatch):
    """Test error handling when aggregator backend is not found"""
    
    import quorum.oai_proxy
    quorum.oai_proxy.config["strategy"]["aggregate"]["aggregator_backend"] = "NonExistentLLM"
    
    response = test_client_aggregate_strategy.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test question"}],
            "stream": False
        },
        headers={"Authorization": "Bearer test-key"},
    )
    
    response.status_code | should.equal(500)
    result = response.json()
    result["error"]["message"] | should.contain("Aggregator backend not found")
    
    quorum.oai_proxy.config["strategy"]["aggregate"]["aggregator_backend"] = "LLM3"

@pytest.mark.asyncio
async def test_aggregate_strategy_source_failure(test_client_aggregate_strategy, monkeypatch):
    """Test error handling when source backends fail"""
    
    async def mock_post(*args, **kwargs):
        url = str(args[1])
        
        if "test1.example.com" in url or "test2.example.com" in url:
            return MockResponse(500, {"error": {"message": "Backend error", "type": "backend_error"}})
            
        elif "test3.example.com" in url:
            return MockResponse(200, {
                "id": "chatcmpl-agg",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4-3",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Aggregated response.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
            })
        
        return MockResponse(500, {"error": {"message": "Unknown backend", "type": "backend_error"}})
    
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    
    response = test_client_aggregate_strategy.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test question"}],
            "stream": False
        },
        headers={"Authorization": "Bearer test-key"},
    )
    
    response.status_code | should.equal(500)
    result = response.json()
    result["error"]["message"] | should.contain("All backends failed")
