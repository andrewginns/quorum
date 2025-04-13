"""FastAPI application and routes for Quorum proxy."""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from .config import config, OPENAI_API_BASE, DEFAULT_MODEL, TIMEOUT
from .backends import call_backend
from .streaming import stream_with_role
from .aggregation import progress_streaming_aggregator, aggregate_responses
from .utils import strip_thinking_tags

logger = logging.getLogger(__name__)

app = FastAPI(title="Quorum Proxy")


@app.post("/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    """
    Primary proxy endpoint for chat completions:
    - Routes requests to multiple backends
    - Supports streaming or non-streaming
    - Aggregates responses according to config
    """
    body = await request.body()
    headers = dict(request.headers)
    
    auth_header = headers.get("authorization") or headers.get("Authorization")
    api_key_env = os.environ.get("OPENAI_API_KEY")
    
    if not auth_header and not api_key_env:
        return Response(
            content=json.dumps({
                "error": {
                    "message": "Authorization header is required and OPENAI_API_KEY environment variable is not set",
                    "type": "auth_error"
                }
            }),
            status_code=401,
            media_type="application/json"
        )

    try:
        request_data = json.loads(body)
    except json.JSONDecodeError:
        is_stream = False
        content_type = headers.get("content-type", "").lower()
        if "stream=true" in content_type or headers.get("accept") == "text/event-stream":
            is_stream = True

        if is_stream:
            async def error_stream():
                initial_event = {
                    "id": "chatcmpl-role",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": "proxy",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(initial_event)}\n\n".encode()

                error_event = {
                    "id": "error",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": "proxy",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "Error: Invalid JSON"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                }
                yield f"data: {json.dumps(error_event)}\n\n".encode()

                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
            )
        else:
            return Response(
                content=json.dumps(
                    {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}
                ),
                status_code=400,
                media_type="application/json",
            )

    model = request_data.get("model", DEFAULT_MODEL)
    is_stream = request_data.get("stream", False)

    if not model:
        if is_stream:
            async def error_stream():
                initial_event = {
                    "id": "chatcmpl-role",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": "proxy",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(initial_event)}\n\n".encode()

                error_event = {
                    "id": "error",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": "proxy",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "Error: All backends failed"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                }
                yield f"data: {json.dumps(error_event)}\n\n".encode()

                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
            )
        else:
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "message": "Model must be specified when config.yaml model is blank",
                            "type": "invalid_request_error",
                        }
                    }
                ),
                status_code=400,
                media_type="application/json",
            )

    is_stream = request_data.get("stream", False)

    valid_backends = []
    for backend in config.get("primary_backends", []):
        if backend.get("url"):
            valid_backends.append(backend)

    if not valid_backends:
        if is_stream:
            async def error_stream():
                initial_event = {
                    "id": "chatcmpl-role",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": "proxy",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(initial_event)}\n\n".encode()

                error_event = {
                    "id": "error",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": "proxy",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "Error: No valid backends configured"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                }
                yield f"data: {json.dumps(error_event)}\n\n".encode()

                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
            )
        else:
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "message": "No valid backends configured",
                            "type": "server_error",
                        }
                    }
                ),
                status_code=500,
                media_type="application/json",
            )

    iterations_config = config.get("iterations", {})
    aggregation_config = iterations_config.get("aggregation", {})
    strategy_type = aggregation_config.get("strategy", "")

    if not strategy_type:
        strategy_config = config.get("strategy", {})
        strategy_type = strategy_config.get("type", "single")

    strategy_config = config.get("strategy", {}).get(strategy_type, {})

    if strategy_type == "single":
        backend = valid_backends[0]
        try:
            response = await call_backend(backend, body, headers, TIMEOUT)

            if response["status_code"] != 200:
                return Response(
                    content=json.dumps(response["content"]),
                    status_code=response["status_code"],
                    media_type="application/json",
                )

            if is_stream:
                return StreamingResponse(
                    stream_with_role(response["content"], model),
                    media_type="text/event-stream",
                )
            else:
                return Response(
                    content=json.dumps(response["content"]),
                    status_code=200,
                    media_type="application/json",
                )
        except Exception as e:
            logger.error(f"Error calling backend: {str(e)}")
            return Response(
                content=json.dumps(
                    {"error": {"message": str(e), "type": "server_error"}}
                ),
                status_code=500,
                media_type="application/json",
            )

    elif strategy_type == "fallback":
        for backend in valid_backends:
            try:
                response = await call_backend(backend, body, headers, TIMEOUT)

                if response["status_code"] == 200:
                    if is_stream:
                        return StreamingResponse(
                            stream_with_role(response["content"], model),
                            media_type="text/event-stream",
                        )
                    else:
                        return Response(
                            content=json.dumps(response["content"]),
                            status_code=200,
                            media_type="application/json",
                        )
            except Exception as e:
                logger.error(f"Error calling backend {backend['name']}: {str(e)}")
                continue

        return Response(
            content=json.dumps(
                {
                    "error": {
                        "message": "All backends failed",
                        "type": "server_error",
                    }
                }
            ),
            status_code=500,
            media_type="application/json",
        )

    elif strategy_type == "aggregate" or strategy_type == "concatenate":
        source_backend_names = strategy_config.get("source_backends", [])
        aggregator_backend_name = strategy_config.get("aggregator_backend", "")

        source_backends = []
        aggregator_backend = None

        for backend in valid_backends:
            if backend.get("name") == aggregator_backend_name:
                aggregator_backend = backend
                break

        for backend in valid_backends:
            if source_backend_names and backend.get("name") in source_backend_names:
                source_backends.append(backend)
            elif not source_backend_names and backend.get("name") != aggregator_backend_name:
                source_backends.append(backend)

        tasks = [
            asyncio.create_task(call_backend(backend, body, headers, TIMEOUT))
            for backend in source_backends
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = []
        successful_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Backend {i} failed: {str(response)}")
            else:
                valid_responses.append(response)
                if response["status_code"] == 200:
                    successful_responses.append(response)

        if not valid_responses or not successful_responses:
            if is_stream:
                async def error_stream():
                    initial_event = {
                        "id": "chatcmpl-role",
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": "aggregate-proxy",
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(initial_event)}\n\n".encode()

                    error_event = {
                        "id": "error",
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": "aggregate-proxy",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": "Error: All backends failed"
                                },
                                "finish_reason": "stop"
                            }
                        ],
                    }
                    yield f"data: {json.dumps(error_event)}\n\n".encode()

                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    error_stream(),
                    media_type="text/event-stream",
                )
            else:
                return Response(
                    content=json.dumps(
                        {
                            "error": {
                                "message": "All backends failed",
                                "type": "proxy_error",
                            }
                        }
                    ),
                    status_code=500,
                    media_type="application/json",
                )

        if is_stream and aggregator_backend:
            hide_intermediate_think = strategy_config.get("strip_intermediate_thinking", True)
            hide_final_think = strategy_config.get("hide_aggregator_thinking", False)
            thinking_tags = strategy_config.get("thinking_tags", ["think", "reason", "reasoning", "thought"])

            return StreamingResponse(
                progress_streaming_aggregator(
                    source_backends,
                    body,
                    headers,
                    TIMEOUT,
                    strategy_config.get("intermediate_separator", "\n\n---\n\n"),
                    hide_intermediate_think,
                    hide_final_think,
                    thinking_tags,
                    False,  # skip_final_aggregation
                    False,  # suppress_individual_responses
                ),
                media_type="text/event-stream",
            )
        elif not is_stream:
            contents = []
            for response in valid_responses:
                if response["status_code"] == 200:
                    if "content" in response and isinstance(response["content"], dict):
                        if "choices" in response["content"] and response["content"]["choices"]:
                            content = response["content"]["choices"][0]["message"]["content"]
                            contents.append(content)

            hide_final_think = strategy_config.get("hide_aggregator_thinking", False)
            thinking_tags = strategy_config.get("thinking_tags", ["think", "reason", "reasoning", "thought"])

            filtered_contents = [
                strip_thinking_tags(content, thinking_tags, hide_intermediate=hide_final_think)
                for content in contents
            ]

            if aggregator_backend:
                try:
                    request_data = json.loads(body)
                    user_query = ""
                    if "messages" in request_data and request_data["messages"]:
                        for msg in request_data["messages"]:
                            if msg.get("role") == "user":
                                user_query = msg.get("content", "")
                                break

                    prompt_template = strategy_config.get(
                        "prompt_template",
                        "You have received the following responses regarding the user's query:\n\n{responses}\n\nProvide a concise synthesis of these responses."
                    )

                    combined_text = await aggregate_responses(
                        filtered_contents,
                        aggregator_backend,
                        user_query,
                        strategy_config.get("intermediate_separator", "\n\n---\n\n"),
                        strategy_config.get("include_original_query", True),
                        strategy_config.get("query_format", "Original query: {query}\n\n"),
                        strategy_config.get("include_source_names", False),
                        strategy_config.get("source_label_format", "Response from {backend_name}:\n"),
                        prompt_template,
                        headers
                    )

                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    total_tokens = 0

                    for response in valid_responses:
                        if response["status_code"] == 200 and "content" in response and isinstance(response["content"], dict):
                            usage = response["content"].get("usage", {})
                            total_prompt_tokens += usage.get("prompt_tokens", 0)
                            total_completion_tokens += usage.get("completion_tokens", 0)
                            total_tokens += usage.get("total_tokens", 0)

                    return Response(
                        content=json.dumps({
                            "id": "chatcmpl-aggregate",
                            "object": "chat.completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": "aggregate-proxy",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": combined_text
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": total_prompt_tokens,
                                "completion_tokens": total_completion_tokens,
                                "total_tokens": total_tokens
                            }
                        }),
                        status_code=200,
                        media_type="application/json"
                    )
                except Exception as e:
                    logger.error(f"Error during aggregation: {str(e)}")
                    combined_text = "\n\n---\n\n".join(filtered_contents)
                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    total_tokens = 0

                    for response in valid_responses:
                        if response["status_code"] == 200 and "content" in response and isinstance(response["content"], dict):
                            usage = response["content"].get("usage", {})
                            total_prompt_tokens += usage.get("prompt_tokens", 0)
                            total_completion_tokens += usage.get("completion_tokens", 0)
                            total_tokens += usage.get("total_tokens", 0)

                    return Response(
                        content=json.dumps({
                            "id": "chatcmpl-aggregate",
                            "object": "chat.completion",
                            "created": int(asyncio.get_event_loop().time()),
                            "model": "aggregate-proxy",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": combined_text
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": total_prompt_tokens,
                                "completion_tokens": total_completion_tokens,
                                "total_tokens": total_tokens
                            }
                        }),
                        status_code=200,
                        media_type="application/json"
                    )

            combined_text = "\n\n---\n\n".join(filtered_contents)

            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0

            for response in valid_responses:
                if response["status_code"] == 200 and "content" in response and isinstance(response["content"], dict):
                    usage = response["content"].get("usage", {})
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)

            return Response(
                content=json.dumps({
                    "id": "chatcmpl-aggregate",
                    "object": "chat.completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": "aggregate-proxy",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": combined_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_tokens
                    }
                }),
                status_code=200,
                media_type="application/json"
            )

    elif strategy_type == "parallel":
        tasks = [
            asyncio.create_task(call_backend(backend, body, headers, TIMEOUT))
            for backend in valid_backends
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = []
        successful_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Backend {i} failed: {str(response)}")
            else:
                valid_responses.append(response)
                if response["status_code"] == 200:
                    successful_responses.append(response)

        if not valid_responses:
            if is_stream:
                async def error_stream():
                    initial_event = {
                        "id": "chatcmpl-role",
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": "parallel-proxy",
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(initial_event)}\n\n".encode()

                    error_event = {
                        "id": "error",
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": "parallel-proxy",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": "Error: All backends failed"
                                },
                                "finish_reason": "stop"
                            }
                        ],
                    }
                    yield f"data: {json.dumps(error_event)}\n\n".encode()

                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    error_stream(),
                    media_type="text/event-stream",
                )
            else:
                return Response(
                    content=json.dumps(
                        {
                            "error": {
                                "message": "All backends failed",
                                "type": "proxy_error",
                            }
                        }
                    ),
                    status_code=500,
                    media_type="application/json",
                )

        if is_stream:
            aggregator_strategy = strategy_config.get("aggregate", {})

            hide_intermediate_think = aggregator_strategy.get("hide_intermediate_think", True)
            hide_final_think = aggregator_strategy.get("hide_final_think", False)
            thinking_tags = aggregator_strategy.get("thinking_tags", ["think", "reason", "reasoning", "thought"])

            skip_final_aggregation = aggregator_strategy.get("skip_final_aggregation", False)

            suppress_individual_responses = aggregator_strategy.get("suppress_individual_responses", False)

            return StreamingResponse(
                progress_streaming_aggregator(
                    valid_backends,
                    body,
                    headers,
                    TIMEOUT,
                    aggregator_strategy.get("intermediate_separator", "\n-------------\n"),
                    hide_intermediate_think,
                    hide_final_think,
                    thinking_tags,
                    skip_final_aggregation,
                    suppress_individual_responses,
                ),
                media_type="text/event-stream",
            )
        else:
            contents = []
            for response in valid_responses:
                if response["status_code"] == 200:
                    if "content" in response and isinstance(response["content"], dict):
                        if "choices" in response["content"] and response["content"]["choices"]:
                            content = response["content"]["choices"][0]["message"]["content"]
                            contents.append(content)

            aggregator_strategy = strategy_config.get("aggregate", {})

            hide_final_think = aggregator_strategy.get("hide_final_think", False)
            thinking_tags = aggregator_strategy.get("thinking_tags", ["think", "reason", "reasoning", "thought"])

            filtered_contents = [
                strip_thinking_tags(content, thinking_tags, hide_intermediate=hide_final_think)
                for content in contents
            ]

            aggregator_backend_name = aggregator_strategy.get("aggregator_backend")

            if aggregator_backend_name:
                aggregator_backend = None
                for backend in config.get("primary_backends", []):
                    if backend.get("name") == aggregator_backend_name:
                        aggregator_backend = backend
                        break

                if aggregator_backend:
                    try:
                        user_query = ""
                        if "messages" in request_data and request_data["messages"]:
                            for msg in request_data["messages"]:
                                if msg.get("role") == "user":
                                    user_query = msg.get("content", "")
                                    break

                        prompt_template = aggregator_strategy.get(
                            "prompt_template",
                            "You have received the following responses regarding the user's query:\n\n{responses}\n\nProvide a concise synthesis of these responses.",
                        )

                        if "{intermediate_results}" in prompt_template:
                            prompt_template = prompt_template.replace(
                                "{intermediate_results}", "{responses}"
                            )

                        combined_text = await aggregate_responses(
                            filtered_contents,
                            aggregator_backend,
                            user_query,
                            aggregator_strategy.get(
                                "intermediate_separator", "\n\n---\n\n"
                            ),
                            aggregator_strategy.get("include_original_query", True),
                            aggregator_strategy.get(
                                "query_format", "Original query: {query}\n\n"
                            ),
                            aggregator_strategy.get("include_source_names", False),
                            aggregator_strategy.get(
                                "source_label_format", "Response from {backend_name}:\n"
                            ),
                            prompt_template,
                            headers,
                        )

                        total_prompt_tokens = 0
                        total_completion_tokens = 0
                        total_tokens = 0

                        for response in valid_responses:
                            if response["status_code"] == 200 and "content" in response and isinstance(response["content"], dict):
                                usage = response["content"].get("usage", {})
                                total_prompt_tokens += usage.get("prompt_tokens", 0)
                                total_completion_tokens += usage.get("completion_tokens", 0)
                                total_tokens += usage.get("total_tokens", 0)

                        return Response(
                            content=json.dumps(
                                {
                                    "id": "chatcmpl-parallel",
                                    "object": "chat.completion",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": "parallel-proxy",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": combined_text,
                                            },
                                            "finish_reason": "stop",
                                        }
                                    ],
                                    "usage": {
                                        "prompt_tokens": total_prompt_tokens,
                                        "completion_tokens": total_completion_tokens,
                                        "total_tokens": total_tokens
                                    }
                                }
                            ),
                            status_code=200,
                            media_type="application/json",
                        )
                    except Exception as e:
                        logger.error(f"Error during aggregation: {str(e)}")
                        combined_text = "\n\n---\n\n".join(filtered_contents)
                        total_prompt_tokens = 0
                        total_completion_tokens = 0
                        total_tokens = 0

                        for response in valid_responses:
                            if response["status_code"] == 200 and "content" in response and isinstance(response["content"], dict):
                                usage = response["content"].get("usage", {})
                                total_prompt_tokens += usage.get("prompt_tokens", 0)
                                total_completion_tokens += usage.get("completion_tokens", 0)
                                total_tokens += usage.get("total_tokens", 0)

                        return Response(
                            content=json.dumps(
                                {
                                    "id": "chatcmpl-parallel",
                                    "object": "chat.completion",
                                    "created": int(asyncio.get_event_loop().time()),
                                    "model": "parallel-proxy",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": combined_text,
                                            },
                                            "finish_reason": "stop",
                                        }
                                    ],
                                    "usage": {
                                        "prompt_tokens": total_prompt_tokens,
                                        "completion_tokens": total_completion_tokens,
                                        "total_tokens": total_tokens
                                    }
                                }
                            ),
                            status_code=200,
                            media_type="application/json",
                        )

            combined_text = "\n\n---\n\n".join(filtered_contents)

            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0

            for response in valid_responses:
                if response["status_code"] == 200 and "content" in response and isinstance(response["content"], dict):
                    usage = response["content"].get("usage", {})
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)

            return Response(
                content=json.dumps(
                    {
                        "id": "chatcmpl-parallel",
                        "object": "chat.completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": "parallel-proxy",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": combined_text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": total_tokens
                        }
                    }
                ),
                status_code=200,
                media_type="application/json",
            )

    else:
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "message": f"Unknown strategy: {strategy_type}",
                        "type": "server_error",
                    }
                }
            ),
            status_code=500,
            media_type="application/json",
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
