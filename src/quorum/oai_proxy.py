from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import logging
import json
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """
    Load configuration from config.yaml file.
    Returns a dictionary containing the configuration.
    """
    try:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        config_yaml = config_path.read_text()
        config = yaml.safe_load(config_yaml)
        logger.info("Successfully loaded configuration from config.yaml")
        return config
    except Exception as e:
        logger.error(f"Error loading config.yaml: {str(e)}")
        # Return default configuration
        return {
            "primary_backends": [
                {
                    "name": "default",
                    "url": "https://api.openai.com/v1",
                    "model": "",
                }
            ],
            "settings": {"timeout": 60},
        }


# Load configuration
config = load_config()


# Initialize FastAPI app
app = FastAPI(title="OpenAI API Proxy")


# Get the target URL and model from the first backend in the configuration
target_backend = config["primary_backends"][0]
OPENAI_API_BASE = target_backend["url"]
DEFAULT_MODEL = target_backend.get("model", "")

if not OPENAI_API_BASE:
    logger.warning("Backend URL not set in config.yaml, using default value")
    OPENAI_API_BASE = "https://api.openai.com/v1"

# Get timeout from configuration
TIMEOUT = config["settings"].get("timeout", 60)

# Create async HTTP client
http_client = httpx.AsyncClient()


async def call_backend(
    backend: Dict[str, str],
    body: bytes,
    headers: Dict[str, str],
    timeout: float
) -> Dict[str, Any]:
    """
    Helper function to call a single backend and return the response.
    
    Args:
        backend: Dictionary containing backend configuration (name, url, model)
        body: Request body as bytes
        headers: Request headers
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing the response or error details
    """
    try:
        # Parse body and handle model
        json_body = json.loads(body)
        
        # Always use model from config if specified, regardless of request
        if backend["model"]:
            json_body["model"] = backend["model"]
            body = json.dumps(json_body).encode()
        elif "model" not in json_body:
            # If no model in config or request, return error
            return {
                "backend_name": backend["name"],
                "status_code": 400,
                "content": {
                    "error": {
                        "message": "No model specified in config.yaml or request",
                        "type": "invalid_request_error"
                    }
                },
                "is_stream": False
            }

        # Update content length in headers to match actual body length
        headers = headers.copy()  # Make a copy to avoid modifying the original
        headers["content-length"] = str(len(body))
        
        target_url = f"{backend['url']}/chat/completions"
        logger.info(f"Calling backend {backend['name']} at {target_url}")
        
        client = httpx.AsyncClient()
        try:
            response = await client.post(
                target_url,
                content=body,  # Use content instead of json to send exact bytes
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                if json_body.get("stream", False):
                    # For streaming responses, return the response object directly
                    return {
                        "backend_name": backend["name"],
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "content": response,
                        "is_stream": True
                    }
                else:
                    # For non-streaming responses, read the entire content
                    content = await response.aread()
                    try:
                        # Try to parse as JSON
                        if isinstance(content, bytes):
                            content = content.decode()
                        if isinstance(content, str):
                            json_content = json.loads(content)
                        else:
                            json_content = content
                        # Add backend identifier to response
                        json_content["backend"] = backend["name"]
                        return {
                            "backend_name": backend["name"],
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "content": json_content,
                            "is_stream": False
                        }
                    except json.JSONDecodeError:
                        # If not JSON, return raw content
                        return {
                            "backend_name": backend["name"],
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "content": content if isinstance(content, str) else content.decode(),
                            "is_stream": False
                        }
            else:
                content = await response.aread()
                try:
                    # Try to parse error content as JSON
                    if isinstance(content, bytes):
                        content = content.decode()
                    if isinstance(content, str):
                        try:
                            error_content = json.loads(content)
                        except json.JSONDecodeError:
                            error_content = {"error": {"message": content, "type": "backend_error"}}
                    else:
                        error_content = content
                except Exception:
                    error_content = {"error": {"message": str(content), "type": "backend_error"}}
                    
                return {
                    "backend_name": backend["name"],
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": error_content,
                    "is_stream": False
                }
        finally:
            await client.aclose()
            
    except Exception as e:
        logger.error(f"Error calling backend {backend['name']}: {str(e)}")
        return {
            "backend_name": backend["name"],
            "status_code": 500,
            "content": {
                "error": {
                    "message": str(e),
                    "type": "proxy_error"
                }
            },
            "is_stream": False
        }


async def progress_streaming_aggregator(
    valid_backends: List[Dict[str, str]],
    body: bytes,
    headers: Dict[str, str],
    timeout: float,
    separator: str = "\n-------------\n"
) -> AsyncGenerator[bytes, None]:
    logger.info("Starting progress_streaming_aggregator")
    # Send initial SSE event (role event)
    initial_event = {
        "id": "chatcmpl-parallel",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": "parallel-proxy",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    initial_data = f"data: {json.dumps(initial_event)}\n\n".encode()
    logger.info("Yielding initial event: %s", initial_data.decode())
    yield initial_data

    # Create tasks for each backend call
    tasks = [
        asyncio.create_task(call_backend(backend, body, headers, timeout))
        for backend in valid_backends
    ]

    streaming_started = set()
    all_content = []

    # Loop until every task is processed
    while len(streaming_started) < len(tasks):
        for i, task in enumerate(tasks):
            if task.done() and i not in streaming_started:
                streaming_started.add(i)
                try:
                    response = await task
                    logger.info("Processing task %d with status_code %s", i, response.get("status_code"))
                    if response.get("status_code") == 200 and response.get("is_stream", False):
                        content = ""
                        # Process every chunk from this backend’s streaming response
                        async for chunk in response["content"].aiter_bytes():
                            try:
                                chunk_decoded = chunk.decode()
                            except UnicodeDecodeError as e:
                                logger.error("Unicode decoding error for chunk from backend %d: %s; error: %s", i, chunk, str(e))
                                continue

                            logger.info("Received chunk from backend %d: %s", i, chunk_decoded.strip())
                            # Split the chunk into individual SSE events
                            events = chunk_decoded.strip().split("\n\n")
                            for event in events:
                                if not event.strip():
                                    continue
                                if event.startswith("data: "):
                                    # Remove the SSE prefix
                                    event_data = event[6:].strip()
                                    # If this is the [DONE] marker, skip it.
                                    if event_data == "[DONE]":
                                        logger.info("Received [DONE] marker from backend %d", i)
                                        continue
                                    try:
                                        parsed = json.loads(event_data)
                                        logger.debug("Decoded event from backend %d: %s", i, parsed)
                                        # Check if the event contains a streaming content piece
                                        if "choices" in parsed and parsed["choices"]:
                                            delta = parsed["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                c = delta["content"]
                                                content += c
                                                # Build a new SSE event from this piece
                                                stream_event = {
                                                    "id": f"chatcmpl-parallel-{i}",
                                                    "object": "chat.completion.chunk",
                                                    "created": int(asyncio.get_event_loop().time()),
                                                    "model": "parallel-proxy",
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"content": c},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                event_payload = f"data: {json.dumps(stream_event)}\n\n".encode()
                                                logger.info("Yielding streaming event for backend %d: %s", i, event_payload.decode())
                                                yield event_payload
                                    except json.JSONDecodeError as e:
                                        logger.warning("JSON decoding failed for event: %s; error: %s", event_data, e)
                                        continue
                        all_content.append(content)
                    else:
                        logger.info("Backend %d did not return streamable content.", i)
                except Exception as e:
                    logger.error("Error processing backend %d: %s", i, e)
        await asyncio.sleep(0.1)

    # After processing all tasks, send the final aggregated SSE event (or an error if nothing was received)
    if all_content:
        combined_text = f"\n{separator}".join(all_content)
        final_event = {
            "id": "chatcmpl-parallel-final",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": "parallel-proxy",
            "choices": [{
                "index": 0,
                "delta": {"content": combined_text},
                "finish_reason": "stop"
            }]
        }
        final_data = f"data: {json.dumps(final_event)}\n\n".encode()
        logger.info("Yielding final aggregated event: %s", final_data.decode())
        yield final_data
    else:
        error_event = {
            "id": "error",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": "parallel-proxy",
            "choices": [{
                "index": 0,
                "delta": {"content": "Error: All backends failed to provide content"},
                "finish_reason": "error"
            }]
        }
        error_data = f"data: {json.dumps(error_event)}\n\n".encode()
        logger.info("Yielding error event: %s", error_data.decode())
        yield error_data

    done_data = b"data: [DONE]\n\n"
    logger.info("Yielding [DONE] marker.")
    yield done_data


async def stream_with_role(backend_response: httpx.Response, model: str) -> AsyncGenerator[bytes, None]:
    """
    Wraps a backend streaming response to ensure proper SSE format and initial role event.
    
    Args:
        backend_response: The streaming response from the backend
        model: The model name to include in events
        
    Yields:
        Properly formatted SSE events including initial role and [DONE] marker
    """
    logger.info("Starting stream_with_role for model: %s", model)
    # Send initial role event
    initial_event = {
        "id": "chatcmpl-role",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    initial_chunk = f"data: {json.dumps(initial_event)}\n\n".encode()
    logger.info("Yielding initial role event: %s", initial_chunk.decode())
    yield initial_chunk
    
    # Use the backend stream iterator
    backend_iter = backend_response.aiter_bytes()
    saw_done = False
    
    try:
        # Get and process first chunk
        first_chunk = await backend_iter.__anext__()
        decoded_first = first_chunk.decode()
        if first_chunk.strip():
            try:
                # Try to parse the chunk as JSON
                chunk_str = first_chunk.decode()
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[6:]
                first_data = json.loads(chunk_str)
                
                # Skip if it's a role event
                first_delta = first_data.get("choices", [{}])[0].get("delta", {})
                if not (first_delta.get("role") and first_delta.get("content", "") == ""):
                    yield first_chunk
                    if first_chunk.strip() == b"data: [DONE]":
                        saw_done = True
            except Exception as e:
                logger.error("Error decoding first chunk: %s", e)
                decoded_first = "<un-decodable>"
                # If we can't parse it, yield it as-is
                yield first_chunk
                if first_chunk.strip() == b"data: [DONE]":
                    saw_done = True
            logger.info("Received first chunk in stream_with_role: %s", decoded_first)
    except StopAsyncIteration:
        logger.info("No chunks available in backend stream.")

    # Stream remaining chunks
    try:
        async for chunk in backend_iter:
            if chunk.strip():
                try:
                    decoded = chunk.decode()
                except Exception as e:
                    logger.error("Error decoding chunk: %s", e)
                    decoded = "<un-decodable>"
                logger.info("Yielding chunk in stream_with_role: %s", decoded)
                yield chunk
                if decoded.strip() == "data: [DONE]":
                    saw_done = True
    except StopAsyncIteration:
        logger.info("Finished streaming in stream_with_role.")
    
    if not saw_done:
        done_chunk = b"data: [DONE]\n\n"
        logger.info("Yielding [DONE] marker at end of stream_with_role")
        yield done_chunk


@app.post("/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    """
    Proxy for OpenAI's chat completions endpoint that sends requests to all configured backends concurrently.
    Requires a Bearer token in the Authorization header.
    """
    try:
        # Get the raw request body
        body = await request.body()
        json_body = json.loads(body)
        is_streaming = json_body.get("stream", False)

        # Forward all headers except host
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

        # Verify Authorization header is present
        if "authorization" not in {k.lower(): v for k, v in headers.items()}:
            logger.warning("Request received without Authorization header")
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "message": "Authorization header is required",
                            "type": "auth_error",
                        }
                    }
                ),
                status_code=401,
                media_type="application/json",
            )

        # Get all valid backends (with non-empty URLs)
        valid_backends = [b for b in config["primary_backends"] if b.get("url")]
        
        if not valid_backends:
            logger.error("No valid backends configured")
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "message": "No valid backends configured",
                            "type": "configuration_error",
                        }
                    }
                ),
                status_code=500,
                media_type="application/json",
            )

        # Check if model is specified when needed
        if "model" not in json_body:
            # Check if any backend has a model specified
            if not any(b.get("model") for b in valid_backends):
                logger.warning("No model specified in request or config.yaml")
                return Response(
                    content=json.dumps(
                        {
                            "error": {
                                "message": "Model must be specified in request when config.yaml model is blank",
                                "type": "invalid_request_error",
                            }
                        }
                    ),
                    status_code=400,
                    media_type="application/json",
                )

        # Check if parallel backend configuration is present
        has_parallel_config = "iterations" in config and "aggregation" in config["iterations"]
        is_parallel = has_parallel_config and len(valid_backends) > 1

        if is_streaming:
            if is_parallel:
                # For parallel backends with streaming, use progress updates
                separator = config["iterations"]["aggregation"].get("separator", "\n")
                return StreamingResponse(
                    progress_streaming_aggregator(valid_backends, body, headers, float(TIMEOUT), separator),
                    media_type="text/event-stream"
                )
            else:
                # For single backend, allow normal streaming with role injection
                response = await call_backend(valid_backends[0], body, headers, float(TIMEOUT))
                if response["status_code"] == 200 and response.get("is_stream", False):
                    model = json_body.get("model") or valid_backends[0].get("model", "unknown")
                    return StreamingResponse(
                        stream_with_role(response["content"], model),
                        status_code=response["status_code"],
                        headers=response["headers"],
                        media_type="text/event-stream",
                    )
                else:
                    error_content = response.get("content", {})
                    if isinstance(error_content, dict) and "error" in error_content:
                        error_message = error_content["error"].get("message", "Unknown error")
                    elif isinstance(error_content, dict):
                        error_message = str(error_content)
                    else:
                        error_message = str(error_content)
                    return Response(
                        content=json.dumps(
                            {
                                "error": {
                                    "message": f"Backend failed: {error_message}",
                                    "type": "proxy_error",
                                }
                            }
                        ),
                        status_code=response["status_code"],
                        media_type="application/json",
                    )

        # For non-streaming requests, continue with existing logic
        try:
            responses = await asyncio.gather(*[
                call_backend(backend, body, headers, float(TIMEOUT))
                for backend in valid_backends
            ])

            # Check if any backend succeeded
            successful_responses = [r for r in responses if r["status_code"] == 200]
            
            if not successful_responses:
                # If all backends failed, return the first error
                error_response = responses[0]
                error_content = error_response.get("content", {})
                if isinstance(error_content, dict) and "error" in error_content:
                    error_message = error_content["error"].get("message", "Unknown error")
                elif isinstance(error_content, dict):
                    error_message = str(error_content)
                else:
                    error_message = str(error_content)
                return Response(
                    content=json.dumps(
                        {
                            "error": {
                                "message": f"All backends failed. First error: {error_message}",
                                "type": "proxy_error",
                            }
                        }
                    ),
                    status_code=500,
                    media_type="application/json",
                )

            if is_parallel:
                # Combine responses from all successful backends
                try:
                    separator = config["iterations"]["aggregation"].get("separator", "\n")
                    combined_content = separator.join(
                        r["content"]["choices"][0]["message"]["content"]
                        for r in successful_responses
                    )

                    # Sum up usage statistics
                    combined_usage = {
                        "prompt_tokens": sum(r["content"]["usage"]["prompt_tokens"] for r in successful_responses),
                        "completion_tokens": sum(r["content"]["usage"]["completion_tokens"] for r in successful_responses),
                        "total_tokens": sum(r["content"]["usage"]["total_tokens"] for r in successful_responses),
                    }

                    # Create combined response
                    combined_response = {
                        "id": successful_responses[0]["content"]["id"],
                        "object": "chat.completion",
                        "created": successful_responses[0]["content"]["created"],
                        "model": successful_responses[0]["content"]["model"],
                        "system_fingerprint": successful_responses[0]["content"].get("system_fingerprint", ""),
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": combined_content
                            },
                            "logprobs": None,
                            "finish_reason": "stop"
                        }],
                        "usage": combined_usage
                    }

                    return Response(
                        content=json.dumps(combined_response),
                        status_code=200,
                        media_type="application/json",
                    )
                except Exception as e:
                    logger.error(f"Error combining responses: {str(e)}")
                    return Response(
                        content=json.dumps(
                            {
                                "error": {
                                    "message": f"Error combining responses: {str(e)}",
                                    "type": "proxy_error",
                                }
                            }
                        ),
                        status_code=500,
                        media_type="application/json",
                    )
            else:
                # For non-parallel mode, just return the first successful response
                success_response = successful_responses[0]
                
                # Ensure we're sending the correct content type
                content_type = success_response["headers"].get("content-type", "application/json")
                if isinstance(success_response["content"], (dict, list)):
                    content = json.dumps(success_response["content"])
                else:
                    content = success_response["content"]
                    
                # Create response with explicit content length
                response = Response(
                    content=content,
                    status_code=success_response["status_code"],
                    media_type=content_type,
                )
                
                # Copy all headers except those we want to override
                for k, v in success_response["headers"].items():
                    if k.lower() not in {"content-length", "content-type", "transfer-encoding"}:
                        response.headers[k] = v
                        
                return response
        except Exception as e:
            logger.error(f"Error in proxy_chat_completions: {str(e)}")
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "message": f"Error processing request: {str(e)}",
                            "type": "proxy_error",
                        }
                    }
                ),
                status_code=500,
                media_type="application/json",
            )
    except Exception as e:
        logger.error(f"Error in proxy_chat_completions: {str(e)}")
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "message": f"Error processing request: {str(e)}",
                        "type": "proxy_error",
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

    uvicorn.run(app, host="0.0.0.0", port=8000)
