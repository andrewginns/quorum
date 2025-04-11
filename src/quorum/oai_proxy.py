from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import logging
import json
import yaml
import asyncio
import re
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


def strip_thinking_tags(
    content: str, tags: List[str], hide_intermediate: bool = True
) -> str:
    """
    Strip reasoning tags from content based on configuration.

    Args:
        content: The text content to process
        tags: List of tag names to look for (e.g. ["think", "reason"])
        hide_intermediate: Whether to remove the text inside these tags

    Returns:
        Processed content with specified tags removed, if hide_intermediate is True
    """
    if not hide_intermediate:
        return content

    tag_pattern = "|".join(tags)
    pattern = f"<({tag_pattern})>.*?</\\1>"
    return re.sub(pattern, "", content, flags=re.IGNORECASE | re.DOTALL).strip()


async def call_backend(
    backend: Dict[str, str], body: bytes, headers: Dict[str, str], timeout: float
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
                        "type": "invalid_request_error",
                    }
                },
                "is_stream": False,
            }

        # Update content length in headers to match new body length
        headers = headers.copy()
        headers["content-length"] = str(len(body))

        target_url = f"{backend['url']}/chat/completions"
        logger.info(f"Calling backend {backend['name']} at {target_url}")

        client = httpx.AsyncClient()
        try:
            response = await client.post(
                target_url,
                content=body,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code == 200:
                if json_body.get("stream", False):
                    # For streaming responses
                    return {
                        "backend_name": backend["name"],
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "content": response,
                        "is_stream": True,
                    }
                else:
                    # For non-streaming responses
                    content = await response.aread()
                    try:
                        if isinstance(content, bytes):
                            content = content.decode()
                        json_content = json.loads(content)
                        # Add backend identifier
                        json_content["backend"] = backend["name"]
                        return {
                            "backend_name": backend["name"],
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "content": json_content,
                            "is_stream": False,
                        }
                    except json.JSONDecodeError:
                        # If not JSON, return raw
                        return {
                            "backend_name": backend["name"],
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "content": content
                            if isinstance(content, str)
                            else content.decode(),
                            "is_stream": False,
                        }
            else:
                # Handle error responses
                content = await response.aread()
                if isinstance(content, bytes):
                    content = content.decode()
                try:
                    error_content = json.loads(content)
                except json.JSONDecodeError:
                    error_content = {
                        "error": {"message": content, "type": "backend_error"}
                    }
                return {
                    "backend_name": backend["name"],
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": error_content,
                    "is_stream": False,
                }
        finally:
            await client.aclose()

    except Exception as e:
        logger.error(f"Error calling backend {backend['name']}: {str(e)}")
        return {
            "backend_name": backend["name"],
            "status_code": 500,
            "content": {"error": {"message": str(e), "type": "proxy_error"}},
            "is_stream": False,
        }


class ThinkingTagFilter:
    """
    Incrementally removes content wrapped inside thinking tags (and their nested occurrences),
    while buffering partial tag input. If hide_intermediate_think is enabled, text in these
    tags is withheld from the streaming output. Final aggregator can also remove them
    depending on hide_final_think.
    """

    def __init__(self, tags):
        self.allowed_tags = [tag.lower() for tag in tags]
        tag_pattern = "|".join(self.allowed_tags)
        self.open_regex = re.compile(f"<({tag_pattern})>", re.IGNORECASE)
        self.close_regex = re.compile(f"</({tag_pattern})>", re.IGNORECASE)
        self.buffer = ""
        self.thinking_depth = 0

    def feed(self, text: str) -> str:
        """
        Add new text to the filter and return "safe" text outside thinking tags.
        """
        self.buffer += text
        output = ""

        while True:
            # CASE 1: Not inside any thinking block
            if self.thinking_depth == 0:
                open_match = self.open_regex.search(self.buffer)
                if open_match:
                    output += self.buffer[: open_match.start()]
                    self.buffer = self.buffer[open_match.start() :]
                    m = self.open_regex.match(self.buffer)
                    if m:
                        self.thinking_depth = 1
                        self.buffer = self.buffer[m.end() :]
                        continue
                    else:
                        pos = self.buffer.rfind("<")
                        if pos != -1:
                            candidate = self.buffer[pos:]
                            for tag in self.allowed_tags:
                                full_tag = f"<{tag}>"
                                if full_tag.startswith(candidate.lower()):
                                    output += self.buffer[:pos]
                                    self.buffer = self.buffer[pos:]
                                    return output
                        output += self.buffer
                        self.buffer = ""
                        break
                else:
                    pos = self.buffer.rfind("<")
                    if pos != -1:
                        candidate = self.buffer[pos:]
                        valid_partial = False
                        for tag in self.allowed_tags:
                            full_tag = f"<{tag}>"
                            if full_tag.startswith(candidate.lower()):
                                valid_partial = True
                                break
                        if valid_partial:
                            output += self.buffer[:pos]
                            self.buffer = self.buffer[pos:]
                            break
                    output += self.buffer
                    self.buffer = ""
                    break

            # CASE 2: Inside one or more thinking blocks
            else:
                next_open = self.open_regex.search(self.buffer)
                next_close = self.close_regex.search(self.buffer)
                if not next_close and not next_open:
                    # Wait for more text
                    break
                if next_close and (
                    not next_open or next_close.start() < next_open.start()
                ):
                    self.buffer = self.buffer[next_close.end() :]
                    self.thinking_depth -= 1
                    if self.thinking_depth < 0:
                        self.thinking_depth = 0
                    continue
                elif next_open:
                    self.buffer = self.buffer[next_open.end() :]
                    self.thinking_depth += 1
                    continue
                else:
                    break

        return output

    def flush(self) -> str:
        """
        Flush any remaining "safe" text. If still inside a thinking block,
        discard that partial content.
        """
        if self.thinking_depth > 0:
            self.buffer = ""
            return ""
        else:
            pos = self.buffer.rfind("<")
            if pos != -1:
                candidate = self.buffer[pos:]
                for tag in self.allowed_tags:
                    full_tag = f"<{tag}>"
                    if full_tag.startswith(candidate.lower()):
                        self.buffer = self.buffer[:pos]
                        break
            out = self.buffer
            self.buffer = ""
            return out


async def progress_streaming_aggregator(
    valid_backends: List[Dict[str, str]],
    body: bytes,
    headers: Dict[str, str],
    timeout: float,
    separator: str = "\n-------------\n",
    hide_intermediate_think: bool = True,
    hide_final_think: bool = False,
    thinking_tags: List[str] = None,
    skip_final_aggregation: bool = False,
) -> AsyncGenerator[bytes, None]:
    """
    Aggregates streaming responses from multiple backends with progress updates.

    Args:
        valid_backends: List of configured backends
        body: Original request body
        headers: Request headers
        timeout: Per-request timeout
        separator: A string to separate multiple backend responses
        hide_intermediate_think: Whether to remove <think>... from the streaming text
        hide_final_think: Whether to remove <think>... in the final aggregated content
        thinking_tags: Which tags to treat as 'thinking' tags
        skip_final_aggregation: If True, skip sending the final combined SSE chunk
    """
    if thinking_tags is None:
        thinking_tags = ["think", "reason", "reasoning", "thought"]

    logger.info("Starting progress_streaming_aggregator")

    # Send an initial SSE role event
    initial_event = {
        "id": "chatcmpl-parallel",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": "parallel-proxy",
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    initial_data = f"data: {json.dumps(initial_event)}\n\n".encode()
    logger.info("Yielding initial event: %s", initial_data.decode())
    yield initial_data

    tag_filters = {
        i: ThinkingTagFilter(thinking_tags) for i in range(len(valid_backends))
    }

    tasks = [
        asyncio.create_task(call_backend(backend, body, headers, timeout))
        for backend in valid_backends
    ]
    streaming_started = set()
    all_content = ["" for _ in valid_backends]

    while len(streaming_started) < len(tasks):
        for i, task in enumerate(tasks):
            if task.done() and i not in streaming_started:
                streaming_started.add(i)
                try:
                    response = await task
                    logger.info(
                        "Processing task %d with status_code %s",
                        i,
                        response.get("status_code"),
                    )
                    if response.get("status_code") == 200 and response.get(
                        "is_stream", False
                    ):
                        content = ""
                        async for chunk in response["content"].aiter_bytes():
                            try:
                                chunk_decoded = chunk.decode()
                            except UnicodeDecodeError as e:
                                logger.error(
                                    "Unicode decoding error for backend %d: %s; error: %s",
                                    i,
                                    chunk,
                                    str(e),
                                )
                                continue

                            logger.info(
                                "Received chunk from backend %d: %s",
                                i,
                                chunk_decoded.strip(),
                            )
                            events = chunk_decoded.strip().split("\n\n")
                            for event in events:
                                if not event.strip():
                                    continue
                                if event.startswith("data: "):
                                    event_data = event[6:].strip()
                                    if event_data == "[DONE]":
                                        logger.info(
                                            "Received [DONE] marker from backend %d", i
                                        )
                                        continue
                                    try:
                                        parsed = json.loads(event_data)
                                        if "choices" in parsed and parsed["choices"]:
                                            delta = parsed["choices"][0].get(
                                                "delta", {}
                                            )
                                            if "content" in delta:
                                                c = delta["content"]
                                                if hide_intermediate_think:
                                                    safe_text = tag_filters[i].feed(c)
                                                else:
                                                    safe_text = c
                                                content += safe_text
                                                if safe_text:
                                                    stream_event = {
                                                        "id": f"chatcmpl-parallel-{i}",
                                                        "object": "chat.completion.chunk",
                                                        "created": int(
                                                            asyncio.get_event_loop().time()
                                                        ),
                                                        "model": "parallel-proxy",
                                                        "choices": [
                                                            {
                                                                "index": 0,
                                                                "delta": {
                                                                    "content": safe_text
                                                                },
                                                                "finish_reason": None,
                                                            }
                                                        ],
                                                    }
                                                    event_payload = f"data: {json.dumps(stream_event)}\n\n".encode()
                                                    logger.info(
                                                        "Yielding streaming event for backend %d: %s",
                                                        i,
                                                        event_payload.decode(),
                                                    )
                                                    yield event_payload
                                    except json.JSONDecodeError as e:
                                        logger.warning(
                                            "JSON decoding failed for event: %s; error: %s",
                                            event_data,
                                            e,
                                        )
                                        continue
                        flushed = tag_filters[i].flush()
                        content += flushed
                        all_content[i] = content
                    else:
                        logger.info("Backend %d did not return streamable content.", i)
                except Exception as e:
                    logger.error("Error processing backend %d: %s", i, e)
        await asyncio.sleep(0.1)

    # Final SSE event if skip_final_aggregation is False
    if not skip_final_aggregation:
        filtered_contents = [
            strip_thinking_tags(text, thinking_tags, hide_intermediate=hide_final_think)
            for text in all_content
            if text
        ]
        if filtered_contents:
            combined_text = f"\n{separator}".join(filtered_contents)
            final_event = {
                "id": "chatcmpl-parallel-final",
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": "parallel-proxy",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": combined_text},
                        "finish_reason": "stop",
                    }
                ],
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "Error: All backends failed to provide content"
                        },
                        "finish_reason": "error",
                    }
                ],
            }
            error_data = f"data: {json.dumps(error_event)}\n\n".encode()
            logger.info("Yielding error event: %s", error_data.decode())
            yield error_data

    done_data = b"data: [DONE]\n\n"
    logger.info("Yielding [DONE] marker.")
    yield done_data


async def stream_with_role(
    backend_response: httpx.Response, model: str
) -> AsyncGenerator[bytes, None]:
    """
    Wraps a backend streaming response to ensure proper SSE format and initial role event.
    """
    logger.info("Starting stream_with_role for model: %s", model)
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
    logger.info("Yielding initial role event: %s", initial_chunk.decode())
    yield initial_chunk

    backend_iter = backend_response.aiter_bytes()
    saw_done = False

    try:
        first_chunk = await backend_iter.__anext__()
        decoded_first = first_chunk.decode()
        if first_chunk.strip():
            try:
                chunk_str = first_chunk.decode()
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[6:]
                first_data = json.loads(chunk_str)
                first_delta = first_data.get("choices", [{}])[0].get("delta", {})
                # If it's just a role event with empty content, skip sending it again
                if not (
                    first_delta.get("role") and first_delta.get("content", "") == ""
                ):
                    yield first_chunk
                    if first_chunk.strip() == b"data: [DONE]":
                        saw_done = True
            except Exception as e:
                logger.error("Error decoding first chunk: %s", e)
                decoded_first = "<un-decodable>"
                yield first_chunk
                if first_chunk.strip() == b"data: [DONE]":
                    saw_done = True
            logger.info("Received first chunk in stream_with_role: %s", decoded_first)
    except StopAsyncIteration:
        logger.info("No chunks available in backend stream.")

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
    Primary proxy endpoint for chat completions:
    - Routes requests to multiple backends
    - Supports streaming or non-streaming
    - Aggregates responses according to config
    """
    try:
        body = await request.body()
        json_body = json.loads(body)
        is_streaming = json_body.get("stream", False)

        # Forward all headers except host
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

        # Verify Authorization header
        if "authorization" not in {k.lower(): v for k, v in headers.items()}:
            logger.warning("Missing Authorization header")
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

        if "model" not in json_body:
            if not any(b.get("model") for b in valid_backends):
                logger.warning("No model specified in request or config.yaml")
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

        # Check for parallel config
        has_parallel_config = "iterations" in config and "strategy" in config
        is_parallel = has_parallel_config and len(valid_backends) > 1

        if is_streaming:
            if is_parallel:
                # Read aggregator settings from config
                aggregator_strategy = (
                    config.get("iterations", {})
                    .get("aggregation", {})
                    .get("strategy", "concatenate")
                )
                aggregator_config = config.get("strategy", {}).get(
                    aggregator_strategy, {}
                )

                separator = aggregator_config.get("separator", "\n")
                hide_intermediate_think = aggregator_config.get(
                    "hide_intermediate_think", True
                )
                hide_final_think = aggregator_config.get("hide_final_think", False)
                thinking_tags = aggregator_config.get(
                    "thinking_tags", ["think", "reason", "reasoning", "thought"]
                )
                skip_final_aggregation = aggregator_config.get(
                    "skip_final_aggregation", False
                )

                return StreamingResponse(
                    progress_streaming_aggregator(
                        valid_backends,
                        body,
                        headers,
                        float(TIMEOUT),
                        separator,
                        hide_intermediate_think,
                        hide_final_think,
                        thinking_tags,
                        skip_final_aggregation,
                    ),
                    media_type="text/event-stream",
                )
            else:
                # Single backend streaming
                response = await call_backend(
                    valid_backends[0], body, headers, float(TIMEOUT)
                )
                if response["status_code"] == 200 and response.get("is_stream", False):
                    model = json_body.get("model") or valid_backends[0].get(
                        "model", "unknown"
                    )
                    return StreamingResponse(
                        stream_with_role(response["content"], model),
                        status_code=response["status_code"],
                        headers=response["headers"],
                        media_type="text/event-stream",
                    )
                else:
                    error_content = response.get("content", {})
                    if isinstance(error_content, dict) and "error" in error_content:
                        error_message = error_content["error"].get(
                            "message", "Unknown error"
                        )
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

        # Non-streaming path
        try:
            responses = await asyncio.gather(
                *[
                    call_backend(backend, body, headers, float(TIMEOUT))
                    for backend in valid_backends
                ]
            )
            successful_responses = [r for r in responses if r["status_code"] == 200]

            if not successful_responses:
                error_response = responses[0]
                error_content = error_response.get("content", {})
                if isinstance(error_content, dict) and "error" in error_content:
                    error_message = error_content["error"].get(
                        "message", "Unknown error"
                    )
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
                # Read aggregator settings from config for parallel mode
                aggregator_strategy = (
                    config.get("iterations", {})
                    .get("aggregation", {})
                    .get("strategy", "concatenate")
                )
                aggregator_config = config.get("strategy", {}).get(
                    aggregator_strategy, {}
                )

                separator = aggregator_config.get("separator", "\n")
                hide_intermediate_think = aggregator_config.get(
                    "hide_intermediate_think", True
                )
                hide_final_think = aggregator_config.get("hide_final_think", False)
                thinking_tags = aggregator_config.get(
                    "thinking_tags", ["think", "reason", "reasoning", "thought"]
                )

                # Combine responses
                try:
                    processed_contents = []
                    for r in successful_responses:
                        content = r["content"]["choices"][0]["message"]["content"]
                        processed_content = strip_thinking_tags(
                            content, thinking_tags, hide_intermediate=hide_final_think
                        )
                        processed_contents.append(processed_content)

                    combined_content = separator.join(processed_contents)

                    # Sum up usage
                    combined_usage = {
                        "prompt_tokens": sum(
                            r["content"]["usage"]["prompt_tokens"]
                            for r in successful_responses
                        ),
                        "completion_tokens": sum(
                            r["content"]["usage"]["completion_tokens"]
                            for r in successful_responses
                        ),
                        "total_tokens": sum(
                            r["content"]["usage"]["total_tokens"]
                            for r in successful_responses
                        ),
                    }

                    combined_response = {
                        "id": successful_responses[0]["content"]["id"],
                        "object": "chat.completion",
                        "created": successful_responses[0]["content"]["created"],
                        "model": successful_responses[0]["content"]["model"],
                        "system_fingerprint": successful_responses[0]["content"].get(
                            "system_fingerprint", ""
                        ),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": combined_content,
                                },
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": combined_usage,
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
                # Non-parallel: return first successful response
                success_response = successful_responses[0]
                content_type = success_response["headers"].get(
                    "content-type", "application/json"
                )
                if isinstance(success_response["content"], (dict, list)):
                    content = json.dumps(success_response["content"])
                else:
                    content = success_response["content"]

                response = Response(
                    content=content,
                    status_code=success_response["status_code"],
                    media_type=content_type,
                )

                for k, v in success_response["headers"].items():
                    if k.lower() not in {
                        "content-length",
                        "content-type",
                        "transfer-encoding",
                    }:
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

    uvicorn.run(app, host="0.0.0.0", port=8006)
