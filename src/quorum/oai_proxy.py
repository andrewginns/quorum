from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import logging
import json
import yaml
import asyncio
import re
import os
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

aggregation_logger = logging.getLogger("aggregation")
aggregation_logger.setLevel(logging.INFO)

log_dir = Path(__file__).parent.parent.parent / "logs"
os.makedirs(log_dir, exist_ok=True)

aggregation_log_file = log_dir / "aggregation.log"
file_handler = logging.FileHandler(str(aggregation_log_file), mode="a")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

aggregation_logger.addHandler(file_handler)
aggregation_logger.propagate = True

try:
    with open(str(aggregation_log_file), "a") as f:
        f.write("Test direct write to log file\n")
    logger.info(f"Successfully wrote to log file at {aggregation_log_file}")
except Exception as e:
    logger.error(f"Failed to write to log file: {str(e)}")


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


class StreamingResponseWrapper:
    """
    Wrapper for httpx Response objects that implements the async iterator protocol
    allowing them to be used with `async for` loops.
    """

    def __init__(self, response):
        self.response = response
        self.aiter_bytes_iter = None

    async def aiter_bytes(self):
        """Original method from httpx Response that yields bytes"""
        async for chunk in self.response.aiter_bytes():
            yield chunk

    def __aiter__(self):
        """Make this object an async iterator"""
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
                    wrapped_response = StreamingResponseWrapper(response)
                    return {
                        "backend_name": backend["name"],
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "content": wrapped_response,
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


async def aggregate_responses(
    source_responses: List[str],
    aggregator_backend: Dict[str, str],
    user_query: str,
    separator: str,
    include_original_query: bool = True,
    query_format: str = "Original query: {query}\n\n",
    include_source_names: bool = False,
    source_label_format: str = "Response from {backend_name}:\n",
    prompt_template: str = "You have received the following responses regarding the user's query:\n\n{responses}\n\nProvide a concise synthesis of these responses.",
    headers: Dict[str, str] = None,
) -> str:
    """
    Send source responses to the aggregator backend for synthesis.

    Args:
        source_responses: List of responses from source backends
        aggregator_backend: Configuration of the aggregator backend
        user_query: The original user query
        separator: Separator to use between source responses
        include_original_query: Whether to include the original query in the prompt
        query_format: Format string for the original query
        include_source_names: Whether to include source backend names
        source_label_format: Format string for source labels
        prompt_template: Template for the aggregator prompt
        headers: Request headers from the original request, used for authorization

    Returns:
        The aggregated response from the aggregator backend
    """
    aggregation_logger.info("Sending responses to aggregator backend")

    formatted_responses = []
    for i, response in enumerate(source_responses):
        if include_source_names and i < len(source_responses):
            backend_name = f"LLM{i + 1}"
            formatted_response = (
                source_label_format.format(backend_name=backend_name) + response
            )
        else:
            formatted_response = response
        formatted_responses.append(formatted_response)

    intermediate_results = separator.join(formatted_responses)

    prompt = ""
    if include_original_query:
        prompt += query_format.format(query=user_query)

    prompt += prompt_template.replace("{responses}", intermediate_results)

    aggregation_logger.info(f"Prompt for aggregator: {prompt}")

    messages = [{"role": "user", "content": prompt}]

    request_body = {
        "model": aggregator_backend.get("model", ""),
        "messages": messages,
        "stream": False,
    }

    # Only include essential headers for the aggregator request
    clean_headers = {}

    if headers:
        # Extract authorization header (prioritizing properly capitalized version)
        if "Authorization" in headers:
            clean_headers["Authorization"] = headers["Authorization"]
        elif "authorization" in headers:
            clean_headers["Authorization"] = headers["authorization"]
        # Fallback to env var if no auth header found
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                clean_headers["Authorization"] = f"Bearer {api_key}"
            else:
                aggregation_logger.error(
                    "No authorization header or OPENAI_API_KEY found"
                )
                return separator.join(source_responses)
    else:
        # Fallback to environment variable if no headers provided
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            clean_headers["Authorization"] = f"Bearer {api_key}"
        else:
            aggregation_logger.error(
                "OPENAI_API_KEY environment variable not set and no headers provided"
            )
            return separator.join(source_responses)

    # Always set content-type to application/json
    clean_headers["Content-Type"] = "application/json"

    aggregation_logger.info(f"Using clean headers for aggregator: {clean_headers}")

    try:
        aggregator_response = await call_backend(
            aggregator_backend, json.dumps(request_body).encode(), clean_headers, 60.0
        )

        if aggregator_response["status_code"] == 200:
            content = aggregator_response["content"]["choices"][0]["message"]["content"]
            aggregation_logger.info(f"Aggregator response: {content}")
            return content
        else:
            aggregation_logger.error(
                f"Aggregator backend failed: {aggregator_response}"
            )
            return separator.join(source_responses)
    except Exception as e:
        aggregation_logger.error(f"Error calling aggregator backend: {str(e)}")
        return separator.join(source_responses)


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
    suppress_individual_responses: bool = False,
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
        suppress_individual_responses: If True, suppress streaming individual LLM responses
                                      and only stream the final aggregated response
    """
    if thinking_tags is None:
        thinking_tags = ["think", "reason", "reasoning", "thought"]

    logger.info("Starting progress_streaming_aggregator")
    aggregation_logger.info("Starting streaming aggregation process")

    try:
        request_data = json.loads(body)
        aggregation_logger.info(f"Request data: {json.dumps(request_data, indent=2)}")
    except Exception as e:
        aggregation_logger.error(f"Error parsing request body: {str(e)}")

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
                        content_gen = response["content"]

                        if not content_gen:
                            logger.error(
                                "Error processing backend %d: content_gen is None", i
                            )
                            continue

                        try:
                            async for chunk in content_gen:
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
                                                "Received [DONE] marker from backend %d",
                                                i,
                                            )
                                            continue
                                        try:
                                            parsed = json.loads(event_data)
                                            if (
                                                "choices" in parsed
                                                and parsed["choices"]
                                            ):
                                                delta = parsed["choices"][0].get(
                                                    "delta", {}
                                                )
                                                if "content" in delta:
                                                    c = delta["content"]
                                                    if hide_intermediate_think:
                                                        safe_text = tag_filters[i].feed(
                                                            c
                                                        )
                                                    else:
                                                        safe_text = c
                                                    content += safe_text
                                                    if (
                                                        safe_text
                                                        and not suppress_individual_responses
                                                    ):
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
                        except AttributeError as e:
                            logger.error(
                                "Error processing backend %d: %s - The response object doesn't support async iteration. Make sure it's properly wrapped.",
                                i,
                                str(e),
                            )
                            continue
                        except Exception as e:
                            logger.error(
                                "Unexpected error iterating over content from backend %d: %s",
                                i,
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
                                                if (
                                                    safe_text
                                                    and not suppress_individual_responses
                                                ):
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
                    aggregation_logger.error(f"Error processing backend {i}: {str(e)}")
        await asyncio.sleep(0.1)

    aggregation_logger.info(
        "Content collected from backends (may be empty if backends failed):"
    )
    for i, content in enumerate(all_content):
        if content:
            aggregation_logger.info(f"Backend {i} content: {content}")
        else:
            aggregation_logger.info(f"Backend {i} content: No content received")

    # Final SSE event if skip_final_aggregation is False
    if not skip_final_aggregation:
        filtered_contents = [
            strip_thinking_tags(text, thinking_tags, hide_intermediate=hide_final_think)
            for text in all_content
            if text
        ]
        if filtered_contents:
            aggregation_logger.info(
                "Individual LLM responses for streaming aggregation:"
            )
            for i, content in enumerate(filtered_contents):
                aggregation_logger.info(f"LLM {i + 1} streaming response: {content}")

            aggregator_strategy = config.get("strategy", {}).get("aggregate", {})

            source_backends_config = aggregator_strategy.get("source_backends", "all")
            if source_backends_config == "all":
                source_backends_names = [
                    b.get("name") for b in config.get("primary_backends", [])
                ]
            else:
                source_backends_names = source_backends_config

            aggregator_backend_name = aggregator_strategy.get("aggregator_backend")

            if aggregator_backend_name:
                aggregator_backend = None
                for backend in config.get("primary_backends", []):
                    if backend.get("name") == aggregator_backend_name:
                        aggregator_backend = backend
                        break

                if aggregator_backend:
                    try:
                        body_json = json.loads(body)
                        user_query = ""
                        if "messages" in body_json and body_json["messages"]:
                            for msg in body_json["messages"]:
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
                        aggregation_logger.info(
                            "Used aggregator backend for final response"
                        )
                    except Exception as e:
                        aggregation_logger.error(f"Error during aggregation: {str(e)}")
                        combined_text = f"\n{separator}".join(filtered_contents)
                else:
                    aggregation_logger.error(
                        f"Aggregator backend {aggregator_backend_name} not found"
                    )
                    combined_text = f"\n{separator}".join(filtered_contents)
            else:
                combined_text = f"\n{separator}".join(filtered_contents)

            aggregation_logger.info(
                f"Final aggregated streaming content: {combined_text}"
            )

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

        # Check for Authorization header and add from environment if missing
        headers_lower = {k.lower(): k for k in headers}
        if "authorization" not in headers_lower:
            logger.warning(
                "Missing Authorization header, checking environment variable"
            )
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                logger.info("Using API key from environment variable")
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                logger.error("No API key in header or environment variable")
                return Response(
                    content=json.dumps(
                        {
                            "error": {
                                "message": "Authorization header is required and OPENAI_API_KEY environment variable is not set",
                                "type": "auth_error",
                            }
                        }
                    ),
                    status_code=401,
                    media_type="application/json",
                )
        # Ensure consistent header capitalization
        elif "authorization" in headers_lower and "Authorization" not in headers:
            auth_key = headers_lower["authorization"]
            headers["Authorization"] = headers[auth_key]
            if auth_key != "Authorization":
                del headers[auth_key]

        # Ensure Content-Type is set properly
        if "content-type" not in headers_lower:
            headers["Content-Type"] = "application/json"

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
                suppress_individual_responses = aggregator_config.get(
                    "suppress_individual_responses", False
                )
                if "suppress_individual_responses" in json_body:
                    suppress_individual_responses = json_body.get(
                        "suppress_individual_responses"
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
                        suppress_individual_responses,
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
                suppress_individual_responses = aggregator_config.get(
                    "suppress_individual_responses", False
                )
                if "suppress_individual_responses" in json_body:
                    suppress_individual_responses = json_body.get(
                        "suppress_individual_responses"
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

                    aggregation_logger.info("Individual LLM responses for aggregation:")
                    for i, content in enumerate(processed_contents):
                        aggregation_logger.info(f"LLM {i + 1} response: {content}")

                    aggregator_strategy = config.get("strategy", {}).get(
                        "aggregate", {}
                    )

                    source_backends_config = aggregator_strategy.get(
                        "source_backends", "all"
                    )
                    if source_backends_config == "all":
                        source_backends_names = [
                            b.get("name") for b in config.get("primary_backends", [])
                        ]
                    else:
                        source_backends_names = source_backends_config

                    aggregator_backend_name = aggregator_strategy.get(
                        "aggregator_backend"
                    )

                    if aggregator_backend_name:
                        aggregator_backend = None
                        for backend in config.get("primary_backends", []):
                            if backend.get("name") == aggregator_backend_name:
                                aggregator_backend = backend
                                break

                        if aggregator_backend:
                            try:
                                user_query = ""
                                if "messages" in json_body and json_body["messages"]:
                                    for msg in json_body["messages"]:
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

                                combined_content = await aggregate_responses(
                                    processed_contents,
                                    aggregator_backend,
                                    user_query,
                                    aggregator_strategy.get(
                                        "intermediate_separator", "\n\n---\n\n"
                                    ),
                                    aggregator_strategy.get(
                                        "include_original_query", True
                                    ),
                                    aggregator_strategy.get(
                                        "query_format", "Original query: {query}\n\n"
                                    ),
                                    aggregator_strategy.get(
                                        "include_source_names", False
                                    ),
                                    aggregator_strategy.get(
                                        "source_label_format",
                                        "Response from {backend_name}:\n",
                                    ),
                                    prompt_template,
                                    headers,
                                )
                                aggregation_logger.info(
                                    "Used aggregator backend for final response"
                                )
                            except Exception as e:
                                aggregation_logger.error(
                                    f"Error during aggregation: {str(e)}"
                                )
                                combined_content = separator.join(processed_contents)
                        else:
                            aggregation_logger.error(
                                f"Aggregator backend {aggregator_backend_name} not found"
                            )
                            combined_content = separator.join(processed_contents)
                    else:
                        combined_content = separator.join(processed_contents)

                    if suppress_individual_responses:
                        aggregation_logger.info(
                            "Individual responses suppressed, only final aggregated response will be shown"
                        )

                    aggregation_logger.info(
                        f"Final aggregated content: {combined_content}"
                    )

                    logger.info("Logged aggregation data to aggregation.log")

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
