"""Aggregation strategies for Quorum proxy."""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, AsyncGenerator

from .config import config, aggregation_logger
from .backends import call_backend
from .utils import strip_thinking_tags, ThinkingTagFilter

logger = logging.getLogger(__name__)


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

    clean_headers = {}

    if headers:
        if "Authorization" in headers:
            clean_headers["Authorization"] = headers["Authorization"]
        elif "authorization" in headers:
            clean_headers["Authorization"] = headers["authorization"]
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
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            clean_headers["Authorization"] = f"Bearer {api_key}"
        else:
            aggregation_logger.error(
                "OPENAI_API_KEY environment variable not set and no headers provided"
            )
            return separator.join(source_responses)

    clean_headers["Content-Type"] = "application/json"

    aggregation_logger.info(f"Using clean headers for aggregator: {clean_headers}")

    try:
        aggregator_response = await call_backend(
            aggregator_backend, json.dumps(request_body).encode(), clean_headers, 60.0
        )

        if aggregator_response["status_code"] == 200:
            if isinstance(aggregator_response["content"], dict) and "choices" in aggregator_response["content"]:
                content = aggregator_response["content"]["choices"][0]["message"]["content"]
                aggregation_logger.info(f"Aggregator response (direct dict): {content}")
                return content
            
            elif hasattr(aggregator_response["content"], "json"):
                try:
                    json_content = aggregator_response["content"].json()
                    aggregation_logger.info(f"Mock response json type: {type(json_content)}")
                    
                    if isinstance(json_content, dict) and "choices" in json_content:
                        content = json_content["choices"][0]["message"]["content"]
                        aggregation_logger.info(f"Mock response json content: {content}")
                        return content
                except Exception as json_err:
                    aggregation_logger.error(f"Error extracting json from MockResponse: {str(json_err)}")
            
            elif hasattr(aggregator_response["content"], "_content"):
                try:
                    mock_content = aggregator_response["content"]._content
                    aggregation_logger.info(f"Mock content type: {type(mock_content)}")
                    
                    if isinstance(mock_content, dict) and "choices" in mock_content:
                        content = mock_content["choices"][0]["message"]["content"]
                        aggregation_logger.info(f"Mock response content: {content}")
                        return content
                    elif isinstance(mock_content, bytes):
                        content_str = mock_content.decode()
                        try:
                            content_json = json.loads(content_str)
                            if "choices" in content_json:
                                content = content_json["choices"][0]["message"]["content"]
                                aggregation_logger.info(f"Mock response decoded content: {content}")
                                return content
                        except json.JSONDecodeError:
                            pass
                except Exception as e:
                    aggregation_logger.error(f"Error extracting content from MockResponse: {str(e)}")
            
            elif aggregator_response.get("is_stream", False) and hasattr(aggregator_response["content"], "aiter_bytes"):
                aggregation_logger.info("Received streaming response from aggregator")
                content_buffer = ""
                try:
                    content_gen = aggregator_response["content"]
                    async for chunk in content_gen:
                        chunk_decoded = chunk.decode()
                        events = chunk_decoded.strip().split("\n\n")
                        for event in events:
                            if event.startswith("data: "):
                                event_data = event[6:].strip()
                                if event_data == "[DONE]":
                                    continue
                                try:
                                    parsed = json.loads(event_data)
                                    if "choices" in parsed and parsed["choices"]:
                                        delta = parsed["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            content_buffer += delta["content"]
                                except json.JSONDecodeError:
                                    continue
                    aggregation_logger.info(f"Aggregator streaming response: {content_buffer}")
                    return content_buffer
                except Exception as e:
                    aggregation_logger.error(f"Error processing streaming response: {str(e)}")
                    return separator.join(source_responses)
            
            else:
                aggregation_logger.error(f"Unexpected response format: {aggregator_response}")
                return separator.join(source_responses)
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
        asyncio.create_task(
            call_backend(backend, body, headers, timeout)
        ) for backend in valid_backends
    ]

    all_content = [""] * len(valid_backends)

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error("Backend %d failed with exception: %s", i, response)
            aggregation_logger.error(f"Backend {i} failed with exception: {str(response)}")
        else:
            try:
                content = ""
                if response["status_code"] == 200:
                    if response.get("is_stream", False):
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
                                                    if safe_text:
                                                        logger.info(
                                                            "Collected streaming content from backend %d: %s",
                                                            i,
                                                            safe_text,
                                                        )
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
                        
                        flushed = tag_filters[i].flush()
                        content += flushed
                        all_content[i] = content
                    else:
                        logger.info("Backend %d did not return streamable content.", i)
                else:
                    logger.info("Backend %d returned non-200 status code: %d", i, response["status_code"])
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
                            "content": "Error: All backends failed"
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            error_data = f"data: {json.dumps(error_event)}\n\n".encode()
            logger.info("Yielding error event: %s", error_data.decode())
            yield error_data

    done_data = b"data: [DONE]\n\n"
    logger.info("Yielding [DONE] marker.")
    yield done_data
