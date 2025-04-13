"""Backend handling for Quorum proxy."""

import json
import logging
import httpx
from typing import Dict, Any

from .streaming import StreamingResponseWrapper

logger = logging.getLogger(__name__)


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

        if "stream" in json_body:
            json_body["stream"] = True
            
        if backend["model"]:
            json_body["model"] = backend["model"]
            body = json.dumps(json_body).encode()
        elif "model" not in json_body:
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

        headers = headers.copy() if headers else {}
        
        headers["content-length"] = str(len(body))
        
        if "Authorization" in headers:
            pass
        elif "authorization" in headers:
            headers["Authorization"] = headers["authorization"]
        else:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
        logger.info(f"Using headers for backend call: {headers}")

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
                    content = await response.aread()
                    try:
                        if isinstance(content, bytes):
                            content = content.decode()
                        json_content = json.loads(content)
                        json_content["backend"] = backend["name"]
                        return {
                            "backend_name": backend["name"],
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "content": json_content,
                            "is_stream": False,
                        }
                    except json.JSONDecodeError:
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
