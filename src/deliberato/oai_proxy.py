from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import os
import logging
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="OpenAI API Proxy")

# Get OpenAI API base URL from environment variable, default to official API
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
if not OPENAI_API_BASE:
    logger.warning("OPENAI_API_BASE not set in .env file, using default value")
    OPENAI_API_BASE = "https://api.openai.com/v1"

# Create async HTTP client
http_client = httpx.AsyncClient()


@app.post("/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    """
    Transparent proxy for OpenAI's chat completions endpoint.
    Forwards the request exactly as received and returns the response as-is.

    Authentication:
    - Requires a Bearer token in the Authorization header
    - The token should be your OpenAI API key
    - Header format: "Authorization: Bearer sk-..."
    - The token is forwarded as-is to the OpenAI API (not stored or validated)
    - No token validation is performed by the proxy
    """
    try:
        # Get the raw request body
        body = await request.body()

        # Parse JSON to check for streaming
        json_body = json.loads(body)
        is_streaming = json_body.get("stream", False)

        # Construct target URL
        target_url = f"{OPENAI_API_BASE}/chat/completions"

        # Forward all headers except host
        # This includes the Authorization header containing the OpenAI API key
        # The header is forwarded exactly as received, maintaining the security context
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

        # Make the request to OpenAI
        async with httpx.AsyncClient() as client:
            response = await client.post(
                target_url, content=body, headers=headers, timeout=60.0
            )

            if is_streaming:
                # For streaming responses, we need to return a StreamingResponse
                return StreamingResponse(
                    response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="text/event-stream",
                )
            else:
                # For regular responses, return the response as-is
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type"),
                )

    except Exception as e:
        logger.error(f"Error in proxy: {str(e)}")
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "message": "An error occurred while processing your request",
                        "type": "proxy_error",
                        "details": str(e),
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
