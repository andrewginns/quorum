from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import logging
import json
import yaml
from pathlib import Path

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

        # Parse JSON to check for streaming and set default model if not provided
        json_body = json.loads(body)
        is_streaming = json_body.get("stream", False)

        # Set model based on config.yaml if not provided in request
        # Only use the model from request if config.yaml model is blank
        if "model" not in json_body:
            if DEFAULT_MODEL:  # If config.yaml has a non-blank model
                json_body["model"] = DEFAULT_MODEL
                body = json.dumps(json_body).encode()
            else:
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
        elif (
            DEFAULT_MODEL and DEFAULT_MODEL != ""
        ):  # Override request model if config model is set
            json_body["model"] = DEFAULT_MODEL
            body = json.dumps(json_body).encode()

        # Construct target URL using the first backend from config.yaml
        target_url = f"{OPENAI_API_BASE}/chat/completions"
        logger.info(
            f"Forwarding request to: {target_url} with model: {json_body['model']}"
        )

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
                target_url, content=body, headers=headers, timeout=float(TIMEOUT)
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
