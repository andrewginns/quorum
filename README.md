# OpenAI API Proxy

A transparent proxy for OpenAI's chat completions API endpoint. This proxy captures all incoming requests, forwards them to OpenAI's API, and returns the responses exactly as received.

## Features

- Transparent proxying of `/chat/completions` endpoint
- Support for both streaming and non-streaming responses
- Multiple configurable backend endpoints via config.json
- Configurable model overrides per backend
- Configurable request timeouts
- Secure authentication forwarding
- Health check endpoint
- Proper error handling and logging

## Requirements

- Python 3.13+
- `uv` (https://github.com/astral-sh/uv)

## Installation

1. Clone the repository

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv install
   ```

## Configuration

Configure the proxy by editing `config.json`:

```json
{
  "primary_backends": [
    { 
      "name": "LLM1",
      "url": "https://api.openai.com/v1",  # Primary OpenAI API endpoint
      "model": ""  # Optional: Set to override model in all requests
    },
    { 
      "name": "LLM2",  # Additional backend configuration
      "url": "",  # Alternative API endpoint
      "model": ""  # Optional model override
    }
  ],
  "settings": {
    "timeout": 30  # Request timeout in seconds
  }
}
```

Configuration options:
- `primary_backends`: List of backend configurations
  - `name`: Identifier for the backend
  - `url`: Base URL for the API endpoint (default for LLM1: https://api.openai.com/v1)
  - `model`: If set, overrides the model parameter in all requests to this backend
- `settings`:
  - `timeout`: Request timeout in seconds

## Running the Proxy

For development with auto-reload:
```bash
uv run dev
```

For production:
```bash
uv run start
```

## Usage

The proxy exposes the following endpoints:

### Chat Completions

```bash
curl http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",  # Required if not set in backend config
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Error Handling

The proxy includes error handling for:
- Network errors
- Invalid requests
- Timeouts
- Server errors

Errors are logged and returned to the client with appropriate status codes and error messages.

## Authentication

The proxy uses Bearer token authentication:
- Your OpenAI API key must be provided in the Authorization header of each request
- The proxy forwards the Authorization header unchanged to the OpenAI API
- API keys are never logged or stored by the proxy
- Never commit API keys to version control

## Security Considerations

Authentication and API Keys:
- Your OpenAI API key must be provided in the Authorization header of each request
- The proxy forwards the Authorization header unchanged to the OpenAI API
- API keys are never logged or stored by the proxy
- Use secure methods to provide the API key to your applications
- Never commit API keys to version control

Network Security:
- All headers (except Host) are forwarded to maintain security context
- HTTPS is required for production use to protect API key transmission
- Consider implementing rate limiting for production deployments
- Use secure methods to provide the API key to your applications

## Development

For development:

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv install

# Run tests
uv run pytest

# Start development server with auto-reload
uv run dev

# Start production server
uv run start
```
