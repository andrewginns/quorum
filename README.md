# OpenAI API Proxy

A transparent proxy for OpenAI's chat completions API endpoint. This proxy captures all incoming requests, forwards them to OpenAI's API, and returns the responses exactly as received.

## Features

- Transparent proxying of `/chat/completions` endpoint
- Support for both streaming and non-streaming responses
- Configurable target API endpoint via environment variables
- Secure authentication forwarding
- Health check endpoint
- Proper error handling and logging

## Requirements

- Python 3.8+
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

1. Create your environment file by copying the example:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to configure your settings:
   ```bash
   # Using your preferred editor
   nano .env  # or vim .env, etc.
   ```

The proxy can be configured using the following environment variables:

- `OPENAI_API_BASE`: Base URL for the OpenAI API (default: https://api.openai.com/v1)

Note: The `.env` file is ignored by git to prevent committing sensitive information. Always use `.env.example` as a template for required environment variables.

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
    "model": "gpt-4",
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

1. Every request to the `/chat/completions` endpoint must include an Authorization header with your OpenAI API key
2. Format: `Authorization: Bearer your-openai-api-key`
3. The API key is forwarded directly to the OpenAI API (not stored or validated)
4. Requests without an Authorization header will receive a 401 Unauthorized response
5. The proxy does not store, modify, or validate the API key

Example with authentication:
```bash
# Using environment variable
curl http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Using direct API key (not recommended for production)
curl http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-..." \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Security Considerations

Authentication and API Keys:
- Your OpenAI API key must be provided in the Authorization header of each request
- The proxy forwards the Authorization header unchanged to the OpenAI API
- API keys are never logged or stored by the proxy
- Use environment variables to manage API keys securely
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
