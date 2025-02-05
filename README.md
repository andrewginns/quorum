# quorum

A flexible proxy service for routing requests to multiple LLM backends. quorum provides configurable proxy routing and aggregation of responses from multiple LLM providers.

## Features

- **Multiple Backend Support**: Route requests to multiple LLM providers in parallel or sequence
- **Configurable Response Aggregation**: Combine responses from multiple backends with custom aggregation strategies
- **OpenAI-Compatible API**: Implements the OpenAI Chat Completions API format for compatibility with existing tools
- **Streaming Support**: Full support for streaming responses from LLM backends
- **Robust Error Handling**: Gracefully handles backend failures and invalid configurations

## Configuration

quorum is configured via YAML files that specify:

- Primary backend providers and their endpoints
- Response aggregation strategies
- Timeout and retry settings
- Model mappings and routing rules

Example configuration:
```yaml
primary_backends:
  - name: LLM1
    url: http://llm1.example.com/v1
    model: gpt-4-1
  - name: LLM2 
    url: http://llm2.example.com/v1
    model: gpt-4-2

iterations:
  aggregation:
    strategy: concatenate
    separator: "\n-------------\n"

settings:
  timeout: 30
```

## Getting Started

1. Clone the repository

2. Create and activate a virtual environment:
   ```bash
   make install
   ```

3. Configure your backends in the config file

4. Run the proxy server

For development with auto-reload on http://localhost:8006:
```bash
make run
```

For production:
```bash
make run-prod
```


## Testing

Run the test suite:

```bash
make test
```
