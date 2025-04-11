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
settings:
  timeout: 30

primary_backends:
  - name: LLM1
    url: http://llm1.example.com/v1
    model: gpt-4o
  - name: LLM2 
    url: http://llm2.example.com/v1
    model: gpt-4o-mini

iterations:
    aggregation:
      strategy: concatenate

strategy:
  concatenate:
    separator: "\n\-------\n\n"
    # Hide intermediate thinking
    hide_intermediate_think: false
    # Hide final thinking
    hide_final_think: true
    # Tags to identify thinking
    thinking_tags: ["think", "reason", "reasoning", "thought", "Thought"]
    # Skip duplicated info from streamed and final aggregation
    skip_final_aggregation: true
```

## Flag Documentation
Below is a summary of how each relevant configuration flag is used to customize the output:

“iterations.aggregation.strategy”
- Specifies which aggregator method to apply for parallel requests. In this example, it defaults to “concatenate.”

“separator”
- Controls the delimiter used to join multiple responses in the final output (both non-streaming and final step of streaming).

“hide_intermediate_think”
- Determines if content inside thinking tags (<think>, <reason>, etc.) is hidden in streaming chunks. If True, partial “chain-of-thought” text is stripped out before reaching the client in real time.

“hide_final_think”
- Determines whether thinking tags are removed in the final aggregated message. If True, all text inside the thinking tags is removed in the combined final output event (streaming) or the merged response (non-stream).

“thinking_tags”
- The list of tag names considered as “thinking content.” Tags like <think>, <reason>, <reasoning>, <thought> are removed if hide_intermediate_think or hide_final_think is True.

“skip_final_aggregation”
- If True (in streaming mode), the final combined chunk of text from all backends is not sent, preventing duplication if you prefer only incremental streaming results.

“timeout”
- The maximum time (in seconds) to wait for each backend to respond before returning an error or partial data.

By adjusting these flags in your config.yaml, you can precisely control what content is streamed, how multiple responses are combined, and whether internal reasoning text appears in the final results.

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
