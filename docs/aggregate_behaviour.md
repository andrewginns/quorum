# Aggregation Strategy Configuration Behavior Analysis

## Overview
The aggregation strategy sends all initial responses to a final LLM for synthesis. This document explains how different configuration settings affect the behavior of this strategy, based on observed behaviors with real queries.

## Settings Explained

### `source_backends`
- **Purpose**: Specifies which backends to use as input sources for aggregation
- **Values**: 
  - `"all"`: Use all primary_backends (default if omitted)
  - List of specific backend names: Only use the specified backends
- **Behavior**: Controls which LLMs contribute responses to be aggregated

**Example**:
```yaml
# Using all available backends
source_backends: "all"

# Using only specific backends
source_backends: ["LLM1", "LLM3"]
```

### `aggregator_backend`
- **Purpose**: Specifies which backend will synthesize all source outputs
- **Value**: Must be a backend defined in primary_backends
- **Behavior**: This LLM receives all source responses and creates the final aggregated response

**Example**:
```yaml
aggregator_backend: "LLM1"
```

### `intermediate_separator`
- **Purpose**: Defines how to combine source outputs in the aggregator prompt
- **Format**: In the example, it's set to `"\n\n---\n\n"` (triple dashes with double newlines before and after)
- **Behavior**: Separates individual LLM responses in the prompt sent to the aggregator

**Example**:
```
# With intermediate_separator: "\n\n---\n\n"

Response from LLM1

---

Response from LLM2

---

Response from LLM3
```

### `include_source_names`
- **Purpose**: Controls whether to include source backend names with their responses
- **Values**: 
  - `false`: Source names are not included
  - `true`: Each response is labeled with its source backend name
- **Behavior**: When true, each response in the aggregator prompt is prefaced with its source

**Example**:
```
# With include_source_names: true
Response from LLM1:
The capital of France is Paris.

---

Response from LLM2:
The capital of France is Paris.

# With include_source_names: false
The capital of France is Paris.

---

The capital of France is Paris.
```

### `source_label_format`
- **Purpose**: Defines the format for labeling each source response when include_source_names is true
- **Format**: In the example, it's set to `"Response from {backend_name}:\n"`
- **Behavior**: Only has an effect when include_source_names is true

**Example**:
```
# With source_label_format: "Response from {backend_name}:\n"
Response from LLM1:
The capital of France is Paris.

# With source_label_format: "Output ({backend_name}):\n"
Output (LLM1):
The capital of France is Paris.
```

### `prompt_template`
- **Purpose**: Defines the prompt template sent to the aggregator backend
- **Format**: A multi-line string with a placeholder for intermediate results
- **Behavior**: Controls how the aggregator LLM is instructed to synthesize the responses

**Example**:
```
# Default prompt template
You have received the following responses regarding the user's query:

{{intermediate_results}}

Synthesize these responses into a single, comprehensive answer that captures
the best information and insights from all sources. Resolve any contradictions
and provide a coherent, unified response.
```

### `strip_intermediate_thinking`
- **Purpose**: Controls whether thinking blocks are removed from source responses before aggregation
- **Values**: 
  - `false`: Thinking blocks are included in the aggregator prompt
  - `true`: Thinking blocks are removed before sending to the aggregator
- **Behavior**: When true, any content within thinking tags is stripped from source responses

**Example**:
```
# Original response with thinking
The capital of France is <think>I know this is Paris</think> Paris.

# With strip_intermediate_thinking: true (sent to aggregator)
The capital of France is Paris.

# With strip_intermediate_thinking: false (sent to aggregator)
The capital of France is <think>I know this is Paris</think> Paris.
```

### `hide_aggregator_thinking`
- **Purpose**: Controls visibility of the aggregator's thinking process in final output
- **Values**: 
  - `false`: Aggregator's thinking is visible in the final output
  - `true`: Aggregator's thinking is hidden in the final output
- **Behavior**: When true, any content within thinking tags is stripped from the aggregator's response

**Example**:
```
# Aggregator's response
<think>Let me combine these responses.</think>
The capital of France is Paris.

# With hide_aggregator_thinking: true (final output)
The capital of France is Paris.

# With hide_aggregator_thinking: false (final output)
<think>Let me combine these responses.</think>
The capital of France is Paris.
```

### `thinking_tags`
- **Purpose**: Defines which tags are identified as thinking sections
- **Value**: In the example set to `["think", "reason", "reasoning", "thought", "Thought"]`
- **Behavior**: These tags identify sections that will be affected by the `strip_intermediate_thinking` and `hide_aggregator_thinking` settings

**Example**:
```
# The following tags will be treated as thinking sections:

<reason>This is reasoning content</reason>
<thought>This is a thought</thought>
```

### `include_original_query`
- **Purpose**: Controls whether to include the original user query in the aggregator prompt
- **Values**: 
  - `false`: Original query is not included
  - `true`: Original query is included in the aggregator prompt
- **Behavior**: When true, the user's original query is included in the prompt to the aggregator

**Example**:
```
# With include_original_query: true
Original query: What is the capital of France?

You have received the following responses regarding the user's query:
...

# With include_original_query: false
You have received the following responses regarding the user's query:
...
```

### `query_format`
- **Purpose**: Defines the format for including the original query when include_original_query is true
- **Format**: In the example, it's set to `"Original query: {query}\n\n"`
- **Behavior**: Only has an effect when include_original_query is true

**Example**:
```
# With query_format: "Original query: {query}\n\n"
Original query: What is the capital of France?

# With query_format: "User asked: {query}\n\n"
User asked: What is the capital of France?
```

### `suppress_individual_responses`
- **Purpose**: Controls whether to suppress streaming individual LLM responses and only stream the final aggregated response
- **Values**: 
  - `false`: Individual responses are streamed/shown before the final aggregated response
  - `true`: Only the final aggregated response is streamed/shown
- **Behavior**: When true, individual responses are suppressed and only the final aggregated response is returned

**Example**:
```
# With suppress_individual_responses: false (streaming)
data: {"id": "chatcmpl-parallel-0", "object": "chat.completion.chunk", "created": 903, "model": "parallel-proxy", "choices": [{"index": 0, "delta": {"content": "The capital of France is Paris."}, "finish_reason": null}]}

data: {"id": "chatcmpl-parallel-1", "object": "chat.completion.chunk", "created": 904, "model": "parallel-proxy", "choices": [{"index": 0, "delta": {"content": "The capital of France is Paris."}, "finish_reason": null}]}

data: {"id": "chatcmpl-parallel-2", "object": "chat.completion.chunk", "created": 905, "model": "parallel-proxy", "choices": [{"index": 0, "delta": {"content": "The capital of France is Paris."}, "finish_reason": null}]}

data: {"id": "chatcmpl-parallel-final", "object": "chat.completion.chunk", "created": 906, "model": "parallel-proxy", "choices": [{"index": 0, "delta": {"content": "The capital of France is Paris."}, "finish_reason": "stop"}]}

# With suppress_individual_responses: true (streaming)
data: {"id": "chatcmpl-parallel", "object": "chat.completion.chunk", "created": 903, "model": "parallel-proxy", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}

data: {"id": "chatcmpl-parallel-final", "object": "chat.completion.chunk", "created": 904, "model": "parallel-proxy", "choices": [{"index": 0, "delta": {"content": "The capital of France is Paris."}, "finish_reason": "stop"}]}
```

## Interactions Between Settings

### Source Names and Intermediate Separator
- The `include_source_names` and `source_label_format` settings work together to control how source responses are labeled in the aggregator prompt
- The `intermediate_separator` is always applied between responses regardless of whether source names are included

**Example**:
```
# Configuration:
# include_source_names: true
# source_label_format: "Response from {backend_name}:\n"
# intermediate_separator: "\n\n---\n\n"

Response from LLM1:
The capital of France is Paris.

---

Response from LLM2:
The capital of France is Paris.

# Same with include_source_names: false

The capital of France is Paris.

---

The capital of France is Paris.
```

### Thinking Tags and Hiding Thinking
- The `thinking_tags` setting defines which tags are considered thinking sections
- The `strip_intermediate_thinking` and `hide_aggregator_thinking` settings control whether these sections are visible in the intermediate and final outputs

**Example**:
```
# Configuration:
# thinking_tags: ["think", "reason", "reasoning", "thought", "Thought"]
# strip_intermediate_thinking: true
# hide_aggregator_thinking: true

# Original source response
The capital of France is <think>I know this is Paris</think> Paris.

# After stripping intermediate thinking (sent to aggregator)
The capital of France is Paris.

# Aggregator's response
<think>All responses agree that the capital of France is Paris.</think>
The capital of France is Paris.

# Final output after hiding aggregator thinking
The capital of France is Paris.
```

### Suppress Individual Responses and Streaming
- The `suppress_individual_responses` setting has different effects on streaming and non-streaming requests
- For streaming requests, it controls whether individual responses are streamed before the final response
- For non-streaming requests, it controls whether individual responses are included in the final response

**Example for streaming requests**:
```
# With suppress_individual_responses: false
# Individual responses are streamed first, then the final response
data: {"id": "chatcmpl-parallel-0", "choices": [{"delta": {"content": "The capital of France is Paris."}}]}
data: {"id": "chatcmpl-parallel-1", "choices": [{"delta": {"content": "The capital of France is Paris."}}]}
data: {"id": "chatcmpl-parallel-2", "choices": [{"delta": {"content": "The capital of France is Paris."}}]}
data: {"id": "chatcmpl-parallel-final", "choices": [{"delta": {"content": "The capital of France is Paris."}}]}

# With suppress_individual_responses: true
# Only the final response is streamed
data: {"id": "chatcmpl-parallel", "choices": [{"delta": {"role": "assistant"}}]}
data: {"id": "chatcmpl-parallel-final", "choices": [{"delta": {"content": "The capital of France is Paris."}}]}
```

**Example for non-streaming requests**:
```
# With suppress_individual_responses: false
# Response contains all individual responses concatenated
{
  "choices": [
    {
      "message": {
        "content": "The capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris."
      }
    }
  ]
}

# With suppress_individual_responses: true
# Response contains only the first response
{
  "choices": [
    {
      "message": {
        "content": "The capital of France is Paris."
      }
    }
  ]
}
```

## Configuration Patterns and Effects

| suppress_individual_responses | strip_intermediate_thinking | hide_aggregator_thinking | Result |
|-------------------------------|----------------------------|--------------------------|--------|
| false | false | false | Shows all individual responses; includes thinking in intermediate and final responses |
| true | false | false | Shows only final response; includes thinking in intermediate and final responses |
| false | true | false | Shows all individual responses; strips thinking from intermediate responses but shows it in final response |
| false | false | true | Shows all individual responses; includes thinking in intermediate responses but hides it in final response |
| true | true | false | Shows only final response; strips thinking from intermediate responses but shows it in final response |
| true | false | true | Shows only final response; includes thinking in intermediate responses but hides it in final response |
| false | true | true | Shows all individual responses; hides thinking in both intermediate and final responses |
| true | true | true | Shows only final response; hides thinking in both intermediate and final responses |

## Practical Example

**User Query**: "What is the capital of France?"

**Raw LLM Responses**:
```
# LLM1
<think>The capital of France is Paris.</think>
The capital of France is Paris.

# LLM2
<think>I know this one - it's Paris.</think>
The capital of France is Paris.

# LLM3
<think>This is a simple geography question.</think>
The capital of France is Paris.
```

**Aggregator Input with strip_intermediate_thinking: true, include_source_names: true**:
```
Original query: What is the capital of France?

You have received the following responses regarding the user's query:

Response from LLM1:
The capital of France is Paris.

---

Response from LLM2:
The capital of France is Paris.

---

Response from LLM3:
The capital of France is Paris.

Synthesize these responses into a single, comprehensive answer that captures
the best information and insights from all sources. Resolve any contradictions
and provide a coherent, unified response.
```

**Aggregator Response**:
```
<think>All three responses agree that the capital of France is Paris. There are no contradictions to resolve.</think>
The capital of France is Paris.
```

**Final Output with hide_aggregator_thinking: true, suppress_individual_responses: true**:
```
The capital of France is Paris.
```

**Final Output with hide_aggregator_thinking: false, suppress_individual_responses: false**:
```
The capital of France is Paris.

---

The capital of France is Paris.

---

The capital of France is Paris.

---

<think>All three responses agree that the capital of France is Paris. There are no contradictions to resolve.</think>
The capital of France is Paris.
```

## Summary
The aggregation strategy configuration settings provide fine-grained control over:
1. Which backends contribute responses to be aggregated
2. How these responses are formatted and presented to the aggregator
3. Whether thinking processes are visible in intermediate and final outputs
4. Whether individual responses are shown or suppressed
5. How the aggregator is instructed to synthesize the responses

This allows for customizing the appearance and behavior of the aggregation strategy based on different use cases, whether focusing on the final synthesized response or showing the individual responses and reasoning processes.
