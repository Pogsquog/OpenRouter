# OpenRouter Python Library Documentation

## Overview

The OpenRouter Python Library provides a comprehensive, production-ready interface to the OpenRouter API. OpenRouter provides a unified API that gives you access to hundreds of AI models through a single endpoint, while automatically handling fallbacks and routing.

## Installation

```bash
pip install openrouter
```

## Quick Start

### Async Usage

```python
import asyncio
from openrouter import AsyncOpenRouter

async def main():
    # Initialize the client
    client = AsyncOpenRouter(
        api_key="your-api-key",
        http_referer="https://your-app.com",  # Optional but recommended
        x_title="Your App Name"  # Optional but recommended
    )
    
    # Create a chat completion
    response = await client.chat_completions(
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        model="openai/gpt-3.5-turbo"
    )
    
    print(response.choices[0].message.content)
    
    # Close the client
    await client.close()

# Run the async function
asyncio.run(main())
```

### Sync Usage

```python
from openrouter import OpenRouter

# Initialize the client
client = OpenRouter(
    api_key="your-api-key",
    http_referer="https://your-app.com",  # Optional but recommended
    x_title="Your App Name"  # Optional but recommended
)

# Create a chat completion
response = client.chat_completions(
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    model="openai/gpt-3.5-turbo"
)

print(response.choices[0].message.content)

# Close the client
client.close()
```

## API Reference

### AsyncOpenRouter

The main async client class with the following methods:

#### `__init__(api_key, base_url, http_referer, x_title, timeout, max_retries, retry_delay, backoff_factor)`

Initialize the OpenRouter client.

- `api_key` (str): OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var.
- `base_url` (str): Base URL for the OpenRouter API. Default: "https://openrouter.ai/api/v1"
- `http_referer` (str, optional): HTTP Referer header value (recommended for rankings).
- `x_title` (str, optional): X-Title header value (recommended for application attribution).
- `timeout` (int): Request timeout in seconds. Default: 30
- `max_retries` (int): Maximum number of retries for failed requests. Default: 3
- `retry_delay` (float): Base delay between retries in seconds. Default: 1.0
- `backoff_factor` (float): Factor by which to multiply the delay between retries. Default: 2.0

#### `chat_completions(messages, model, fallback_models, **kwargs)`

Create a chat completion using the OpenRouter API.

- `messages` (List[Dict]): List of message dictionaries with 'role' and 'content' keys
- `model` (str): Primary model identifier to use for completion
- `fallback_models` (List[str], optional): List of fallback model identifiers to try if the primary model fails
- `**kwargs`: Additional parameters for the completion request
- Returns: `ChatCompletionResponse`

#### `chat_completions_with_fallback(messages, primary_model, fallback_strategy, **kwargs)`

Create a chat completion with automatic fallback based on strategy.

- `messages` (List[Dict]): List of message dictionaries with 'role' and 'content' keys
- `primary_model` (str): Primary model identifier to use for completion
- `fallback_strategy` (str): Strategy for selecting fallbacks ("provider", "performance", "cost")
- `**kwargs`: Additional parameters for the completion request
- Returns: `ChatCompletionResponse`

#### `stream_chat_completions(messages, model, **kwargs)`

Stream chat completions from the OpenRouter API.

- `messages` (List[Dict]): List of message dictionaries with 'role' and 'content' keys
- `model` (str): Model identifier to use for completion
- `**kwargs`: Additional parameters for the completion request
- Yields: Stream chunks as they arrive

#### `list_models(**kwargs)`

List available models from OpenRouter API.

- `**kwargs`: Additional query parameters for filtering
- Returns: List of `ModelInfo` objects

#### `get_model_info(model_id)`

Get detailed information about a specific model.

- `model_id` (str): ID of the model to get information for
- Returns: `ModelInfo` object

#### `get_rate_limits()`

Get current rate limit information.

- Returns: Dict containing rate limit information

#### `get_account_info()`

Get account information including rate limits and usage.

- Returns: Dict containing account information

#### `calculate_cost(model_id, usage)`

Calculate the estimated cost of a request based on model and token usage.

- `model_id` (str): The model identifier
- `usage` (Dict): Dictionary containing 'prompt_tokens' and 'completion_tokens'
- Returns: Estimated cost in USD

#### `search_models(search)`

Search for models by name or description.

- `search` (str): Search term to look for in model names or descriptions
- Returns: List of matching `ModelInfo` objects

#### `get_models_by_provider(provider)`

Get models filtered by provider.

- `provider` (str): Provider name to filter by (e.g., "openai", "anthropic", "google")
- Returns: List of `ModelInfo` objects from the specified provider

#### `parse_rate_limit_headers(response)`

Parse rate limit information from response headers.

- `response` (httpx.Response): HTTP response object
- Returns: Dictionary containing rate limit information

### OpenRouter

Synchronous wrapper around AsyncOpenRouter with the same methods as described above.

## Error Handling

The library provides custom exceptions:

- `OpenRouterError` - Base exception
- `AuthenticationError` - Authentication failures
- `RateLimitError` - Rate limit exceeded
- `ModelNotFoundError` - Requested model not found
- `InvalidRequestError` - Invalid request parameters

## Environment Variables

- `OPENROUTER_API_KEY` - Your OpenRouter API key

## Configuration

The client accepts the following parameters:

- `api_key`: Your OpenRouter API key (required)
- `base_url`: API base URL (defaults to OpenRouter's endpoint)
- `http_referer`: HTTP Referer header (recommended for rankings)
- `x_title`: X-Title header (recommended for attribution)
- `timeout`: Request timeout in seconds (default: 30)
- `max_retries`: Maximum number of retries (default: 3)
- `retry_delay`: Base delay between retries in seconds (default: 1.0)
- `backoff_factor`: Factor by which to multiply the delay between retries (default: 2.0)

## Contributing

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.