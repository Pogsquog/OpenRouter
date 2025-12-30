# OpenRouter Python Library

A Python library for interfacing with the OpenRouter API. OpenRouter provides a unified API that gives you access to hundreds of AI models through a single endpoint, while automatically handling fallbacks and routing.

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
        x_title="Your App Name",  # Optional but recommended
        log_level="INFO"  # Set logging level (DEBUG, INFO, WARNING, ERROR)
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
    x_title="Your App Name",  # Optional but recommended
    log_level="INFO"  # Set logging level
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

### Streaming Responses

```python
import asyncio
from openrouter import AsyncOpenRouter

async def stream_example():
    client = AsyncOpenRouter(api_key="your-api-key")

    async for chunk in client.stream_chat_completions(
        messages=[{"role": "user", "content": "Write a short story"}],
        model="openai/gpt-3.5-turbo"
    ):
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)

    await client.close()

asyncio.run(stream_example())
```

### Model Management

```python
import asyncio
from openrouter import AsyncOpenRouter

async def model_examples():
    client = AsyncOpenRouter(api_key="your-api-key")

    # List all available models
    models = await client.list_models()
    print(f"Found {len(models)} models")

    # Get specific model information
    model_info = await client.get_model_info("openai/gpt-3.5-turbo")
    print(f"Model: {model_info.name}")
    print(f"Description: {model_info.description}")

    # Search for models
    gpt_models = await client.search_models("gpt")
    print(f"Found {len(gpt_models)} GPT models")

    # Get models by provider
    openai_models = await client.get_models_by_provider("openai")
    print(f"Found {len(openai_models)} OpenAI models")

    await client.close()

asyncio.run(model_examples())
```

### Fallback Mechanisms

```python
import asyncio
from openrouter import AsyncOpenRouter

async def fallback_example():
    client = AsyncOpenRouter(api_key="your-api-key")

    # Use automatic fallback with different strategies
    response = await client.chat_completions_with_fallback(
        messages=[{"role": "user", "content": "Tell me a joke"}],
        primary_model="nonexistent-model",  # This will fail
        fallback_strategy="provider"  # Try other models from same provider
    )

    print(f"Response from: {response.route}")
    print(response.choices[0].message.content)

    await client.close()

asyncio.run(fallback_example())
```

### Cost Tracking

```python
import asyncio
from openrouter import AsyncOpenRouter
from openrouter.utils import format_cost

async def cost_example():
    client = AsyncOpenRouter(api_key="your-api-key")

    response = await client.chat_completions(
        messages=[{"role": "user", "content": "Calculate 2+2"}],
        model="openai/gpt-3.5-turbo"
    )

    # Calculate cost based on token usage
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens
    }
    cost = client.calculate_cost("openai/gpt-3.5-turbo", usage)

    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Total cost: {format_cost(cost)}")

    await client.close()

asyncio.run(cost_example())
```

## Features

- **Async-first design** with sync wrappers
- **Streaming support** for real-time responses
- **Automatic retry logic** with exponential backoff
- **Rate limit handling** with automatic retries
- **Model management** to list and query available models
- **Dynamic pricing** with automatic updates from API
- **Cost tracking** utilities with accurate pricing
- **Fallback mechanisms** for improved reliability
- **Comprehensive error handling**
- **Structured logging** with multiple log levels
- **Security best practices** for API key handling
- **Full OpenAI API compatibility** with OpenRouter-specific features

## API Reference

### AsyncOpenRouter

The main async client class with the following methods:

- `chat_completions()` - Create chat completions with optional fallbacks
- `chat_completions_with_fallback()` - Create chat completions with automatic fallback strategies
- `stream_chat_completions()` - Stream chat completions in real-time
- `list_models()` - List all available models
- `get_model_info(model_id)` - Get detailed information about a specific model
- `search_models(search_term)` - Search for models by name or description
- `get_models_by_provider(provider)` - Get models filtered by provider
- `get_rate_limits()` - Get current rate limit status
- `get_account_info()` - Get account information and usage
- `calculate_cost(model_id, usage)` - Calculate cost for a request
- `parse_rate_limit_headers(response)` - Parse rate limit information from response headers

### OpenRouter

Synchronous wrapper around AsyncOpenRouter with the same methods.

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
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
- `enable_request_logging`: Enable detailed request logging (WARNING: may log sensitive data) (default: False)

## Environment Variables

You can set your API key using an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Error Handling

The library provides custom exceptions:

- `OpenRouterError` - Base exception for all OpenRouter errors
- `AuthenticationError` - Authentication failures
- `RateLimitError` - Rate limit exceeded
- `ModelNotFoundError` - Requested model not found
- `InvalidRequestError` - Invalid request parameters

## Security Best Practices

- API keys are never logged directly; only hashed identifiers are used for logging
- All API keys are validated using regex patterns before use
- Secure HTTPS connections are enforced
- Request logging can be disabled to prevent sensitive data exposure
- Follows secure coding practices to prevent common vulnerabilities

## Testing

The library includes comprehensive tests:

- Unit tests for all core functionality
- Integration tests for live API calls
- Performance benchmarks for critical operations
- Mock tests for error scenarios

Run tests with:
```bash
python -m pytest tests/
```

## Documentation

Complete API documentation is available through Sphinx:

```bash
cd docs
make html
# Documentation will be available in docs/_build/html/
```

Or view the documentation online once deployed to a hosting service.

## Contributing

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
