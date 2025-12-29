# OpenRouter Python Library

A comprehensive, production-ready Python library for interfacing with the OpenRouter API. OpenRouter provides a unified API that gives you access to hundreds of AI models through a single endpoint, while automatically handling fallbacks and routing.

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

## Features

- **Async-first design** with sync wrappers
- **Streaming support** for real-time responses
- **Automatic retry logic** with exponential backoff
- **Rate limit handling** with automatic retries
- **Model management** to list and query available models
- **Cost tracking** utilities
- **Fallback mechanisms** for improved reliability
- **Comprehensive error handling**

## API Reference

### AsyncOpenRouter

The main async client class with the following methods:

- `chat_completions()` - Create chat completions
- `stream_chat_completions()` - Stream chat completions
- `list_models()` - List available models
- `get_model_info(model_id)` - Get detailed model information
- `get_rate_limits()` - Get current rate limit status

### OpenRouter

Synchronous wrapper around AsyncOpenRouter with the same methods.

## Environment Variables

You can set your API key using an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Configuration

The client accepts the following parameters:

- `api_key`: Your OpenRouter API key (required)
- `base_url`: API base URL (defaults to OpenRouter's endpoint)
- `http_referer`: HTTP Referer header (recommended for rankings)
- `x_title`: X-Title header (recommended for attribution)
- `timeout`: Request timeout in seconds (default: 30)
- `max_retries`: Maximum number of retries (default: 3)
- `retry_delay`: Base delay between retries in seconds (default: 1.0)

## Error Handling

The library provides custom exceptions:

- `OpenRouterError` - Base exception
- `AuthenticationError` - Authentication failures
- `RateLimitError` - Rate limit exceeded
- `ModelNotFoundError` - Requested model not found
- `InvalidRequestError` - Invalid request parameters

## Contributing

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.