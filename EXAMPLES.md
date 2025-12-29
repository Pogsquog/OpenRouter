# Examples

Here are some examples of how to use the OpenRouter Python library in different scenarios.

## Basic Chat Completion

```python
from openrouter import AsyncOpenRouter
import asyncio

async def basic_example():
    client = AsyncOpenRouter(api_key="your-api-key")
    
    response = await client.chat_completions(
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
        model="openai/gpt-3.5-turbo"
    )
    
    print(response.choices[0].message.content)
    await client.close()

asyncio.run(basic_example())
```

## Streaming Response

```python
from openrouter import AsyncOpenRouter
import asyncio

async def streaming_example():
    client = AsyncOpenRouter(api_key="your-api-key")
    
    print("Response: ", end="", flush=True)
    async for chunk in client.stream_chat_completions(
        messages=[
            {"role": "user", "content": "Count from 1 to 10."}
        ],
        model="openai/gpt-3.5-turbo"
    ):
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)
    print()  # New line
    
    await client.close()

asyncio.run(streaming_example())
```

## Using Fallback Models

```python
from openrouter import AsyncOpenRouter
import asyncio

async def fallback_example():
    client = AsyncOpenRouter(api_key="your-api-key")
    
    # With explicit fallback models
    response = await client.chat_completions(
        messages=[
            {"role": "user", "content": "Explain quantum computing."}
        ],
        model="nonexistent-model",  # This will fail
        fallback_models=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"]
    )
    
    print(f"Response from: {response.route}")  # Shows which model was actually used
    print(response.choices[0].message.content)
    
    await client.close()

asyncio.run(fallback_example())
```

## Automatic Fallback Strategy

```python
from openrouter import AsyncOpenRouter
import asyncio

async def auto_fallback_example():
    client = AsyncOpenRouter(api_key="your-api-key")
    
    # With automatic fallback by provider
    response = await client.chat_completions_with_fallback(
        messages=[
            {"role": "user", "content": "Write a short poem."}
        ],
        primary_model="openai/gpt-4",
        fallback_strategy="provider"  # Will try other OpenAI models first
    )
    
    print(f"Response from: {response.route}")
    print(response.choices[0].message.content)
    
    await client.close()

asyncio.run(auto_fallback_example())
```

## Cost Calculation

```python
from openrouter import AsyncOpenRouter
import asyncio

async def cost_example():
    client = AsyncOpenRouter(api_key="your-api-key")
    
    response = await client.chat_completions(
        messages=[
            {"role": "user", "content": "Hello, world!"}
        ],
        model="openai/gpt-3.5-turbo"
    )
    
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens
    }
    
    cost = client.calculate_cost("openai/gpt-3.5-turbo", usage)
    print(f"Cost: ${cost:.6f}")
    
    await client.close()

asyncio.run(cost_example())
```

## Model Discovery

```python
from openrouter import AsyncOpenRouter
import asyncio

async def model_discovery_example():
    client = AsyncOpenRouter(api_key="your-api-key")
    
    # List all models
    models = await client.list_models()
    print(f"Found {len(models)} models")
    
    # Search for specific models
    gpt_models = await client.search_models("gpt")
    print(f"Found {len(gpt_models)} GPT models")
    
    # Get models by provider
    openai_models = await client.get_models_by_provider("openai")
    print(f"Found {len(openai_models)} OpenAI models")
    
    await client.close()

asyncio.run(model_discovery_example())
```

## Error Handling

```python
from openrouter import AsyncOpenRouter
from openrouter.exceptions import RateLimitError, AuthenticationError
import asyncio

async def error_handling_example():
    client = AsyncOpenRouter(api_key="your-api-key")
    
    try:
        response = await client.chat_completions(
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            model="openai/gpt-3.5-turbo"
        )
        print(response.choices[0].message.content)
    except RateLimitError:
        print("Rate limit exceeded. Please wait before making more requests.")
    except AuthenticationError:
        print("Authentication failed. Please check your API key.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    await client.close()

asyncio.run(error_handling_example())
```