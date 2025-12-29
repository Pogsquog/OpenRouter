Usage Examples
==============

This section provides practical examples of how to use the OpenRouter Python library for various common tasks.

Basic Usage
-----------

Async Client
~~~~~~~~~~~~

The primary interface for the library is the ``AsyncOpenRouter`` class:

.. code-block:: python

    import asyncio
    from openrouter import AsyncOpenRouter

    async def basic_example():
        client = AsyncOpenRouter(
            api_key="your-api-key",
            http_referer="https://your-app.com",
            x_title="Your App Name"
        )
        
        response = await client.chat_completions(
            messages=[{"role": "user", "content": "Hello, world!"}],
            model="openai/gpt-3.5-turbo"
        )
        
        print(response.choices[0].message.content)
        await client.close()

    asyncio.run(basic_example())

Synchronous Client
~~~~~~~~~~~~~~~~~~

For synchronous applications, use the ``OpenRouter`` wrapper:

.. code-block:: python

    from openrouter import OpenRouter

    client = OpenRouter(api_key="your-api-key")
    
    response = client.chat_completions(
        messages=[{"role": "user", "content": "Hello, world!"}],
        model="openai/gpt-3.5-turbo"
    )
    
    print(response.choices[0].message.content)
    client.close()

Streaming Responses
-------------------

For real-time responses, use the streaming functionality:

.. code-block:: python

    import asyncio
    from openrouter import AsyncOpenRouter

    async def stream_example():
        client = AsyncOpenRouter(api_key="your-api-key")
        
        async for chunk in client.stream_chat_completions(
            messages=[{"role": "user", "content": "Write a poem"}],
            model="openai/gpt-3.5-turbo"
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    print(delta["content"], end="", flush=True)
        
        await client.close()

    asyncio.run(stream_example())

Model Management
----------------

List and query available models:

.. code-block:: python

    import asyncio
    from openrouter import AsyncOpenRouter

    async def model_examples():
        client = AsyncOpenRouter(api_key="your-api-key")
        
        # List all models
        models = await client.list_models()
        print(f"Available models: {len(models)}")
        
        # Get specific model info
        model_info = await client.get_model_info("openai/gpt-3.5-turbo")
        print(f"Model: {model_info.name}")
        
        # Search for models
        gpt_models = await client.search_models("gpt")
        print(f"GPT models: {len(gpt_models)}")
        
        # Filter by provider
        openai_models = await client.get_models_by_provider("openai")
        print(f"OpenAI models: {len(openai_models)}")
        
        await client.close()

Fallback Mechanisms
-------------------

Use automatic fallback strategies:

.. code-block:: python

    import asyncio
    from openrouter import AsyncOpenRouter

    async def fallback_example():
        client = AsyncOpenRouter(api_key="your-api-key")
        
        # With automatic fallback
        response = await client.chat_completions_with_fallback(
            messages=[{"role": "user", "content": "Hello"}],
            primary_model="nonexistent-model",  # Will fail
            fallback_strategy="provider"  # Try other models from same provider
        )
        
        print(f"Response from: {response.route}")
        print(response.choices[0].message.content)
        
        await client.close()

Cost Tracking
-------------

Calculate costs for API usage:

.. code-block:: python

    import asyncio
    from openrouter import AsyncOpenRouter
    from openrouter.utils import format_cost

    async def cost_example():
        client = AsyncOpenRouter(api_key="your-api-key")
        
        response = await client.chat_completions(
            messages=[{"role": "user", "content": "Calculate 2+2"}],
            model="openai/gpt-3.5-turbo"
        )
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
        cost = client.calculate_cost("openai/gpt-3.5-turbo", usage)
        
        print(f"Cost: {format_cost(cost)}")
        
        await client.close()

Rate Limits and Monitoring
--------------------------

Check your usage and rate limits:

.. code-block:: python

    import asyncio
    from openrouter import AsyncOpenRouter

    async def limits_example():
        client = AsyncOpenRouter(api_key="your-api-key")
        
        # Get rate limit information
        limits = await client.get_rate_limits()
        print(f"Rate limits: {limits}")
        
        # Get account information
        account_info = await client.get_account_info()
        print(f"Account info: {account_info}")
        
        await client.close()