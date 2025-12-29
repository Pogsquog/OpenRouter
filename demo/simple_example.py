#!/usr/bin/env python3
"""
Simple example script demonstrating OpenRouter library usage.
"""

import asyncio
import os
from openrouter import AsyncOpenRouter


async def main():
    # Initialize the client
    # You can set OPENROUTER_API_KEY environment variable or pass it directly
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return
    
    client = AsyncOpenRouter(
        api_key=api_key,
        http_referer="https://github.com/yourusername/openrouter-demo",
        x_title="OpenRouter Example"
    )
    
    try:
        # List available models
        print("Fetching available models...")
        models = await client.list_models()
        print(f"Found {len(models)} models")
        
        # Get info about a specific model
        print("\nGetting info for GPT-4...")
        model_info = await client.get_model_info("openai/gpt-4")
        print(f"Model: {model_info.name}")
        print(f"Description: {model_info.description}")
        
        # Create a chat completion
        print("\nCreating chat completion...")
        response = await client.chat_completions(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            model="openai/gpt-3.5-turbo"
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        
        # Calculate cost
        cost = client.calculate_cost("openai/gpt-3.5-turbo", {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        })
        print(f"Estimated cost: ${cost:.6f}")
        
        # Try streaming
        print("\nTrying streaming response...")
        print("Streaming response: ", end="", flush=True)
        async for chunk in client.stream_chat_completions(
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            model="openai/gpt-3.5-turbo"
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    print(delta["content"], end="", flush=True)
        print()  # New line
        
        # Check rate limits
        print("\nChecking rate limits...")
        rate_limits = await client.get_rate_limits()
        print(f"Rate limit info: {rate_limits}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())