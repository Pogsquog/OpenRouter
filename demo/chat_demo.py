#!/usr/bin/env python3
"""
Demo application for the OpenRouter Python library.

This command-line application demonstrates the usage of the OpenRouter library
with interactive chat, model selection, cost tracking, and rate limit monitoring.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import List, Dict, Optional

import openrouter
from openrouter import AsyncOpenRouter
from openrouter.utils import format_cost


class OpenRouterDemo:
    """Demo application for OpenRouter API."""
    
    def __init__(self):
        self.client: Optional[AsyncOpenRouter] = None
        self.current_model = "openai/gpt-3.5-turbo"
        self.conversation_history: List[Dict[str, str]] = []
        self.total_cost = 0.0
        self.api_key = None
        
    def load_config(self):
        """Load configuration from environment or config file."""
        # Try to get API key from environment
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            # Try to load from config file
            config_path = os.path.expanduser("~/.openrouter_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key")
        
        if not self.api_key:
            print("Error: No API key found.")
            print("Please set OPENROUTER_API_KEY environment variable or create ~/.openrouter_config.json")
            print("Example config file content:")
            print('{"api_key": "your-api-key-here"}')
            sys.exit(1)
    
    async def initialize_client(self):
        """Initialize the OpenRouter client."""
        self.client = AsyncOpenRouter(
            api_key=self.api_key,
            http_referer="https://github.com/yourusername/openrouter-demo",
            x_title="OpenRouter Demo App"
        )
    
    async def list_available_models(self):
        """List available models."""
        if not self.client:
            await self.initialize_client()
        
        try:
            print("\nAvailable Models:")
            models = await self.client.list_models()
            
            # Group models by provider
            providers = {}
            for model in models:
                provider = model.id.split('/')[0] if '/' in model.id else 'unknown'
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(model)
            
            for provider, models_list in providers.items():
                print(f"\n{provider.upper()}:")
                for model in models_list[:10]:  # Show first 10 models per provider
                    print(f"  - {model.id}")
                
                if len(models_list) > 10:
                    print(f"  ... and {len(models_list) - 10} more")
        
        except Exception as e:
            print(f"Error fetching models: {e}")
    
    async def chat_with_model(self, message: str, stream: bool = True):
        """Send a message to the current model and return the response."""
        if not self.client:
            await self.initialize_client()
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            if stream:
                print("\nAssistant: ", end="", flush=True)
                
                # Stream the response
                full_response = ""
                async for chunk in self.client.stream_chat_completions(
                    messages=self.conversation_history,
                    model=self.current_model
                ):
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            print(content, end="", flush=True)
                            full_response += content
                
                print()  # New line after streaming
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": full_response})
                
                # Get token usage from the final response
                # For streaming, we'll estimate based on character count
                prompt_tokens = sum(len(msg["content"]) for msg in self.conversation_history if msg["role"] == "user")
                completion_tokens = len(full_response)
                
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
                cost = self.client.calculate_cost(self.current_model, usage)
                self.total_cost += cost
                
                print(f"\n[Cost: {format_cost(cost)}, Total: {format_cost(self.total_cost)}]")
                
            else:
                # Non-streaming version
                response = await self.client.chat_completions(
                    messages=self.conversation_history,
                    model=self.current_model
                )
                
                assistant_message = response.choices[0].message.content
                print(f"\nAssistant: {assistant_message}")
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Calculate cost
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
                cost = self.client.calculate_cost(self.current_model, usage)
                self.total_cost += cost
                
                print(f"\n[Cost: {format_cost(cost)}, Total: {format_cost(self.total_cost)}]")
                
                return assistant_message
                
        except Exception as e:
            print(f"\nError during chat: {e}")
            # Remove the user message from history since it failed
            self.conversation_history.pop()
            return None
    
    async def get_rate_limit_info(self):
        """Get and display rate limit information."""
        if not self.client:
            await self.initialize_client()
        
        try:
            rate_limits = await self.client.get_rate_limits()
            print("\nRate Limit Information:")
            for key, value in rate_limits.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"\nError getting rate limits: {e}")
    
    async def switch_model(self, model_name: str):
        """Switch to a different model."""
        # Validate the model exists
        if not self.client:
            await self.initialize_client()

        try:
            # Try to get model info to validate it exists
            await self.client.get_model_info(model_name)
            self.current_model = model_name
            print(f"Switched to model: {model_name}")
        except Exception as e:
            # For demo purposes, warn the user but still allow the model switch
            # since some models might not be available for model info lookup
            # but could still work for chat completion
            print(f"Warning: Could not validate model {model_name} ({e})")
            print(f"Proceeding with model: {model_name}")
            self.current_model = model_name
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    async def run_interactive(self):
        """Run the interactive chat interface."""
        print("OpenRouter Demo Application")
        print("=" * 40)
        print("Commands:")
        print("  /models - List available models")
        print("  /model <name> - Switch to a different model")
        print("  /rate - Check rate limit status")
        print("  /clear - Clear conversation history")
        print("  /cost - Show total cost so far")
        print("  /quit or /exit - Exit the application")
        print("\nCurrent model:", self.current_model)
        print("Type your message below (or a command starting with /):")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_parts = user_input.split(' ', 1)
                    command = command_parts[0].lower()
                    
                    if command in ['/quit', '/exit']:
                        print("Goodbye!")
                        break
                    elif command == '/models':
                        await self.list_available_models()
                    elif command == '/model':
                        if len(command_parts) > 1:
                            await self.switch_model(command_parts[1])
                        else:
                            print("Usage: /model <model_name>")
                    elif command == '/rate':
                        await self.get_rate_limit_info()
                    elif command == '/clear':
                        self.clear_conversation()
                    elif command == '/cost':
                        print(f"Total cost so far: {format_cost(self.total_cost)}")
                    else:
                        print(f"Unknown command: {command}. Type /help for available commands.")
                else:
                    # Regular chat message
                    await self.chat_with_model(user_input)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
    
    async def run(self, args):
        """Main run method."""
        self.load_config()

        if args.list_models:
            await self.list_available_models()
        elif args.message:
            # If a message is provided, process it (with optional model switch)
            if args.model:
                try:
                    await self.switch_model(args.model)
                except Exception:
                    # If model switching fails, exit with error
                    sys.exit(1)
            await self.chat_with_model(args.message, stream=not args.no_stream)
        elif args.interactive or not sys.stdin.isatty():
            await self.run_interactive()
        else:
            # Default to interactive mode
            await self.run_interactive()


def main():
    parser = argparse.ArgumentParser(description="OpenRouter Demo Application")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Model to use (default: openai/gpt-3.5-turbo)")
    parser.add_argument("--message", type=str, help="Message to send to the model")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    demo = OpenRouterDemo()
    
    if args.model:
        demo.current_model = args.model
    
    asyncio.run(demo.run(args))


if __name__ == "__main__":
    main()