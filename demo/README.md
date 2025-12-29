# OpenRouter Demo Application

This demo application showcases the features of the OpenRouter Python library through an interactive command-line interface.

## Features

- Interactive chat interface
- Model selection and switching
- Streaming responses
- Cost tracking per message and session total
- Rate limit monitoring
- Conversation history management
- Clean, user-friendly interface

## Installation

Make sure you have the OpenRouter library installed:

```bash
pip install openrouter
```

## Configuration

The demo application looks for your OpenRouter API key in the following order:

1. Environment variable: `OPENROUTER_API_KEY`
2. Configuration file: `~/.openrouter_config.json`

### Setting up API Key

**Option 1: Environment Variable**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

**Option 2: Configuration File**
Create a file at `~/.openrouter_config.json` with the following content:
```json
{
    "api_key": "your-api-key-here"
}
```

## Usage

### Interactive Mode (Default)
```bash
python -m demo.chat_demo
```

### Command Line Options
```bash
# List available models
python -m demo.chat_demo --list-models

# Send a single message
python -m demo.chat_demo --message "Hello, how are you?"

# Use a specific model
python -m demo.chat_demo --model "anthropic/claude-3-haiku" --message "Write a poem"

# Disable streaming
python -m demo.chat_demo --no-stream --message "Hello"

# Run in interactive mode explicitly
python -m demo.chat_demo --interactive
```

## Available Commands in Interactive Mode

- `/models` - List available models
- `/model <name>` - Switch to a different model
- `/rate` - Check rate limit status
- `/clear` - Clear conversation history
- `/cost` - Show total cost so far
- `/quit` or `/exit` - Exit the application

## Example Usage

```
OpenRouter Demo Application
========================================
Commands:
  /models - List available models
  /model <name> - Switch to a different model
  /rate - Check rate limit status
  /clear - Clear conversation history
  /cost - Show total cost so far
  /quit or /exit - Exit the application

Current model: openai/gpt-3.5-turbo
Type your message below (or a command starting with /):

> Hello, world!
Assistant: Hello! How can I assist you today?

[Cost: $0.000001, Total: $0.000001]

> /model anthropic/claude-3-haiku
Switched to model: anthropic/claude-3-haiku

> Tell me a joke
Assistant: Why don't scientists trust atoms? Because they make up everything!

[Cost: $0.000002, Total: $0.000003]
```

## Error Handling

The demo application handles various error scenarios:

- Rate limit exceeded: Shows user-friendly message and suggests waiting
- Authentication failure: Provides clear instructions
- Invalid model: Lists available models
- Network timeouts: Implements retry logic