# Agent Guidance for OpenRouter Python Library

## Project Overview
This is a comprehensive Python library for interfacing with the OpenRouter API, providing access to hundreds of AI models through a unified endpoint with automatic fallbacks and routing.

## Virtual Environment Setup
**CRITICAL**: Always use the virtual environment for this project to avoid system package conflicts.

```bash
cd /home/stuart/code/stu/openrouter
source venv/bin/activate
```

## Development Commands

### Installing Dependencies
```bash
source venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests
```bash
source venv/bin/activate
python -m pytest tests/ -v
```

### Running Tests with Coverage
```bash
source venv/bin/activate
python -m pytest tests/ --cov=openrouter --cov-report=html
```

### Running Demo Application
```bash
source venv/bin/activate
python -m demo.chat_demo --interactive
```

### Linting
```bash
source venv/bin/activate
ruff check openrouter/
black --check openrouter/
mypy openrouter/
```

### Formatting
```bash
source venv/bin/activate
black openrouter/
ruff check openrouter/ --fix
```

## Key Files and Directories
- `openrouter/client.py` - Main async/sync client implementation
- `openrouter/models.py` - Pydantic data models
- `openrouter/exceptions.py` - Custom exception classes
- `openrouter/utils.py` - Utility functions
- `tests/` - Test suite
- `demo/` - Demo applications
- `pyproject.toml` - Project configuration

## API Key Setup
For testing and development, set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Testing Notes
- Use fake API keys (starting with `sk-fake-`) for unit tests
- Integration tests require a real API key
- Mock responses appropriately for network calls

## Grep usage
Prefer ripgrep (rg) to grep eg:
rg -n "TODO.*Qwen" src/
rg --type=py "def main" .
rg --json "\\buser_id\\b" | jq '.'
