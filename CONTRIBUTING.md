# CONTRIBUTING.md

## Contributing to OpenRouter Python Library

Thank you for your interest in contributing to the OpenRouter Python Library! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a branch for your changes
4. Make your changes
5. Submit a pull request

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/openrouter.git
   cd openrouter
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Project Structure

```
openrouter/
├── openrouter/           # Main library source code
│   ├── __init__.py
│   ├── client.py         # Main client classes (AsyncOpenRouter, OpenRouter)
│   ├── models.py         # Data models using Pydantic
│   ├── exceptions.py     # Custom exception classes
│   ├── utils.py          # Utility functions (pricing, validation, etc.)
│   └── api/              # API-specific modules (if any)
├── tests/                # Test suite
│   ├── test_openrouter.py    # Core functionality tests
│   ├── test_integration.py   # Integration tests with live API
│   ├── test_benchmarks.py    # Performance benchmarks
│   └── test_utils.py         # Utility function tests
├── docs/                 # Sphinx documentation
├── demo/                 # Demo applications
├── .github/workflows/    # CI/CD configuration
└── ...
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all public functions and methods
- Write comprehensive docstrings for all public functions, classes, and modules (Google-style)
- Use descriptive variable and function names
- Keep functions focused and small when possible
- Follow async-first design principles
- Implement proper error handling with custom exceptions
- Include logging for debugging and monitoring

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage (>90%)
- Include unit tests, integration tests, and performance benchmarks
- Run tests with:
  ```bash
  python -m pytest tests/
  ```
- Run tests with coverage:
  ```bash
  python -m pytest tests/ --cov=openrouter --cov-report=html
  ```
- Run specific test files:
  ```bash
  python -m pytest tests/test_openrouter.py
  ```

### Test Categories
- **Unit tests**: Test individual functions and methods in isolation
- **Integration tests**: Test with live API (requires API key)
- **Benchmark tests**: Performance testing for critical operations
- **Mock tests**: Test error scenarios and edge cases

## Documentation

- Update README.md when adding new features
- Add/update Sphinx documentation in the `docs/` directory
- Include docstrings for all public APIs
- Document configuration options and parameters
- Provide usage examples in README.md

To build documentation:
```bash
cd docs
make html
```

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass
5. Include performance benchmarks if adding critical functionality
6. Update CHANGELOG.md with a summary of changes (if applicable)
7. Submit your pull request with a clear description of changes
8. Link any relevant issues in the pull request description

## Reporting Issues

When reporting issues, please include:
- Python version
- Library version
- Operating system
- Detailed description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Any relevant error messages or logs
- Example code that reproduces the issue