# CONTRIBUTING.md

## Contributing to OpenRouter Python Library

Thank you for your interest in contributing to the OpenRouter Python Library! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
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

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all public functions and methods
- Write docstrings for all public functions, classes, and modules
- Use descriptive variable and function names
- Keep functions focused and small when possible

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage
- Run tests with:
  ```bash
  python -m pytest tests/
  ```

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass
5. Submit your pull request with a clear description of changes

## Reporting Issues

When reporting issues, please include:
- Python version
- Library version
- Detailed description of the issue
- Steps to reproduce
- Expected vs actual behavior