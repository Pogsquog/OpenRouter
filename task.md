Project Prompt: Develop Python Library for OpenRouter API

Project Overview:
Develop a comprehensive, production-ready Python library for interfacing with the OpenRouter API. OpenRouter provides a unified API that gives you access to hundreds of AI models through a single endpoint, while automatically handling fallbacks and routing. 13 The library should be designed for developers building AI applications that need to access multiple LLM providers through a single interface.

Core Requirements:

1. Library Architecture
Create an asynchronous-first library with synchronous wrappers
Support both streaming and non-streaming chat completions
Implement proper error handling and retries with exponential backoff
Include type hints and comprehensive docstrings
Follow PEP 8 standards and modern Python best practices
Support Python 3.8+ with dependency management via pyproject.toml
2. API Implementation Details
Base URL: https://openrouter.ai/api/v1
Primary Endpoint: /chat/completions for chat completions requests 26
Authentication: Use Bearer tokens in Authorization header for API key authentication 15
Required Headers:
Authorization: Bearer <API_KEY>
HTTP-Referer: <YOUR_SITE_URL> (Optional but recommended for rankings on openrouter.ai) 30
X-Title: <YOUR_SITE_NAME> (Optional but recommended for application attribution) 30
Content-Type: application/json
3. Key Features to Implement
Model Management: Methods to list available models and filter by provider preferences 17
Chat Completions: Full support for chat completion requests with all OpenRouter parameters
Streaming Support: Async generator for streaming responses with proper chunk handling
Rate Limit Handling: Automatic detection and handling of 429 errors with retry logic 36
Rate Limit Monitoring: Method to check current rate limits and remaining credits via GET /api/v1/key endpoint 33
Cost Tracking: Include token usage and cost estimation helpers
Fallback Mechanisms: Implement automatic model fallbacks for failed requests
4. Essential Documentation & Resources
Primary API Documentation:

OpenRouter API Reference: https://openrouter.ai/docs/api-reference 16
Quickstart Guide: https://openrouter.ai/docs/quickstart 13
Authentication Guide: https://openrouter.ai/docs/api-authentication 15
API Compatibility Notes:

OpenRouter's request and response schemas are very similar to the OpenAI Chat API, with a few small differences 5
The API syntax is largely compatible with OpenAI's API (e.g., /v1/chat/completions) but uses a different base URL 29
Rate Limit Information:

Free accounts are limited to 50 free model API requests per day 35
Free model variants (IDs ending in :free) are limited to 20 requests per minute and 200 daily 36
Paid accounts have higher rate limits that can be checked programmatically 34
5. Test Suite Requirements
Unit Tests: 100% coverage for core functionality using pytest
Integration Tests: Live tests against OpenRouter API (with test API key)
Mock Tests: Comprehensive mocking of API responses for error scenarios
Test Coverage: Minimum 90% test coverage with coverage.py
Test Scenarios to Include:
Successful chat completion (sync and async)
Streaming response handling
Rate limit error handling (429 responses)
Authentication failures
Invalid model requests
Network timeout handling
Header validation (HTTP-Referer, X-Title)
Cost calculation accuracy
6. Demo Application Requirements
Create a command-line chat interface demo application that:

Uses the library to interact with OpenRouter models
Supports multiple conversation modes (single-turn, multi-turn chat)
Allows model selection from available models
Displays token usage and cost estimates
Supports streaming responses with real-time display
Handles rate limits gracefully with user-friendly messages
Includes configuration file support for API keys and default settings
Provides help documentation and example usage
Demo Features:

Interactive chat session with history persistence
Model switching during conversation
Cost tracking per message and session total
Rate limit status display
Error handling with recovery suggestions
Clean, user-friendly interface with colored output
7. Development Guidelines
Dependencies: Minimize dependencies; use httpx for async HTTP, pydantic for validation
Configuration: Support environment variables, configuration files, and constructor arguments
Logging: Implement structured logging with debug levels for troubleshooting
Error Handling: Create custom exception hierarchy for OpenRouter-specific errors
Documentation: Generate API documentation using Sphinx or similar tool
Packaging: Prepare for PyPI publication with proper metadata and versioning
CI/CD: Include GitHub Actions workflow for testing, linting, and deployment
8. Quality Assurance
Code must pass all linters (flake8, mypy, black)
All public methods must have comprehensive docstrings
Include example usage in README and documentation
Performance benchmarks for critical operations
Security review for API key handling and data protection
Type checking with mypy for all public interfaces
9. Deliverables
Complete Python library package with proper structure
Comprehensive test suite with coverage reports
Demo chat application with full source code
API documentation (auto-generated)
README with installation, usage examples, and configuration guide
CONTRIBUTING.md and CODE_OF_CONDUCT.md files
GitHub repository with proper organization and CI setup
Success Criteria:

Library passes all tests and meets coverage requirements
Demo application works seamlessly with multiple models
Documentation is clear and comprehensive
Code follows Python best practices and is maintainable
Library handles edge cases and errors gracefully
Performance is acceptable for production use