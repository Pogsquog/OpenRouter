"""
Custom exceptions for OpenRouter API errors.
"""


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class AuthenticationError(OpenRouterError):
    """Raised when API authentication fails."""
    pass


class RateLimitError(OpenRouterError):
    """Raised when API rate limits are exceeded."""
    pass


class ModelNotFoundError(OpenRouterError):
    """Raised when a requested model is not found."""
    pass


class InvalidRequestError(OpenRouterError):
    """Raised when the API request is invalid."""
    pass