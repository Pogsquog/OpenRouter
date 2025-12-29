import pytest
from unittest.mock import patch, MagicMock
import asyncio

from openrouter.exceptions import (
    OpenRouterError, 
    AuthenticationError, 
    RateLimitError, 
    ModelNotFoundError, 
    InvalidRequestError
)


class TestExceptions:
    """Test exception classes."""
    
    def test_base_exception(self):
        """Test the base OpenRouterError exception."""
        with pytest.raises(OpenRouterError):
            raise OpenRouterError("Test error message")
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Authentication failed")
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limit exceeded")
    
    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("Model not found")
    
    def test_invalid_request_error(self):
        """Test InvalidRequestError."""
        with pytest.raises(InvalidRequestError):
            raise InvalidRequestError("Invalid request")
    
    def test_exception_inheritance(self):
        """Test that custom exceptions inherit from base exception."""
        assert issubclass(AuthenticationError, OpenRouterError)
        assert issubclass(RateLimitError, OpenRouterError)
        assert issubclass(ModelNotFoundError, OpenRouterError)
        assert issubclass(InvalidRequestError, OpenRouterError)