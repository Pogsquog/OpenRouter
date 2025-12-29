import pytest
from openrouter.utils import (
    calculate_cost,
    normalize_model_id,
    validate_api_key,
    parse_rate_limit_headers,
    format_cost,
    estimate_cost_for_request
)


class TestUtils:
    """Test utility functions."""
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        # Test with known model
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = calculate_cost("openai/gpt-3.5-turbo", usage)
        
        expected = (1000 / 1000) * 0.0015 + (500 / 1000) * 0.002
        assert abs(cost - expected) < 0.0001
        
        # Test with unknown model (should use default)
        cost_unknown = calculate_cost("unknown/model", usage)
        expected_default = (1000 / 1000) * 0.1 + (500 / 1000) * 0.1
        assert abs(cost_unknown - expected_default) < 0.0001
        
        # Test with model variants (e.g., :free)
        cost_variant = calculate_cost("openai/gpt-3.5-turbo:free", usage)
        # Should use base model pricing
        assert abs(cost_variant - expected) < 0.0001
    
    def test_normalize_model_id(self):
        """Test model ID normalization."""
        # Test with provider prefix
        assert normalize_model_id("openai/gpt-3.5-turbo") == "openai/gpt-3.5-turbo"
        
        # Test without provider prefix (should default to openai)
        assert normalize_model_id("gpt-3.5-turbo") == "openai/gpt-3.5-turbo"
        
        # Test with whitespace
        assert normalize_model_id(" gpt-3.5-turbo ") == "openai/gpt-3.5-turbo"
        
        # Test with other providers
        assert normalize_model_id("anthropic/claude-3-haiku") == "anthropic/claude-3-haiku"
    
    def test_validate_api_key(self):
        """Test API key validation."""
        # Valid API keys
        assert validate_api_key("sk-1234567890") is True
        assert validate_api_key("sk-abcdefghijklmnopqrstuvwxyz") is True
        
        # Invalid API keys
        assert validate_api_key("invalid-key") is False
        assert validate_api_key("sk-") is False
        assert validate_api_key("") is False
        assert validate_api_key("1234567890") is False
        assert validate_api_key("SK-1234567890") is False  # Case sensitive
    
    def test_parse_rate_limit_headers(self):
        """Test parsing rate limit headers."""
        headers = {
            'x-ratelimit-limit-requests': '100',
            'x-ratelimit-remaining-requests': '99',
            'x-ratelimit-reset-requests': '3600',
            'x-ratelimit-limit-tokens': '1000000',
            'x-ratelimit-remaining-tokens': '999000',
            'x-ratelimit-reset-tokens': '3600',
        }
        
        parsed = parse_rate_limit_headers(headers)
        
        assert parsed['limit_requests'] == 100
        assert parsed['remaining_requests'] == 99
        assert parsed['reset_requests'] == '3600'
        assert parsed['limit_tokens'] == 1000000
        assert parsed['remaining_tokens'] == 999000
        assert parsed['reset_tokens'] == '3600'
        
        # Test with alternative headers
        alt_headers = {
            'x-rateLimit-limit': '200',
            'x-rateLimit-remaining': '199',
            'x-rateLimit-reset': '1800',
        }
        
        parsed_alt = parse_rate_limit_headers(alt_headers)
        assert parsed_alt['limit_requests'] == 200
        assert parsed_alt['remaining_requests'] == 199
        assert parsed_alt['reset_requests'] == '1800'
        
        # Test with empty headers
        empty_parsed = parse_rate_limit_headers({})
        assert empty_parsed == {}
    
    def test_format_cost(self):
        """Test cost formatting."""
        from openrouter.utils import format_cost
        
        assert format_cost(0.001234) == "$0.001234"
        assert format_cost(1.234567) == "$1.234567"
        assert format_cost(0) == "$0.000000"
    
    def test_estimate_cost_for_request(self):
        """Test cost estimation for requests."""
        estimated = estimate_cost_for_request(
            "openai/gpt-3.5-turbo",
            prompt_tokens_estimate=100,
            completion_tokens_estimate=50
        )
        
        expected = (100 / 1000) * 0.0015 + (50 / 1000) * 0.002
        assert abs(estimated - expected) < 0.0001