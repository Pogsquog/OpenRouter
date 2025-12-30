"""
Additional utility functions for the OpenRouter library.
"""

import re
import time
from typing import Dict, Any, Optional
from .models import ModelInfo


class PricingService:
    """
    Service for managing dynamic pricing information for OpenRouter models.
    Fetches pricing data from the API and caches it to avoid repeated requests.
    """

    def __init__(self):
        self._pricing_cache = {}
        self._last_updated = 0
        self._cache_duration = 3600  # Cache for 1 hour (3600 seconds)

    def _is_cache_valid(self) -> bool:
        """Check if the pricing cache is still valid."""
        return (time.time() - self._last_updated) < self._cache_duration

    def _get_default_pricing(self) -> Dict[str, Dict[str, float]]:
        """
        Get default pricing as fallback when API pricing is not available.
        Note: These are example values and should be updated with real OpenRouter pricing.
        """
        return {
            "openai/gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
            "openai/gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "openai/gpt-4-turbo": {"input": 0.01, "output": 0.03},  # per 1K tokens
            "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125},  # per 1K tokens
            "anthropic/claude-3-sonnet": {"input": 0.003, "output": 0.015},  # per 1K tokens
            "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},  # per 1K tokens
            "google/gemini-pro": {"input": 0.0005, "output": 0.0015},  # per 1K tokens
            "mistral/mistral-7b-instruct": {"input": 0.0002, "output": 0.0002},  # per 1K tokens
            "nousresearch/nous-hermes-2-mixtral-8x7b-dpo": {"input": 0.002, "output": 0.002},  # per 1K tokens
        }

    def update_pricing_from_models(self, models: list) -> None:
        """
        Update pricing cache from a list of ModelInfo objects.

        Args:
            models: List of ModelInfo objects containing pricing information
        """
        new_pricing = {}

        for model in models:
            if hasattr(model, 'pricing') and model.pricing:
                pricing_info = model.pricing
                # Extract input and output pricing if available
                input_cost = pricing_info.get('prompt', 0.0)
                output_cost = pricing_info.get('completion', 0.0)

                # Convert to per 1K tokens format if needed
                if 'per_token' in pricing_info:
                    input_cost *= 1000  # Convert per-token to per-1K
                    output_cost *= 1000

                new_pricing[model.id] = {
                    "input": input_cost,
                    "output": output_cost
                }
            else:
                # Use default pricing if model doesn't have pricing info
                default_pricing = self._get_default_pricing()
                if model.id in default_pricing:
                    new_pricing[model.id] = default_pricing[model.id]

        self._pricing_cache = new_pricing
        self._last_updated = time.time()

    def get_pricing_for_model(self, model_id: str) -> Dict[str, float]:
        """
        Get pricing information for a specific model.

        Args:
            model_id: The model identifier

        Returns:
            Dictionary with 'input' and 'output' pricing per 1K tokens
        """
        # Extract base model name without variant suffixes (e.g., ":free", ":extended")
        base_model_id = model_id.split(':')[0]

        if base_model_id in self._pricing_cache:
            return self._pricing_cache[base_model_id]

        # If not in cache, try to get from default pricing
        default_pricing = self._get_default_pricing()
        if base_model_id in default_pricing:
            return default_pricing[base_model_id]

        # Default to high cost if model not found
        return {"input": 0.1, "output": 0.1}


# Global pricing service instance
_pricing_service = PricingService()


def calculate_cost(model_id: str, usage: Dict[str, int]) -> float:
    """
    Calculate the estimated cost of a request based on model and token usage.

    Args:
        model_id: The model identifier
        usage: Dictionary containing 'prompt_tokens' and 'completion_tokens'

    Returns:
        Estimated cost in USD
    """
    pricing_service = get_pricing_service()
    model_pricing = pricing_service.get_pricing_for_model(model_id)

    input_cost = (usage.get("prompt_tokens", 0) / 1000) * model_pricing["input"]
    output_cost = (usage.get("completion_tokens", 0) / 1000) * model_pricing["output"]

    return input_cost + output_cost


def get_pricing_service() -> PricingService:
    """
    Get the global pricing service instance.

    Returns:
        PricingService instance
    """
    return _pricing_service


def update_pricing_from_models(models: list) -> None:
    """
    Update the pricing cache from a list of ModelInfo objects.

    Args:
        models: List of ModelInfo objects containing pricing information
    """
    _pricing_service.update_pricing_from_models(models)


def normalize_model_id(model_id: str) -> str:
    """
    Normalize a model ID by removing any provider prefixes if needed.
    
    Args:
        model_id: The raw model identifier
        
    Returns:
        Normalized model identifier
    """
    # Remove any leading/trailing whitespace
    model_id = model_id.strip()
    
    # Ensure it follows the provider/model format
    if "/" not in model_id:
        # If no provider specified, default to openai
        model_id = f"openai/{model_id}"
    
    return model_id


def validate_api_key(api_key: str) -> bool:
    """
    Validate an OpenRouter API key format.

    Args:
        api_key: The API key to validate

    Returns:
        True if valid, False otherwise
    """
    # OpenRouter API keys typically start with "sk-" and are followed by alphanumeric characters
    # For testing, we also allow "sk-fake-" prefix
    # Also support the new "sk-or-v1-" format
    pattern = r"^(sk-[a-zA-Z0-9]+|sk-fake-[a-zA-Z0-9-]+|sk-or-v1-[a-zA-Z0-9-]+)$"
    return bool(re.match(pattern, api_key))


def parse_rate_limit_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse rate limit information from response headers.
    
    Args:
        headers: HTTP response headers
        
    Returns:
        Dictionary containing rate limit information
    """
    rate_limit_info = {}
    
    # Standard rate limit headers
    if 'x-ratelimit-limit-requests' in headers:
        rate_limit_info['limit_requests'] = int(headers['x-ratelimit-limit-requests'])
    if 'x-ratelimit-remaining-requests' in headers:
        rate_limit_info['remaining_requests'] = int(headers['x-ratelimit-remaining-requests'])
    if 'x-ratelimit-reset-requests' in headers:
        rate_limit_info['reset_requests'] = headers['x-ratelimit-reset-requests']
    
    if 'x-ratelimit-limit-tokens' in headers:
        rate_limit_info['limit_tokens'] = int(headers['x-ratelimit-limit-tokens'])
    if 'x-ratelimit-remaining-tokens' in headers:
        rate_limit_info['remaining_tokens'] = int(headers['x-ratelimit-remaining-tokens'])
    if 'x-ratelimit-reset-tokens' in headers:
        rate_limit_info['reset_tokens'] = headers['x-ratelimit-reset-tokens']
    
    # Alternative headers that might be used
    if 'x-rateLimit-limit' in headers:
        rate_limit_info['limit_requests'] = int(headers['x-rateLimit-limit'])
    if 'x-rateLimit-remaining' in headers:
        rate_limit_info['remaining_requests'] = int(headers['x-rateLimit-remaining'])
    if 'x-rateLimit-reset' in headers:
        rate_limit_info['reset_requests'] = headers['x-rateLimit-reset']
    
    return rate_limit_info


def format_cost(cost: float) -> str:
    """
    Format a cost value for display.
    
    Args:
        cost: Cost value in USD
        
    Returns:
        Formatted cost string
    """
    return f"${cost:.6f}"


def estimate_cost_for_request(model_id: str, prompt_tokens_estimate: int, 
                             completion_tokens_estimate: int) -> float:
    """
    Estimate the cost for a request based on token estimates.
    
    Args:
        model_id: The model identifier
        prompt_tokens_estimate: Estimated number of prompt tokens
        completion_tokens_estimate: Estimated number of completion tokens
        
    Returns:
        Estimated cost in USD
    """
    usage = {
        "prompt_tokens": prompt_tokens_estimate,
        "completion_tokens": completion_tokens_estimate
    }
    return calculate_cost(model_id, usage)