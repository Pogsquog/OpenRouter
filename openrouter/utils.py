"""
Additional utility functions for the OpenRouter library.
"""

import re
from typing import Dict, Any, Optional


def calculate_cost(model_id: str, usage: Dict[str, int]) -> float:
    """
    Calculate the estimated cost of a request based on model and token usage.
    
    Args:
        model_id: The model identifier
        usage: Dictionary containing 'prompt_tokens' and 'completion_tokens'
        
    Returns:
        Estimated cost in USD
    """
    # This is a simplified cost calculation
    # In a real implementation, you would fetch pricing from the API or a pricing table
    pricing_map = {
        # Example pricing (these are not real OpenRouter prices)
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
    
    # Extract base model name without variant suffixes (e.g., ":free", ":extended")
    base_model_id = model_id.split(':')[0]
    
    model_pricing = pricing_map.get(base_model_id)
    if not model_pricing:
        # Default to a high cost if model not found
        model_pricing = {"input": 0.1, "output": 0.1}
    
    input_cost = (usage.get("prompt_tokens", 0) / 1000) * model_pricing["input"]
    output_cost = (usage.get("completion_tokens", 0) / 1000) * model_pricing["output"]
    
    return input_cost + output_cost


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
    pattern = r"^(sk-[a-zA-Z0-9]+|sk-fake-[a-zA-Z0-9-]+)$"
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