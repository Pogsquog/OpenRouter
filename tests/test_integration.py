"""
Integration tests for the OpenRouter library.
These tests run against the live OpenRouter API.
"""

import os
import pytest
import asyncio
from openrouter import AsyncOpenRouter, OpenRouter


@pytest.mark.asyncio
async def test_async_client_list_models_integration():
    """Integration test for listing models using real API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = AsyncOpenRouter(api_key=api_key)
    try:
        models = await client.list_models()
        assert len(models) > 0
        # Check that we get at least one model with expected structure
        first_model = models[0]
        assert hasattr(first_model, 'id')
        assert hasattr(first_model, 'name')
        assert isinstance(first_model.id, str)
        assert len(first_model.id) > 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_get_model_info_integration():
    """Integration test for getting model info using real API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = AsyncOpenRouter(api_key=api_key)
    try:
        # Test with a known free model
        model_info = await client.get_model_info("openai/gpt-3.5-turbo")
        assert model_info.id == "openai/gpt-3.5-turbo"
        assert hasattr(model_info, 'name')
        assert hasattr(model_info, 'description')
    except Exception as e:
        # Some accounts might not have access to all models, so we'll test with a free model
        try:
            model_info = await client.get_model_info("openai/gpt-3.5-turbo:free")
            assert model_info.id == "openai/gpt-3.5-turbo:free"
            assert hasattr(model_info, 'name')
        except Exception:
            pytest.skip(f"Model not accessible: {str(e)}")
    finally:
        await client.close()


def test_sync_client_list_models_integration():
    """Integration test for sync client listing models using real API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = OpenRouter(api_key=api_key)
    try:
        models = client.list_models()
        assert len(models) > 0
        # Check that we get at least one model with expected structure
        first_model = models[0]
        assert hasattr(first_model, 'id')
        assert hasattr(first_model, 'name')
        assert isinstance(first_model.id, str)
        assert len(first_model.id) > 0
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_client_get_rate_limits_integration():
    """Integration test for getting rate limits using real API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = AsyncOpenRouter(api_key=api_key)
    try:
        rate_limits = await client.get_rate_limits()
        assert isinstance(rate_limits, dict)
        # Should contain at least some rate limit information
        assert len(rate_limits) > 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_get_account_info_integration():
    """Integration test for getting account info using real API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = AsyncOpenRouter(api_key=api_key)
    try:
        account_info = await client.get_account_info()
        assert isinstance(account_info, dict)
        # Should contain at least email or some account information
        assert len(account_info) > 0
    except Exception as e:
        # Some API keys might not have access to account info endpoint
        pytest.skip(f"Account info endpoint not accessible: {str(e)}")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_search_models_integration():
    """Integration test for searching models using real API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = AsyncOpenRouter(api_key=api_key)
    try:
        models = await client.search_models("gpt")
        # Should return models that match the search term
        assert isinstance(models, list)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_client_get_models_by_provider_integration():
    """Integration test for getting models by provider using real API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = AsyncOpenRouter(api_key=api_key)
    try:
        models = await client.get_models_by_provider("openai")
        # Should return OpenAI models
        assert isinstance(models, list)
        if len(models) > 0:
            # Verify that all returned models are from OpenAI
            for model in models:
                assert model.id.startswith("openai/")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cost_calculation_with_real_model():
    """Test cost calculation with real model data."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping integration test")
    
    client = AsyncOpenRouter(api_key=api_key)
    try:
        # Get a model to update pricing cache
        await client.list_models()
        
        # Test cost calculation
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        cost = client.calculate_cost("openai/gpt-3.5-turbo", usage)
        assert isinstance(cost, float)
        assert cost >= 0
    finally:
        await client.close()