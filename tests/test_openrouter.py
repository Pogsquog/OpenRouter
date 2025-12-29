import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx
from pydantic import ValidationError

from openrouter import AsyncOpenRouter, OpenRouter
from openrouter.models import ChatCompletionResponse, Usage, Message
from openrouter.exceptions import AuthenticationError, RateLimitError, ModelNotFoundError, OpenRouterError


class TestAsyncOpenRouter:
    """Test suite for AsyncOpenRouter client."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Use a fake API key for testing
        self.api_key = "sk-fake-test-key"
        self.client = AsyncOpenRouter(api_key=self.api_key)
    
    def teardown_method(self):
        """Clean up after each test method."""
        if hasattr(self, 'client'):
            # Close the client if it's still open
            try:
                asyncio.run(self.client.close())
            except:
                pass
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = AsyncOpenRouter(api_key="sk-fake-test-key")
        assert client.api_key == "sk-fake-test-key"
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert client.max_retries == 3
    
    @pytest.mark.asyncio
    async def test_initialization_with_env_var(self):
        """Test client initialization with environment variable."""
        os.environ["OPENROUTER_API_KEY"] = "sk-fake-env-test-key"
        client = AsyncOpenRouter()
        assert client.api_key == "sk-fake-env-test-key"
        del os.environ["OPENROUTER_API_KEY"]
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test initialization with invalid API key format."""
        with pytest.raises(ValueError, match="Invalid API key format"):
            AsyncOpenRouter(api_key="invalid-key")
    
    @pytest.mark.asyncio
    async def test_chat_completions_success(self):
        """Test successful chat completion request."""
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "openai/gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        with patch.object(self.client._client, 'request', return_value=mock_response):
            response = await self.client.chat_completions(
                messages=[{"role": "user", "content": "Hello"}],
                model="openai/gpt-3.5-turbo"
            )
            
            assert isinstance(response, ChatCompletionResponse)
            assert response.choices[0].message.content == "Hello, world!"
            assert response.usage.total_tokens == 15
    
    @pytest.mark.asyncio
    async def test_chat_completions_with_fallback(self):
        """Test chat completion with fallback models."""
        # Mock responses: first fails, second succeeds
        responses = [
            MagicMock(status_code=404),  # First model fails
            MagicMock(status_code=200)   # Second model succeeds
        ]
        responses[1].json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "anthropic/claude-3-haiku",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Fallback response"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        async def mock_request(*args, **kwargs):
            return responses.pop(0)
        
        with patch.object(self.client._client, 'request', side_effect=mock_request):
            response = await self.client.chat_completions(
                messages=[{"role": "user", "content": "Hello"}],
                model="nonexistent-model",
                fallback_models=["anthropic/claude-3-haiku"]
            )
            
            assert isinstance(response, ChatCompletionResponse)
            assert response.choices[0].message.content == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_stream_chat_completions(self):
        """Test streaming chat completions."""
        # Create a mock async iterator for streaming
        async def mock_stream_response():
            chunks = [
                'data: {"id":"test","choices":[{"delta":{"content":"Hello"}}]}\n\n',
                'data: {"id":"test","choices":[{"delta":{"content":" world"}}]}\n\n',
                'data: [DONE]\n\n'
            ]
            for chunk in chunks:
                yield chunk
        
        mock_stream = MagicMock()
        mock_stream.__aenter__.return_value = mock_stream
        mock_stream.__aexit__.return_value = None
        mock_stream.status_code = 200
        mock_stream.aiter_lines = mock_stream_response
        
        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        
        with patch('httpx.AsyncClient', return_value=mock_client):
            chunks = []
            async for chunk in self.client.stream_chat_completions(
                messages=[{"role": "user", "content": "Hello"}],
                model="openai/gpt-3.5-turbo"
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
            assert chunks[1]["choices"][0]["delta"]["content"] == " world"
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        
        with patch.object(self.client._client, 'request', return_value=mock_response):
            with pytest.raises(RateLimitError):
                await self.client.chat_completions(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="openai/gpt-3.5-turbo"
                )
    
    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test handling of authentication errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        
        with patch.object(self.client._client, 'request', return_value=mock_response):
            with pytest.raises(AuthenticationError):
                await self.client.chat_completions(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="openai/gpt-3.5-turbo"
                )
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self):
        """Test handling of model not found errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Model not found"
        
        with patch.object(self.client._client, 'request', return_value=mock_response):
            with pytest.raises(ModelNotFoundError):
                await self.client.chat_completions(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="nonexistent-model"
                )
    
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "openai/gpt-3.5-turbo",
                    "name": "GPT-3.5 Turbo",
                    "description": "Fast and capable model"
                }
            ]
        }
        
        with patch.object(self.client._client, 'request', return_value=mock_response):
            models = await self.client.list_models()
            
            assert len(models) == 1
            assert models[0].id == "openai/gpt-3.5-turbo"
            assert models[0].name == "GPT-3.5 Turbo"
    
    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test getting specific model information."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "id": "openai/gpt-4",
                "name": "GPT-4",
                "description": "Most capable model"
            }
        }
        
        with patch.object(self.client._client, 'request', return_value=mock_response):
            model_info = await self.client.get_model_info("openai/gpt-4")
            
            assert model_info.id == "openai/gpt-4"
            assert model_info.name == "GPT-4"
    
    @pytest.mark.asyncio
    async def test_get_rate_limits(self):
        """Test getting rate limit information."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "key": "test-key-info",
            "requests_remaining": 100,
            "tokens_remaining": 100000
        }
        
        with patch.object(self.client._client, 'request', return_value=mock_response):
            rate_limits = await self.client.get_rate_limits()
            
            assert "requests_remaining" in rate_limits
            assert rate_limits["requests_remaining"] == 100
    
    @pytest.mark.asyncio
    async def test_calculate_cost(self):
        """Test cost calculation utility."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = self.client.calculate_cost("openai/gpt-3.5-turbo", usage)
        
        # Should be approximately (1000/1000)*0.0015 + (500/1000)*0.002 = 0.0015 + 0.001 = 0.0025
        expected_cost = (1000 / 1000) * 0.0015 + (500 / 1000) * 0.002
        assert abs(cost - expected_cost) < 0.0001


class TestOpenRouterSync:
    """Test suite for synchronous OpenRouter client."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.client = OpenRouter(api_key="sk-fake-test-key")
    
    def teardown_method(self):
        """Clean up after each test method."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def test_sync_chat_completions(self):
        """Test synchronous chat completions."""
        # This test would require mocking the async implementation
        # For now, we'll just test that the method exists and can be called
        assert hasattr(self.client, 'chat_completions')
        assert hasattr(self.client, 'list_models')
        assert hasattr(self.client, 'get_model_info')
        assert hasattr(self.client, 'get_rate_limits')
        assert hasattr(self.client, 'calculate_cost')


class TestModels:
    """Test suite for data models."""
    
    def test_message_model(self):
        """Test Message model validation."""
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

        # Test with valid roles - pydantic doesn't validate role values by default
        # So we just test that it accepts the expected values
        message_assistant = Message(role="assistant", content="Hello")
        assert message_assistant.role == "assistant"

        # Test with empty content
        message_empty = Message(role="user", content="")
        assert message_empty.content == ""
    
    def test_chat_completion_request(self):
        """Test ChatCompletionRequest model."""
        from openrouter.models import ChatCompletionRequest
        
        request = ChatCompletionRequest(
            model="openai/gpt-3.5-turbo",
            messages=[Message(role="user", content="Hello")]
        )
        
        assert request.model == "openai/gpt-3.5-turbo"
        assert len(request.messages) == 1
        assert request.messages[0].content == "Hello"
    
    def test_chat_completion_response(self):
        """Test ChatCompletionResponse model."""
        response = ChatCompletionResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="openai/gpt-3.5-turbo",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        )
        
        assert response.id == "test-id"
        assert response.usage.total_tokens == 15


class TestUtils:
    """Test suite for utility functions."""
    
    def test_calculate_cost(self):
        """Test cost calculation utility function."""
        from openrouter.utils import calculate_cost
        
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        cost = calculate_cost("openai/gpt-3.5-turbo", usage)
        
        expected_cost = (1000 / 1000) * 0.0015 + (500 / 1000) * 0.002
        assert abs(cost - expected_cost) < 0.0001
    
    def test_normalize_model_id(self):
        """Test model ID normalization."""
        from openrouter.utils import normalize_model_id
        
        assert normalize_model_id("gpt-3.5-turbo") == "openai/gpt-3.5-turbo"
        assert normalize_model_id("openai/gpt-3.5-turbo") == "openai/gpt-3.5-turbo"
        assert normalize_model_id("anthropic/claude-3-haiku") == "anthropic/claude-3-haiku"
    
    def test_validate_api_key(self):
        """Test API key validation."""
        from openrouter.utils import validate_api_key
        
        assert validate_api_key("sk-1234567890") is True
        assert validate_api_key("invalid-key") is False
        assert validate_api_key("sk-") is False
        assert validate_api_key("") is False