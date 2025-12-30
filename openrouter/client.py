"""
Core OpenRouter API client implementation.
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx
from pydantic import ValidationError

from .exceptions import OpenRouterError, RateLimitError, AuthenticationError, ModelNotFoundError, InvalidRequestError
from .models import ChatCompletionRequest, ChatCompletionResponse, ModelInfo, ModelListResponse
from .utils import validate_api_key


# Set up module-level logger
logger = logging.getLogger(__name__)


class AsyncOpenRouter:
    """
    Asynchronous OpenRouter API client.

    This client provides access to OpenRouter's unified API for multiple LLM providers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        log_level: str = "INFO",
        enable_request_logging: bool = False,  # Security: Don't log sensitive data by default
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var.
            base_url: Base URL for the OpenRouter API.
            http_referer: HTTP Referer header value (recommended for rankings).
            x_title: X-Title header value (recommended for application attribution).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Base delay between retries in seconds.
            backoff_factor: Factor by which to multiply the delay between retries.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            enable_request_logging: Whether to enable detailed request logging (WARNING: may log sensitive data).
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENROUTER_API_KEY environment variable")

        if not validate_api_key(self.api_key):
            raise ValueError("Invalid API key format")

        # Security: Sanitize API key to prevent logging
        self._api_key_hash = self._hash_api_key(self.api_key)

        self.base_url = base_url
        self.http_referer = http_referer
        self.x_title = x_title
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.enable_request_logging = enable_request_logging

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=self._get_headers()
        )

        self.logger.info(f"Initialized OpenRouter client with base URL: {base_url}. API key hash: {self._api_key_hash}")

    def _hash_api_key(self, api_key: str) -> str:
        """
        Create a hash of the API key for logging purposes without exposing the actual key.

        Args:
            api_key: The API key to hash

        Returns:
            A hash representation of the API key
        """
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]  # First 16 chars of hash

    def _get_headers(self) -> Dict[str, str]:
        """Get the default headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        return headers

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make an HTTP request to the OpenRouter API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/chat/completions")
            data: Request data to send
            **kwargs: Additional arguments to pass to the request

        Returns:
            httpx.Response: The API response

        Raises:
            OpenRouterError: For various API errors
        """
        # Properly join the base URL with the endpoint
        # Remove trailing slash from base_url and leading slash from endpoint to avoid issues with urljoin
        base_url = self.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        url = f"{base_url}/{endpoint}"
        headers = self._get_headers()

        self.logger.debug(f"Making {method} request to {url} (API key hash: {self._api_key_hash})")
        if data and self.enable_request_logging:
            # Only log request data if explicitly enabled for security
            self.logger.debug(f"Request data: {data}")

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                    **kwargs
                )

                self.logger.debug(f"Received response with status {response.status_code}")

                if response.status_code == 429:  # Rate limit
                    self.logger.warning(f"Rate limit hit on attempt {attempt + 1}, retrying...")
                    if attempt < self.max_retries:
                        # Exponential backoff
                        delay = self.retry_delay * (self.backoff_factor ** attempt)
                        self.logger.debug(f"Waiting {delay}s before retry")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                        raise RateLimitError(
                            f"Rate limit exceeded after {self.max_retries} retries. "
                            f"Response: {response.text}"
                        )
                elif response.status_code == 401:  # Unauthorized
                    self.logger.error("Authentication failed")
                    raise AuthenticationError(
                        f"Authentication failed. Response: {response.text}"
                    )
                elif response.status_code == 404:  # Not found
                    self.logger.warning(f"Resource not found: {response.text}")
                    raise ModelNotFoundError(
                        f"Resource not found: {response.text}"
                    )
                elif response.status_code == 422:  # Validation error
                    self.logger.error(f"Invalid request: {response.text}")
                    raise InvalidRequestError(
                        f"Invalid request: {response.text}"
                    )
                elif response.status_code >= 400:
                    self.logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    raise OpenRouterError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                self.logger.debug(f"Request successful, status: {response.status_code}")
                return response

            except httpx.TimeoutException as e:
                self.logger.warning(f"Request timed out on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    self.logger.debug(f"Waiting {delay}s before retry")
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.logger.error(f"Request timed out after {self.max_retries} retries")
                    raise OpenRouterError(f"Request timed out after {self.max_retries} retries")
            except httpx.RequestError as e:
                self.logger.warning(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    self.logger.debug(f"Waiting {delay}s before retry")
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                    raise OpenRouterError(f"Request failed after {self.max_retries} retries: {str(e)}")
    
    async def chat_completions(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Create a chat completion using the OpenRouter API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Primary model identifier to use for completion
            fallback_models: List of fallback model identifiers to try if the primary model fails
            **kwargs: Additional parameters for the completion request

        Returns:
            ChatCompletionResponse: The completion response
        """
        if fallback_models is None:
            fallback_models = []

        self.logger.info(f"Starting chat completion request with model: {model}")
        if fallback_models:
            self.logger.debug(f"Fallback models: {fallback_models}")

        # Try primary model first
        all_models = [model] + fallback_models

        last_exception = None
        for i, current_model in enumerate(all_models):
            self.logger.debug(f"Attempting model: {current_model} ({i+1}/{len(all_models)})")
            try:
                request_data = ChatCompletionRequest(
                    model=current_model,
                    messages=messages,
                    **kwargs
                )

                response = await self._make_request(
                    "POST",
                    "/chat/completions",
                    data=request_data.model_dump(exclude_none=True)
                )

                result = ChatCompletionResponse.model_validate(response.json())
                # Add model that was actually used to the response
                result.route = current_model
                self.logger.info(f"Chat completion successful using model: {current_model}")
                return result
            except (ModelNotFoundError, RateLimitError, OpenRouterError) as e:
                self.logger.warning(f"Model {current_model} failed: {str(e)}")
                last_exception = e
                # If this was the last model in the fallback list, raise the exception
                if i == len(all_models) - 1:
                    self.logger.error(f"All models failed, raising last exception: {str(e)}")
                    raise e
                # Otherwise, continue to the next model
                continue

        # This should not be reached, but just in case
        if last_exception:
            self.logger.error("No models were successful")
            raise last_exception
        else:
            self.logger.error("No models were attempted")
            raise OpenRouterError("No models were attempted")

    async def chat_completions_with_fallback(
        self,
        messages: List[Dict[str, str]],
        primary_model: str = "openai/gpt-3.5-turbo",
        fallback_strategy: str = "provider",
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Create a chat completion with automatic fallback based on strategy.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            primary_model: Primary model identifier to use for completion
            fallback_strategy: Strategy for selecting fallbacks ("provider", "performance", "cost")
            **kwargs: Additional parameters for the completion request

        Returns:
            ChatCompletionResponse: The completion response
        """
        if fallback_strategy == "provider":
            # Get models from the same provider as the primary model
            provider = primary_model.split('/')[0] if '/' in primary_model else "openai"
            available_models = await self.get_models_by_provider(provider)
            fallback_models = [model.id for model in available_models if model.id != primary_model]
        elif fallback_strategy == "performance":
            # For performance-based fallback, we might want faster/cheaper alternatives
            # This would require more complex logic based on model capabilities
            fallback_models = [
                "openai/gpt-3.5-turbo",
                "anthropic/claude-3-haiku",
                "google/gemini-pro"
            ]
            # Remove primary model if it's in the list
            fallback_models = [m for m in fallback_models if m != primary_model]
        elif fallback_strategy == "cost":
            # For cost-based fallback, use cheaper alternatives
            fallback_models = [
                "openai/gpt-3.5-turbo:free",
                "mistral/mistral-7b-instruct",
                "openai/gpt-3.5-turbo"
            ]
            fallback_models = [m for m in fallback_models if m != primary_model]
        else:
            # Default fallback
            fallback_models = [
                "openai/gpt-3.5-turbo",
                "anthropic/claude-3-haiku",
                "google/gemini-pro"
            ]
            fallback_models = [m for m in fallback_models if m != primary_model]

        return await self.chat_completions(
            messages=messages,
            model=primary_model,
            fallback_models=fallback_models,
            **kwargs
        )
    
    async def stream_chat_completions(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completions from the OpenRouter API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model identifier to use for completion
            **kwargs: Additional parameters for the completion request

        Yields:
            Dict: Stream chunks as they arrive
        """
        self.logger.info(f"Starting streaming chat completion with model: {model}")

        request_data = ChatCompletionRequest(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )

        # For streaming, we need a separate client instance to avoid conflicts
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers=self._get_headers()
        ) as client:
            async with client.stream(
                "POST",
                urljoin(self.base_url, "/chat/completions"),
                json=request_data.model_dump(exclude_none=True)
            ) as response:
                if response.status_code != 200:
                    response_text = await response.aread()
                    self.logger.error(f"Stream request failed with status {response.status_code}")
                    if response.status_code == 429:
                        raise RateLimitError(f"Rate limit exceeded: {response_text}")
                    elif response.status_code == 401:
                        raise AuthenticationError(f"Authentication failed: {response_text}")
                    elif response.status_code == 404:
                        raise ModelNotFoundError(f"Model not found: {response_text}")
                    else:
                        raise OpenRouterError(
                            f"Stream request failed with status {response.status_code}: {response_text}"
                        )

                self.logger.debug("Stream connection established, starting to yield chunks")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]  # Remove "data: " prefix
                        if chunk_data.strip() == "[DONE]":
                            self.logger.debug("Received [DONE] signal, ending stream")
                            break
                        try:
                            chunk = json.loads(chunk_data)
                            self.logger.debug(f"Yielding stream chunk: {chunk.get('id', 'no-id')}")
                            yield chunk
                        except json.JSONDecodeError:
                            self.logger.debug("Skipping invalid JSON in stream")
                            continue
    
    async def list_models(self, **kwargs) -> List[ModelInfo]:
        """
        List available models from OpenRouter API.

        Args:
            **kwargs: Additional query parameters for filtering

        Returns:
            List[ModelInfo]: List of available models
        """
        params = {}
        if kwargs:
            params.update(kwargs)

        # Build query string
        query_string = "&".join([f"{k}={v}" for k, v in params.items()]) if params else ""
        endpoint = f"/models{'?' + query_string if query_string else ''}"

        # Debug: Log the full URL that will be constructed
        import urllib.parse
        full_url = urllib.parse.urljoin(self.base_url, endpoint)
        self.logger.debug(f"Making request to: {full_url}")

        response = await self._make_request("GET", endpoint)

        # Debug: Print raw response content before parsing
        self.logger.debug(f"Raw response content: {response.text}")
        self.logger.debug(f"Response status code: {response.status_code}")
        self.logger.debug(f"Response headers: {dict(response.headers)}")

        # Check if response is empty
        if not response.text:
            self.logger.error("Received empty response from API")
            raise OpenRouterError("Received empty response from API")

        # Check if response is HTML (which would indicate we're getting the website instead of API)
        if response.text.strip().startswith('<!DOCTYPE html') or response.text.strip().startswith('<html'):
            self.logger.error("Received HTML response instead of JSON - likely incorrect API endpoint or redirect")
            raise OpenRouterError("Received HTML response instead of JSON from API")

        try:
            response_json = response.json()
            self.logger.debug(f"Parsed JSON response: {response_json}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON response: {e}")
            self.logger.error(f"Raw response text: {response.text}")
            raise OpenRouterError(f"Invalid JSON response from API: {e}")

        # Check if the response format matches what we expect
        # The API might return data in a different format than expected
        if "data" in response_json and isinstance(response_json["data"], list):
            # If there's no 'object' field, add it to match our expected format
            if "object" not in response_json:
                response_json["object"] = "list"

        try:
            response_data = ModelListResponse.model_validate(response_json)
        except Exception as e:
            self.logger.error(f"Failed to validate response: {e}")
            self.logger.error(f"Response JSON: {response_json}")
            raise OpenRouterError(f"Invalid response format from API: {e}")

        # Update pricing cache with the new model information
        from .utils import update_pricing_from_models
        update_pricing_from_models(response_data.data)

        return response_data.data

    async def get_model_info(self, model_id: str) -> ModelInfo:
        """
        Get detailed information about a specific model.

        Args:
            model_id: ID of the model to get information for

        Returns:
            ModelInfo: Detailed model information
        """
        response = await self._make_request("GET", f"/models/{model_id}")
        model_info = ModelInfo.model_validate(response.json()["data"])

        # Update pricing cache with this model information
        from .utils import update_pricing_from_models
        update_pricing_from_models([model_info])

        return model_info

    async def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get current rate limit information.

        Returns:
            Dict containing rate limit information
        """
        response = await self._make_request("GET", "/key")
        return response.json()

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including rate limits and usage.

        Returns:
            Dict containing account information
        """
        response = await self._make_request("GET", "/user")
        return response.json()

    def parse_rate_limit_headers(self, response: httpx.Response) -> Dict[str, Any]:
        """
        Parse rate limit information from response headers.

        Args:
            response: HTTP response object

        Returns:
            Dictionary containing rate limit information
        """
        from .utils import parse_rate_limit_headers as parse_headers
        return parse_headers(dict(response.headers))

    async def search_models(self, search: str) -> List[ModelInfo]:
        """
        Search for models by name or description.

        Args:
            search: Search term to look for in model names or descriptions

        Returns:
            List of matching ModelInfo objects
        """
        return await self.list_models(search=search)

    async def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """
        Get models filtered by provider.

        Args:
            provider: Provider name to filter by (e.g., "openai", "anthropic", "google")

        Returns:
            List of ModelInfo objects from the specified provider
        """
        # OpenRouter uses provider prefixes like "openai/gpt-3.5-turbo"
        all_models = await self.list_models()
        return [model for model in all_models if model.id.startswith(f"{provider}/")]

    def calculate_cost(self, model_id: str, usage: Dict[str, int]) -> float:
        """
        Calculate the estimated cost of a request based on model and token usage.

        Args:
            model_id: The model identifier
            usage: Dictionary containing 'prompt_tokens' and 'completion_tokens'

        Returns:
            Estimated cost in USD
        """
        from .utils import calculate_cost as calculate_cost_util
        return calculate_cost_util(model_id, usage)
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class OpenRouter:
    """
    Synchronous wrapper for the AsyncOpenRouter client.
    """

    def __init__(self, *args, **kwargs):
        # Extract log_level from kwargs to pass to async client
        self._async_client = AsyncOpenRouter(*args, **kwargs)

    def _hash_api_key(self, api_key: str) -> str:
        """
        Create a hash of the API key for logging purposes without exposing the actual key.

        Args:
            api_key: The API key to hash

        Returns:
            A hash representation of the API key
        """
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]  # First 16 chars of hash

    @property
    def _api_key_hash(self):
        """Get the API key hash from the async client."""
        return self._async_client._api_key_hash

    def _run_sync(self, coro):
        """Run an async coroutine synchronously."""
        # Use a simple approach with a new event loop each time to avoid threading issues
        import asyncio
        import concurrent.futures
        import threading

        def run_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        # If we're already in an event loop, run in a separate thread
        try:
            asyncio.get_running_loop()
            # We're in an event loop, use thread executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
        except RuntimeError:
            # No event loop running, safe to create and run one
            return run_in_new_loop()

    @property
    def logger(self):
        """Expose the logger from the async client."""
        return self._async_client.logger

    def chat_completions(self, messages: List[Dict[str, str]], model: str = "openai/gpt-3.5-turbo", fallback_models: Optional[List[str]] = None, **kwargs):
        """Synchronous version of chat_completions."""
        return self._run_sync(
            self._async_client.chat_completions(messages, model, fallback_models, **kwargs)
        )

    def chat_completions_with_fallback(self, messages: List[Dict[str, str]], primary_model: str = "openai/gpt-3.5-turbo", fallback_strategy: str = "provider", **kwargs):
        """Synchronous version of chat_completions_with_fallback."""
        return self._run_sync(
            self._async_client.chat_completions_with_fallback(messages, primary_model, fallback_strategy, **kwargs)
        )

    def list_models(self, **kwargs) -> List[ModelInfo]:
        """Synchronous version of list_models."""
        return self._run_sync(self._async_client.list_models(**kwargs))

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Synchronous version of get_model_info."""
        return self._run_sync(self._async_client.get_model_info(model_id))

    def get_rate_limits(self) -> Dict[str, Any]:
        """Synchronous version of get_rate_limits."""
        return self._run_sync(self._async_client.get_rate_limits())

    def search_models(self, search: str) -> List[ModelInfo]:
        """Synchronous version of search_models."""
        return self._run_sync(self._async_client.search_models(search))

    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Synchronous version of get_models_by_provider."""
        return self._run_sync(self._async_client.get_models_by_provider(provider))

    def get_account_info(self) -> Dict[str, Any]:
        """Synchronous version of get_account_info."""
        return self._run_sync(self._async_client.get_account_info())

    def calculate_cost(self, model_id: str, usage: Dict[str, int]) -> float:
        """Synchronous version of calculate_cost."""
        return self._async_client.calculate_cost(model_id, usage)

    def parse_rate_limit_headers(self, response) -> Dict[str, Any]:
        """Synchronous version of parse_rate_limit_headers."""
        return self._async_client.parse_rate_limit_headers(response)

    def close(self):
        """Close the underlying async client."""
        return self._run_sync(self._async_client.close())