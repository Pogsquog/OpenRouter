"""
Data models for OpenRouter API requests and responses.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a chat conversation."""
    role: str  # "system", "user", or "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion API."""
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # OpenRouter specific parameters
    route: Optional[str] = None  # "fallback" or "direct"
    provider: Optional[Dict[str, Any]] = None  # Provider preferences


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response model for chat completion API."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    # OpenRouter specific fields
    route: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    name: str
    description: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    pricing: Optional[Dict[str, Any]] = None
    context_length: Optional[int] = None
    architecture: Optional[Dict[str, Any]] = None
    top_provider: Optional[Dict[str, Any]] = None
    per_request_limits: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    object: str
    data: List[ModelInfo]