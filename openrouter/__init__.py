"""
OpenRouter Python Library
A comprehensive, production-ready library for interfacing with the OpenRouter API.
"""

__version__ = "0.1.0"

from .client import AsyncOpenRouter, OpenRouter

__all__ = ["AsyncOpenRouter", "OpenRouter"]