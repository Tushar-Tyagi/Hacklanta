"""
OpenRouter Client - A comprehensive client for OpenRouter API with rate limiting,
error handling, retry logic, cost tracking, and response caching.

Exports:
    OpenRouterClient: Main client for interacting with OpenRouter API
    ResponseCache: Cache for API responses with similarity hashing
    RateLimiter: Token bucket rate limiter
    CostTracker: Track API usage costs
    Custom Exceptions: OpenRouterError, RateLimitError, AuthenticationError, etc.
"""

from .client import OpenRouterClient
from .cache import ResponseCache, CacheError
from .rate_limiter import RateLimiter
from .cost_tracker import CostTracker
from .exceptions import (
    OpenRouterError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    InsufficientCreditsError,
    APIError,
)

__all__ = [
    "OpenRouterClient",
    "ResponseCache",
    "RateLimiter",
    "CostTracker",
    "OpenRouterError",
    "RateLimitError",
    "AuthenticationError",
    "ValidationError",
    "InsufficientCreditsError",
    "APIError",
    "CacheError",
]

__version__ = "1.0.0"
