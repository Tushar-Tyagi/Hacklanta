"""
Custom exceptions for OpenRouter client.
"""


class OpenRouterError(Exception):
    """Base exception for all OpenRouter client errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}


class RateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(OpenRouterError):
    """Raised when authentication fails."""
    pass


class ValidationError(OpenRouterError):
    """Raised when request validation fails."""
    pass


class InsufficientCreditsError(OpenRouterError):
    """Raised when account has insufficient credits."""
    
    def __init__(self, message: str = "Insufficient credits", **kwargs):
        super().__init__(message, **kwargs)


class APIError(OpenRouterError):
    """Raised when API returns an error."""
    
    def __init__(self, message: str, error_code: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = error_code


class CacheError(Exception):
    """Raised when cache operations fail."""
    pass


class NetworkError(OpenRouterError):
    """Raised when network connection fails."""
    pass


class TimeoutError(OpenRouterError):
    """Raised when request times out."""
    pass
