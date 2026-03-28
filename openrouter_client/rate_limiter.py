"""
Rate limiter using token bucket algorithm for API rate limiting.
"""

import time
import threading
from typing import Optional
from collections import defaultdict


class RateLimiter:
    """
    Token bucket rate limiter for controlling API request rates.
    
    Supports per-model rate limiting and global rate limiting.
    Uses token bucket algorithm for smooth rate control.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
        per_model_limit: Optional[int] = None,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute (global)
            tokens_per_minute: Maximum tokens per minute (global)
            per_model_limit: Maximum requests per minute per model
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.per_model_limit = per_model_limit or requests_per_minute
        
        # Token bucket state
        self._request_tokens = requests_per_minute
        self._token_tokens = tokens_per_minute
        self._last_refill = time.time()
        
        # Per-model tracking
        self._model_timestamps: dict[str, list[float]] = defaultdict(list)
        self._model_tokens: dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        
        # Refill request tokens (per minute = per 60 seconds)
        refill_rate = self.requests_per_minute / 60.0
        self._request_tokens = min(
            self.requests_per_minute,
            self._request_tokens + elapsed * refill_rate
        )
        
        # Refill token bucket (per minute = per 60 seconds)
        token_refill_rate = self.tokens_per_minute / 60.0
        self._token_tokens = min(
            self.tokens_per_minute,
            self._token_tokens + elapsed * token_refill_rate
        )
        
        self._last_refill = now
    
    def _cleanup_model_timestamps(self, model: str) -> None:
        """Remove timestamps older than 1 minute."""
        now = time.time()
        cutoff = now - 60
        self._model_timestamps[model] = [
            ts for ts in self._model_timestamps[model]
            if ts > cutoff
        ]
    
    def acquire(
        self,
        model: str = None,
        tokens: int = 0,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire tokens for making a request.
        
        Args:
            model: Model identifier for per-model rate limiting
            tokens: Number of tokens (for token-based limiting)
            wait: Whether to wait for tokens to become available
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if tokens acquired, False if would block
            
        Raises:
            TimeoutError: If timeout is reached while waiting
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                self._refill_tokens()
                
                # Check global request limit
                if self._request_tokens >= 1:
                    # Check global token limit
                    if self._token_tokens >= tokens:
                        # Check per-model limit if model specified
                        if model:
                            self._cleanup_model_timestamps(model)
                            if len(self._model_timestamps[model]) < self.per_model_limit:
                                # Consume tokens
                                self._request_tokens -= 1
                                self._token_tokens -= tokens
                                self._model_timestamps[model].append(time.time())
                                return True
                        else:
                            # Consume tokens
                            self._request_tokens -= 1
                            self._token_tokens -= tokens
                            return True
                
                if not wait:
                    return False
            
            # Calculate wait time
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                time.sleep(min(0.1, timeout - elapsed))
            else:
                time.sleep(0.1)
    
    def wait_if_needed(self, model: str = None, tokens: int = 0, timeout: float = 60.0) -> None:
        """
        Wait until tokens are available.
        
        Args:
            model: Model identifier
            tokens: Number of tokens needed
            timeout: Maximum time to wait
            
        Raises:
            TimeoutError: If timeout is reached
        """
        if not self.acquire(model, tokens, wait=True, timeout=timeout):
            raise TimeoutError(f"Rate limit wait timeout after {timeout}s")
    
    def get_available_tokens(self) -> float:
        """Get current available request tokens."""
        with self._lock:
            self._refill_tokens()
            return self._request_tokens
    
    def get_model_request_count(self, model: str) -> int:
        """Get number of requests in last minute for a model."""
        with self._lock:
            self._cleanup_model_timestamps(model)
            return len(self._model_timestamps[model])
    
    def reset(self) -> None:
        """Reset all rate limiter state."""
        with self._lock:
            self._request_tokens = self.requests_per_minute
            self._token_tokens = self.tokens_per_minute
            self._last_refill = time.time()
            self._model_timestamps.clear()
            self._model_tokens.clear()
