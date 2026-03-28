"""
Main OpenRouter client with rate limiting, error handling, retry logic,
cost tracking, and response caching.
"""

import time
import json
import threading
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .rate_limiter import RateLimiter
from .cache import ResponseCache
from .cost_tracker import CostTracker
from .exceptions import (
    OpenRouterError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    InsufficientCreditsError,
    APIError,
    NetworkError,
    TimeoutError,
)


@dataclass
class ChatMessage:
    """Chat message structure."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class ChatCompletionRequest:
    """Chat completion request parameters."""
    messages: List[Union[ChatMessage, Dict[str, Any]]]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    

@dataclass
class ChatCompletionResponse:
    """Chat completion response structure."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    provider: Optional[str] = None
    system_fingerprint: Optional[str] = None
    
    def get_content(self) -> str:
        """Get the content of the first choice."""
        if self.choices and len(self.choices) > 0:
            return self.choices[0].get("message", {}).get("content", "")
        return ""
    
    def get_first_message(self) -> Optional[ChatMessage]:
        """Get the first message from choices."""
        if self.choices and len(self.choices) > 0:
            msg = self.choices[0].get("message", {})
            return ChatMessage(
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                name=msg.get("name"),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
            )
        return None


class OpenRouterClient:
    """
    Client for interacting with OpenRouter API.
    
    Features:
    - Rate limiting with token bucket algorithm
    - Automatic retry with exponential backoff
    - Response caching with similarity hashing
    - Cost tracking and analytics
    - Comprehensive error handling
    
    Usage:
        client = OpenRouterClient(api_key="your-api-key")
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ]
        )
        print(response.get_content())
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        rate_limit_requests: int = 60,
        rate_limit_tokens: int = 100000,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        retry_on_status_codes: List[int] = None,
        timeout: float = 60.0,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
        cache_similarity_threshold: int = 3,
        cache_max_size: int = 1000,
        enable_cost_tracking: bool = True,
        cost_persist_file: str = None,
        custom_pricing: Dict[str, Dict[str, float]] = None,
        headers: Dict[str, str] = None,
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            base_url: Custom API base URL (for proxy/development)
            rate_limit_requests: Max requests per minute
            rate_limit_tokens: Max tokens per minute
            max_retries: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries (seconds)
            max_retry_delay: Maximum delay between retries (seconds)
            retry_on_status_codes: HTTP status codes to retry on
            timeout: Request timeout in seconds
            enable_caching: Enable response caching
            cache_ttl_seconds: Cache time-to-live in seconds
            cache_similarity_threshold: Max Hamming distance for similarity
            cache_max_size: Maximum cache size
            enable_cost_tracking: Enable cost tracking
            cost_persist_file: File path for cost data persistence
            custom_pricing: Custom pricing per 1M tokens
            headers: Additional headers for API requests
        """
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_limit_requests,
            tokens_per_minute=rate_limit_tokens,
        )
        
        # Initialize cache
        self.cache = ResponseCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
            similarity_threshold=cache_similarity_threshold,
        ) if enable_caching else None
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker(
            persist_file=cost_persist_file,
            pricing=custom_pricing,
        ) if enable_cost_tracking else None
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.retry_on_status_codes = retry_on_status_codes or [429, 500, 502, 503, 504]
        
        # Build session with retry logic
        self.session = self._build_session()
        
        # Default headers
        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openrouter/openrouter-python",
            "X-Title": "OpenRouter-Python-Client",
        }
        
        if headers:
            self.default_headers.update(headers)
        
        # Chat completions interface
        self.chat = _ChatCompletionsEndpoint(self)
    
    def _build_session(self) -> requests.Session:
        """Build requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0,  # We handle backoff manually
            status_forcelist=self.retry_on_status_codes,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"],
            raise_on_status=False,
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _calculate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages (approximate)."""
        # Rough estimate: ~4 characters per token
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4
    
    def _handle_response_error(
        self,
        response: requests.Response,
    ) -> None:
        """Handle API response errors and raise appropriate exceptions."""
        try:
            error_data = response.json()
        except json.JSONDecodeError:
            error_data = {"error": {"message": response.text}}
        
        error_info = error_data.get("error", {})
        error_message = error_info.get("message", "Unknown error")
        error_code = error_info.get("code")
        
        if response.status_code == 401:
            raise AuthenticationError(
                f"Authentication failed: {error_message}",
                status_code=response.status_code,
                response_data=error_data,
            )
        elif response.status_code == 403:
            raise ValidationError(
                f"Validation error: {error_message}",
                status_code=response.status_code,
                response_data=error_data,
            )
        elif response.status_code == 402 or error_code == "insufficient_quota":
            raise InsufficientCreditsError(
                f"Insufficient credits: {error_message}",
                status_code=response.status_code,
                response_data=error_data,
            )
        elif response.status_code == 429:
            retry_after = int(response.headers.get("retry-after", 60))
            raise RateLimitError(
                f"Rate limit exceeded: {error_message}",
                status_code=response.status_code,
                response_data=error_data,
                retry_after=retry_after,
            )
        elif response.status_code >= 500:
            raise APIError(
                f"Server error: {error_message}",
                error_code=error_code,
                status_code=response.status_code,
                response_data=error_data,
            )
        else:
            raise APIError(
                f"API error: {error_message}",
                error_code=error_code,
                status_code=response.status_code,
                response_data=error_data,
            )
    
    def _retry_with_backoff(
        self,
        request_func,
        *args,
        **kwargs,
    ) -> requests.Response:
        """Execute request with exponential backoff retry."""
        last_exception = None
        delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                response = request_func(*args, **kwargs)
                
                if response.status_code < 400:
                    return response
                
                if response.status_code not in self.retry_on_status_codes:
                    self._handle_response_error(response)
                    return response
                
                # Rate limit specific handling
                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", delay))
                    time.sleep(retry_after)
                    continue
                
                # Server error - retry with backoff
                last_exception = APIError(
                    f"Server error (attempt {attempt + 1}): {response.status_code}",
                    status_code=response.status_code,
                )
                
            except (RateLimitError, NetworkError, TimeoutError):
                raise
            except requests.exceptions.Timeout as e:
                last_exception = TimeoutError(
                    f"Request timeout (attempt {attempt + 1})",
                )
            except requests.exceptions.ConnectionError as e:
                last_exception = NetworkError(
                    f"Connection error (attempt {attempt + 1}): {str(e)}",
                )
            except requests.exceptions.RequestException as e:
                last_exception = APIError(f"Request failed: {str(e)}")
            
            # Wait before retry
            time.sleep(delay)
            delay = min(delay * 2, self.max_retry_delay)
        
        raise last_exception or APIError("Max retries exceeded")
    
    def request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        headers: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to OpenRouter API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        request_headers = {**self.default_headers}
        if headers:
            request_headers.update(headers)
        
        def do_request():
            return self.session.request(
                method=method.upper(),
                url=url,
                json=data,
                params=params,
                headers=request_headers,
                timeout=self.timeout,
            )
        
        response = self._retry_with_backoff(do_request)
        
        if response.status_code >= 400:
            self._handle_response_error(response)
        
        return response.json()
    
    def get_models(self) -> Dict[str, Any]:
        """Get list of available models."""
        return self.request("GET", "/models")
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get user account information."""
        return self.request("GET", "/auth/info")
    
    def get_generation(self, generation_id: str) -> Dict[str, Any]:
        """Get details of a specific generation."""
        return self.request("GET", f"/generation/{generation_id}")
    
    def inject_request(
        self,
        prompt: str,
        model: str,
        parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Inject a prompt into the OpenRouter queue (for async generation).
        """
        data = {
            "prompt": prompt,
            "model": model,
        }
        if parameters:
            data.update(parameters)
        
        return self.request("POST", "/inject", data=data)
    
    def get_config(self) -> Dict[str, Any]:
        """Get client configuration."""
        return {
            "base_url": self.base_url,
            "has_cache": self.cache is not None,
            "has_cost_tracker": self.cost_tracker is not None,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }
    
    def close(self) -> None:
        """Close the client session."""
        self.session.close()


class _ChatCompletionsEndpoint:
    """Chat completions endpoint interface."""
    
    def __init__(self, client: OpenRouterClient):
        self._client = client
    
    @property
    def completions(self):
        """Return self so client.chat.completions.create() works (OpenAI-compatible)."""
        return self
    
    def create(
        self,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        model: str = "openai/gpt-4o-mini",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ) -> ChatCompletionResponse:
        """
        Create a chat completion.
        
        Args:
            messages: List of message dictionaries or ChatMessage objects
            model: Model identifier (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-haiku")
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in completion
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Enable streaming response
            tools: Function tools for the model
            tool_choice: Tool choice configuration
            response_format: Response format (e.g., {"type": "json_object"})
            seed: Random seed for reproducibility
            use_cache: Whether to use response cache (if enabled)
            
        Returns:
            ChatCompletionResponse object
        """
        # Convert ChatMessage objects to dicts
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized_messages.append(msg.to_dict())
            else:
                normalized_messages.append(msg)
        
        # Check cache first (only for non-streaming)
        cache_hit = False
        cached_response = None
        
        if not stream and use_cache and self._client.cache:
            cached_response = self._client.cache.get(model, normalized_messages)
            if cached_response:
                cache_hit = True
                return ChatCompletionResponse(**cached_response)
        
        # Build request data
        data = {
            "model": model,
            "messages": normalized_messages,
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if top_p is not None:
            data["top_p"] = top_p
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if stop is not None:
            data["stop"] = stop
        if stream:
            data["stream"] = True
        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice
        if response_format:
            data["response_format"] = response_format
        if seed is not None:
            data["seed"] = seed
        
        # Wait for rate limit
        estimated_tokens = self._client._calculate_tokens(normalized_messages)
        self._client.rate_limiter.wait_if_needed(model=model, tokens=estimated_tokens)
        
        # Make request
        start_time = time.time()
        
        try:
            response_data = self._client.request("POST", "/chat/completions", data=data)
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract token usage
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Record cost
            if self._client.cost_tracker:
                self._client.cost_tracker.record_request(
                    model=model,
                    operation="chat",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    response_time_ms=response_time_ms,
                    success=True,
                    cache_hit=cache_hit,
                )
            
            # Cache the response (only non-streaming)
            if not stream and use_cache and self._client.cache and not cache_hit:
                self._client.cache.set(model, normalized_messages, response_data)
            
            return ChatCompletionResponse(**response_data)
            
        except OpenRouterError:
            # Record failed request
            if self._client.cost_tracker:
                self._client.cost_tracker.record_request(
                    model=model,
                    operation="chat",
                    prompt_tokens=estimated_tokens,
                    completion_tokens=0,
                    response_time_ms=int((time.time() - start_time) * 1000),
                    success=False,
                    error_message=str(OpenRouterError),
                    cache_hit=cache_hit,
                )
            raise


# Convenience function for quick initialization
def create_client(
    api_key: str,
    **kwargs,
) -> OpenRouterClient:
    """Create and return an OpenRouterClient instance."""
    return OpenRouterClient(api_key=api_key, **kwargs)
