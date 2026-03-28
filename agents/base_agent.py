"""
Base Agent class with dual-processing logic, confidence scoring,
API fallback, caching layer, and graceful degradation.
"""

import time
import hashlib
import json
import threading
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Try importing local ML dependencies
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LOCAL_ML_AVAILABLE = True
except ImportError:
    LOCAL_ML_AVAILABLE = False

from openrouter_client import OpenRouterClient, ResponseCache
from openrouter_client.exceptions import OpenRouterError, APIError


class ProcessingMode(Enum):
    """Processing mode for the agent."""
    LOCAL_ONLY = "local_only"
    API_ONLY = "api_only"
    HYBRID = "hybrid"  # Local first, then API fallback
    API_FALLBACK = "api_fallback"  # Try local, fallback to API on failure


@dataclass
class AgentResponse:
    """Standardized response object for agent outputs."""
    content: str
    confidence: float
    mode_used: str
    source: str  # "local" or "api"
    model_used: Optional[str] = None
    processing_time_ms: int = 0
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "confidence": self.confidence,
            "mode_used": self.mode_used,
            "source": self.source,
            "model_used": self.model_used,
            "processing_time_ms": self.processing_time_ms,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "metadata": self.metadata,
        }


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class LocalModelError(AgentError):
    """Raised when local model processing fails."""
    pass


class APIFallbackError(AgentError):
    """Raised when API fallback processing fails."""
    pass


class BaseAgent(ABC):
    """
    Base Agent class with dual-processing capabilities.

    Features:
    - Local processing using ML models (when available)
    - API processing using OpenRouter
    - Confidence scoring for responses
    - Caching layer for both local and API responses
    - Graceful degradation when local models unavailable
    - API fallback on local model failure

    Usage:
        agent = BaseAgent(
            api_key="your-api-key",
            mode=ProcessingMode.HYBRID,
            local_model_name="gpt2",
            api_model="openai/gpt-4o-mini",
        )
        response = agent.process("Your prompt here")
        print(response.content, response.confidence)
    """

    DEFAULT_API_MODEL = "openai/gpt-4o-mini"
    DEFAULT_LOCAL_MODEL = "gpt2"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        api_key: str,
        mode: ProcessingMode = ProcessingMode.HYBRID,
        local_model_name: str = DEFAULT_LOCAL_MODEL,
        api_model: str = DEFAULT_API_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,
        max_retries: int = 3,
        local_model_device: str = "cpu",
        local_temperature: float = 0.7,
        api_temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the Base Agent.

        Args:
            api_key: OpenRouter API key
            mode: Processing mode (LOCAL_ONLY, API_ONLY, HYBRID, API_FALLBACK)
            local_model_name: Name of local model to use
            api_model: OpenRouter model identifier
            confidence_threshold: Minimum confidence for local model acceptance
            enable_cache: Enable response caching
            cache_ttl_seconds: Cache time-to-live in seconds
            max_retries: Maximum API retry attempts
            local_model_device: Device for local model ("cpu", "cuda", "mps")
            local_temperature: Temperature for local model
            api_temperature: Temperature for API model
            max_tokens: Maximum tokens in response
            system_prompt: System prompt for the agent
        """
        self.api_key = api_key
        self.mode = mode
        self.local_model_name = local_model_name
        self.api_model = api_model
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.local_model_device = local_model_device
        self.local_temperature = local_temperature
        self.api_temperature = api_temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        # Track availability
        self._local_model_available = False
        self._local_model = None
        self._local_tokenizer = None

        # Initialize OpenRouter client
        self._client = OpenRouterClient(
            api_key=api_key,
            enable_caching=enable_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            max_retries=max_retries,
        )

        # Initialize cache for agent responses
        self._agent_cache = ResponseCache(
            ttl_seconds=cache_ttl_seconds,
            max_size=500,
        ) if enable_cache else None

        # Initialize local model if requested and available
        self._initialize_local_model()

        # Statistics
        self._stats = {
            "local_requests": 0,
            "api_requests": 0,
            "cache_hits": 0,
            "local_failures": 0,
            "api_failures": 0,
        }
        self._stats_lock = threading.RLock()

    def _initialize_local_model(self) -> None:
        """Initialize local ML model for offline processing."""
        if not LOCAL_ML_AVAILABLE:
            self._local_model_available = False
            return

        try:
            # Load local model and tokenizer
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_name
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name
            )

            # Move to appropriate device
            if self.local_model_device == "mps" and hasattr(torch, 'mps'):
                self._local_model = self._local_model.to("mps")
            elif self.local_model_device == "cuda" and torch.cuda.is_available():
                self._local_model = self._local_model.to("cuda")
            else:
                self._local_model = self._local_model.to("cpu")

            self._local_model.eval()
            self._local_model_available = True

        except Exception as e:
            self._local_model_available = False

    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate cache key for a prompt."""
        combined = f"{system_prompt or ''}:{prompt}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _check_cache(self, prompt: str) -> Optional[AgentResponse]:
        """Check cache for cached response."""
        if not self._agent_cache:
            return None

        cache_key = self._get_cache_key(prompt, self.system_prompt)
        cached = self._agent_cache.get("agent", [{"role": "user", "content": cache_key}])

        if cached:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            return AgentResponse(
                content=cached.get("content", ""),
                confidence=cached.get("confidence", 0.0),
                mode_used=cached.get("mode_used", "cache"),
                source="cache",
                model_used=cached.get("model_used"),
                processing_time_ms=cached.get("processing_time_ms", 0),
                cache_hit=True,
            )
        return None

    def _set_cache(self, prompt: str, response: AgentResponse) -> None:
        """Cache the response."""
        if not self._agent_cache:
            return

        cache_data = response.to_dict()
        self._agent_cache.set(
            "agent",
            [{"role": "user", "content": self._get_cache_key(prompt, self.system_prompt)}],
            cache_data,
        )

    def _calculate_confidence(
        self,
        prompt: str,
        response: str,
        local_model_logprobs: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate confidence score for the response.

        Args:
            prompt: Input prompt
            response: Generated response
            local_model_logprobs: Log probabilities from local model (if available)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # If local model with logprobs, use them
        if local_model_logprobs:
            avg_logprob = sum(local_model_logprobs) / len(local_model_logprobs)
            # Convert log probabilities to probability (approximate)
            prob = min(1.0, max(0.0, (avg_logprob + 2) / 2))
            confidence = prob
        else:
            # Heuristic-based confidence for API responses
            # Length-based scoring
            if len(response) > 10:
                confidence += 0.1
            if len(response) > 50:
                confidence += 0.1

            # Check for meaningful response (not just repetition)
            if len(set(response.split())) > 5:
                confidence += 0.1

            # Check for error indicators
            error_indicators = ["error", "sorry", "cannot", "unable", "don't know"]
            if any(ind in response.lower() for ind in error_indicators):
                confidence -= 0.2

            # Check for proper sentence structure
            if response[0].isupper() if response else False:
                confidence += 0.1

        return min(1.0, max(0.0, confidence))

    @abstractmethod
    def _local_processing_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Implement local processing logic. Subclasses can override.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt

        Returns:
            AgentResponse with local processing results
        """
        pass

    def _default_local_processing(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Default local processing using transformers library.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt

        Returns:
            AgentResponse with local processing results
        """
        if not self._local_model_available:
            raise LocalModelError("Local model not available")

        start_time = time.time()

        try:
            # Build full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            elif self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"

            # Tokenize
            inputs = self._local_tokenizer(
                full_prompt,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            if self.local_model_device == "mps" and hasattr(torch, 'mps'):
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            elif self.local_model_device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._local_model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.local_temperature,
                    do_sample=True,
                    pad_token_id=self._local_tokenizer.eos_token_id,
                )

            # Decode response
            generated_text = self._local_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Extract response (remove prompt from output)
            response = generated_text[len(full_prompt):].strip()

            # Calculate confidence
            confidence = self._calculate_confidence(prompt, response)

            processing_time = int((time.time() - start_time) * 1000)

            return AgentResponse(
                content=response,
                confidence=confidence,
                mode_used=self.mode.value,
                source="local",
                model_used=self.local_model_name,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            raise LocalModelError(f"Local processing failed: {str(e)}")

    def _api_processing(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process prompt using OpenRouter API.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt

        Returns:
            AgentResponse with API processing results
        """
        start_time = time.time()

        # Build messages
        messages = []
        if system_prompt or self.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.system_prompt,
            })
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                temperature=self.api_temperature,
                max_tokens=self.max_tokens,
            )

            content = response.get_content()
            processing_time = int((time.time() - start_time) * 1000)

            # Calculate confidence (API responses typically get higher confidence)
            confidence = self._calculate_confidence(prompt, content)
            confidence = min(1.0, confidence + 0.15)  # Boost for API responses

            return AgentResponse(
                content=content,
                confidence=confidence,
                mode_used=self.mode.value,
                source="api",
                model_used=self.api_model,
                processing_time_ms=processing_time,
            )

        except OpenRouterError as e:
            raise APIFallbackError(f"API processing failed: {str(e)}")

    def process(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        force_mode: Optional[ProcessingMode] = None,
    ) -> AgentResponse:
        """
        Process a prompt using dual-processing logic.

        Args:
            prompt: Input prompt to process
            system_prompt: Optional system prompt override
            force_mode: Force a specific processing mode

        Returns:
            AgentResponse with processed results
        """
        # Check cache first
        cached = self._check_cache(prompt)
        if cached:
            return cached

        mode = force_mode or self.mode
        start_time = time.time()

        # Determine processing strategy based on mode
        if mode == ProcessingMode.LOCAL_ONLY:
            response = self._process_local_only(prompt, system_prompt)
        elif mode == ProcessingMode.API_ONLY:
            response = self._process_api_only(prompt, system_prompt)
        elif mode == ProcessingMode.HYBRID:
            response = self._process_hybrid(prompt, system_prompt)
        elif mode == ProcessingMode.API_FALLBACK:
            response = self._process_with_fallback(prompt, system_prompt)
        else:
            raise AgentError(f"Unknown processing mode: {mode}")

        # Cache the response
        if not response.error:
            self._set_cache(prompt, response)

        return response

    def _process_local_only(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """Process using local model only."""
        if not self._local_model_available:
            return AgentResponse(
                content="",
                confidence=0.0,
                mode_used=self.mode.value,
                source="local",
                error="Local model not available",
            )

        with self._stats_lock:
            self._stats["local_requests"] += 1

        # Use custom or default local processing
        if self._local_processing_impl.__func__ is BaseAgent._local_processing_impl:
            response = self._default_local_processing(prompt, system_prompt)
        else:
            response = self._local_processing_impl(prompt, system_prompt)

        return response

    def _process_api_only(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """Process using API only."""
        with self._stats_lock:
            self._stats["api_requests"] += 1

        return self._api_processing(prompt, system_prompt)

    def _process_hybrid(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process using hybrid mode - local first, then API if confidence low.
        """
        # Try local first if available
        if self._local_model_available:
            with self._stats_lock:
                self._stats["local_requests"] += 1

            try:
                if self._local_processing_impl.__func__ is BaseAgent._local_processing_impl:
                    response = self._default_local_processing(prompt, system_prompt)
                else:
                    response = self._local_processing_impl(prompt, system_prompt)

                # Check if confidence meets threshold
                if response.confidence >= self.confidence_threshold:
                    return response

                # Low confidence - fallback to API
            except LocalModelError as e:
                with self._stats_lock:
                    self._stats["local_failures"] += 1

        # Use API
        with self._stats_lock:
            self._stats["api_requests"] += 1

        return self._api_processing(prompt, system_prompt)

    def _process_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process with API fallback - try local, fallback to API on failure.
        """
        # Try local first if available
        if self._local_model_available:
            with self._stats_lock:
                self._stats["local_requests"] += 1

            try:
                if self._local_processing_impl.__func__ is BaseAgent._local_processing_impl:
                    response = self._default_local_processing(prompt, system_prompt)
                else:
                    response = self._local_processing_impl(prompt, system_prompt)

                return response

            except LocalModelError as e:
                with self._stats_lock:
                    self._stats["local_failures"] += 1

        # Fallback to API
        with self._stats_lock:
            self._stats["api_requests"] += 1

        return self._api_processing(prompt, system_prompt)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        with self._stats_lock:
            stats = dict(self._stats)

        stats["local_model_available"] = self._local_model_available
        stats["local_model_name"] = self.local_model_name
        stats["api_model"] = self.api_model
        stats["current_mode"] = self.mode.value
        stats["cache_stats"] = (
            self._agent_cache.get_stats() if self._agent_cache else {}
        )

        return stats

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        with self._stats_lock:
            self._stats = {
                "local_requests": 0,
                "api_requests": 0,
                "cache_hits": 0,
                "local_failures": 0,
                "api_failures": 0,
            }

    def clear_cache(self) -> None:
        """Clear agent response cache."""
        if self._agent_cache:
            self._agent_cache.clear()

    @property
    def is_local_available(self) -> bool:
        """Check if local model is available."""
        return self._local_model_available

    def switch_mode(self, mode: ProcessingMode) -> None:
        """Switch processing mode."""
        self.mode = mode

    def close(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()

        # Clear local model to free memory
        if self._local_model is not None:
            del self._local_model
            self._local_model = None
        if self._local_tokenizer is not None:
            del self._local_tokenizer
            self._local_tokenizer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
