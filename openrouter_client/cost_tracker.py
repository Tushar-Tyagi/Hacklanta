"""
Cost tracker for monitoring OpenRouter API usage and costs.
"""

import json
import threading
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict


@dataclass
class TokenUsage:
    """Token usage for a single request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CostEntry:
    """Cost tracking entry for a single API call."""
    timestamp: str
    model: str
    operation: str  # 'chat' or 'completion'
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    response_time_ms: int
    success: bool
    error_message: str = ""
    cache_hit: bool = False
    

@dataclass
class CostSummary:
    """Summary of cost statistics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    cache_hits: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    avg_response_time_ms: float
    cost_by_model: Dict[str, float]
    requests_by_model: Dict[str, int]


class CostTracker:
    """
    Track and analyze OpenRouter API usage costs.
    
    Supports per-model cost tracking, cost history, and cost summaries.
    Can persist data to JSON file for long-term tracking.
    """
    
    # Default pricing per 1M tokens (as of late 2024)
    DEFAULT_PRICING = {
        # Anthropic models
        "claude-3-5-sonnet-20241022": {"prompt": 3.0, "completion": 15.0},
        "claude-3-5-sonnet-20241022:beta": {"prompt": 3.0, "completion": 15.0},
        "claude-3-opus-20240229": {"prompt": 15.0, "completion": 75.0},
        "claude-3-5-sonnet-20240620": {"prompt": 3.0, "completion": 15.0},
        "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
        
        # OpenAI models
        "gpt-4o": {"prompt": 2.5, "completion": 10.0},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
        "gpt-4": {"prompt": 30.0, "completion": 60.0},
        "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
        
        # Google models
        "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.0},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.3},
        
        # Meta models
        "meta-llama-3.1-70b-instruct": {"prompt": 0.9, "completion": 0.9},
        "meta-llama-3.1-8b-instruct": {"prompt": 0.2, "completion": 0.2},
        
        # Mistral
        "mistral-large-2407": {"prompt": 2.0, "completion": 6.0},
        "mistral-small-2401": {"prompt": 0.2, "completion": 0.6},
        
        # DeepSeek
        "deepseek-chat": {"prompt": 0.14, "completion": 0.28},
        "deepseek-coder": {"prompt": 0.14, "completion": 0.28},
        
        # Others
        "qwen-2.5-72b-instruct": {"prompt": 0.9, "completion": 0.9},
        "qwen-2.5-32b-instruct": {"prompt": 0.4, "completion": 0.4},
    }
    
    def __init__(
        self,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
        persist_file: Optional[str] = None,
    ):
        """
        Initialize the cost tracker.
        
        Args:
            pricing: Custom pricing per 1M tokens {"model": {"prompt": X, "completion": Y}}
            persist_file: Path to JSON file for persistent storage
        """
        self.pricing = {**self.DEFAULT_PRICING}
        if pricing:
            self.pricing.update(pricing)
        
        self.persist_file = Path(persist_file) if persist_file else None
        self._entries: List[CostEntry] = []
        self._lock = threading.RLock()
        
        # In-memory aggregation for efficiency
        self._model_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "success": 0,
                "failures": 0,
                "cache_hits": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
                "response_times": [],
            }
        )
        
        # Load existing data if file exists
        if self.persist_file and self.persist_file.exists():
            self._load_from_file()
    
    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost in USD based on token usage."""
        model_base = model.split(":")[0]  # Remove any :beta suffix
        
        # Try exact match first, then base model
        prices = self.pricing.get(model, self.pricing.get(model_base))
        
        if prices is None:
            # Default estimate: $1/M prompt, $2/M completion
            prompt_cost = prompt_tokens / 1_000_000 * 1.0
            completion_cost = completion_tokens / 1_000_000 * 2.0
            return prompt_cost + completion_cost
        
        prompt_cost = prompt_tokens / 1_000_000 * prices["prompt"]
        completion_cost = completion_tokens / 1_000_000 * prices["completion"]
        
        return prompt_cost + completion_cost
    
    def record_request(
        self,
        model: str,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: int,
        success: bool = True,
        error_message: str = "",
        cache_hit: bool = False,
    ) -> CostEntry:
        """
        Record a request for cost tracking.
        
        Args:
            model: Model identifier
            operation: Operation type ('chat' or 'completion')
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            response_time_ms: Response time in milliseconds
            success: Whether request succeeded
            error_message: Error message if failed
            cache_hit: Whether response was from cache
            
        Returns:
            CostEntry for the recorded request
        """
        cost_usd = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        entry = CostEntry(
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_message,
            cache_hit=cache_hit,
        )
        
        with self._lock:
            self._entries.append(entry)
            self._update_model_stats(entry)
            
            # Persist to file periodically
            if self.persist_file and len(self._entries) % 10 == 0:
                self._save_to_file()
        
        return entry
    
    def _update_model_stats(self, entry: CostEntry) -> None:
        """Update in-memory model statistics."""
        stats = self._model_stats[entry.model]
        stats["requests"] += 1
        if entry.success:
            stats["success"] += 1
        else:
            stats["failures"] += 1
        if entry.cache_hit:
            stats["cache_hits"] += 1
        stats["prompt_tokens"] += entry.prompt_tokens
        stats["completion_tokens"] += entry.completion_tokens
        stats["total_cost"] += entry.cost_usd
        stats["response_times"].append(entry.response_time_ms)
        
        # Keep only last 1000 response times to avoid memory issues
        if len(stats["response_times"]) > 1000:
            stats["response_times"] = stats["response_times"][-1000:]
    
    def get_summary(self) -> CostSummary:
        """Get summary of all tracked costs."""
        with self._lock:
            total_requests = len(self._entries)
            successful_requests = sum(1 for e in self._entries if e.success)
            failed_requests = total_requests - successful_requests
            cache_hits = sum(1 for e in self._entries if e.cache_hit)
            
            total_prompt_tokens = sum(e.prompt_tokens for e in self._entries)
            total_completion_tokens = sum(e.completion_tokens for e in self._entries)
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost_usd = sum(e.cost_usd for e in self._entries)
            
            all_response_times = [e.response_time_ms for e in self._entries]
            avg_response_time = (
                sum(all_response_times) / len(all_response_times)
                if all_response_times else 0.0
            )
            
            cost_by_model = {}
            requests_by_model = {}
            for model, stats in self._model_stats.items():
                cost_by_model[model] = stats["total_cost"]
                requests_by_model[model] = stats["requests"]
            
            return CostSummary(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                cache_hits=cache_hits,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                avg_response_time_ms=avg_response_time,
                cost_by_model=cost_by_model,
                requests_by_model=requests_by_model,
            )
    
    def get_model_summary(self, model: str) -> Optional[Dict[str, Any]]:
        """Get cost summary for a specific model."""
        with self._lock:
            if model not in self._model_stats:
                return None
            
            stats = self._model_stats[model]
            response_times = stats["response_times"]
            
            return {
                "model": model,
                "requests": stats["requests"],
                "successful_requests": stats["success"],
                "failed_requests": stats["failures"],
                "cache_hits": stats["cache_hits"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                "total_cost_usd": stats["total_cost"],
                "avg_response_time_ms": (
                    sum(response_times) / len(response_times)
                    if response_times else 0.0
                ),
            }
    
    def get_recent_entries(self, n: int = 10) -> List[CostEntry]:
        """Get the most recent N cost entries."""
        with self._lock:
            return list(self._entries[-n:])
    
    def reset(self) -> None:
        """Reset all tracked data."""
        with self._lock:
            self._entries.clear()
            self._model_stats.clear()
            if self.persist_file and self.persist_file.exists():
                self.persist_file.unlink()
    
    def _save_to_file(self) -> None:
        """Save data to JSON file."""
        if not self.persist_file:
            return
        
        data = {
            "entries": [asdict(e) for e in self._entries],
            "last_updated": datetime.utcnow().isoformat(),
        }
        
        with open(self.persist_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_from_file(self) -> None:
        """Load data from JSON file."""
        if not self.persist_file or not self.persist_file.exists():
            return
        
        try:
            with open(self.persist_file, "r") as f:
                data = json.load(f)
            
            for entry_data in data.get("entries", []):
                entry = CostEntry(**entry_data)
                self._entries.append(entry)
                self._update_model_stats(entry)
                
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # If file is corrupted, start fresh
    
    def export_to_csv(self, filepath: str) -> None:
        """Export cost data to CSV file."""
        import csv
        
        with self._lock:
            entries = list(self._entries)
        
        if not entries:
            return
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "model", "operation", "prompt_tokens",
                "completion_tokens", "total_tokens", "cost_usd",
                "response_time_ms", "success", "error_message", "cache_hit"
            ])
            
            for e in entries:
                writer.writerow([
                    e.timestamp, e.model, e.operation, e.prompt_tokens,
                    e.completion_tokens, e.total_tokens, f"{e.cost_usd:.6f}",
                    e.response_time_ms, e.success, e.error_message, e.cache_hit
                ])
