"""
Response cache with similarity hashing for OpenRouter API calls.

Uses SimHash for efficient similarity detection, allowing cached responses
to be reused for semantically similar prompts.
"""

import hashlib
import threading
import time
import json
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


class CacheError(Exception):
    """Raised when cache operations fail."""
    pass


@dataclass
class CacheEntry:
    """Cached API response."""
    key_hash: str
    prompt_hash: str
    model: str
    messages: List[Dict[str, Any]]
    response: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int = 1
    hit_count: int = 0


class SimHash:
    """
    SimHash implementation for similarity detection.
    
    SimHash is a fingerprinting technique that allows quick similarity comparison
    by comparing the Hamming distance between hash fingerprints.
    """
    
    def __init__(self, features: str):
        """Initialize with feature string."""
        self.features = features
        self.fingerprint = self._compute_hash(features)
    
    def _compute_hash(self, features: str) -> int:
        """Compute SimHash fingerprint from features."""
        md5_hash = hashlib.md5(features.encode("utf-8")).hexdigest()
        return int(md5_hash, 16)
    
    def hamming_distance(self, other: "SimHash") -> int:
        """Calculate Hamming distance between two fingerprints."""
        xor = self.fingerprint ^ other.fingerprint
        return bin(xor).count("1")
    
    @staticmethod
    def from_hash(fingerprint: int, features: str = "") -> "SimHash":
        """Create SimHash from existing fingerprint."""
        simhash = SimHash(features)
        simhash.fingerprint = fingerprint
        return simhash


class ResponseCache:
    """
    Response cache with similarity-based lookup for OpenRouter API.
    
    Features:
    - Exact match caching
    - SimHash-based similarity detection
    - TTL (time-to-live) support
    - LRU eviction
    - Persistent storage
    """
    
    DEFAULT_TTL_SECONDS = 3600
    DEFAULT_MAX_SIZE = 1000
    DEFAULT_SIMILARITY_THRESHOLD = 3
    
    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
        persist_file: Optional[str] = None,
    ):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Default time-to-live for cache entries
            similarity_threshold: Max Hamming distance for similar prompts
            persist_file: Path to JSON file for persistent storage
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.persist_file = Path(persist_file) if persist_file else None
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        self._stats = {
            "hits": 0,
            "misses": 0,
            "similarity_hits": 0,
            "evictions": 0,
        }
        
        if self.persist_file and self.persist_file.exists():
            self._load_from_file()
    
    def _normalize_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize messages for consistent caching."""
        normalized = []
        for msg in messages:
            normalized_msg = {
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
            }
            if "name" in msg:
                normalized_msg["name"] = msg["name"]
            if "tool_calls" in msg:
                normalized_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                normalized_msg["tool_call_id"] = msg["tool_call_id"]
            normalized.append(normalized_msg)
        return normalized
    
    def _get_message_hash(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Generate deterministic hash from messages."""
        normalized = self._normalize_messages(messages)
        content = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def _get_key(
        self,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, SimHash]:
        """Generate cache key and SimHash for messages."""
        message_hash = self._get_message_hash(messages)
        key = f"{model}:{message_hash}"
        features = f"{model}:{message_hash}"
        simhash = SimHash(features)
        return key, simhash
    
    def _get_similar_keys(
        self,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """Find similar cached entries based on SimHash."""
        _, simhash = self._get_key(model, messages)
        similar_keys = []
        model_prefix = f"{model}:"
        
        for key in self._cache:
            if not key.startswith(model_prefix):
                continue
            entry = self._cache[key]
            entry_simhash = SimHash(entry.prompt_hash)
            distance = simhash.hamming_distance(entry_simhash)
            if distance <= self.similarity_threshold:
                similar_keys.append(key)
        return similar_keys
    
    def get(
        self,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Get cached response for messages."""
        with self._lock:
            key, _ = self._get_key(model, messages)
            
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry.created_at > self.ttl_seconds:
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return None
                entry.last_accessed = time.time()
                entry.access_count += 1
                entry.hit_count += 1
                self._stats["hits"] += 1
                return {
                    **entry.response,
                    "_cache_hit": True,
                    "_cache_hit_type": "exact",
                }
            
            similar_keys = self._get_similar_keys(model, messages)
            if similar_keys:
                best_key = None
                best_distance = float("inf")
                _, target_simhash = self._get_key(model, messages)
                
                for skey in similar_keys:
                    entry = self._cache[skey]
                    entry_simhash = SimHash(entry.prompt_hash)
                    distance = target_simhash.hamming_distance(entry_simhash)
                    if distance < best_distance:
                        best_distance = distance
                        best_key = skey
                
                if best_key and best_distance <= self.similarity_threshold:
                    entry = self._cache[best_key]
                    if time.time() - entry.created_at > self.ttl_seconds:
                        self._stats["misses"] += 1
                        return None
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    entry.hit_count += 1
                    self._stats["hits"] += 1
                    self._stats["similarity_hits"] += 1
                    return {
                        **entry.response,
                        "_cache_hit": True,
                        "_cache_hit_type": "similar",
                        "_similarity_distance": best_distance,
                    }
            
            self._stats["misses"] += 1
            return None
    
    def set(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response: Dict[str, Any],
    ) -> None:
        """Cache a response."""
        with self._lock:
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            key, _ = self._get_key(model, messages)
            message_hash = self._get_message_hash(messages)
            
            entry = CacheEntry(
                key_hash=key,
                prompt_hash=message_hash,
                model=model,
                messages=self._normalize_messages(messages),
                response=response,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                hit_count=0,
            )
            self._cache[key] = entry
            
            if self.persist_file and len(self._cache) % 10 == 0:
                self._save_to_file()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]
        self._stats["evictions"] += 1
    
    def invalidate(
        self,
        model: str = None,
        messages: List[Dict[str, Any]] = None,
    ) -> int:
        """Invalidate cache entries."""
        with self._lock:
            if model and not messages:
                keys_to_delete = [k for k in self._cache if k.startswith(f"{model}:")]
                for key in keys_to_delete:
                    del self._cache[key]
                return len(keys_to_delete)
            
            if messages:
                key, _ = self._get_key(model, messages)
                if key in self._cache:
                    del self._cache[key]
                    return 1
            return 0
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0, "similarity_hits": 0, "evictions": 0}
            if self.persist_file and self.persist_file.exists():
                self.persist_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "similarity_hits": self._stats["similarity_hits"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate,
            }
    
    def _save_to_file(self) -> None:
        """Save cache to JSON file."""
        if not self.persist_file:
            return
        data = {
            "entries": [
                {**asdict(entry), "response": entry.response}
                for entry in self._cache.values()
            ],
            "stats": self._stats,
            "saved_at": time.time(),
        }
        with open(self.persist_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_from_file(self) -> None:
        """Load cache from JSON file."""
        if not self.persist_file or not self.persist_file.exists():
            return
        try:
            with open(self.persist_file, "r") as f:
                data = json.load(f)
            for entry_data in data.get("entries", []):
                response = entry_data.pop("response")
                entry = CacheEntry(**entry_data, response=response)
                if time.time() - entry.created_at <= self.ttl_seconds:
                    self._cache[entry.key_hash] = entry
            if "stats" in data:
                self._stats = data["stats"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
