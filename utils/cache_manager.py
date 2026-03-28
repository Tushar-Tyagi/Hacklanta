import os
import json
import hashlib
from typing import Dict, Any, Optional

class CacheManager:
    """
    Persistent caching manager for video and audio analysis results.
    Uses SHA256 hashes of files to identify and retrieve previous results.
    """
    
    def __init__(self, cache_dir: str = "results"):
        """Initialize the cache manager with a results directory."""
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def hash_file(self, file_path: str) -> str:
        """Generate a SHA256 hash for a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large video files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def get_cached_result(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result by its file hash."""
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None
        
    def save_result(self, file_hash: str, data: Dict[str, Any]) -> bool:
        """Save analysis results to the cache."""
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except IOError:
            return False

# Singleton instance for the app
default_cache = CacheManager()
