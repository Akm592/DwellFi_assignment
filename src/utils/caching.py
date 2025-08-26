from functools import lru_cache
import redis
import pickle
from typing import Optional

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)

    async def get_cached_response(self, query_hash: str) -> Optional[str]:
        """Retrieve cached response"""
        cached = self.redis_client.get(f"response:{query_hash}")
        return pickle.loads(cached) if cached else None

    async def cache_response(self, query_hash: str, response: str, ttl: int = 3600):
        """Cache response with TTL"""
        self.redis_client.setex(
            f"response:{query_hash}",
            ttl,
            pickle.dumps(response)
        )