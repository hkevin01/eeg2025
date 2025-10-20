"""
Redis Feature Cache for EEG2025
================================
High-performance caching layer for features and intermediate results.
"""
import redis
import pickle
import hashlib
import logging
from typing import Any, Optional, List, Dict
import numpy as np
import os

logger = logging.getLogger(__name__)


class RedisFeatureCache:
    """Redis-backed feature and data cache with pub/sub support."""
    
    def __init__(self, host: str = None, port: int = None, db: int = 0):
        """
        Initialize Redis connection.
        
        Args:
            host: Redis host (default: from REDIS_HOST env or 'localhost')
            port: Redis port (default: from REDIS_PORT env or 6379)
            db: Redis database number
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = int(port or os.getenv('REDIS_PORT', 6379))
        self.db = db
        
        try:
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,  # for binary data
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis.ping()
            logger.info(f"✅ Connected to Redis: {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
        
        self.default_ttl = 3600  # 1 hour
    
    def cache_features(self, key: str, features: np.ndarray, ttl: int = None):
        """
        Cache computed features with optional TTL.
        
        Args:
            key: Cache key
            features: Feature array to cache
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        ttl = ttl or self.default_ttl
        
        try:
            serialized = pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL)
            self.redis.setex(f"features:{key}", ttl, serialized)
            logger.debug(f"💾 Cached features: {key} ({len(serialized)} bytes)")
        except Exception as e:
            logger.error(f"❌ Failed to cache features: {e}")
    
    def get_features(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve cached features.
        
        Args:
            key: Cache key
        
        Returns:
            Cached feature array or None if not found
        """
        try:
            data = self.redis.get(f"features:{key}")
            if data:
                features = pickle.loads(data)
                logger.debug(f"✅ Retrieved features from cache: {key}")
                return features
            return None
        except Exception as e:
            logger.error(f"❌ Failed to retrieve features: {e}")
            return None
    
    def cache_preprocessed_window(self, window_id: str, data: np.ndarray, 
                                   ttl: int = None):
        """
        Cache preprocessed EEG window with checksum.
        
        Args:
            window_id: Window identifier
            data: EEG data array
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl
        
        try:
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            checksum = hashlib.sha256(serialized).hexdigest()[:8]
            
            pipe = self.redis.pipeline()
            pipe.setex(f"window:{window_id}", ttl, serialized)
            pipe.setex(f"window:{window_id}:checksum", ttl, checksum)
            pipe.execute()
            
            logger.debug(f"💾 Cached window: {window_id} (checksum: {checksum})")
        except Exception as e:
            logger.error(f"❌ Failed to cache window: {e}")
    
    def get_window(self, window_id: str) -> Optional[np.ndarray]:
        """
        Retrieve cached preprocessed window.
        
        Args:
            window_id: Window identifier
        
        Returns:
            Cached window data or None if not found
        """
        try:
            data = self.redis.get(f"window:{window_id}")
            if data:
                window = pickle.loads(data)
                logger.debug(f"✅ Retrieved window from cache: {window_id}")
                return window
            return None
        except Exception as e:
            logger.error(f"❌ Failed to retrieve window: {e}")
            return None
    
    def increment_metric(self, key: str, amount: float = 1.0) -> float:
        """
        Atomically increment a metric counter.
        
        Args:
            key: Metric key
            amount: Amount to increment by
        
        Returns:
            New value after increment
        """
        try:
            new_value = self.redis.incrbyfloat(f"metric:{key}", amount)
            return new_value
        except Exception as e:
            logger.error(f"❌ Failed to increment metric: {e}")
            return 0.0
    
    def get_metric(self, key: str) -> float:
        """
        Get current value of a metric.
        
        Args:
            key: Metric key
        
        Returns:
            Current metric value or 0.0 if not found
        """
        try:
            value = self.redis.get(f"metric:{key}")
            return float(value) if value else 0.0
        except Exception as e:
            logger.error(f"❌ Failed to get metric: {e}")
            return 0.0
    
    def publish_training_update(self, channel: str, message: Dict):
        """
        Publish training progress update to a channel.
        
        Args:
            channel: Pub/sub channel name
            message: Message dictionary
        """
        try:
            serialized = pickle.dumps(message)
            self.redis.publish(channel, serialized)
            logger.debug(f"📡 Published update to channel: {channel}")
        except Exception as e:
            logger.error(f"❌ Failed to publish update: {e}")
    
    def subscribe_training_updates(self, channel: str):
        """
        Subscribe to training updates.
        
        Args:
            channel: Pub/sub channel name
        
        Returns:
            PubSub object for receiving messages
        """
        try:
            pubsub = self.redis.pubsub()
            pubsub.subscribe(channel)
            logger.info(f"📻 Subscribed to channel: {channel}")
            return pubsub
        except Exception as e:
            logger.error(f"❌ Failed to subscribe: {e}")
            return None
    
    def set_worker_state(self, worker_id: str, state: Dict, ttl: int = 300):
        """
        Set worker state with automatic expiration (heartbeat).
        
        Args:
            worker_id: Worker identifier
            state: Worker state dictionary
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        try:
            serialized = pickle.dumps(state)
            self.redis.setex(f"worker:{worker_id}", ttl, serialized)
            logger.debug(f"💓 Updated worker state: {worker_id}")
        except Exception as e:
            logger.error(f"❌ Failed to set worker state: {e}")
    
    def get_worker_state(self, worker_id: str) -> Optional[Dict]:
        """
        Get current worker state.
        
        Args:
            worker_id: Worker identifier
        
        Returns:
            Worker state dictionary or None if expired
        """
        try:
            data = self.redis.get(f"worker:{worker_id}")
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get worker state: {e}")
            return None
    
    def get_active_workers(self) -> List[str]:
        """
        Get list of currently active workers.
        
        Returns:
            List of active worker IDs
        """
        try:
            pattern = "worker:*"
            keys = self.redis.keys(pattern)
            worker_ids = [k.decode('utf-8').split(':', 1)[1] for k in keys]
            logger.info(f"👥 Found {len(worker_ids)} active workers")
            return worker_ids
        except Exception as e:
            logger.error(f"❌ Failed to get active workers: {e}")
            return []
    
    def cache_batch(self, batch_id: str, data: Dict, ttl: int = None):
        """
        Cache a training batch for reuse.
        
        Args:
            batch_id: Batch identifier
            data: Batch data dictionary
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl
        
        try:
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            self.redis.setex(f"batch:{batch_id}", ttl, serialized)
            logger.debug(f"💾 Cached batch: {batch_id}")
        except Exception as e:
            logger.error(f"❌ Failed to cache batch: {e}")
    
    def get_batch(self, batch_id: str) -> Optional[Dict]:
        """
        Retrieve cached batch.
        
        Args:
            batch_id: Batch identifier
        
        Returns:
            Cached batch dictionary or None
        """
        try:
            data = self.redis.get(f"batch:{batch_id}")
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to retrieve batch: {e}")
            return None
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "features:*")
        
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.redis.keys(pattern)
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"🗑️  Cleared {deleted} keys matching: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to clear pattern: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict:
        """
        Get Redis cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            info = self.redis.info('stats')
            memory = self.redis.info('memory')
            
            return {
                'total_keys': self.redis.dbsize(),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
                ) * 100,
                'memory_used_mb': memory.get('used_memory', 0) / (1024 * 1024),
                'memory_peak_mb': memory.get('used_memory_peak', 0) / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get cache stats: {e}")
            return {}
    
    def close(self):
        """Close Redis connection."""
        if self.redis:
            self.redis.close()
            logger.info("👋 Closed Redis connection")


# Convenience function for quick access
def get_cache() -> RedisFeatureCache:
    """Get a RedisFeatureCache instance with default settings."""
    return RedisFeatureCache()
