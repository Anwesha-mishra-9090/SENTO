import time
import json
from datetime import datetime, timedelta
from collections import OrderedDict


class CacheManager:
    def __init__(self, max_size=1000, default_ttl=3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key):
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]

            # Check if entry has expired
            if time.time() > entry['expires_at']:
                del self.cache[key]
                self.miss_count += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return entry['value']

        self.miss_count += 1
        return None

    def set(self, key, value, ttl=None):
        """Set value in cache"""
        if ttl is None:
            ttl = self.default_ttl

        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }

    def delete(self, key):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self):
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            'total_entries': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'max_size': self.max_size,
            'memory_usage_estimate': self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self):
        """Estimate memory usage of cache"""
        total_size = 0
        for key, value in self.cache.items():
            total_size += len(str(key)) + len(str(value))
        return total_size

    def cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)


class EmotionalCacheManager:
    def __init__(self):
        self.emotion_cache = CacheManager(max_size=500, default_ttl=1800)  # 30 minutes
        self.pattern_cache = CacheManager(max_size=100, default_ttl=7200)  # 2 hours
        self.coaching_cache = CacheManager(max_size=200, default_ttl=3600)  # 1 hour

    def cache_emotion_analysis(self, text, emotion_data):
        """Cache emotion analysis results"""
        key = f"emotion_{hash(text)}"
        self.emotion_cache.set(key, emotion_data)

    def get_cached_emotion_analysis(self, text):
        """Get cached emotion analysis"""
        key = f"emotion_{hash(text)}"
        return self.emotion_cache.get(key)

    def cache_emotional_patterns(self, user_id, patterns):
        """Cache emotional patterns for user"""
        key = f"patterns_{user_id}"
        self.pattern_cache.set(key, patterns)

    def get_cached_emotional_patterns(self, user_id):
        """Get cached emotional patterns"""
        key = f"patterns_{user_id}"
        return self.pattern_cache.get(key)

    def cache_coaching_response(self, emotion_context, response):
        """Cache coaching responses"""
        key = f"coaching_{hash(str(emotion_context))}"
        self.coaching_cache.set(key, response)

    def get_cached_coaching_response(self, emotion_context):
        """Get cached coaching response"""
        key = f"coaching_{hash(str(emotion_context))}"
        return self.coaching_cache.get(key)

    def cache_stress_prediction(self, user_id, prediction_data):
        """Cache stress prediction results"""
        key = f"stress_{user_id}"
        # Shorter TTL for predictions as they time-sensitive
        self.pattern_cache.set(key, prediction_data, ttl=900)  # 15 minutes

    def get_cached_stress_prediction(self, user_id):
        """Get cached stress prediction"""
        key = f"stress_{user_id}"
        return self.pattern_cache.get(key)

    def get_all_stats(self):
        """Get statistics for all caches"""
        return {
            'emotion_cache': self.emotion_cache.get_stats(),
            'pattern_cache': self.pattern_cache.get_stats(),
            'coaching_cache': self.coaching_cache.get_stats()
        }

    def cleanup_all_caches(self):
        """Clean up all expired cache entries"""
        emotion_cleaned = self.emotion_cache.cleanup_expired()
        pattern_cleaned = self.pattern_cache.cleanup_expired()
        coaching_cleaned = self.coaching_cache.cleanup_expired()

        return {
            'emotion_cache_cleaned': emotion_cleaned,
            'pattern_cache_cleaned': pattern_cleaned,
            'coaching_cache_cleaned': coaching_cleaned
        }

    def clear_user_data(self, user_id):
        """Clear all cached data for a specific user"""
        pattern_key = f"patterns_{user_id}"
        stress_key = f"stress_{user_id}"

        self.pattern_cache.delete(pattern_key)
        self.pattern_cache.delete(stress_key)

        return True