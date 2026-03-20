import hashlib
import json
from typing import Iterable
from .models import LLMCache, Message

class LLMCacheService:
    @staticmethod
    def compute_cache_key(
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        msg_data = [{"role": m.type, "content": m.text} for m in messages]
        key_data = {
            "model": model_name,
            "messages": msg_data,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    @staticmethod
    def lookup_cache(cache_key: str) -> LLMCache | None:
        try:
            cache_item = LLMCache.objects.get(cache_key=cache_key)
            cache_item.hit_count += 1
            cache_item.save(update_fields=["hit_count"])
            return cache_item
        except LLMCache.DoesNotExist:
            return None

    @staticmethod
    def save_to_cache(
        cache_key: str, model_name: str, response_text: str, response_data: dict
    ):
        LLMCache.objects.get_or_create(
            cache_key=cache_key,
            defaults={
                "model_name": model_name,
                "response_text": response_text,
                "response_data": response_data,
            },
        )
