"""LiteSpeed-style TTL + LRU cache - stdlib only (OrderedDict + threading).

Why not `functools.lru_cache(..., ttl=300)`: Python 3.13's lru_cache has no
`ttl` argument (signature is (maxsize, typed)), and installing cachetools /
redis just for 300s freshness is banned on the 1GB server. This module gives
the same decorator ergonomics with:

  * a STRICT `maxsize` bound (default 50) - the store can never exceed
    `maxsize` entries, so cache RAM is capped (blog payloads are ~2-40KB
    each, i.e. at most a couple of MB for 50 items),
  * a per-entry TTL (default 300s) - stale entries are dropped lazily,
  * thread safety (sync endpoints run in FastAPI's threadpool),
  * no caching of exceptions (a failed call is simply not stored).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Every store created by ttl_lru_cache - cleared together by clear_all().
_REGISTRY: list[Callable[[], None]] = []


def ttl_lru_cache(
    *, maxsize: int = 50, ttl: float = 300.0
) -> Callable[[F], F]:
    """LRU cache decorator with a per-entry TTL.

    Usage: @ttl_lru_cache(maxsize=50, ttl=300)
    """
    if maxsize < 1:
        raise ValueError("maxsize must be >= 1")
    if ttl <= 0:
        raise ValueError("ttl must be > 0")

    store: OrderedDict[tuple[Any, ...], tuple[float, Any]] = OrderedDict()
    lock = threading.Lock()

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Unhashable args (e.g. a list) -> bypass the cache entirely.
            try:
                key = (args, tuple(sorted(kwargs.items())))
                hash(key)
            except TypeError:
                return fn(*args, **kwargs)

            now = time.monotonic()
            with lock:
                hit = store.get(key)
                if hit is not None:
                    stored_at, value = hit
                    if now - stored_at < ttl:
                        store.move_to_end(key)  # LRU: mark as recently used
                        return value
                    store.pop(key, None)  # expired -> recompute below
            # Compute OUTSIDE the lock: slow fetches never block other readers.
            value = fn(*args, **kwargs)
            with lock:
                store[key] = (time.monotonic(), value)
                store.move_to_end(key)
                while len(store) > maxsize:  # strict maxsize bound
                    store.popitem(last=False)  # evict least-recently-used
            return value

        def cache_clear() -> None:
            with lock:
                store.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        _REGISTRY.append(cache_clear)
        return wrapper  # type: ignore[return-value]

    return decorator


def clear_all() -> None:
    """Drop every ttl_lru_cache store (called after a new post is published)."""
    for clear in _REGISTRY:
        clear()
