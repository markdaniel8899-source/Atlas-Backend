import time
import unittest

from app.cache import clear_all, ttl_lru_cache


class TestTtlLruCache(unittest.TestCase):
    def test_hit_avoids_recompute(self):
        calls: list[int] = []

        @ttl_lru_cache(maxsize=5, ttl=60)
        def double(x: int) -> int:
            calls.append(x)
            return x * 2

        self.assertEqual(double(2), 4)
        self.assertEqual(double(2), 4)
        self.assertEqual(calls, [2])

    def test_strict_maxsize_evicts_least_recently_used(self):
        calls: list[int] = []

        @ttl_lru_cache(maxsize=2, ttl=60)
        def ident(x: int) -> int:
            calls.append(x)
            return x

        ident(1)
        ident(2)
        ident(3)  # maxsize=2 -> key 1 evicted
        ident(1)  # recompute
        self.assertEqual(calls, [1, 2, 3, 1])

    def test_lru_order_prefers_recently_used(self):
        calls: list[int] = []

        @ttl_lru_cache(maxsize=2, ttl=60)
        def ident(x: int) -> int:
            calls.append(x)
            return x

        ident(1)
        ident(2)
        ident(1)  # cache HIT -> refreshes 1, no recompute
        ident(3)  # evicts LRU (2), not the refreshed 1
        ident(2)  # 2 was evicted -> recompute; 1 would be a hit
        self.assertEqual(calls, [1, 2, 3, 2])

    def test_entry_expires_after_ttl(self):
        calls: list[int] = []

        @ttl_lru_cache(maxsize=5, ttl=0.05)
        def ident(x: int) -> int:
            calls.append(x)
            return x

        ident(1)
        time.sleep(0.06)
        ident(1)
        self.assertEqual(calls, [1, 1])

    def test_exceptions_are_not_cached(self):
        calls: list[int] = []

        @ttl_lru_cache(maxsize=5, ttl=60)
        def flaky(x: int) -> str:
            calls.append(x)
            if len(calls) == 1:
                raise ValueError("boom")
            return "ok"

        with self.assertRaises(ValueError):
            flaky(1)
        self.assertEqual(flaky(1), "ok")
        self.assertEqual(len(calls), 2)

    def test_kwargs_are_part_of_the_key(self):
        calls: list[int] = []

        @ttl_lru_cache(maxsize=5, ttl=60)
        def f(*, x: int) -> int:
            calls.append(x)
            return x

        f(x=1)
        f(x=2)
        f(x=1)  # cache HIT -> no recompute
        self.assertEqual(calls, [1, 2])

    def test_clear_all_drops_stores(self):
        calls: list[int] = []

        @ttl_lru_cache(maxsize=5, ttl=60)
        def ident(x: int) -> int:
            calls.append(x)
            return x

        ident(1)
        clear_all()
        ident(1)
        self.assertEqual(calls, [1, 1])

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            ttl_lru_cache(maxsize=0, ttl=60)
        with self.assertRaises(ValueError):
            ttl_lru_cache(maxsize=5, ttl=0)
