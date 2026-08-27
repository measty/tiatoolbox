"""Byte-bounded memory and persistent caches for immutable tile payloads."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class TilePayload:
    """An encoded HTTP tile response independent of Flask."""

    data: bytes
    content_type: str
    content_encoding: str | None = None
    etag: str | None = None
    representation: str | None = None
    feature_count: int = 0
    vertex_count: int = 0
    cache_status: str | None = None
    server_timing: str | None = None

    @property
    def size(self) -> int:
        """Return encoded payload size in bytes."""
        return len(self.data)

    def with_etag(self) -> TilePayload:
        """Return this payload with a deterministic strong ETag."""
        if self.etag is not None:
            return self
        digest = hashlib.sha256(self.data).hexdigest()
        return TilePayload(
            data=self.data,
            content_type=self.content_type,
            content_encoding=self.content_encoding,
            etag=f'"{digest}"',
            representation=self.representation,
            feature_count=self.feature_count,
            vertex_count=self.vertex_count,
            cache_status=self.cache_status,
            server_timing=self.server_timing,
        )


class ByteLRUCache:
    """Thread-safe LRU whose limit is encoded bytes rather than entry count."""

    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        """Initialise the cache."""
        if max_bytes <= 0:
            msg = "Cache byte limit must be positive."
            raise ValueError(msg)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[str, TilePayload] = OrderedDict()
        self._size = 0
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        """Return the number of cached bytes."""
        with self._lock:
            return self._size

    def get(self, key: str) -> TilePayload | None:
        """Return and promote an entry, or ``None`` on a miss."""
        with self._lock:
            payload = self._entries.pop(key, None)
            if payload is None:
                return None
            self._entries[key] = payload
            return payload

    def put(self, key: str, payload: TilePayload) -> None:
        """Insert an entry and evict least-recently-used bytes."""
        payload = payload.with_etag()
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._size -= previous.size
            if payload.size > self.max_bytes:
                return
            self._entries[key] = payload
            self._size += payload.size
            while self._size > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._size -= evicted.size

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()
            self._size = 0


class PersistentTileCache:
    """A small SQLite BLOB cache for on-demand immutable tiles."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
        touch_batch_size: int = 128,
    ) -> None:
        """Open or create a cache sidecar."""
        if max_bytes <= 0:
            msg = "Cache byte limit must be positive."
            raise ValueError(msg)
        if touch_batch_size <= 0:
            msg = "Cache touch batch size must be positive."
            raise ValueError(msg)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.touch_batch_size = int(touch_batch_size)
        self._lock = threading.RLock()
        # Repeated hits coalesce by key. Recency is deliberately approximate:
        # losing a few unflushed touches on process termination can only affect
        # eviction order, never payload or byte-accounting correctness.
        self._pending_touches: dict[str, int] = {}
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS tiles(
                cache_key TEXT PRIMARY KEY,
                payload BLOB NOT NULL,
                content_type TEXT NOT NULL,
                content_encoding TEXT,
                etag TEXT NOT NULL,
                representation TEXT,
                feature_count INTEGER NOT NULL,
                vertex_count INTEGER NOT NULL,
                byte_size INTEGER NOT NULL,
                accessed_ns INTEGER NOT NULL
            )
            """,
        )
        self._con.execute(
            """
            CREATE INDEX IF NOT EXISTS tiles_accessed_ns
                ON tiles(accessed_ns, cache_key)
            """,
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata(
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                logical_bytes INTEGER NOT NULL CHECK(logical_bytes >= 0)
            )
            """,
        )
        # Reconcile once when opening an existing/old-format sidecar. All normal
        # writes maintain this total incrementally in the same transaction as
        # their tile mutations.
        self._con.execute(
            """
            INSERT INTO cache_metadata(singleton, logical_bytes)
            VALUES (1, (SELECT COALESCE(SUM(byte_size), 0) FROM tiles))
            ON CONFLICT(singleton) DO UPDATE SET
                logical_bytes = excluded.logical_bytes
            """,
        )
        row = self._con.execute(
            "SELECT logical_bytes FROM cache_metadata WHERE singleton = 1",
        ).fetchone()
        total = self._evict(int(row[0]))
        self._con.execute(
            "UPDATE cache_metadata SET logical_bytes = ? WHERE singleton = 1",
            (total,),
        )
        self._con.commit()

    @property
    def size(self) -> int:
        """Return the persisted logical payload size in bytes."""
        with self._lock:
            row = self._con.execute(
                "SELECT logical_bytes FROM cache_metadata WHERE singleton = 1",
            ).fetchone()
        return int(row[0])

    def get(self, key: str) -> TilePayload | None:
        """Return a cached payload."""
        with self._lock:
            row = self._con.execute(
                """
                SELECT payload, content_type, content_encoding, etag,
                       representation, feature_count, vertex_count
                  FROM tiles
                 WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            self._pending_touches[key] = time.time_ns()
            if len(self._pending_touches) >= self.touch_batch_size:
                self._flush_touches()
        return TilePayload(
            data=bytes(row[0]),
            content_type=row[1],
            content_encoding=row[2],
            etag=row[3],
            representation=row[4],
            feature_count=int(row[5]),
            vertex_count=int(row[6]),
        )

    def put(self, key: str, payload: TilePayload) -> None:
        """Persist a tile and evict old bytes when needed."""
        payload = payload.with_etag()
        if payload.size > self.max_bytes:
            return
        with self._lock:
            touches = tuple(self._pending_touches.items())
            try:
                # A write lock makes the persisted byte counter authoritative
                # even when multiple viewer processes share the sidecar.
                self._con.execute("BEGIN IMMEDIATE")
                self._apply_touches(touches)
                previous = self._con.execute(
                    "SELECT byte_size FROM tiles WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                row = self._con.execute(
                    "SELECT logical_bytes FROM cache_metadata WHERE singleton = 1",
                ).fetchone()
                total = int(row[0]) + payload.size
                if previous is not None:
                    total -= int(previous[0])
                self._con.execute(
                    """
                    INSERT INTO tiles(
                        cache_key, payload, content_type, content_encoding, etag,
                        representation, feature_count, vertex_count, byte_size,
                        accessed_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload=excluded.payload,
                        content_type=excluded.content_type,
                        content_encoding=excluded.content_encoding,
                        etag=excluded.etag,
                        representation=excluded.representation,
                        feature_count=excluded.feature_count,
                        vertex_count=excluded.vertex_count,
                        byte_size=excluded.byte_size,
                        accessed_ns=excluded.accessed_ns
                    """,
                    (
                        key,
                        payload.data,
                        payload.content_type,
                        payload.content_encoding,
                        payload.etag,
                        payload.representation,
                        payload.feature_count,
                        payload.vertex_count,
                        payload.size,
                        time.time_ns(),
                    ),
                )
                total = self._evict(total)
                self._con.execute(
                    """
                    UPDATE cache_metadata
                       SET logical_bytes = ?
                     WHERE singleton = 1
                    """,
                    (total,),
                )
                self._con.commit()
            except BaseException:
                self._con.rollback()
                raise
            else:
                self._discard_applied_touches(touches)

    def close(self) -> None:
        """Close the sidecar connection."""
        with self._lock:
            self._flush_touches()
            self._con.close()

    def _flush_touches(self) -> None:
        """Persist coalesced LRU touches in one small transaction."""
        touches = tuple(self._pending_touches.items())
        if not touches:
            return
        try:
            self._apply_touches(touches)
            self._con.commit()
        except BaseException:
            self._con.rollback()
            raise
        else:
            self._discard_applied_touches(touches)

    def _apply_touches(self, touches: tuple[tuple[str, int], ...]) -> None:
        self._con.executemany(
            "UPDATE tiles SET accessed_ns = ? WHERE cache_key = ?",
            ((accessed_ns, key) for key, accessed_ns in touches),
        )

    def _discard_applied_touches(self, touches: tuple[tuple[str, int], ...]) -> None:
        for key, accessed_ns in touches:
            if self._pending_touches.get(key) == accessed_ns:
                self._pending_touches.pop(key, None)

    def _evict(self, total: int) -> int:
        """Evict the indexed oldest entries and return the new logical size."""
        while total > self.max_bytes:
            victims = self._con.execute(
                """
                SELECT cache_key, byte_size
                  FROM tiles
                 ORDER BY accessed_ns, cache_key
                 LIMIT 64
                """,
            ).fetchall()
            if not victims:
                break
            selected: list[tuple[str, int]] = []
            for victim_key, byte_size in victims:
                selected.append((str(victim_key), int(byte_size)))
                total -= int(byte_size)
                if total <= self.max_bytes:
                    break
            self._con.executemany(
                "DELETE FROM tiles WHERE cache_key = ?",
                ((victim[0],) for victim in selected),
            )
        return total


class SingleFlight:
    """Deduplicate concurrent work for the same cache key."""

    def __init__(self) -> None:
        """Initialise the in-flight key map."""
        self._guard = threading.Lock()
        self._flights: dict[str, Future[TilePayload]] = {}

    def run(self, key: str, build: Callable[[], TilePayload]) -> TilePayload:
        """Run one ``build`` and share its result with concurrent callers."""
        with self._guard:
            future = self._flights.get(key)
            owner = future is None
            if future is None:
                future = Future()
                self._flights[key] = future
        if not owner:
            return future.result()
        try:
            result = build()
        except BaseException as error:
            future.set_exception(error)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            with self._guard:
                self._flights.pop(key, None)
