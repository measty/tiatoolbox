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
    ) -> None:
        """Open or create a cache sidecar."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock = threading.RLock()
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
        self._con.commit()

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
            self._con.execute(
                "UPDATE tiles SET accessed_ns = ? WHERE cache_key = ?",
                (time.time_ns(), key),
            )
            self._con.commit()
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
            self._evict()
            self._con.commit()

    def close(self) -> None:
        """Close the sidecar connection."""
        with self._lock:
            self._con.close()

    def _evict(self) -> None:
        row = self._con.execute(
            "SELECT COALESCE(SUM(byte_size), 0) FROM tiles",
        ).fetchone()
        total = int(row[0])
        while total > self.max_bytes:
            victims = self._con.execute(
                """
                SELECT cache_key, byte_size
                  FROM tiles
                 ORDER BY accessed_ns
                 LIMIT 64
                """,
            ).fetchall()
            if not victims:
                break
            self._con.executemany(
                "DELETE FROM tiles WHERE cache_key = ?",
                ((victim[0],) for victim in victims),
            )
            total -= sum(int(victim[1]) for victim in victims)


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
