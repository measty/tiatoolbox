"""Application services shared by the versioned viewer API routes."""

from __future__ import annotations

import copy
import hashlib
import os
import secrets
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.annotation.storage import (
    PROPERTY_FILTER_MAX_BYTES,
    PROPERTY_FILTER_MAX_DEPTH,
    PROPERTY_FILTER_MAX_IN_VALUES,
    PROPERTY_FILTER_MAX_NODES,
)
from tiatoolbox.tools.pyramid import ZoomifyGenerator
from tiatoolbox.utils.misc import store_from_dat
from tiatoolbox.visualization.annotation_tiles.cache import ByteLRUCache
from tiatoolbox.visualization.annotation_tiles.grid import TileMatrix
from tiatoolbox.visualization.annotation_tiles.lod import LODBuildCancelled
from tiatoolbox.visualization.annotation_tiles.source import AnnotationTileSource
from tiatoolbox.visualization.resources import ResourceRegistry, ViewerResource
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from tiatoolbox.visualization.tileserver import TileServer


# Bump whenever tile geometry/properties or derived LOD semantics change. It is
# part of the public store identity so browser/CDN immutable URLs cannot reuse
# bytes produced by an older representation pipeline after an upgrade.
_ANNOTATION_PIPELINE_VERSION = 8
_DEFAULT_ANNOTATION_MEMORY_CACHE_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_ANNOTATION_SOURCES = 32
_DEFAULT_MAX_IDLE_ANNOTATION_SOURCES = 4
_DEFAULT_MAX_VIEWER_SESSIONS = 128
_DEFAULT_SESSION_IDLE_SECONDS = 30 * 60


@dataclass(slots=True)
class ViewerSession:
    """Modern viewer state scoped to one HTTP session cookie."""

    id: str
    slide_resource_id: str | None = None
    # ``slide_selection_sequence`` orders in-flight opens. ``slide_generation``
    # only advances when the newest open commits, so documents for the current
    # slide never advertise an in-progress selection's tile namespace.
    slide_selection_sequence: int = 0
    slide_generation: int = 0
    annotation_sources: list[str] = field(default_factory=list)
    raster_layers: dict[str, dict[str, Any]] = field(default_factory=dict)


class VisualizationServices:
    """Composition root for resources, sessions, tile sources and LOD workers."""

    def __init__(
        self,
        tile_server: TileServer,
        *,
        slide_roots: tuple[str | Path, ...] | list[str | Path] = (),
        overlay_roots: tuple[str | Path, ...] | list[str | Path] = (),
        cache_dir: str | Path | None = None,
        annotation_memory_cache_bytes: int = _DEFAULT_ANNOTATION_MEMORY_CACHE_BYTES,
        max_annotation_sources: int = _DEFAULT_MAX_ANNOTATION_SOURCES,
        max_idle_annotation_sources: int = _DEFAULT_MAX_IDLE_ANNOTATION_SOURCES,
        max_sessions: int = _DEFAULT_MAX_VIEWER_SESSIONS,
        session_idle_seconds: float = _DEFAULT_SESSION_IDLE_SECONDS,
    ) -> None:
        """Initialise viewer services without changing legacy routes."""
        if max_annotation_sources <= 0:
            msg = "Annotation source limit must be positive."
            raise ValueError(msg)
        if not 0 <= max_idle_annotation_sources <= max_annotation_sources:
            msg = "Idle annotation source limit must be between zero and source limit."
            raise ValueError(msg)
        if max_sessions <= 0:
            msg = "Viewer session limit must be positive."
            raise ValueError(msg)
        if session_idle_seconds <= 0:
            msg = "Viewer session idle timeout must be positive."
            raise ValueError(msg)
        self.tile_server = tile_server
        self.registry = ResourceRegistry(
            slide_roots=slide_roots,
            overlay_roots=overlay_roots,
        )
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self._sessions: OrderedDict[str, ViewerSession] = OrderedDict()
        self._session_last_access: dict[str, float] = {}
        self._sources: OrderedDict[str, AnnotationTileSource] = OrderedDict()
        self._source_sessions: dict[str, set[str]] = {}
        self._source_pins: dict[str, int] = {}
        self._source_reservations: set[str] = set()
        self._source_creation_locks: dict[str, threading.Lock] = {}
        self._source_creation_users: dict[str, int] = {}
        self._observed_lod_futures: set[object] = set()
        self.annotation_memory_cache = ByteLRUCache(annotation_memory_cache_bytes)
        self.max_annotation_sources = int(max_annotation_sources)
        self.max_idle_annotation_sources = int(max_idle_annotation_sources)
        self.max_sessions = int(max_sessions)
        self.session_idle_seconds = float(session_idle_seconds)
        self._lock = threading.RLock()
        self._lease_condition = threading.Condition(self._lock)
        self._closed = False
        self._import_locks: dict[str, threading.Lock] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="tiatoolbox-lod",
        )

    def ensure_session(self, requested_id: str | None) -> tuple[ViewerSession, bool]:
        """Return a valid session and whether a new cookie is required."""
        if self.tile_server.default_session_id:
            session_id = "default"
        else:
            session_id = requested_id or secrets.token_urlsafe(18)
        now = time.monotonic()
        with self._lock:
            sources_to_close = self._reap_sessions_locked(
                now,
                protected_id=session_id,
            )
            session = self._sessions.get(session_id)
            if session is not None:
                is_new = False
            else:
                session = ViewerSession(session_id)
                self._sessions[session_id] = session
                self.tile_server.layers.setdefault(session_id, {})
                self.tile_server.pyramids.setdefault(session_id, {})
                self.tile_server.renderers.setdefault(
                    session_id,
                    copy.deepcopy(self.tile_server.renderer),
                )
                self.tile_server.overlaps.setdefault(session_id, 0)
                is_new = requested_id != session_id
            self._sessions.move_to_end(session_id)
            self._session_last_access[session_id] = now
            sources_to_close.extend(
                self._reap_sessions_locked(now, protected_id=session_id),
            )
        self._close_sources(sources_to_close)
        return session, is_new

    def bootstrap(self, session: ViewerSession) -> dict[str, Any]:
        """Return the path-free application bootstrap document."""
        return {
            "apiVersion": "1.0",
            "title": self.tile_server.title,
            "catalog": self.registry.catalog(),
            "session": self.session_document(session),
            "capabilities": {
                "renderers": ["canvas", "webgl"],
                "annotationRepresentations": [
                    "auto",
                    "aggregate",
                    "centroid",
                    "polygon",
                    "labels",
                ],
                "linkedViews": True,
                "clientStyleFilters": True,
                "serverFilters": True,
                "serverFilter": {
                    "queryParameter": "filter",
                    "encoding": "urlencoded-json",
                    "comparisonOperators": [
                        "eq",
                        "ne",
                        "lt",
                        "lte",
                        "gt",
                        "gte",
                        "in",
                    ],
                    "logicalOperators": ["and", "or"],
                    "limits": {
                        "maxBytes": PROPERTY_FILTER_MAX_BYTES,
                        "maxDepth": PROPERTY_FILTER_MAX_DEPTH,
                        "maxNodes": PROPERTY_FILTER_MAX_NODES,
                        "maxInValues": PROPERTY_FILTER_MAX_IN_VALUES,
                    },
                },
            },
        }

    def session_document(self, session: ViewerSession) -> dict[str, Any]:
        """Return current layer state for a session."""
        with self._lock:
            slide = self._slide_document(session) if session.slide_resource_id else None
            stores = [
                self.store_manifest(store_id) for store_id in session.annotation_sources
            ]
            return {
                "id": session.id,
                "slide": slide,
                "annotationStores": stores,
                "rasterLayers": list(session.raster_layers.values()),
            }

    def slide_metadata(self, resource_id: str) -> dict[str, Any]:
        """Open a slide long enough to return its tile-grid metadata."""
        resource = self.registry.get(resource_id, "slide")
        reader = WSIReader.open(resource.path)
        matrix = TileMatrix(*map(int, reader.info.slide_dimensions))
        return self._slide_metadata_document(resource, reader, matrix)

    def select_slide(
        self,
        session: ViewerSession,
        resource_id: str,
    ) -> dict[str, Any]:
        """Select a slide and reset its dependent display layers."""
        resource = self.registry.get(resource_id, "slide")
        with self._lock:
            if self._sessions.get(session.id) is not session:
                msg = "Viewer session expired; reload and retry."
                raise RuntimeError(msg)
            session.slide_selection_sequence += 1
            selection_sequence = session.slide_selection_sequence
        reader = WSIReader.open(resource.path)
        pyramid = ZoomifyGenerator(reader, tile_size=256)
        if reader.info.mpp is None:
            reader.info.mpp = [1, 1]
        sources_to_close: list[AnnotationTileSource] = []
        with self._lock:
            if (
                self._sessions.get(session.id) is not session
                or selection_sequence != session.slide_selection_sequence
            ):
                # WSIReader has no common close API. Drop both owners before
                # raising so native reader handles can be released immediately.
                del pyramid, reader
                msg = "Slide selection was superseded by a newer request."
                raise RuntimeError(msg)
            for store_id in tuple(session.annotation_sources):
                self._detach_source_locked(session, store_id)
            self.tile_server.layers[session.id] = {"slide": reader}
            self.tile_server.pyramids[session.id] = {"slide": pyramid}
            self.tile_server.slide_mpps[session.id] = reader.info.mpp
            session.slide_resource_id = resource.id
            session.slide_generation = selection_sequence
            session.raster_layers.clear()
            document = self._slide_document(session)
            sources_to_close = self._collect_idle_evictions_locked()
        self._close_sources(sources_to_close)
        return document

    def add_overlay(
        self,
        session: ViewerSession,
        resource_id: str,
        *,
        slide_generation: int | None = None,
    ) -> dict[str, Any]:
        """Add an annotation or raster overlay to a selected slide."""
        resource = self.registry.get(resource_id)
        with self._lock:
            generation = self._validate_slide_generation_locked(
                session,
                slide_generation,
            )
            slide_reader = self.tile_server.layers[session.id]["slide"]
            matrix = TileMatrix(*map(int, slide_reader.info.slide_dimensions))
        if resource.kind == "annotation":
            source = self._acquire_annotation_source(resource, matrix)
            try:
                with self._lock:
                    self._validate_slide_generation_locked(session, generation)
                    if source.store_id not in session.annotation_sources:
                        session.annotation_sources.append(source.store_id)
                        self._source_sessions.setdefault(source.store_id, set()).add(
                            session.id,
                        )
                    self._touch_source_locked(source.store_id)
                    lod_future = source.ensure_lod(self._executor)
                    self._watch_lod_locked(source, lod_future)
                return {
                    "kind": "annotation",
                    "store": self._store_manifest_document(source),
                }
            finally:
                self._release_source(source.store_id)
        if resource.kind == "raster-overlay":
            reader, pyramid = self._open_raster_layer(resource, slide_reader)
            layer_name = f"r-{resource.id}"
            with self._lock:
                try:
                    self._validate_slide_generation_locked(session, generation)
                except RuntimeError:
                    # WSIReader has no common close API. Drop both owners before
                    # propagating the superseded request.
                    del pyramid, reader
                    raise
                self.tile_server.layers[session.id][layer_name] = reader
                self.tile_server.pyramids[session.id][layer_name] = pyramid
                document = self._raster_layer_document(
                    session,
                    resource,
                    layer_name,
                    reader,
                )
                session.raster_layers[resource.id] = document
            return {"kind": "raster", "layer": document}
        msg = "Slides cannot be added as overlays."
        raise ValueError(msg)

    def remove_overlay(
        self,
        session: ViewerSession,
        layer_id: str,
        *,
        slide_generation: int | None = None,
    ) -> None:
        """Remove a session layer without invalidating shared immutable caches."""
        sources_to_close: list[AnnotationTileSource] = []
        with self._lock:
            self._validate_slide_generation_locked(session, slide_generation)
            if layer_id in session.annotation_sources:
                self._detach_source_locked(session, layer_id)
                sources_to_close = self._collect_idle_evictions_locked()
            else:
                layer = session.raster_layers.pop(layer_id, None)
                if layer is None:
                    msg = "Unknown session layer."
                    raise KeyError(msg)
                layer_name = layer["layerName"]
                self.tile_server.layers[session.id].pop(layer_name, None)
                self.tile_server.pyramids[session.id].pop(layer_name, None)
        self._close_sources(sources_to_close)

    def get_source(
        self,
        session: ViewerSession,
        store_id: str,
    ) -> AnnotationTileSource:
        """Resolve a store only when assigned to the requesting session."""
        with self._lock:
            return self._get_source_locked(session, store_id)

    @contextmanager
    def source_lease(
        self,
        session: ViewerSession,
        store_id: str,
    ) -> Iterator[AnnotationTileSource]:
        """Pin an assigned source for the duration of one API operation."""
        with self._lock:
            source = self._get_source_locked(session, store_id)
            self._pin_source_locked(store_id)
        try:
            yield source
        finally:
            self._release_source(store_id)

    def store_manifest(self, store_id: str) -> dict[str, Any]:
        """Return a source manifest with API URL templates."""
        with self._lock:
            source = self._sources[store_id]
            self._pin_source_locked(store_id)
        try:
            return self._store_manifest_document(source)
        finally:
            self._release_source(store_id)

    def _store_manifest_document(
        self,
        source: AnnotationTileSource,
    ) -> dict[str, Any]:
        """Return one already-pinned source manifest with API URL templates."""
        manifest = source.manifest()
        revision_base = f"/api/v1/stores/{source.store_id}/revisions/{source.revision}"
        lod_generation = manifest["lodStatus"]
        generation_query = f"?lod={lod_generation}&tileVersion={source.tile_revision}"
        manifest["urls"] = {
            "manifest": revision_base,
            "tiles": (
                f"{revision_base}/tiles/auto/{{z}}/{{x}}/{{y}}.mvt{generation_query}"
            ),
            "representations": {
                representation: (
                    f"{revision_base}/tiles/{representation}/{{z}}/{{x}}/{{y}}.mvt"
                    f"{generation_query}"
                )
                for representation in (
                    "aggregate",
                    "centroid",
                    "polygon",
                )
            },
            "labels": (
                f"{revision_base}/tiles/labels/{{z}}/{{x}}/{{y}}.bin{generation_query}"
            ),
            "pick": f"{revision_base}/features/pick",
            "feature": f"{revision_base}/features/{{featureId}}",
        }
        return manifest

    def close(self) -> None:
        """Stop builders and release source/caching connections."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sources = list(self._sources.values())
            for source in sources:
                source.cancel_lod()
        # Running builders own thread-local read connections. Wait for those finite
        # ingestion jobs before closing stores so Windows can release DB handles.
        self._executor.shutdown(wait=True, cancel_futures=True)
        # Werkzeug request threads may still be unwinding when the development
        # server returns. A source lease protects every API use, so drain those
        # finite operations before closing their SQLite connections.
        with self._lease_condition:
            self._lease_condition.wait_for(
                lambda: not any(self._source_pins.values()),
            )
        self._close_sources(sources)
        self.annotation_memory_cache.clear()
        with self._lock:
            self._sources.clear()
            self._source_sessions.clear()
            self._source_pins.clear()
            self._source_reservations.clear()
            self._source_creation_locks.clear()
            self._source_creation_users.clear()
            self._observed_lod_futures.clear()
            self._session_last_access.clear()
            for session_id in self._sessions:
                self.tile_server.layers.pop(session_id, None)
                self.tile_server.pyramids.pop(session_id, None)
                self.tile_server.slide_mpps.pop(session_id, None)
                self.tile_server.renderers.pop(session_id, None)
                self.tile_server.overlaps.pop(session_id, None)
            self._sessions.clear()

    def _acquire_annotation_source(  # noqa: PLR0912, PLR0915
        self,
        resource: ViewerResource,
        matrix: TileMatrix,
    ) -> AnnotationTileSource:
        """Return a pinned source, creating it within the process-wide bound."""
        source_id = hashlib.blake2b(
            (
                f"{resource.id}:{resource.revision}:"
                f"{matrix.width}:{matrix.height}:{matrix.tile_size}:"
                f"pipeline-{_ANNOTATION_PIPELINE_VERSION}"
            ).encode(),
            digest_size=12,
        ).hexdigest()
        with self._lock:
            if self._closed:
                msg = "Visualization services are closed."
                raise RuntimeError(msg)
            existing = self._sources.get(source_id)
            if existing is not None:
                self._pin_source_locked(source_id)
                self._touch_source_locked(source_id)
                return existing
            creation_lock = self._source_creation_locks.setdefault(
                source_id,
                threading.Lock(),
            )
            self._source_creation_users[source_id] = (
                self._source_creation_users.get(source_id, 0) + 1
            )

        creation_lock.acquire()
        try:
            sources_to_close: list[AnnotationTileSource] = []
            with self._lock:
                if self._closed:
                    msg = "Visualization services are closed."
                    raise RuntimeError(msg)
                existing = self._sources.get(source_id)
                if existing is not None:
                    self._pin_source_locked(source_id)
                    self._touch_source_locked(source_id)
                    return existing
                sources_to_close = self._collect_idle_evictions_locked(
                    required_capacity=1,
                )
                if (
                    len(self._sources) + len(self._source_reservations)
                    >= self.max_annotation_sources
                ):
                    msg = (
                        "Annotation source limit is currently occupied; "
                        "retry after obsolete preprocessing stops."
                    )
                    raise RuntimeError(msg)
                self._source_reservations.add(source_id)
            self._close_sources(sources_to_close)

            source: AnnotationTileSource | None = None
            try:
                store = self._open_store(resource)
                try:
                    source = AnnotationTileSource(
                        store,
                        matrix,
                        store_id=source_id,
                        revision=resource.revision,
                        name=resource.name,
                        cache_dir=self.cache_dir,
                        owns_store=True,
                        memory_cache=self.annotation_memory_cache,
                    )
                except BaseException:
                    store.close()
                    raise
                with self._lock:
                    self._source_reservations.discard(source_id)
                    if self._closed:
                        msg = "Visualization services are closed."
                        raise RuntimeError(msg)  # noqa: TRY301
                    self._sources[source_id] = source
                    self._source_sessions[source_id] = set()
                    self._source_pins[source_id] = 1
                    self._touch_source_locked(source_id)
                    return source
            except BaseException:
                with self._lock:
                    self._source_reservations.discard(source_id)
                if source is not None:
                    source.close()
                raise
        finally:
            creation_lock.release()
            with self._lock:
                remaining_users = self._source_creation_users.get(source_id, 1) - 1
                if remaining_users > 0:
                    self._source_creation_users[source_id] = remaining_users
                else:
                    self._source_creation_users.pop(source_id, None)
                if (
                    remaining_users == 0
                    and self._source_creation_locks.get(source_id) is creation_lock
                ):
                    self._source_creation_locks.pop(source_id, None)

    def _get_source_locked(
        self,
        session: ViewerSession,
        store_id: str,
    ) -> AnnotationTileSource:
        """Resolve and touch a source already attached to a session."""
        if store_id not in session.annotation_sources:
            msg = "Unknown annotation store."
            raise KeyError(msg)
        source = self._sources.get(store_id)
        if source is None:
            msg = "Unknown annotation store."
            raise KeyError(msg)
        self._touch_source_locked(store_id)
        return source

    def _pin_source_locked(self, store_id: str) -> None:
        """Prevent source eviction until the corresponding lease is released."""
        if self._closed:
            msg = "Visualization services are closed."
            raise RuntimeError(msg)
        if store_id not in self._sources:
            msg = "Unknown annotation store."
            raise KeyError(msg)
        self._source_pins[store_id] = self._source_pins.get(store_id, 0) + 1

    def _release_source(self, store_id: str) -> None:
        """Release one source lease and perform any newly possible eviction."""
        with self._lease_condition:
            pins = self._source_pins.get(store_id)
            if pins is None:
                return
            if pins <= 0:
                msg = "Annotation source lease underflow."
                raise RuntimeError(msg)
            self._source_pins[store_id] = pins - 1
            self._lease_condition.notify_all()
            sources_to_close = self._collect_idle_evictions_locked()
        self._close_sources(sources_to_close)

    def _detach_source_locked(self, session: ViewerSession, store_id: str) -> None:
        """Detach one session and cancel preprocessing when no session needs it."""
        if store_id in session.annotation_sources:
            session.annotation_sources.remove(store_id)
        sessions = self._source_sessions.get(store_id)
        if sessions is None:
            return
        sessions.discard(session.id)
        source = self._sources.get(store_id)
        if source is not None and not sessions:
            source.cancel_lod()
        self._touch_source_locked(store_id)

    def _touch_source_locked(self, store_id: str) -> None:
        """Promote a source in the process-wide source LRU."""
        if store_id in self._sources:
            self._sources.move_to_end(store_id)

    def _collect_idle_evictions_locked(
        self,
        *,
        required_capacity: int = 0,
    ) -> list[AnnotationTileSource]:
        """Remove oldest safe idle sources while retaining their disk sidecars."""
        evicted: list[AnnotationTileSource] = []

        def idle_ids() -> list[str]:
            return [
                store_id
                for store_id, source in self._sources.items()
                if not self._source_sessions.get(store_id)
                and self._source_pins.get(store_id, 0) == 0
                and not source.lod_running
            ]

        candidates = idle_ids()
        while candidates and (
            len(candidates) > self.max_idle_annotation_sources
            or len(self._sources) + len(self._source_reservations) + required_capacity
            > self.max_annotation_sources
        ):
            store_id = candidates.pop(0)
            source = self._sources.pop(store_id)
            self._source_sessions.pop(store_id, None)
            self._source_pins.pop(store_id, None)
            evicted.append(source)
        return evicted

    def _watch_lod_locked(self, source: AnnotationTileSource, future) -> None:  # noqa: ANN001
        """Observe one build so cancelled active sources restart and idle ones evict."""
        if future is None or future in self._observed_lod_futures:
            return
        self._observed_lod_futures.add(future)
        future.add_done_callback(
            lambda completed, watched=source: self._lod_finished(watched, completed),
        )

    def _lod_finished(self, source: AnnotationTileSource, future) -> None:  # noqa: ANN001
        """Handle asynchronous LOD completion without publishing obsolete work."""
        sources_to_close: list[AnnotationTileSource] = []
        with self._lock:
            self._observed_lod_futures.discard(future)
            if self._sources.get(source.store_id) is not source:
                return
            cancelled = future.cancelled()
            error = None if cancelled else future.exception()
            expected_cancel = cancelled or isinstance(error, LODBuildCancelled)
            active = bool(self._source_sessions.get(source.store_id)) or (
                self._source_pins.get(source.store_id, 0) > 0
            )
            if expected_cancel and active and not self._closed:
                replacement = source.ensure_lod(self._executor)
                self._watch_lod_locked(source, replacement)
            else:
                sources_to_close = self._collect_idle_evictions_locked()
        self._close_sources(sources_to_close)

    def _validate_slide_generation_locked(
        self,
        session: ViewerSession,
        expected: int | None,
    ) -> int:
        """Return the current slide generation or reject stale overlay work."""
        if self._sessions.get(session.id) is not session:
            msg = "Viewer session expired; reload and retry."
            raise RuntimeError(msg)
        if session.slide_resource_id is None:
            msg = "Select a slide before adding an overlay."
            raise RuntimeError(msg)
        if expected is not None and (
            isinstance(expected, bool) or not isinstance(expected, int) or expected < 0
        ):
            msg = "slideGeneration must be a non-negative integer."
            raise ValueError(msg)
        if expected is not None and expected != session.slide_generation:
            msg = "Slide selection was superseded by a newer request."
            raise RuntimeError(msg)
        return session.slide_generation

    def _reap_sessions_locked(
        self,
        now: float,
        *,
        protected_id: str | None,
    ) -> list[AnnotationTileSource]:
        """Expire idle/LRU sessions and detach every source they referenced."""
        expired = [
            session_id
            for session_id in self._sessions
            if session_id != protected_id
            and now - self._session_last_access.get(session_id, now)
            >= self.session_idle_seconds
        ]
        for session_id in expired:
            self._drop_session_locked(session_id)
        while len(self._sessions) > self.max_sessions:
            victim = next(
                (
                    session_id
                    for session_id in self._sessions
                    if session_id != protected_id
                ),
                None,
            )
            if victim is None:
                break
            self._drop_session_locked(victim)
        return self._collect_idle_evictions_locked()

    def _drop_session_locked(self, session_id: str) -> None:
        """Remove one session and all of its mutable TileServer state."""
        session = self._sessions.pop(session_id, None)
        self._session_last_access.pop(session_id, None)
        if session is None:
            return
        for store_id in tuple(session.annotation_sources):
            self._detach_source_locked(session, store_id)
        session.raster_layers.clear()
        self.tile_server.layers.pop(session_id, None)
        self.tile_server.pyramids.pop(session_id, None)
        self.tile_server.slide_mpps.pop(session_id, None)
        self.tile_server.renderers.pop(session_id, None)
        self.tile_server.overlaps.pop(session_id, None)

    @staticmethod
    def _close_sources(sources: list[AnnotationTileSource]) -> None:
        """Close each evicted source at most once."""
        seen: set[int] = set()
        for source in sources:
            identity = id(source)
            if identity in seen:
                continue
            seen.add(identity)
            source.close()

    def _open_store(self, resource: ViewerResource) -> SQLiteStore:
        if resource.path.suffix.lower() == ".db":
            return SQLiteStore(resource.path, read_only=True)
        import_dir = self.cache_dir / "imports" / resource.id
        import_dir.mkdir(parents=True, exist_ok=True)
        target = import_dir / f"{resource.revision}.db"
        lock = self._import_locks.setdefault(resource.id, threading.Lock())
        with lock:
            if not target.exists():
                temporary = target.with_suffix(".building.db")
                temporary.unlink(missing_ok=True)
                if resource.path.suffix.lower() == ".geojson":
                    converted = SQLiteStore.from_geojson(resource.path)
                else:
                    converted = store_from_dat(resource.path)
                try:
                    converted.dump(temporary)
                finally:
                    converted.close()
                temporary.replace(target)
        return SQLiteStore(target, read_only=True)

    @staticmethod
    def _open_raster_layer(
        resource: ViewerResource,
        slide_reader: WSIReader,
    ) -> tuple[WSIReader, ZoomifyGenerator]:
        """Open raster work without mutating session state until generation commit."""
        suffix = resource.path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            info = copy.deepcopy(slide_reader.info)
            info.file_path = str(resource.path)
            with Image.open(resource.path) as image:
                array = np.asarray(image).copy()
            reader = VirtualWSIReader(array, info=info)
        else:
            reader = WSIReader.open(resource.path)
        return reader, ZoomifyGenerator(reader)

    def _slide_document(self, session: ViewerSession) -> dict[str, Any]:
        resource = self.registry.get(session.slide_resource_id or "", "slide")
        reader = self.tile_server.layers[session.id]["slide"]
        matrix = TileMatrix(*map(int, reader.info.slide_dimensions))
        document = self._slide_metadata_document(resource, reader, matrix)
        document["tileUrl"] = (
            f"/tileserver/layer/slide/{session.id}/zoomify/"
            "{TileGroup}/{z}-{x}-{y}@1x.jpg"
            f"?slideGeneration={session.slide_generation}"
            f"&resource={resource.id}&revision={resource.revision}"
        )
        document["associatedOverlays"] = self.registry.associated_overlays(resource.id)
        document["slideGeneration"] = session.slide_generation
        return document

    @staticmethod
    def _slide_metadata_document(
        resource: ViewerResource,
        reader: WSIReader,
        matrix: TileMatrix,
    ) -> dict[str, Any]:
        mpp = reader.info.mpp
        return {
            **resource.public_dict(),
            "dimensions": [matrix.width, matrix.height],
            "mpp": None if mpp is None else [float(value) for value in mpp],
            "objectivePower": (
                None
                if reader.info.objective_power is None
                else float(reader.info.objective_power)
            ),
            "tileMatrix": matrix.as_dict(),
        }

    @staticmethod
    def _raster_layer_document(
        session: ViewerSession,
        resource: ViewerResource,
        layer_name: str,
        reader: WSIReader,
    ) -> dict[str, Any]:
        return {
            **resource.public_dict(),
            "layerName": layer_name,
            "dimensions": [int(value) for value in reader.info.slide_dimensions],
            "tileUrl": (
                f"/tileserver/layer/{layer_name}/{session.id}/zoomify/"
                "{TileGroup}/{z}-{x}-{y}@1x.jpg"
                f"?slideGeneration={session.slide_generation}"
                f"&resource={resource.id}&revision={resource.revision}"
            ),
        }


def default_cache_dir() -> Path:
    """Return the configurable per-user visualization cache directory."""
    configured = os.environ.get("TIATOOLBOX_VIS_CACHE_DIR")
    if configured:
        return Path(configured)
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "tiatoolbox" / "visualization"
    return Path(tempfile.gettempdir()) / "tiatoolbox" / "visualization"
