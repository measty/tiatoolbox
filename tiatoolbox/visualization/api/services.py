"""Application services shared by the versioned viewer API routes."""

from __future__ import annotations

import copy
import hashlib
import os
import secrets
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
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
from tiatoolbox.visualization.annotation_tiles.grid import TileMatrix
from tiatoolbox.visualization.annotation_tiles.source import AnnotationTileSource
from tiatoolbox.visualization.resources import ResourceRegistry, ViewerResource
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.visualization.tileserver import TileServer


# Bump whenever tile geometry/properties or derived LOD semantics change. It is
# part of the public store identity so browser/CDN immutable URLs cannot reuse
# bytes produced by an older representation pipeline after an upgrade.
_ANNOTATION_PIPELINE_VERSION = 8


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
    ) -> None:
        """Initialise viewer services without changing legacy routes."""
        self.tile_server = tile_server
        self.registry = ResourceRegistry(
            slide_roots=slide_roots,
            overlay_roots=overlay_roots,
        )
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self._sessions: dict[str, ViewerSession] = {}
        self._sources: dict[str, AnnotationTileSource] = {}
        self._lock = threading.RLock()
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
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                return session, False
            session = ViewerSession(session_id)
            self._sessions[session_id] = session
            self.tile_server.layers.setdefault(session_id, {})
            self.tile_server.pyramids.setdefault(session_id, {})
            self.tile_server.renderers.setdefault(
                session_id,
                copy.deepcopy(self.tile_server.renderer),
            )
            self.tile_server.overlaps.setdefault(session_id, 0)
            return session, requested_id != session_id

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
            session.slide_selection_sequence += 1
            selection_sequence = session.slide_selection_sequence
        reader = WSIReader.open(resource.path)
        pyramid = ZoomifyGenerator(reader, tile_size=256)
        if reader.info.mpp is None:
            reader.info.mpp = [1, 1]
        with self._lock:
            if selection_sequence != session.slide_selection_sequence:
                # WSIReader has no common close API. Drop both owners before
                # raising so native reader handles can be released immediately.
                del pyramid, reader
                msg = "Slide selection was superseded by a newer request."
                raise RuntimeError(msg)
            self.tile_server.layers[session.id] = {"slide": reader}
            self.tile_server.pyramids[session.id] = {"slide": pyramid}
            self.tile_server.slide_mpps[session.id] = reader.info.mpp
            session.slide_resource_id = resource.id
            session.slide_generation = selection_sequence
            session.annotation_sources.clear()
            session.raster_layers.clear()
            return self._slide_document(session)

    def add_overlay(
        self,
        session: ViewerSession,
        resource_id: str,
    ) -> dict[str, Any]:
        """Add an annotation or raster overlay to a selected slide."""
        if session.slide_resource_id is None:
            msg = "Select a slide before adding an overlay."
            raise RuntimeError(msg)
        resource = self.registry.get(resource_id)
        if resource.kind == "annotation":
            source = self._annotation_source(session, resource)
            with self._lock:
                if source.store_id not in session.annotation_sources:
                    session.annotation_sources.append(source.store_id)
            source.ensure_lod(self._executor)
            return {"kind": "annotation", "store": self.store_manifest(source.store_id)}
        if resource.kind == "raster-overlay":
            return {"kind": "raster", "layer": self._add_raster(session, resource)}
        msg = "Slides cannot be added as overlays."
        raise ValueError(msg)

    def remove_overlay(self, session: ViewerSession, layer_id: str) -> None:
        """Remove a session layer without invalidating shared immutable caches."""
        with self._lock:
            if layer_id in session.annotation_sources:
                session.annotation_sources.remove(layer_id)
                return
            layer = session.raster_layers.pop(layer_id, None)
            if layer is None:
                msg = "Unknown session layer."
                raise KeyError(msg)
            layer_name = layer["layerName"]
            self.tile_server.layers[session.id].pop(layer_name, None)
            self.tile_server.pyramids[session.id].pop(layer_name, None)

    def get_source(
        self,
        session: ViewerSession,
        store_id: str,
    ) -> AnnotationTileSource:
        """Resolve a store only when assigned to the requesting session."""
        if store_id not in session.annotation_sources:
            msg = "Unknown annotation store."
            raise KeyError(msg)
        source = self._sources.get(store_id)
        if source is None:
            msg = "Unknown annotation store."
            raise KeyError(msg)
        return source

    def store_manifest(self, store_id: str) -> dict[str, Any]:
        """Return a source manifest with API URL templates."""
        source = self._sources[store_id]
        manifest = source.manifest()
        revision_base = f"/api/v1/stores/{store_id}/revisions/{source.revision}"
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
        # Running builders own thread-local read connections. Wait for those finite
        # ingestion jobs before closing stores so Windows can release DB handles.
        self._executor.shutdown(wait=True, cancel_futures=True)
        for source in self._sources.values():
            source.close()
        self._sources.clear()

    def _annotation_source(
        self,
        session: ViewerSession,
        resource: ViewerResource,
    ) -> AnnotationTileSource:
        reader = self.tile_server.layers[session.id]["slide"]
        matrix = TileMatrix(*map(int, reader.info.slide_dimensions))
        source_id = hashlib.blake2b(
            (
                f"{resource.id}:{resource.revision}:"
                f"{matrix.width}:{matrix.height}:{matrix.tile_size}:"
                f"pipeline-{_ANNOTATION_PIPELINE_VERSION}"
            ).encode(),
            digest_size=12,
        ).hexdigest()
        with self._lock:
            existing = self._sources.get(source_id)
            if existing is not None:
                return existing
        store = self._open_store(resource)
        source = AnnotationTileSource(
            store,
            matrix,
            store_id=source_id,
            revision=resource.revision,
            name=resource.name,
            cache_dir=self.cache_dir,
            owns_store=True,
        )
        with self._lock:
            winner = self._sources.setdefault(source_id, source)
        if winner is not source:
            source.close()
        return winner

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

    def _add_raster(
        self,
        session: ViewerSession,
        resource: ViewerResource,
    ) -> dict[str, Any]:
        layer_name = f"r-{resource.id}"
        slide_reader = self.tile_server.layers[session.id]["slide"]
        suffix = resource.path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            info = copy.deepcopy(slide_reader.info)
            info.file_path = str(resource.path)
            reader = VirtualWSIReader(np.asarray(Image.open(resource.path)), info=info)
        else:
            reader = WSIReader.open(resource.path)
        self.tile_server.layers[session.id][layer_name] = reader
        self.tile_server.pyramids[session.id][layer_name] = ZoomifyGenerator(reader)
        document = self._raster_layer_document(session, resource, layer_name, reader)
        session.raster_layers[resource.id] = document
        return document

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
