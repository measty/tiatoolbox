"""Annotation-store tile source with uniform zoom LOD and bounded encoding."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import struct
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import shapely
from shapely import wkb as shapely_wkb
from shapely.geometry import MultiPolygon, Point, Polygon, mapping

from tiatoolbox.annotation.storage import normalize_property_filter
from tiatoolbox.visualization.annotation_tiles.cache import (
    ByteLRUCache,
    PersistentTileCache,
    SingleFlight,
    TilePayload,
)
from tiatoolbox.visualization.annotation_tiles.lod import LODIndex
from tiatoolbox.visualization.annotation_tiles.mvt import (
    DEFAULT_BUFFER,
    DEFAULT_EXTENT,
    TileFeature,
    encode_empty_mvt,
    encode_mvt,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from concurrent.futures import Executor, Future

    from tiatoolbox.annotation.storage import AnnotationRecord, SQLiteStore
    from tiatoolbox.visualization.annotation_tiles.grid import TileMatrix

Representation = Literal["auto", "aggregate", "centroid", "polygon"]

_PROPERTY_NAME_PATTERN = re.compile(r"^[^\x00-\x1f]{1,128}$")
_HEX_COLOR_PATTERN = re.compile(r"^#(?P<hex>[0-9a-f]{3}|[0-9a-f]{6})$", re.I)
_RGB_COLOR_PATTERN = re.compile(
    r"^rgb\(\s*(?P<red>\d{1,3})\s*,\s*(?P<green>\d{1,3})\s*,"
    r"\s*(?P<blue>\d{1,3})\s*\)$",
    re.I,
)
_COLOR_CHANNEL_MAX = 255
_DIRECT_COLOR_FALLBACK = "#9ca3af"
_DATA_TILE_HEADER = struct.Struct("<4sBBHHBB")
_DATA_TILE_MAGIC = b"TIAD"
_DATA_TILE_VERSION = 1
_DATA_TILE_DTYPE_UINT32 = 4
_LOD_FAILED_MESSAGE = "Overview LOD preprocessing failed; reload the overlay to retry."
_VECTOR_TILE_CACHE_VERSION = 9


class TileBudgetExceededError(RuntimeError):
    """Raised when an exact tile representation exceeds an absolute limit."""


@dataclass(frozen=True, slots=True)
class TileBudgets:
    """Pathology-calibrated tile targets and absolute safety limits.

    The target values describe the expected operating envelope and are exposed
    for diagnostics. They do not change an individual tile's representation.
    The larger ``hard_*`` values are terminal safety limits: exceeding one
    raises an explicit error instead of silently returning another geometry
    family and producing a patchwork overlay.
    """

    polygon_features: int = 4_000
    point_features: int = 32_000
    vertices: int = 120_000
    compressed_bytes: int = 256 * 1024
    polygon_max_downsample: int = 4
    promoted_polygon_min_screen_area: float = 36.0
    density_bins: int = 16
    hard_features: int = 100_000
    hard_vertices: int = 500_000
    hard_compressed_bytes: int = 1024 * 1024


class AnnotationTileSource:
    """Serve one immutable store revision through renderer-neutral tiles."""

    def __init__(
        self,
        store: SQLiteStore,
        matrix: TileMatrix,
        *,
        store_id: str,
        revision: str,
        name: str,
        cache_dir: str | Path,
        budgets: TileBudgets | None = None,
        category_property: str = "type",
        owns_store: bool = False,
        max_concurrent_tile_builds: int = 1,
    ) -> None:
        """Initialise a revisioned tile source."""
        if max_concurrent_tile_builds <= 0:
            msg = "Concurrent tile build limit must be positive."
            raise ValueError(msg)
        self.store = store
        self.matrix = matrix
        self.store_id = store_id
        self.revision = revision
        self.name = name
        self.budgets = budgets or TileBudgets()
        self.category_property = category_property
        self.owns_store = owns_store
        tile_contract = json.dumps(
            {
                "version": _VECTOR_TILE_CACHE_VERSION,
                "matrix": matrix.as_dict(),
                "categoryProperty": category_property,
                "budgets": asdict(self.budgets),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self._tile_contract_digest = hashlib.blake2b(
            tile_contract.encode(),
            digest_size=12,
        ).hexdigest()
        revision_dir = Path(cache_dir) / store_id / revision
        revision_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = ByteLRUCache()
        self.persistent_cache = PersistentTileCache(revision_dir / "tiles.sqlite")
        self.lod = LODIndex(
            revision_dir / "lod.sqlite",
            matrix,
            revision,
            category_property=category_property,
            bins_per_tile=self.budgets.density_bins,
            polygon_max_downsample=self.budgets.polygon_max_downsample,
            promoted_polygon_min_screen_area=(
                self.budgets.promoted_polygon_min_screen_area
            ),
        )
        self._single_flight = SingleFlight()
        # SQLite queries and Python MVT encoding contend when a cold viewport
        # starts many tiles at once. One builder gave both faster first output
        # and higher total throughput than competing builds on the dense store.
        self._tile_build_slots = threading.BoundedSemaphore(
            max_concurrent_tile_builds,
        )
        self.max_concurrent_tile_builds = max_concurrent_tile_builds
        self._lod_future: Future[None] | None = None
        self._lod_lock = threading.Lock()

    @property
    def tile_revision(self) -> str:
        """Return the renderer-facing tile contract revision."""
        return self._tile_contract_digest

    @property
    def lod_status(self) -> str:
        """Return ``ready``, ``building``, ``failed`` or ``not-built``."""
        if self.lod.ready:
            return "ready"
        future = self._lod_future
        if future is not None and not future.done():
            return "building"
        if future is not None and future.done():
            if future.cancelled():
                return "failed"
            return "failed" if future.exception() else "not-built"
        return "not-built"

    def ensure_lod(self, executor: Executor | None = None) -> Future[None] | None:
        """Start a single background LOD build, or build synchronously."""
        if self.lod.ready:
            return self._lod_future
        with self._lod_lock:
            if self._lod_future is not None and not self._lod_future.done():
                return self._lod_future
            if executor is None:
                self.lod.build(self.store)
                return None
            self._lod_future = executor.submit(self.lod.build, self.store)
            return self._lod_future

    def manifest(self) -> dict[str, Any]:
        """Return current store metadata and immutable representation contract."""
        # Read status before metadata. If a background build commits between
        # these calls, returning ``building`` with ready metadata is harmless
        # and will be polled once more; the inverse ``ready`` plus empty
        # metadata would make clients stop polling before properties arrive.
        lod_status = self.lod_status
        persisted = self.lod.manifest()
        feature_count = (
            int(persisted["featureCount"]) if persisted is not None else len(self.store)
        )
        bounds = persisted["bounds"] if persisted is not None else self._store_bounds()
        geometry_types = (
            persisted["geometryTypes"]
            if persisted is not None
            else self._geometry_types()
        )
        properties = persisted["properties"] if persisted is not None else {}
        return {
            "id": self.store_id,
            "name": self.name,
            "revision": self.revision,
            "featureCount": feature_count,
            "bounds": bounds,
            "geometryTypes": geometry_types,
            "properties": properties,
            "tileMatrix": self.matrix.as_dict(),
            "lodStatus": lod_status,
            "categoryProperty": self.category_property,
            "representations": {
                "auto": {
                    "format": "mvt",
                    "policy": {
                        "scope": "store-zoom",
                        "ranges": self._auto_representation_ranges(),
                        "geometryPromotion": self._geometry_promotion_policy(),
                    },
                },
                "aggregate": {
                    "format": "mvt",
                    "maxZoom": self.lod.overview_max_zoom,
                    "binsPerTile": self.budgets.density_bins,
                },
                "centroid": {"format": "mvt"},
                "polygon": {
                    "format": "mvt",
                    "maxDownsample": self.budgets.polygon_max_downsample,
                },
                "labels": {
                    "format": "tiatoolbox-uint32",
                    "bands": ["category", "feature_id"],
                    "available": self._supports_label_tiles(geometry_types),
                },
            },
            "budgets": asdict(self.budgets),
            "maxConcurrentTileBuilds": self.max_concurrent_tile_builds,
        }

    def feature(self, feature_id: int) -> dict[str, Any] | None:
        """Return exact authoritative geometry/properties for a numeric feature ID."""
        row = self.store.con.execute(
            """
            SELECT id, [key], objtype, cx, cy, geometry, properties, area
              FROM annotations
             WHERE id = ?
            """,
            (feature_id,),
        ).fetchone()
        if row is None:
            return None
        row_id, key, object_type, cx, cy, geometry_blob, properties, area = row
        geometry = shapely_wkb.loads(self.store._unpack_wkb(geometry_blob, cx, cy))  # noqa: SLF001
        return {
            "type": "Feature",
            "id": str(row_id),
            "geometry": mapping(geometry),
            "properties": {
                **json.loads(properties or "{}"),
                "_tiatoolbox": {
                    "canonicalId": key,
                    "objectType": object_type,
                    "area": area,
                    "revision": self.revision,
                },
            },
        }

    def pick(self, x: float, y: float, tolerance: float = 0) -> int | None:
        """Resolve a close-detail slide coordinate to an authoritative row ID.

        Browser renderers use local hit detection first. This bounded spatial
        fallback covers WebGL/embedded-browser implementations whose vector-tile
        hit buffer is unavailable or unreliable.
        """
        values = np.asarray([x, y, tolerance], dtype=float)
        if not np.all(np.isfinite(values)):
            msg = "Pick coordinates and tolerance must be finite."
            raise ValueError(msg)
        if tolerance < 0 or tolerance > 64:  # noqa: PLR2004
            msg = "Pick tolerance must be between 0 and 64 slide pixels."
            raise ValueError(msg)
        bounds = (x - tolerance, y - tolerance, x + tolerance, y + tolerance)
        # A zero-area RTree query has no overlap under strict interval tests.
        if tolerance == 0:
            bounds = (x - 1e-9, y - 1e-9, x + 1e-9, y + 1e-9)
        records = list(self.store.query_records(bounds, (), include_geometry=True))
        if not records:
            return None
        geometries = shapely.from_wkb(
            np.asarray([record.wkb for record in records], dtype=object),
        )
        point = Point(x, y)
        distances = shapely.distance(geometries, point)
        covered = shapely.covers(geometries, point)
        candidates = [
            (
                not bool(covered[index]),
                float(distances[index]),
                float(record.area) if record.area is not None else float("inf"),
                record.id,
            )
            for index, record in enumerate(records)
            if bool(covered[index]) or float(distances[index]) <= tolerance
        ]
        return None if not candidates else min(candidates)[-1]

    def vector_tile(
        self,
        representation: Representation,
        z: int,
        x: int,
        y: int,
        *,
        fields: Iterable[str] = (),
        property_filter: Mapping[str, object] | None = None,
    ) -> TilePayload:
        """Return a gzip-compressed vector tile with uniform zoom LOD."""
        self.matrix.validate_tile(z, x, y)
        selected_fields = self._normalise_fields(fields)
        persisted_manifest = self.lod.manifest()
        if persisted_manifest is not None:
            allowed_fields = set(persisted_manifest.get("properties", {}))
            unknown_fields = sorted(set(selected_fields).difference(allowed_fields))
            if unknown_fields:
                msg = "Unknown display properties: " + ", ".join(unknown_fields)
                raise ValueError(msg)
        normalized_filter = normalize_property_filter(property_filter)
        canonical_filter = (
            json.dumps(
                normalized_filter,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if normalized_filter is not None
            else None
        )
        lod_generation = self.lod_status
        cache_key = json.dumps(
            {
                "version": _VECTOR_TILE_CACHE_VERSION,
                "contract": self._tile_contract_digest,
                "store": self.store_id,
                "revision": self.revision,
                "lod": lod_generation,
                "representation": representation,
                "fields": selected_fields,
                "filter": canonical_filter,
                "z": z,
                "x": x,
                "y": y,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        cached = self.memory_cache.get(cache_key)
        if cached is not None:
            return replace(cached, cache_status="memory", server_timing="cache;dur=0")
        cached = self.persistent_cache.get(cache_key)
        if cached is not None:
            self.memory_cache.put(cache_key, cached)
            return replace(cached, cache_status="disk", server_timing="cache;dur=0")

        def build_once() -> TilePayload:
            queued_at = time.perf_counter()
            with self._tile_build_slots:
                started = time.perf_counter()
                second_hit = self.memory_cache.get(cache_key)
                if second_hit is not None:
                    return replace(
                        second_hit,
                        cache_status="memory",
                        server_timing="cache;dur=0",
                    )
                payload = self._build_vector_tile(
                    representation,
                    z,
                    x,
                    y,
                    selected_fields,
                    normalized_filter,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                queued_ms = (started - queued_at) * 1000
                stages = payload.server_timing
                timing = f"queue;dur={queued_ms:.3f}, tile;dur={elapsed_ms:.3f}"
                payload = replace(
                    payload.with_etag(),
                    cache_status="miss",
                    server_timing=timing if not stages else f"{timing}, {stages}",
                )
                # A pending overview is deliberately not persisted: once the LOD
                # build completes, its cache key changes and the aggregate replaces
                # it.
                self.memory_cache.put(cache_key, payload)
                if payload.representation != "building":
                    self.persistent_cache.put(cache_key, payload)
                return payload

        return self._single_flight.run(cache_key, build_once)

    def label_tile(
        self,
        z: int,
        x: int,
        y: int,
        *,
        category_property: str | None = None,
    ) -> TilePayload:
        """Return an interleaved uint32 category/feature-ID data tile.

        This representation is intended for non-overlapping segmentation output.
        It is style independent and can be palette-coloured by a WebGL data layer.
        """
        self.matrix.validate_tile(z, x, y)
        category_property = category_property or self.category_property
        cache_key = (
            f"labels:1:{self._tile_contract_digest}:{self.store_id}:{self.revision}:"
            f"{category_property}:{z}:{x}:{y}"
        )
        cached = self.memory_cache.get(cache_key)
        if cached is not None:
            return replace(cached, cache_status="memory", server_timing="cache;dur=0")
        cached = self.persistent_cache.get(cache_key)
        if cached is not None:
            self.memory_cache.put(cache_key, cached)
            return replace(cached, cache_status="disk", server_timing="cache;dur=0")

        def build_once() -> TilePayload:
            queued_at = time.perf_counter()
            with self._tile_build_slots:
                started = time.perf_counter()
                second_hit = self.memory_cache.get(cache_key)
                if second_hit is not None:
                    return replace(second_hit, cache_status="memory")
                payload = self._build_label_tile(
                    z,
                    x,
                    y,
                    category_property,
                ).with_etag()
                payload = replace(
                    payload,
                    cache_status="miss",
                    server_timing=(
                        f"queue;dur={(started - queued_at) * 1000:.3f}, "
                        f"tile;dur={(time.perf_counter() - started) * 1000:.3f}"
                    ),
                )
                self.memory_cache.put(cache_key, payload)
                self.persistent_cache.put(cache_key, payload)
                return payload

        return self._single_flight.run(cache_key, build_once)

    def close(self) -> None:
        """Release caches, sidecars and optionally the source store."""
        self.persistent_cache.close()
        self.lod.close()
        self.memory_cache.clear()
        if self.owns_store:
            self.store.close()

    def _build_vector_tile(
        self,
        requested: Representation,
        z: int,
        x: int,
        y: int,
        fields: tuple[str, ...],
        property_filter: Mapping[str, object] | None,
    ) -> TilePayload:
        select_started = time.perf_counter()
        representation = self._choose_representation(requested, z)
        selected_at = time.perf_counter()
        tile_bounds = self.matrix.tile_bounds(z, x, y)
        if representation == "building":
            data = gzip.compress(encode_empty_mvt(), compresslevel=5, mtime=0)
            return TilePayload(
                data=data,
                content_type="application/vnd.mapbox-vector-tile",
                content_encoding="gzip",
                representation="building",
                server_timing=(
                    f"select;dur={(selected_at - select_started) * 1000:.3f}"
                ),
            )

        promote_large = representation == "aggregate" or (
            requested == "auto" and representation == "centroid"
        )
        features = self._features(
            representation,
            z,
            x,
            y,
            fields,
            property_filter,
            promote_large=promote_large,
        )
        queried_at = time.perf_counter()
        downsample = self.matrix.downsample(z)
        result = encode_mvt(
            "annotations",
            features,
            tile_bounds=tile_bounds,
            extent=DEFAULT_EXTENT,
            buffer=DEFAULT_BUFFER,
            simplify_tolerance=(
                downsample * 0.25 if representation == "polygon" or promote_large else 0
            ),
            max_features=self.budgets.hard_features,
            max_vertices=self.budgets.hard_vertices,
        )
        encoded_at = time.perf_counter()
        compressed = gzip.compress(result.data, compresslevel=5, mtime=0)
        compressed_at = time.perf_counter()
        if (
            result.budget_exceeded
            or len(compressed) > self.budgets.hard_compressed_bytes
        ):
            msg = (
                f"Deterministic {representation}-base tile {z}/{x}/{y} exceeds "
                "the absolute safety limit; its representation rule was not "
                "changed. "
                f"Input features: {result.input_features:,}; encoded vertices: "
                f"{result.output_vertices:,}; compressed bytes: {len(compressed):,}."
            )
            raise TileBudgetExceededError(msg)
        return TilePayload(
            data=compressed,
            content_type="application/vnd.mapbox-vector-tile",
            content_encoding="gzip",
            representation=representation,
            feature_count=result.output_features,
            vertex_count=result.output_vertices,
            server_timing=(
                f"select;dur={(selected_at - select_started) * 1000:.3f}, "
                f"query;dur={(queried_at - selected_at) * 1000:.3f}, "
                f"encode;dur={(encoded_at - queried_at) * 1000:.3f}, "
                f"compress;dur={(compressed_at - encoded_at) * 1000:.3f}"
            ),
        )

    def _choose_representation(
        self,
        requested: Representation,
        z: int,
    ) -> str:
        if requested == "aggregate":
            if z <= self.lod.overview_max_zoom and not self.lod.ready:
                if self.lod_status == "failed":
                    raise RuntimeError(_LOD_FAILED_MESSAGE)
                return "building"
            return "aggregate"
        if requested != "auto":
            return requested
        if z <= self.lod.overview_max_zoom:
            if self.lod_status == "failed":
                raise RuntimeError(_LOD_FAILED_MESSAGE)
            return "aggregate" if self.lod.ready else "building"
        if self.matrix.downsample(z) > self.budgets.polygon_max_downsample:
            return "centroid"
        return "polygon"

    def _auto_representation_ranges(self) -> list[dict[str, int | str]]:
        """Return the immutable whole-slide representation schedule."""
        ranges: list[dict[str, int | str]] = [
            {
                "minZoom": 0,
                "maxZoom": self.lod.overview_max_zoom,
                "representation": "aggregate",
            },
        ]
        detail_min = self.lod.overview_max_zoom + 1
        polygon_min = next(
            (
                z
                for z in range(detail_min, self.matrix.max_zoom + 1)
                if self.matrix.downsample(z) <= self.budgets.polygon_max_downsample
            ),
            self.matrix.max_zoom + 1,
        )
        if detail_min < polygon_min:
            ranges.append(
                {
                    "minZoom": detail_min,
                    "maxZoom": polygon_min - 1,
                    "representation": "centroid",
                },
            )
        if polygon_min <= self.matrix.max_zoom:
            ranges.append(
                {
                    "minZoom": polygon_min,
                    "maxZoom": self.matrix.max_zoom,
                    "representation": "polygon",
                },
            )
        return ranges

    def _geometry_promotion_policy(self) -> dict[str, int | float | str] | None:
        """Describe deterministic large-polygon promotion in auto tiles."""
        if "area" not in self.store.table_columns or self.lod.promotion_max_zoom < 0:
            return None
        return {
            "metric": "projected-area",
            "minimumPixelsSquared": self.budgets.promoted_polygon_min_screen_area,
            "maximumZoom": self.lod.promotion_max_zoom,
            "representation": "polygon",
        }

    def _features(
        self,
        representation: str,
        z: int,
        x: int,
        y: int,
        fields: tuple[str, ...],
        property_filter: Mapping[str, object] | None,
        *,
        promote_large: bool = False,
    ) -> list[TileFeature]:
        if representation == "aggregate":
            if self.lod.ready and z <= self.lod.overview_max_zoom:
                if property_filter is None:
                    aggregates = self.lod.density_features(z, x, y)
                    return (
                        aggregates
                        + self._promoted_polygon_features(
                            z,
                            x,
                            y,
                            fields,
                            property_filter,
                        )
                        if promote_large
                        else aggregates
                    )
                category_values = _overview_category_filter_values(
                    property_filter,
                    self.category_property,
                )
                if category_values is None:
                    msg = (
                        "Filtered overview tiles support only eq/in filters on "
                        f"the indexed category property {self.category_property!r}."
                    )
                    raise ValueError(msg)
                aggregates = self.lod.density_features(
                    z,
                    x,
                    y,
                    category_values,
                )
                return (
                    aggregates
                    + self._promoted_polygon_features(
                        z,
                        x,
                        y,
                        fields,
                        property_filter,
                    )
                    if promote_large
                    else aggregates
                )
            return self._dynamic_aggregates(z, x, y, property_filter)
        bounds = self._buffered_tile_bounds(z, x, y)
        include_geometry = representation == "polygon"
        records = list(
            self.store.query_records(
                bounds,
                fields,
                include_geometry=include_geometry,
                property_filter=property_filter,
            ),
        )
        if representation == "centroid":
            promoted = (
                self._promoted_polygon_features(
                    z,
                    x,
                    y,
                    fields,
                    property_filter,
                )
                if promote_large
                else []
            )
            promoted_ids = {
                feature.feature_id
                for feature in promoted
                if feature.feature_id is not None
            }
            return [
                TileFeature(
                    record.id,
                    Point(record.cx, record.cy),
                    _normalise_tile_properties(
                        record.properties,
                        ensure_direct_color="color" in fields,
                    ),
                )
                for record in records
                if record.id not in promoted_ids
            ] + promoted
        # OpenLayers uses the MVT feature sequence as the painter's order.
        # Emit large structures first so progressively smaller polygons are
        # painted over them consistently in every independently queried tile.
        records.sort(key=_polygon_draw_order)
        return [
            TileFeature(
                record.id,
                record.wkb,
                _normalise_tile_properties(
                    record.properties,
                    ensure_direct_color="color" in fields,
                ),
            )
            for record in records
            if record.wkb is not None
        ]

    def _promoted_polygon_features(
        self,
        z: int,
        x: int,
        y: int,
        fields: tuple[str, ...],
        property_filter: Mapping[str, object] | None,
    ) -> list[TileFeature]:
        """Return screen-visible polygons selected by the persisted macro index."""
        bounds = self._buffered_tile_bounds(z, x, y)
        promoted_ids = self.lod.promoted_ids(z, bounds)
        records = sorted(
            self.store.records_by_ids(
                promoted_ids,
                fields,
                include_geometry=True,
                property_filter=property_filter,
            ),
            key=_polygon_draw_order,
        )
        return [
            TileFeature(
                record.id,
                record.wkb,
                _normalise_tile_properties(
                    record.properties,
                    ensure_direct_color="color" in fields,
                ),
            )
            for record in records
            if record.wkb is not None
        ]

    def _dynamic_aggregates(
        self,
        z: int,
        x: int,
        y: int,
        property_filter: Mapping[str, object] | None,
    ) -> list[TileFeature]:
        bounds = self.matrix.tile_bounds(z, x, y)
        records = self.store.query_records(
            bounds,
            (self.category_property,),
            include_geometry=False,
            property_filter=property_filter,
        )
        bin_span = (
            self.matrix.tile_size
            * self.matrix.downsample(z)
            / self.budgets.density_bins
        )
        groups: dict[tuple[int, int], Counter[object]] = defaultdict(Counter)
        for record in records:
            bin_x = int((record.cx - bounds[0]) // bin_span)
            bin_y = int((record.cy - bounds[1]) // bin_span)
            category = record.properties.get(self.category_property)
            groups[(bin_x, bin_y)][category] += 1
        output = []
        for (bin_x, bin_y), counts in sorted(groups.items()):
            total = sum(counts.values())
            dominant, dominant_count = counts.most_common(1)[0]
            properties = {
                "tiatoolbox_aggregate": True,
                "count": total,
                "dominant_type": dominant,
                "dominant_fraction": dominant_count / total,
            }
            properties.setdefault(self.category_property, dominant)
            output.append(
                TileFeature(
                    None,
                    Point(
                        bounds[0] + ((bin_x + 0.5) * bin_span),
                        bounds[1] + ((bin_y + 0.5) * bin_span),
                    ),
                    properties,
                ),
            )
        return output

    def _build_label_tile(
        self,
        z: int,
        x: int,
        y: int,
        category_property: str,
    ) -> TilePayload:
        manifest = self.lod.manifest()
        if manifest is None:
            msg = "Label tiles require a completed LOD/property manifest."
            raise RuntimeError(msg)
        category_summary = manifest.get("properties", {}).get(category_property, {})
        categories = category_summary.get("categories", [])
        category_codes = {
            json.dumps(item["value"], separators=(",", ":"), sort_keys=True): index + 1
            for index, item in enumerate(categories)
        }
        if not category_codes:
            msg = f"Property {category_property!r} is not a bounded categorical field."
            raise ValueError(msg)

        bounds = self.matrix.tile_bounds(z, x, y)
        records = self.store.query_records(
            bounds,
            (category_property,),
            include_geometry=True,
        )
        category_image = np.zeros(
            (self.matrix.tile_size, self.matrix.tile_size),
            dtype=np.int32,
        )
        id_image = np.zeros_like(category_image)
        downsample = self.matrix.downsample(z)
        for record in records:
            if record.wkb is None:
                continue
            geometry = shapely_wkb.loads(record.wkb)
            polygons = (
                [geometry]
                if isinstance(geometry, Polygon)
                else list(geometry.geoms)
                if isinstance(geometry, MultiPolygon)
                else []
            )
            category_key = json.dumps(
                record.properties.get(category_property),
                separators=(",", ":"),
                sort_keys=True,
            )
            category_code = category_codes.get(category_key, 0)
            if record.id > np.iinfo(np.int32).max:
                msg = "Label tiles require feature IDs within signed 32-bit range."
                raise OverflowError(msg)
            for polygon in polygons:
                exterior = self._tile_ring(polygon.exterior.coords, bounds, downsample)
                if len(exterior) < 3:  # noqa: PLR2004
                    continue
                cv2.fillPoly(id_image, [exterior], int(record.id))
                cv2.fillPoly(category_image, [exterior], int(category_code))
                for interior in polygon.interiors:
                    hole = self._tile_ring(interior.coords, bounds, downsample)
                    if len(hole) >= 3:  # noqa: PLR2004
                        cv2.fillPoly(id_image, [hole], 0)
                        cv2.fillPoly(category_image, [hole], 0)

        interleaved = np.stack((category_image, id_image), axis=-1).astype(np.uint32)
        header = _DATA_TILE_HEADER.pack(
            _DATA_TILE_MAGIC,
            _DATA_TILE_VERSION,
            _DATA_TILE_DTYPE_UINT32,
            self.matrix.tile_size,
            self.matrix.tile_size,
            2,
            0,
        )
        compressed = gzip.compress(
            header + interleaved.tobytes(),
            compresslevel=5,
            mtime=0,
        )
        if len(compressed) > self.budgets.hard_compressed_bytes:
            msg = (
                "Label tile exceeds the configured compressed-byte budget; "
                "use a vector representation for this viewport."
            )
            raise TileBudgetExceededError(msg)
        return TilePayload(
            data=compressed,
            content_type="application/vnd.tiatoolbox.annotation-tile",
            content_encoding="gzip",
            representation="labels",
            feature_count=int(np.count_nonzero(np.unique(id_image))),
        )

    @staticmethod
    def _tile_ring(
        coordinates: Iterable[tuple[float, float]],
        bounds: tuple[float, float, float, float],
        downsample: int,
    ) -> np.ndarray:
        return np.asarray(
            [
                (
                    round((coordinate[0] - bounds[0]) / downsample),
                    round((coordinate[1] - bounds[1]) / downsample),
                )
                for coordinate in coordinates
            ],
            dtype=np.int32,
        )

    def _buffered_tile_bounds(
        self,
        z: int,
        x: int,
        y: int,
    ) -> tuple[float, float, float, float]:
        """Return source bounds matching the MVT encoder's seam buffer."""
        buffer_pixels = (DEFAULT_BUFFER / DEFAULT_EXTENT) * self.matrix.tile_size
        return self.matrix.tile_bounds(z, x, y, buffer_pixels=buffer_pixels)

    def _store_bounds(self) -> list[float | None]:
        row = self.store.con.execute(
            "SELECT MIN(min_x), MIN(min_y), MAX(max_x), MAX(max_y) FROM rtree",
        ).fetchone()
        return [None if value is None else float(value) for value in row]

    def _geometry_types(self) -> dict[str, int]:
        return {
            str(name): int(count)
            for name, count in self.store.con.execute(
                "SELECT objtype, COUNT(*) FROM annotations GROUP BY objtype",
            )
        }

    @staticmethod
    def _supports_label_tiles(geometry_types: Mapping[str, int]) -> bool:
        supported = {"Polygon", "MultiPolygon"}
        return bool(geometry_types) and set(geometry_types) <= supported

    @staticmethod
    def _normalise_fields(fields: Iterable[str]) -> tuple[str, ...]:
        result = tuple(dict.fromkeys(field for field in fields if field))
        if len(result) > 16:  # noqa: PLR2004
            msg = "At most 16 display properties may be embedded in a tile."
            raise ValueError(msg)
        if not all(_PROPERTY_NAME_PATTERN.fullmatch(field) for field in result):
            msg = "Invalid display property name."
            raise ValueError(msg)
        return result


def _polygon_draw_order(record: AnnotationRecord) -> tuple[float, int]:
    """Return a stable largest-first painter order for one geometry record."""
    area = record.area
    if area is None or not math.isfinite(area) or area < 0:
        min_x, min_y, max_x, max_y = record.bounds
        area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    return -area, record.id


def _normalise_tile_properties(
    properties: Mapping[str, Any],
    *,
    ensure_direct_color: bool = False,
) -> dict[str, Any]:
    """Return renderer-safe tile properties, including legacy RGB colours."""
    result = dict(properties)
    if "color" not in result:
        if ensure_direct_color:
            result["color"] = _DIRECT_COLOR_FALLBACK
        return result
    color = _normalise_direct_color(result["color"])
    if color is None:
        if ensure_direct_color:
            result["color"] = _DIRECT_COLOR_FALLBACK
        else:
            result.pop("color", None)
    else:
        result["color"] = color
    return result


def _normalise_direct_color(value: object) -> str | None:
    """Normalize a CSS colour or legacy 0..1/0..255 RGB array to hex."""
    if isinstance(value, str):
        return _normalise_css_color(value)
    if not isinstance(value, (list, tuple)) or len(value) != 3:  # noqa: PLR2004
        return None
    try:
        channels = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if channels.shape != (3,) or not np.all(np.isfinite(channels)):
        return None
    if np.all((channels >= 0) & (channels <= 1)):
        channels *= _COLOR_CHANNEL_MAX
    elif not np.all((channels >= 0) & (channels <= _COLOR_CHANNEL_MAX)):
        return None
    return "#" + "".join(f"{round(channel):02x}" for channel in channels)


def _normalise_css_color(value: str) -> str | None:
    """Normalize supported CSS hexadecimal and RGB colour strings."""
    hexadecimal = _HEX_COLOR_PATTERN.fullmatch(value.strip())
    if hexadecimal:
        digits = hexadecimal.group("hex").lower()
        if len(digits) == 3:  # noqa: PLR2004
            digits = "".join(channel * 2 for channel in digits)
        return f"#{digits}"
    rgb = _RGB_COLOR_PATTERN.fullmatch(value.strip())
    if not rgb:
        return None
    channels = [int(rgb.group(name)) for name in ("red", "green", "blue")]
    if any(channel > _COLOR_CHANNEL_MAX for channel in channels):
        return None
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _overview_category_filter_values(  # noqa: PLR0911
    property_filter: Mapping[str, object],
    category_property: str,
) -> set[object] | None:
    """Extract a bounded category set that can use the persisted overview LOD."""
    operator = property_filter.get("op")
    if operator in {"eq", "in"}:
        if property_filter.get("property") != category_property:
            return None
        values = (
            [property_filter.get("value")]
            if operator == "eq"
            else property_filter.get("values")
        )
        return set(values) if isinstance(values, list) else None
    if operator not in {"and", "or"}:
        return None
    arguments = property_filter.get("args")
    if not isinstance(arguments, list) or not arguments:
        return None
    extracted = [
        _overview_category_filter_values(argument, category_property)
        if isinstance(argument, Mapping)
        else None
        for argument in arguments
    ]
    if any(values is None for values in extracted):
        return None
    category_sets = [values for values in extracted if values is not None]
    if operator == "or":
        return set().union(*category_sets)
    result = set(category_sets[0])
    for values in category_sets[1:]:
        result.intersection_update(values)
    return result
