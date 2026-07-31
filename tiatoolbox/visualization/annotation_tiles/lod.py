"""Persistent, style-independent metadata and overview aggregates."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapely.geometry import Point

from tiatoolbox.visualization.annotation_tiles.mvt import TileFeature

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.annotation.storage import SQLiteStore
    from tiatoolbox.visualization.annotation_tiles.grid import TileMatrix

_CATEGORY_LIMIT = 64
_NUMERIC_SAMPLE_LIMIT = 16384
_LOD_MANIFEST_VERSION = 2
_DEFAULT_OVERVIEW_ZOOM_OFFSET = 6


@dataclass(slots=True)
class _PropertyAccumulator:
    count: int = 0
    null_count: int = 0
    value_types: set[str] = field(default_factory=set)
    categories: Counter[str] | None = field(default_factory=Counter)
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_sum: float = 0
    numeric_count: int = 0
    numeric_sample: list[float] = field(default_factory=list)

    def add(self, value: Any) -> None:  # noqa: ANN401
        """Add one JSON property value."""
        self.count += 1
        if value is None:
            self.null_count += 1
            self.value_types.add("null")
            return
        value_type = _json_type(value)
        self.value_types.add(value_type)
        if self.categories is not None:
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
            self.categories[key] += 1
            if len(self.categories) > _CATEGORY_LIMIT:
                self.categories = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                self.numeric_count += 1
                self.numeric_sum += numeric
                self.numeric_min = (
                    numeric
                    if self.numeric_min is None
                    else min(self.numeric_min, numeric)
                )
                self.numeric_max = (
                    numeric
                    if self.numeric_max is None
                    else max(self.numeric_max, numeric)
                )
                # A bounded deterministic prefix sample is enough for UI histograms;
                # exact min/max/count remain authoritative.
                if len(self.numeric_sample) < _NUMERIC_SAMPLE_LIMIT:
                    self.numeric_sample.append(numeric)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable property summary."""
        result: dict[str, Any] = {
            "count": self.count,
            "nullCount": self.null_count,
            "valueTypes": sorted(self.value_types),
        }
        if self.categories is not None:
            result["kind"] = "categorical"
            result["categories"] = [
                {"value": json.loads(value), "count": count}
                for value, count in self.categories.most_common()
            ]
        elif self.numeric_count:
            result["kind"] = "numeric"
        else:
            result["kind"] = "text"
        if self.numeric_count:
            result["numeric"] = {
                "count": self.numeric_count,
                "min": self.numeric_min,
                "max": self.numeric_max,
                "mean": self.numeric_sum / self.numeric_count,
                "histogram": _histogram(
                    self.numeric_sample,
                    self.numeric_min,
                    self.numeric_max,
                ),
                "histogramApproximate": self.numeric_count > len(self.numeric_sample),
            }
        return result


class LODIndex:
    """SQLite sidecar containing persisted metadata and density grids."""

    def __init__(
        self,
        path: str | Path,
        matrix: TileMatrix,
        revision: str,
        *,
        category_property: str = "type",
        bins_per_tile: int = 16,
        overview_max_zoom: int | None = None,
    ) -> None:
        """Open a sidecar index."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.matrix = matrix
        self.revision = revision
        self.category_property = category_property
        self.bins_per_tile = bins_per_tile
        self.overview_max_zoom = (
            max(0, matrix.max_zoom - _DEFAULT_OVERVIEW_ZOOM_OFFSET)
            if overview_max_zoom is None
            else min(matrix.max_zoom, max(0, overview_max_zoom))
        )
        self._lock = threading.RLock()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    @property
    def ready(self) -> bool:
        """Return whether a complete index exists for the current revision."""
        with self._lock:
            row = self._con.execute(
                "SELECT status, manifest FROM build WHERE revision = ?",
                (self.revision,),
            ).fetchone()
        return (
            row is not None
            and row[0] == "ready"
            and self._compatible_manifest(row[1]) is not None
        )

    def build(self, store: SQLiteStore) -> None:
        """Build metadata and overview aggregates in one source-table pass."""
        if self.ready:
            return
        started_ns = time.time_ns()
        density: Counter[tuple[int, int, int, str]] = Counter()
        properties: dict[str, _PropertyAccumulator] = defaultdict(_PropertyAccumulator)

        source_rows = store.con.execute("SELECT cx, cy, properties FROM annotations")
        feature_count = 0
        for cx, cy, properties_json in source_rows:
            feature_count += 1
            annotation_properties = json.loads(properties_json or "{}")
            for name, value in annotation_properties.items():
                properties[name].add(value)
            category = json.dumps(
                annotation_properties.get(self.category_property),
                sort_keys=True,
                separators=(",", ":"),
            )
            for z in range(self.overview_max_zoom + 1):
                span = self._cell_span(z)
                density[
                    (z, math.floor(cx / span), math.floor(cy / span), category)
                ] += 1

        bounds_row = store.con.execute(
            "SELECT MIN(min_x), MIN(min_y), MAX(max_x), MAX(max_y) FROM rtree",
        ).fetchone()
        geometry_types = {
            str(name): int(count)
            for name, count in store.con.execute(
                "SELECT objtype, COUNT(*) FROM annotations GROUP BY objtype",
            )
        }
        manifest = {
            "version": _LOD_MANIFEST_VERSION,
            "featureCount": feature_count,
            "bounds": [None if value is None else float(value) for value in bounds_row],
            "geometryTypes": geometry_types,
            "properties": {
                name: accumulator.as_dict() for name, accumulator in properties.items()
            },
            "categoryProperty": self.category_property,
            "binsPerTile": self.bins_per_tile,
            "overviewMaxZoom": self.overview_max_zoom,
            "buildDurationMs": (time.time_ns() - started_ns) / 1_000_000,
        }

        with self._lock, self._con:
            self._con.execute(
                """
                INSERT OR REPLACE INTO build(revision, status, manifest)
                VALUES (?, ?, ?)
                """,
                (self.revision, "building", None),
            )
            self._con.execute(
                "DELETE FROM density WHERE revision = ?",
                (self.revision,),
            )
            self._con.executemany(
                """
                INSERT INTO density(revision, z, cell_x, cell_y, category, count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (self.revision, z, cell_x, cell_y, category, count)
                    for (z, cell_x, cell_y, category), count in density.items()
                ),
            )
            self._con.execute(
                "UPDATE build SET status = ?, manifest = ? WHERE revision = ?",
                (
                    "ready",
                    json.dumps(manifest, separators=(",", ":"), sort_keys=True),
                    self.revision,
                ),
            )

    def manifest(self) -> dict[str, Any] | None:
        """Return persisted metadata when ready."""
        with self._lock:
            row = self._con.execute(
                "SELECT manifest FROM build WHERE revision = ? AND status = 'ready'",
                (self.revision,),
            ).fetchone()
        return None if row is None else self._compatible_manifest(row[0])

    def density_features(
        self,
        z: int,
        x: int,
        y: int,
        category_values: set[object] | None = None,
    ) -> list[TileFeature]:
        """Return bounded aggregate points for one overview tile."""
        if not self.ready or z > self.overview_max_zoom:
            return []
        if category_values is not None and not category_values:
            return []
        self.matrix.validate_tile(z, x, y)
        min_cell_x = x * self.bins_per_tile
        min_cell_y = y * self.bins_per_tile
        max_cell_x = min_cell_x + self.bins_per_tile
        max_cell_y = min_cell_y + self.bins_per_tile
        query = """
                SELECT cell_x, cell_y, category, count
                  FROM density
                 WHERE revision = ? AND z = ?
                   AND cell_x >= ? AND cell_x < ?
                   AND cell_y >= ? AND cell_y < ?
        """
        parameters: list[object] = [
            self.revision,
            z,
            min_cell_x,
            max_cell_x,
            min_cell_y,
            max_cell_y,
        ]
        if category_values is not None:
            category_keys = sorted(_category_key(value) for value in category_values)
            placeholders = ",".join("?" for _ in category_keys)
            query += f" AND category IN ({placeholders})"
            parameters.extend(category_keys)
        query += " ORDER BY cell_y, cell_x, category"
        with self._lock:
            rows = self._con.execute(query, parameters).fetchall()

        grouped: dict[tuple[int, int], list[tuple[Any, int]]] = defaultdict(list)
        for cell_x, cell_y, category, count in rows:
            grouped[(int(cell_x), int(cell_y))].append(
                (json.loads(category), int(count)),
            )
        category_codes = self._category_codes()
        span = self._cell_span(z)
        output: list[TileFeature] = []
        for (cell_x, cell_y), counts in grouped.items():
            total = sum(count for _, count in counts)
            dominant, dominant_count = max(counts, key=lambda item: item[1])
            properties: dict[str, Any] = {
                "count": total,
                "dominant_type": dominant,
                "dominant_fraction": dominant_count / total,
            }
            # Reuse the configured categorical property name so the same client
            # style remains meaningful when auto-LOD changes representation.
            properties.setdefault(self.category_property, dominant)
            for category, count in counts:
                code = category_codes.get(_category_key(category))
                if code is not None:
                    properties[f"c{code}"] = count
            output.append(
                TileFeature(
                    # Aggregates are display primitives, not authoritative
                    # annotations.  Omitting an ID prevents clients from trying
                    # to resolve one through the exact-feature endpoint.
                    feature_id=None,
                    geometry=Point((cell_x + 0.5) * span, (cell_y + 0.5) * span),
                    properties=properties,
                ),
            )
        return output

    def close(self) -> None:
        """Close the sidecar."""
        with self._lock:
            self._con.close()

    def _create_schema(self) -> None:
        with self._con:
            self._con.execute(
                """
                CREATE TABLE IF NOT EXISTS build(
                    revision TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    manifest TEXT
                )
                """,
            )
            self._con.execute(
                """
                CREATE TABLE IF NOT EXISTS density(
                    revision TEXT NOT NULL,
                    z INTEGER NOT NULL,
                    cell_x INTEGER NOT NULL,
                    cell_y INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY(revision, z, cell_x, cell_y, category)
                ) WITHOUT ROWID
                """,
            )

    def _cell_span(self, z: int) -> float:
        return self.matrix.tile_size * self.matrix.downsample(z) / self.bins_per_tile

    def _compatible_manifest(self, value: str | None) -> dict[str, Any] | None:
        """Return a manifest only when its persisted LOD contract still matches."""
        if value is None:
            return None
        try:
            manifest = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(manifest, dict):
            return None
        expected = {
            "version": _LOD_MANIFEST_VERSION,
            "categoryProperty": self.category_property,
            "binsPerTile": self.bins_per_tile,
            "overviewMaxZoom": self.overview_max_zoom,
        }
        if any(
            manifest.get(key) != expected_value
            for key, expected_value in expected.items()
        ):
            return None
        return manifest

    def _category_codes(self) -> dict[str, int]:
        manifest = self.manifest() or {}
        summary = manifest.get("properties", {}).get(self.category_property, {})
        categories = summary.get("categories", [])
        return {
            _category_key(item["value"]): index for index, item in enumerate(categories)
        }


def _json_type(value: Any) -> str:  # noqa: ANN401
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _histogram(
    values: list[float],
    minimum: float | None,
    maximum: float | None,
    bins: int = 32,
) -> dict[str, list[float] | list[int]]:
    if not values or minimum is None or maximum is None:
        return {"edges": [], "counts": []}
    if minimum == maximum:
        return {"edges": [minimum, maximum], "counts": [len(values)]}
    width = (maximum - minimum) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int((value - minimum) / width)))
        counts[index] += 1
    return {
        "edges": [minimum + (index * width) for index in range(bins + 1)],
        "counts": counts,
    }


def _category_key(value: Any) -> str:  # noqa: ANN401
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
