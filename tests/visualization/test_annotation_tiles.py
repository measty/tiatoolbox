"""Tests for tile coordinates, MVT encoding, LOD and data tiles."""

from __future__ import annotations

import gzip
import json
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from tests.visualization._mvt_decode import decode_mvt
from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.visualization.annotation_tiles.cache import (
    ByteLRUCache,
    SingleFlight,
    TilePayload,
)
from tiatoolbox.visualization.annotation_tiles.grid import TileMatrix
from tiatoolbox.visualization.annotation_tiles.lod import LODIndex
from tiatoolbox.visualization.annotation_tiles.mvt import TileFeature, encode_mvt
from tiatoolbox.visualization.annotation_tiles.source import (
    AnnotationTileSource,
    TileBudgetExceededError,
    TileBudgets,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@pytest.fixture
def synthetic_store(tmp_path: Path) -> SQLiteStore:
    """Return a small deterministic polygon store."""
    store = SQLiteStore(tmp_path / "annotations.db")
    annotations = []
    for y in range(4):
        for x in range(4):
            x0 = 16 + (x * 48)
            y0 = 16 + (y * 48)
            annotations.append(
                Annotation(
                    Polygon(
                        [
                            (x0, y0),
                            (x0 + 24, y0),
                            (x0 + 24, y0 + 24),
                            (x0, y0 + 24),
                        ],
                    ),
                    {"type": (x + y) % 2, "prob": x / 4},
                ),
            )
    store.append_many(annotations)
    yield store
    store.close()


def test_tile_matrix_matches_zoomify_contract() -> None:
    """The shared matrix must align overview and baseline coordinates."""
    matrix = TileMatrix(106_496, 85_248)
    assert matrix.max_zoom == 9
    assert matrix.resolutions == (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
    assert matrix.grid_size(0) == (1, 1)
    assert matrix.grid_size(9) == (416, 333)
    assert matrix.tile_bounds(6, 32, 13) == (
        65_536,
        26_624,
        67_584,
        28_672,
    )
    assert matrix.map_extent == (0, -85_248, 106_496, 0)
    with pytest.raises(IndexError):
        matrix.tile_bounds(9, 416, 0)


def test_default_auto_lod_is_uniform_for_each_zoom(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """The supplied slide uses one representation across every tile at a zoom."""
    matrix = TileMatrix(106_496, 85_248)
    source = AnnotationTileSource(
        synthetic_store,
        matrix,
        store_id="overview-boundary",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "overview-boundary-cache",
    )
    try:
        assert matrix.max_zoom == 9
        manifest = source.manifest()
        assert manifest["representations"]["aggregate"]["maxZoom"] == 3
        assert manifest["representations"]["auto"]["policy"] == {
            "scope": "store-zoom",
            "geometryPromotion": {
                "metric": "projected-area",
                "minimumPixelsSquared": 36.0,
                "maximumZoom": 6,
                "representation": "polygon",
            },
            "ranges": [
                {"minZoom": 0, "maxZoom": 3, "representation": "aggregate"},
                {"minZoom": 4, "maxZoom": 6, "representation": "centroid"},
                {"minZoom": 7, "maxZoom": 9, "representation": "polygon"},
            ],
        }
        assert manifest["budgets"]["point_features"] == 32_000
        assert manifest["budgets"]["compressed_bytes"] == 256 * 1024
        assert manifest["budgets"]["hard_features"] == 100_000
        assert manifest["budgets"]["hard_compressed_bytes"] == 1024 * 1024
        assert manifest["maxConcurrentTileBuilds"] == 1
        source.ensure_lod()
        assert source.lod.manifest()["overviewMaxZoom"] == 3
        assert [source._choose_representation("auto", z) for z in range(10)] == [
            "aggregate",
            "aggregate",
            "aggregate",
            "aggregate",
            "centroid",
            "centroid",
            "centroid",
            "polygon",
            "polygon",
            "polygon",
        ]

        overview = source.vector_tile("auto", 3, 0, 0)
        assert overview.representation == "aggregate"

        stale_key = json.dumps(
            {
                "version": 4,
                "store": source.store_id,
                "revision": source.revision,
                "lod": "ready",
                "representation": "auto",
                "fields": [],
                "filter": None,
                "z": 4,
                "x": 0,
                "y": 0,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        source.persistent_cache.put(
            stale_key,
            TilePayload(b"stale-z4-aggregate", "application/x-stale"),
        )
        detail = source.vector_tile("auto", 4, 0, 0)
        assert detail.cache_status == "miss"
        assert detail.representation == "centroid"
        assert detail.data != b"stale-z4-aggregate"

        polygon = source.vector_tile("auto", 7, 0, 0)
        assert polygon.representation == "polygon"
    finally:
        source.close()


@pytest.mark.parametrize(
    ("matrix", "category_property", "budgets", "expected_representation"),
    [
        (TileMatrix(53_248, 42_624), "type", None, "polygon"),
        (TileMatrix(106_496, 85_248), "class", None, "polygon"),
        (
            TileMatrix(106_496, 85_248),
            "type",
            TileBudgets(polygon_max_downsample=2),
            "centroid",
        ),
    ],
)
def test_vector_cache_isolated_by_complete_tile_contract(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    matrix: TileMatrix,
    category_property: str,
    budgets: TileBudgets | None,
    expected_representation: str,
) -> None:
    """Reopening a revision with different semantics never serves stale bytes."""
    cache_dir = tmp_path / "contract-cache"
    baseline = AnnotationTileSource(
        synthetic_store,
        TileMatrix(106_496, 85_248),
        store_id="contract-store",
        revision="revision",
        name="synthetic",
        cache_dir=cache_dir,
    )
    baseline_payload = baseline.vector_tile("auto", 7, 0, 0)
    assert baseline_payload.cache_status == "miss"
    assert baseline_payload.representation == "polygon"
    baseline.close()

    reopened = AnnotationTileSource(
        synthetic_store,
        matrix,
        store_id="contract-store",
        revision="revision",
        name="synthetic",
        cache_dir=cache_dir,
        category_property=category_property,
        budgets=budgets,
    )
    try:
        payload = reopened.vector_tile("auto", 7, 0, 0)
        assert payload.cache_status == "miss"
        assert payload.representation == expected_representation
    finally:
        reopened.close()


def test_label_cache_isolated_by_tile_matrix_contract(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """Label bytes from a different slide matrix are never reused."""
    cache_dir = tmp_path / "label-contract-cache"
    baseline = AnnotationTileSource(
        synthetic_store,
        TileMatrix(256, 256),
        store_id="label-contract-store",
        revision="revision",
        name="synthetic",
        cache_dir=cache_dir,
    )
    baseline.ensure_lod()
    assert baseline.label_tile(0, 0, 0).cache_status == "miss"
    baseline.close()

    reopened = AnnotationTileSource(
        synthetic_store,
        TileMatrix(512, 256),
        store_id="label-contract-store",
        revision="revision",
        name="synthetic",
        cache_dir=cache_dir,
    )
    try:
        payload = reopened.label_tile(0, 0, 0)
        assert payload.cache_status == "miss"
    finally:
        reopened.close()


def test_lod_sidecar_rebuilds_when_overview_boundary_changes(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """A ready sidecar with the former z4 boundary is incompatible."""
    matrix = TileMatrix(106_496, 85_248)
    path = tmp_path / "lod.sqlite"
    former = LODIndex(path, matrix, "revision", overview_max_zoom=4)
    former.build(synthetic_store)
    assert former.ready
    assert former.manifest()["overviewMaxZoom"] == 4
    former.close()

    current = LODIndex(path, matrix, "revision")
    try:
        assert current.overview_max_zoom == 3
        assert not current.ready
        assert current.manifest() is None

        current.build(synthetic_store)
        assert current.ready
        assert current.manifest()["overviewMaxZoom"] == 3
        persisted_zooms = {
            int(row[0])
            for row in current._con.execute(
                "SELECT DISTINCT z FROM density WHERE revision = ?",
                (current.revision,),
            )
        }
        assert persisted_zooms == {0, 1, 2, 3}
    finally:
        current.close()


def test_manifest_never_reports_ready_before_property_metadata(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A commit between status and metadata reads remains pollable."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(1024, 1024),
        store_id="manifest-race",
        revision="revision",
        name="manifest-race",
        cache_dir=tmp_path / "manifest-race-cache",
    )
    committed = False
    manifest_calls = 0
    ready_properties = {
        "type": {
            "kind": "categorical",
            "categories": [{"value": 0, "count": 1}],
        },
    }

    def racing_manifest() -> dict[str, object] | None:
        nonlocal committed, manifest_calls
        manifest_calls += 1
        if manifest_calls == 1:
            committed = True
            return None
        return {
            "featureCount": len(synthetic_store),
            "bounds": [0, 0, 1024, 1024],
            "geometryTypes": {"Polygon": len(synthetic_store)},
            "properties": ready_properties,
        }

    monkeypatch.setattr(
        type(source.lod),
        "ready",
        property(lambda _index: committed),
    )
    monkeypatch.setattr(source.lod, "manifest", racing_manifest)
    try:
        transitional = source.manifest()
        ready = source.manifest()
    finally:
        source.close()

    assert transitional["lodStatus"] == "not-built"
    assert transitional["properties"] == {}
    assert ready["lodStatus"] == "ready"
    assert ready["properties"] == ready_properties


def test_auto_selection_does_not_preflight_or_change_with_tile_density(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soft targets do not create tile-local representation decisions."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(106_496, 85_248),
        store_id="uniform-selection",
        revision="revision",
        name="uniform-selection",
        cache_dir=tmp_path / "uniform-selection-cache",
        budgets=TileBudgets(polygon_features=1, point_features=1),
    )
    feature_builds: list[str] = []
    original_features = source._features

    def recording_features(
        representation: str,
        z: int,
        x: int,
        y: int,
        fields: tuple[str, ...],
        property_filter: Mapping[str, object] | None,
        *,
        promote_large: bool = False,
    ) -> list[TileFeature]:
        feature_builds.append(representation)
        return original_features(
            representation,
            z,
            x,
            y,
            fields,
            property_filter,
            promote_large=promote_large,
        )

    monkeypatch.setattr(source, "_features", recording_features)
    try:
        source.ensure_lod()
        payload = source.vector_tile("auto", 4, 0, 0)
        assert payload.representation == "centroid"
        assert payload.feature_count == 16
        assert feature_builds == ["centroid"]
    finally:
        source.close()


def test_auto_lod_promotes_screen_visible_polygons_without_duplicates(
    tmp_path: Path,
) -> None:
    """Large structures stay polygons while small objects follow base auto LOD."""
    store = SQLiteStore(tmp_path / "mixed-scale.db")
    store.append_many(
        [
            Annotation(
                Polygon([(100, 100), (612, 100), (612, 612), (100, 612)]),
                {"type": "gland"},
            ),
            Annotation(
                Polygon([(700, 100), (716, 100), (716, 116), (700, 116)]),
                {"type": "cell"},
            ),
        ],
    )
    source = AnnotationTileSource(
        store,
        TileMatrix(65_536, 65_536),
        store_id="mixed-scale",
        revision="revision",
        name="mixed-scale",
        cache_dir=tmp_path / "mixed-scale-cache",
    )
    try:
        source.ensure_lod()
        lod_manifest = source.lod.manifest()
        assert lod_manifest is not None
        assert lod_manifest["geometryPromotion"]["candidateCount"] == 1

        aggregate = decode_mvt(
            gzip.decompress(source.vector_tile("auto", 2, 0, 0).data),
        )
        assert sorted(feature.geometry_type for feature in aggregate.features) == [1, 3]
        assert (
            sum(feature.feature_id is not None for feature in aggregate.features) == 1
        )
        aggregate_point = next(
            feature for feature in aggregate.features if feature.geometry_type == 1
        )
        promoted_polygon = next(
            feature for feature in aggregate.features if feature.geometry_type == 3
        )
        assert aggregate_point.properties["tiatoolbox_aggregate"] is True
        assert "tiatoolbox_aggregate" not in promoted_polygon.properties

        centroid = decode_mvt(gzip.decompress(source.vector_tile("auto", 5, 0, 0).data))
        assert sorted(feature.geometry_type for feature in centroid.features) == [1, 3]

        explicit_centroid = decode_mvt(
            gzip.decompress(source.vector_tile("centroid", 5, 0, 0).data),
        )
        assert [feature.geometry_type for feature in explicit_centroid.features] == [
            1,
            1,
        ]
    finally:
        source.close()
        store.close()


def test_mvt_has_numeric_ids_and_is_deterministic() -> None:
    """Display tiles must use compact protobuf IDs, not UUID tags."""
    features = [
        TileFeature(
            7,
            Polygon([(0, 0), (32, 0), (32, 32), (0, 32)]),
            {"type": 2},
        ),
        TileFeature(8, Point(48, 48), {"type": 1}),
    ]
    first = encode_mvt("annotations", features, tile_bounds=(0, 0, 256, 256))
    second = encode_mvt("annotations", features, tile_bounds=(0, 0, 256, 256))
    assert first.data == second.data
    assert first.output_features == 2
    assert first.output_vertices == 5
    assert not first.budget_exceeded

    layers = [value for number, _, value in _protobuf_fields(first.data) if number == 3]
    layer_fields = list(_protobuf_fields(layers[0]))
    feature_messages = [value for number, _, value in layer_fields if number == 2]
    ids = [
        next(value for number, _, value in _protobuf_fields(feature) if number == 1)
        for feature in feature_messages
    ]
    keys = [value.decode() for number, _, value in layer_fields if number == 3]
    assert ids == [7, 8]
    assert keys == ["type"]
    assert b"canonical" not in first.data


def test_byte_cache_and_single_flight_are_bounded() -> None:
    """The hot cache is byte-limited and duplicate cold work is serialised."""
    cache = ByteLRUCache(max_bytes=5)
    cache.put("a", TilePayload(b"123", "x/test"))
    cache.put("b", TilePayload(b"456", "x/test"))
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.size == 3

    flight = SingleFlight()
    calls = 0
    release = threading.Event()

    def build() -> TilePayload:
        nonlocal calls
        calls += 1
        release.wait(timeout=5)
        return TilePayload(b"x", "x/test")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(flight.run, "key", build) for _ in range(4)]
        release.set()
        results = [future.result() for future in futures]
    assert len(results) == 4
    assert calls == 1


def test_cold_tile_builds_are_bounded_per_source(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only two distinct cache misses may query/encode concurrently."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(1024, 1024),
        store_id="cold-build-limit",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "cold-build-limit-cache",
        max_concurrent_tile_builds=2,
    )
    active = 0
    peak = 0
    calls = 0
    lock = threading.Lock()
    saturated = threading.Event()
    release = threading.Event()

    def build_tile(
        _representation: str,
        _z: int,
        x: int,
        y: int,
        _fields: tuple[str, ...],
        _property_filter: Mapping[str, object] | None,
    ) -> TilePayload:
        nonlocal active, calls, peak
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
            if active == 2:
                saturated.set()
        assert release.wait(timeout=5)
        with lock:
            active -= 1
        return TilePayload(
            f"{x},{y}".encode(),
            "application/vnd.mapbox-vector-tile",
            representation="polygon",
        )

    monkeypatch.setattr(source, "_build_vector_tile", build_tile)
    try:
        coordinates = [(0, 0), (1, 0), (2, 0), (3, 0)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(source.vector_tile, "auto", 2, x, y)
                for x, y in coordinates
            ]
            assert saturated.wait(timeout=5)
            assert peak == 2
            release.set()
            payloads = [future.result(timeout=5) for future in futures]

        assert calls == 4
        assert peak == 2
        assert all(payload.cache_status == "miss" for payload in payloads)
        assert all(
            "queue;dur=" in (payload.server_timing or "") for payload in payloads
        )

        cached = source.vector_tile("auto", 2, 0, 0)
        assert cached.cache_status == "memory"
        assert calls == 4
    finally:
        release.set()
        source.close()


def test_lod_overview_never_uses_interactive_query(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ready overview is proportional to bins, never source annotations."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(1024, 768),
        store_id="test-store",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "cache",
    )
    source.ensure_lod()

    def fail_query(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        pytest.fail("overview requested raw annotation records")

    monkeypatch.setattr(synthetic_store, "query_records", fail_query)
    payload = source.vector_tile("auto", 0, 0, 0)
    assert payload.representation == "aggregate"
    assert 0 < payload.feature_count <= source.budgets.density_bins**2
    source.close()


def test_filtered_overview_uses_only_indexed_category_lod(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Common class filters stay bounded; unsupported overview scans fail fast."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(1024, 768),
        store_id="filtered-overview",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "filtered-overview-cache",
    )
    source.ensure_lod()

    def fail_query(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        pytest.fail("filtered overview requested raw annotation records")

    monkeypatch.setattr(synthetic_store, "query_records", fail_query)
    payload = source.vector_tile(
        "auto",
        0,
        0,
        0,
        property_filter={"op": "in", "property": "type", "values": [0]},
    )
    assert payload.representation == "aggregate"
    assert payload.feature_count > 0
    with pytest.raises(ValueError, match="Filtered overview tiles support"):
        source.vector_tile(
            "auto",
            0,
            0,
            0,
            property_filter={"op": "gte", "property": "prob", "value": 0.5},
        )
    source.close()


def test_pending_overview_does_not_count_entire_store(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before preprocessing, overview returns a bounded building tile."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(1024, 768),
        store_id="pending-store",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "pending-cache",
    )

    def fail_query(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        pytest.fail("pending overview queried source candidates")

    monkeypatch.setattr(synthetic_store, "query_records", fail_query)
    payload = source.vector_tile("auto", 0, 0, 0)
    assert payload.representation == "building"
    assert gzip.decompress(payload.data)
    source.close()


@pytest.mark.parametrize("failure", ["exception", "cancelled"])
def test_failed_lod_overview_is_explicit_and_never_scans_source(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed LOD bypasses cached building tiles without a whole-store scan."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(1024, 768),
        store_id=f"failed-lod-{failure}",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / f"failed-lod-{failure}-cache",
    )
    future: Future[None] = Future()
    source._lod_future = future
    building = source.vector_tile("auto", 0, 0, 0)
    assert building.representation == "building"

    if failure == "cancelled":
        assert future.cancel()
    else:
        future.set_exception(RuntimeError("LOD build failed"))

    assert source.lod_status == "failed"

    def fail_query(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        pytest.fail("failed overview requested raw annotation records")

    monkeypatch.setattr(synthetic_store, "query_records", fail_query)
    with pytest.raises(RuntimeError, match="LOD preprocessing failed"):
        source.vector_tile("auto", 0, 0, 0)
    source.close()


def test_explicit_representation_fails_instead_of_cross_family_fallback(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """A hard-limit failure never turns one polygon tile into centroids."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(256, 256),
        store_id="budget-store",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "budget-cache",
        budgets=TileBudgets(hard_features=2),
    )
    with pytest.raises(
        TileBudgetExceededError,
        match=r"Deterministic polygon-base.*not changed",
    ):
        source.vector_tile("polygon", 0, 0, 0)
    source.close()


def test_terminal_representations_fail_at_absolute_byte_limit(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """Vector and label tiles report an absolute limit instead of degrading."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(1024, 768),
        store_id="byte-budget-store",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "byte-budget-cache",
        budgets=TileBudgets(hard_compressed_bytes=64),
    )
    source.ensure_lod()
    with pytest.raises(
        TileBudgetExceededError,
        match=r"Deterministic aggregate-base.*not changed",
    ):
        source.vector_tile("aggregate", 0, 0, 0)
    with pytest.raises(TileBudgetExceededError, match="Label tile exceeds"):
        source.label_tile(0, 0, 0)
    source.close()


def test_vector_tile_filter_is_selective_and_part_of_cache_key(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """Canonical filters select records and isolate immutable tile caches."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(256, 256),
        store_id="filtered-store",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "filter-cache",
    )
    narrow_filter = {
        "op": "and",
        "args": [
            {"op": "eq", "property": "type", "value": 0},
            {"op": "lt", "property": "prob", "value": 0.25},
        ],
    }
    equivalent_filter = {
        "args": list(reversed(narrow_filter["args"])),
        "op": "and",
    }
    broad_filter = {"op": "gte", "property": "prob", "value": 0.25}

    narrow = source.vector_tile(
        "polygon",
        0,
        0,
        0,
        property_filter=narrow_filter,
    )
    equivalent = source.vector_tile(
        "polygon",
        0,
        0,
        0,
        property_filter=equivalent_filter,
    )
    broad = source.vector_tile(
        "polygon",
        0,
        0,
        0,
        property_filter=broad_filter,
    )

    assert narrow.feature_count == 2
    assert narrow.cache_status == "miss"
    assert equivalent.feature_count == narrow.feature_count
    assert equivalent.cache_status == "memory"
    assert broad.feature_count == 12
    assert broad.cache_status == "miss"
    assert broad.etag != narrow.etag
    source.close()


def test_label_tile_contains_category_and_feature_id(
    synthetic_store: SQLiteStore,
    tmp_path: Path,
) -> None:
    """Segmentation data tiles preserve category and exact pick IDs."""
    source = AnnotationTileSource(
        synthetic_store,
        TileMatrix(256, 256),
        store_id="label-store",
        revision="revision",
        name="synthetic",
        cache_dir=tmp_path / "label-cache",
    )
    source.ensure_lod()
    payload = source.label_tile(0, 0, 0)
    raw = gzip.decompress(payload.data)
    magic, version, dtype, width, height, bands, _ = struct.unpack(
        "<4sBBHHBB",
        raw[:12],
    )
    assert (magic, version, dtype, width, height, bands) == (
        b"TIAD",
        1,
        4,
        256,
        256,
        2,
    )
    pixels = np.frombuffer(raw[12:], dtype="<u4").reshape(height, width, bands)
    assert pixels[20, 20, 0] > 0
    assert pixels[20, 20, 1] == 1
    assert pixels[0, 0].tolist() == [0, 0]
    source.close()


def _protobuf_fields(data: bytes):  # noqa: ANN202
    """Yield a minimal subset of protobuf fields used by MVT tests."""
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            value, offset = data[offset : offset + 8], offset + 8
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            value, offset = data[offset : offset + length], offset + length
        elif wire_type == 5:  # pragma: no cover - supported for completeness
            value, offset = data[offset : offset + 4], offset + 4
        else:  # pragma: no cover
            msg = f"Unsupported protobuf wire type {wire_type}"
            raise AssertionError(msg)
        yield number, wire_type, value


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
