"""Phase-4 correctness contract for renderer-neutral annotation tiles."""

from __future__ import annotations

import gzip
from itertools import pairwise
from typing import TYPE_CHECKING

from shapely.geometry import Point, Polygon

from tests.visualization._mvt_decode import decode_mvt
from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.visualization.annotation_tiles.grid import TileMatrix
from tiatoolbox.visualization.annotation_tiles.mvt import TileFeature, encode_mvt
from tiatoolbox.visualization.annotation_tiles.source import AnnotationTileSource

if TYPE_CHECKING:
    from pathlib import Path


def _signed_area(ring: tuple[tuple[int, int], ...]) -> float:
    return 0.5 * sum((x1 * y2) - (x2 * y1) for (x1, y1), (x2, y2) in pairwise(ring))


def test_independent_mvt_decode_preserves_y_orientation_and_holes() -> None:
    """Asymmetric slide-space Y and polygon holes survive MVT encoding."""
    polygon = Polygon(
        [(116, 224), (308, 224), (308, 376), (116, 376)],
        holes=[[(164, 260), (164, 324), (260, 324), (260, 260)]],
    )
    encoded = encode_mvt(
        "annotations",
        [TileFeature(41, polygon, {"type": 3})],
        tile_bounds=(100, 200, 356, 456),
    )

    layer = decode_mvt(encoded.data)
    assert (layer.extent, layer.version) == (4096, 2)
    assert len(layer.features) == 1
    feature = layer.features[0]
    assert feature.feature_id == 41
    assert feature.geometry_type == 3
    assert feature.properties == {"type": 3}
    assert len(feature.paths) == 2

    exterior, hole = feature.paths
    assert {point[1] for point in exterior} == {384, 2816}
    assert {point[1] for point in hole} == {960, 1984}
    assert _signed_area(exterior) > 0
    assert _signed_area(hole) < 0
    decoded_polygon = Polygon(exterior, [hole])
    assert decoded_polygon.contains(Point(512, 512))
    assert not decoded_polygon.contains(Point(1536, 1400))


def test_source_buffer_repeats_shared_edge_geometry(tmp_path: Path) -> None:
    """A feature just across a seam appears in both buffered neighbour tiles."""
    store_path = tmp_path / "seam.db"
    writer = SQLiteStore(store_path)
    writer.append(
        Annotation(
            Polygon([(260, 40), (263, 40), (263, 120), (260, 120)]),
            {"type": 1},
        ),
        "seam-feature",
    )
    writer.close()
    store = SQLiteStore(store_path, read_only=True)
    source = AnnotationTileSource(
        store,
        TileMatrix(512, 256),
        store_id="seam-store",
        revision="r1",
        name="seam",
        cache_dir=tmp_path / "cache",
    )
    try:
        left = decode_mvt(
            gzip.decompress(source.vector_tile("polygon", 1, 0, 0).data),
        ).features[0]
        right = decode_mvt(
            gzip.decompress(source.vector_tile("polygon", 1, 1, 0).data),
        ).features[0]
    finally:
        source.close()
        store.close()

    assert left.feature_id == right.feature_id == 1
    left_ring, right_ring = left.paths[0], right.paths[0]
    assert max(x for x, _ in left_ring) > 4096
    assert min(x for x, _ in right_ring) >= 0
    left_global = {(x / 16, y / 16) for x, y in left_ring}
    right_global = {(256 + (x / 16), y / 16) for x, y in right_ring}
    assert left_global == right_global
    assert {x for x, _ in left_global} == {260, 263}


def test_ready_aggregates_have_no_pickable_feature_ids(tmp_path: Path) -> None:
    """Density primitives deliberately cannot resolve as source annotations."""
    store_path = tmp_path / "aggregate.db"
    writer = SQLiteStore(store_path)
    writer.append_many(
        [
            Annotation(
                Polygon.from_bounds(x, y, x + 8, y + 8),
                {"type": (x + y) % 2},
            )
            for y in range(16, 256, 32)
            for x in range(16, 256, 32)
        ],
    )
    writer.close()
    store = SQLiteStore(store_path, read_only=True)
    source = AnnotationTileSource(
        store,
        TileMatrix(1024, 768),
        store_id="aggregate-store",
        revision="r1",
        name="aggregate",
        cache_dir=tmp_path / "cache",
    )
    try:
        source.ensure_lod()
        payload = source.vector_tile("aggregate", 0, 0, 0)
        features = decode_mvt(gzip.decompress(payload.data)).features
    finally:
        source.close()
        store.close()

    assert payload.representation == "aggregate"
    assert features
    assert all(feature.feature_id is None for feature in features)
    assert all(feature.geometry_type == 1 for feature in features)
    assert all("type" in feature.properties for feature in features)
    assert all(
        feature.properties["type"] == feature.properties["dominant_type"]
        for feature in features
    )


def test_revision_namespaces_isolate_persistent_tile_cache(tmp_path: Path) -> None:
    """A new revision cannot reuse a prior revision's persisted tile entry."""
    store_path = tmp_path / "revision.db"
    writer = SQLiteStore(store_path)
    writer.append(Annotation(Polygon.from_bounds(8, 8, 64, 64), {"type": 1}))
    writer.close()
    store = SQLiteStore(store_path, read_only=True)
    matrix = TileMatrix(256, 256)
    first_source = AnnotationTileSource(
        store,
        matrix,
        store_id="stable-store-id",
        revision="revision-a",
        name="revision",
        cache_dir=tmp_path / "cache",
    )
    first = first_source.vector_tile("polygon", 0, 0, 0)
    repeated = first_source.vector_tile("polygon", 0, 0, 0)
    first_source.close()

    second_source = AnnotationTileSource(
        store,
        matrix,
        store_id="stable-store-id",
        revision="revision-b",
        name="revision",
        cache_dir=tmp_path / "cache",
    )
    second = second_source.vector_tile("polygon", 0, 0, 0)
    second_source.close()
    store.close()

    assert first.cache_status == "miss"
    assert repeated.cache_status == "memory"
    assert second.cache_status == "miss"
    assert (tmp_path / "cache/stable-store-id/revision-a/tiles.sqlite").exists()
    assert (tmp_path / "cache/stable-store-id/revision-b/tiles.sqlite").exists()


def test_legacy_rgb_property_is_normalised_for_both_renderers(tmp_path: Path) -> None:
    """Legacy float RGB arrays become a compact CSS colour in the MVT."""
    store_path = tmp_path / "direct-color.db"
    writer = SQLiteStore(store_path)
    writer.append(
        Annotation(
            Polygon.from_bounds(16, 16, 64, 64),
            {"color": [0.0, 0.5, 1.0]},
        ),
    )
    writer.append(Annotation(Polygon.from_bounds(80, 16, 128, 64), {}))
    writer.close()
    store = SQLiteStore(store_path, read_only=True)
    source = AnnotationTileSource(
        store,
        TileMatrix(256, 256),
        store_id="direct-color",
        revision="r1",
        name="direct-color",
        cache_dir=tmp_path / "cache",
    )
    try:
        payload = source.vector_tile("polygon", 0, 0, 0, fields=("color",))
        features = decode_mvt(gzip.decompress(payload.data)).features
    finally:
        source.close()
        store.close()

    assert len(features) == 2
    colors = {feature.feature_id: feature.properties["color"] for feature in features}
    assert colors == {1: "#0080ff", 2: "#9ca3af"}
