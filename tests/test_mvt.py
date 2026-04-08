"""Tests for lightweight MVT encoding helpers."""

from __future__ import annotations

import importlib.util
import numpy as np
from shapely.geometry import LineString, box
from pathlib import Path

_MVT_SPEC = importlib.util.spec_from_file_location(
    "test_mvt_module",
    Path(__file__).resolve().parents[1] / "tiatoolbox" / "visualization" / "mvt.py",
)
_MVT_MODULE = importlib.util.module_from_spec(_MVT_SPEC)
assert _MVT_SPEC.loader is not None
_MVT_SPEC.loader.exec_module(_MVT_MODULE)

encode_annotation_layer = _MVT_MODULE.encode_annotation_layer
encode_empty_annotation_layer = _MVT_MODULE.encode_empty_annotation_layer


def test_encode_annotation_layer_simplifies_dense_lines() -> None:
    """Simplification should reduce payload size for dense low-value geometry."""
    xs = np.linspace(0, 255, 512)
    dense_line = LineString(
        (float(x), float(128 + np.sin(x / 8) * 3)) for x in xs
    )

    unsimplified = encode_annotation_layer(
        "overlay",
        [(dense_line, {"kind": "line"})],
        tile_bounds=(0, 0, 256, 256),
    )
    simplified = encode_annotation_layer(
        "overlay",
        [(dense_line, {"kind": "line"})],
        tile_bounds=(0, 0, 256, 256),
        simplify_tolerance=2.0,
        min_line_length=1.0,
    )

    assert len(simplified) < len(unsimplified)
    assert len(simplified) > len(encode_empty_annotation_layer("overlay"))


def test_encode_annotation_layer_culls_tiny_polygons() -> None:
    """Sub-pixel polygon fragments can be omitted entirely."""
    tiny_polygon = box(32.0, 32.0, 32.2, 32.2)

    payload = encode_annotation_layer(
        "overlay",
        [(tiny_polygon, {"kind": "tiny"})],
        tile_bounds=(0, 0, 256, 256),
        min_line_length=1.0,
        min_polygon_area=1.0,
    )

    assert payload == encode_empty_annotation_layer("overlay")
