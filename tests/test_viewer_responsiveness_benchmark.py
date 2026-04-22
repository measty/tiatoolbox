"""Tests for the viewer responsiveness benchmark script helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "viewer_responsiveness_benchmark",
    Path(__file__).resolve().parents[1] / "benchmarks" / "viewer_responsiveness.py",
)
_BENCHMARK_MODULE = importlib.util.module_from_spec(_BENCHMARK_SPEC)
sys.modules[_BENCHMARK_SPEC.name] = _BENCHMARK_MODULE
assert _BENCHMARK_SPEC.loader is not None
_BENCHMARK_SPEC.loader.exec_module(_BENCHMARK_MODULE)

TileSpec = _BENCHMARK_MODULE.TileSpec
build_mvt_path = _BENCHMARK_MODULE.build_mvt_path
capture_annotation_perf = _BENCHMARK_MODULE.capture_annotation_perf
evenly_spaced_indices = _BENCHMARK_MODULE.evenly_spaced_indices
first_visible_overlay_proxy = _BENCHMARK_MODULE.first_visible_overlay_proxy
parse_tile_spec = _BENCHMARK_MODULE.parse_tile_spec
summarize_samples = _BENCHMARK_MODULE.summarize_samples


class DummyTileServer:
    """Minimal TileServer-like class for perf capture tests."""

    calls: list[tuple[str, float, dict[str, object]]] = []

    @staticmethod
    def _log_annotation_perf(
        event: str,
        elapsed_seconds: float,
        **fields: object,
    ) -> None:
        """Record calls from the capture wrapper."""
        DummyTileServer.calls.append((event, elapsed_seconds, fields))


def test_parse_tile_spec_accepts_default_and_named_representations() -> None:
    """Tile CLI syntax should support explicit and implicit representations."""
    assert parse_tile_spec("3:4:5") == TileSpec("full", 3, 4, 5)
    assert parse_tile_spec("overview:0:1:2") == TileSpec("overview", 0, 1, 2)


@pytest.mark.parametrize("value", ["1:2", "a:b:c", "1:-2:3"])
def test_parse_tile_spec_rejects_invalid_values(value: str) -> None:
    """Invalid tile syntax should fail clearly."""
    with pytest.raises(ValueError):
        parse_tile_spec(value)


def test_build_mvt_path_encodes_layer_and_query() -> None:
    """MVT paths should match the TileServer route shape."""
    path = build_mvt_path(
        "cell overlay",
        "default",
        TileSpec("centroids", 2, 3, 4),
        color_property="type",
        fields=("score",),
        where='props["type"] == 1',
    )

    assert path.startswith(
        "/tileserver/layer/cell%20overlay/default/mvt/centroids/2/3/4.pbf?",
    )
    assert "cprop=type" in path
    assert "fields=%5B%22score%22%5D" in path
    assert "where=props%5B%22type%22%5D+%3D%3D+1" in path


def test_summarize_samples_reports_distribution() -> None:
    """Summary helper should report stable endpoint statistics."""
    samples = [
        {"elapsed_ms": 3.0, "status_code": 200, "bytes": 30},
        {"elapsed_ms": 1.0, "status_code": 200, "bytes": 10},
        {"elapsed_ms": 2.0, "status_code": 304, "bytes": 20},
    ]

    assert summarize_samples(samples) == {
        "count": 3,
        "elapsed_ms_min": 1.0,
        "elapsed_ms_median": 2.0,
        "elapsed_ms_mean": 2.0,
        "elapsed_ms_p95": 3.0,
        "elapsed_ms_max": 3.0,
        "status_codes": [200, 304],
        "bytes_min": 10,
        "bytes_max": 30,
    }


def test_evenly_spaced_indices_samples_grid_without_duplicates() -> None:
    """Tile-grid sampling should stay stable and bounded."""
    assert evenly_spaced_indices(9, 3) == [0, 4, 8]
    assert evenly_spaced_indices(2, 3) == [0, 1]


def test_capture_annotation_perf_records_and_restores_logger() -> None:
    """Perf capture should collect records without permanently patching logging."""
    original = DummyTileServer._log_annotation_perf
    DummyTileServer.calls = []

    with capture_annotation_perf(DummyTileServer) as records:
        DummyTileServer._log_annotation_perf("mvt_tile", 0.0123, z=1, cache=False)

    assert DummyTileServer._log_annotation_perf is original
    assert DummyTileServer.calls == [("mvt_tile", 0.0123, {"z": 1, "cache": False})]
    assert records == [
        {
            "event": "mvt_tile",
            "elapsed_ms": 12.3,
            "fields": {"z": 1, "cache": False},
        },
    ]


def test_first_visible_overlay_proxy_adds_metadata_and_tile_times() -> None:
    """First overlay proxy should be the first metadata plus first tile timing."""
    metadata = {
        "label": "property_names_all",
        "method": "GET",
        "path": "/tileserver/prop_names/all",
        "samples": [{"elapsed_ms": 12.5, "status_code": 200}],
    }
    tile = {
        "label": "mvt_full:1:2:3",
        "method": "GET",
        "path": "/tile.pbf",
        "samples": [{"elapsed_ms": 3.25, "status_code": 200}],
    }

    proxy = first_visible_overlay_proxy(metadata, tile)

    assert proxy["elapsed_ms"] == 15.75
    assert [step["label"] for step in proxy["sequence"]] == [
        "property_names_all",
        "mvt_full:1:2:3",
    ]
