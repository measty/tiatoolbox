"""Integration tests for the versioned efficient viewer API."""

from __future__ import annotations

import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.annotation.storage import (
    PROPERTY_FILTER_MAX_BYTES,
    PROPERTY_FILTER_MAX_DEPTH,
    PROPERTY_FILTER_MAX_IN_VALUES,
    PROPERTY_FILTER_MAX_NODES,
)
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def viewer_app(tmp_path: Path) -> Iterator[TileServer]:
    """Create a path-confined viewer with one local slide and store."""
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()
    Image.fromarray(np.zeros((384, 512, 3), dtype=np.uint8)).save(
        slides / "sample.tif",
    )
    store = SQLiteStore(overlays / "sample.db")
    store.append_many(
        Annotation(
            Polygon([(x, y), (x + 12, y), (x + 12, y + 12), (x, y + 12)]),
            {"type": (x // 32) % 3, "prob": x / 512},
        )
        for y in range(16, 368, 24)
        for x in range(16, 496, 24)
    )
    store.close()
    app = TileServer(
        "Efficient viewer test",
        {},
        slide_roots=[slides],
        overlay_roots=[overlays],
        viewer_cache_dir=tmp_path / "cache",
    )
    app.config.update(TESTING=True)
    yield app
    app.viewer_services.close()


def _load_resources(client):  # noqa: ANN001, ANN202
    bootstrap = client.get("/api/v1/bootstrap")
    assert bootstrap.status_code == 200
    document = bootstrap.get_json()
    slide_id = document["catalog"]["slides"][0]["id"]
    overlay_id = document["catalog"]["overlays"][0]["id"]
    slide = client.put(
        "/api/v1/session/slide",
        json={"resourceId": slide_id},
    )
    overlay = client.post(
        "/api/v1/session/overlays",
        json={"resourceId": overlay_id},
    )
    return bootstrap, slide, overlay


def _with_filter(url: str, property_filter: object) -> str:
    """Append an encoded property-filter AST to a manifest URL."""
    separator = "&" if "?" in url else "?"
    return separator.join(
        (url, urlencode({"filter": json.dumps(property_filter)})),
    )


def test_bootstrap_select_and_path_redaction(viewer_app: TileServer) -> None:
    """Clients receive opaque resources and a shared raster/vector tile matrix."""
    with viewer_app.test_client() as client:
        bootstrap, slide, overlay = _load_resources(client)
        assert bootstrap.headers["Cache-Control"] == "no-cache"
        assert slide.status_code == 200
        assert slide.get_json()["dimensions"] == [512, 384]
        assert slide.get_json()["tileMatrix"]["maxZoom"] == 1
        assert "{TileGroup}" in slide.get_json()["tileUrl"]
        assert overlay.status_code == 201
        assert overlay.get_json()["kind"] == "annotation"
        combined = b"".join((bootstrap.data, slide.data, overlay.data)).decode()
        assert (
            str(Path(viewer_app.viewer_services.registry.overlay_roots[0]))
            not in combined
        )
        assert "sample.db" in combined

        capabilities = bootstrap.get_json()["capabilities"]
        assert capabilities["serverFilters"] is True
        assert capabilities["serverFilter"] == {
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
        }


def test_reselecting_slide_uses_a_fresh_raster_tile_namespace(
    viewer_app: TileServer,
) -> None:
    """Every committed selection isolates tiles from earlier map instances."""
    raster_path = viewer_app.viewer_services.registry.overlay_roots[0] / "mask.png"
    Image.fromarray(np.zeros((384, 512, 3), dtype=np.uint8)).save(raster_path)
    viewer_app.viewer_services.registry.refresh()
    with viewer_app.test_client() as client:
        bootstrap = client.get("/api/v1/bootstrap").get_json()
        resource = bootstrap["catalog"]["slides"][0]
        raster_resource = next(
            overlay
            for overlay in bootstrap["catalog"]["overlays"]
            if overlay["name"] == raster_path.name
        )

        first = client.put(
            "/api/v1/session/slide",
            json={"resourceId": resource["id"]},
        ).get_json()
        first_raster = client.post(
            "/api/v1/session/overlays",
            json={"resourceId": raster_resource["id"]},
        ).get_json()["layer"]
        second = client.put(
            "/api/v1/session/slide",
            json={"resourceId": resource["id"]},
        ).get_json()
        second_raster = client.post(
            "/api/v1/session/overlays",
            json={"resourceId": raster_resource["id"]},
        ).get_json()["layer"]

        assert first["tileUrl"] != second["tileUrl"]
        first_query = parse_qs(urlparse(first["tileUrl"]).query)
        second_query = parse_qs(urlparse(second["tileUrl"]).query)
        assert first_query == {
            "slideGeneration": ["1"],
            "resource": [resource["id"]],
            "revision": [resource["revision"]],
        }
        assert second_query == {
            **first_query,
            "slideGeneration": ["2"],
        }
        assert first_raster["tileUrl"] != second_raster["tileUrl"]
        assert parse_qs(urlparse(first_raster["tileUrl"]).query) == {
            "slideGeneration": ["1"],
            "resource": [raster_resource["id"]],
            "revision": [raster_resource["revision"]],
        }
        assert parse_qs(urlparse(second_raster["tileUrl"]).query) == {
            "slideGeneration": ["2"],
            "resource": [raster_resource["id"]],
            "revision": [raster_resource["revision"]],
        }
        tile_coordinates = {"TileGroup": "TileGroup0", "z": 0, "x": 0, "y": 0}
        slide_tile = client.get(second["tileUrl"].format(**tile_coordinates))
        raster_tile = client.get(
            second_raster["tileUrl"].format(**tile_coordinates),
        )
        assert (slide_tile.status_code, slide_tile.content_type) == (200, "image/webp")
        assert (raster_tile.status_code, raster_tile.content_type) == (
            200,
            "image/webp",
        )


def test_newer_slide_selection_cannot_be_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow, superseded open cannot commit after the newer selection."""
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()
    Image.fromarray(np.zeros((64, 96, 3), dtype=np.uint8)).save(
        slides / "first.tif",
    )
    Image.fromarray(np.full((64, 96, 3), 255, dtype=np.uint8)).save(
        slides / "second.tif",
    )
    app = TileServer(
        "Concurrent slide selection test",
        {},
        slide_roots=[slides],
        overlay_roots=[overlays],
        viewer_cache_dir=tmp_path / "cache",
    )
    original_open = WSIReader.open
    first_started = threading.Event()
    release_first = threading.Event()

    def delayed_open(input_img, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if Path(input_img).name == "first.tif":
            first_started.set()
            assert release_first.wait(timeout=10)
        return original_open(input_img, *args, **kwargs)

    monkeypatch.setattr(WSIReader, "open", staticmethod(delayed_open))
    services = app.viewer_services
    session, _ = services.ensure_session(None)
    slide_ids = {
        item["name"]: item["id"] for item in services.registry.catalog()["slides"]
    }

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            older = executor.submit(
                services.select_slide,
                session,
                slide_ids["first.tif"],
            )
            assert first_started.wait(timeout=10)
            newer_document = services.select_slide(
                session,
                slide_ids["second.tif"],
            )
            release_first.set()
            with pytest.raises(RuntimeError, match="superseded"):
                older.result(timeout=10)

        assert session.slide_resource_id == slide_ids["second.tif"]
        assert session.slide_generation == 2
        assert newer_document["id"] == slide_ids["second.tif"]
        assert "slideGeneration=2" in newer_document["tileUrl"]
        assert services.session_document(session)["slide"] == newer_document
    finally:
        release_first.set()
        services.close()


def test_manifest_tiles_details_and_etag(  # noqa: PLR0915
    viewer_app: TileServer,
) -> None:
    """Revisioned tiles are immutable, compact and exactly inspectable."""
    with viewer_app.test_client() as client:
        _, _, overlay = _load_resources(client)
        initial = overlay.get_json()["store"]
        source = viewer_app.viewer_services._sources[initial["id"]]
        assert source._lod_future is not None
        source._lod_future.result(timeout=30)

        current_url = f"/api/v1/stores/{initial['id']}"
        current = client.get(current_url)
        manifest = current.get_json()
        assert current.headers["Cache-Control"] == "no-cache"
        assert manifest["lodStatus"] == "ready"
        assert manifest["featureCount"] == 300
        assert set(manifest["properties"]) == {"prob", "type"}
        assert "paths" not in json.dumps(manifest).lower()

        immutable_url = manifest["urls"]["manifest"]
        immutable = client.get(immutable_url)
        assert immutable.status_code == 200
        assert "immutable" in immutable.headers["Cache-Control"]
        immutable_not_modified = client.get(
            immutable_url,
            headers={"If-None-Match": immutable.headers["ETag"]},
        )
        assert immutable_not_modified.status_code == 304

        tile_url = manifest["urls"]["tiles"].format(z=1, x=0, y=0)
        tile = client.get(tile_url)
        assert tile.status_code == 200
        assert tile.content_type == "application/vnd.mapbox-vector-tile"
        assert tile.headers["Content-Encoding"] == "gzip"
        assert tile.headers["X-TIAToolbox-Representation"] == "polygon"
        assert gzip.decompress(tile.data)
        assert int(tile.headers["X-TIAToolbox-Features"]) > 0
        server_timing = tile.headers["Server-Timing"]
        assert "tile;dur=" in server_timing
        assert "query;dur=" in server_timing
        assert "encode;dur=" in server_timing
        assert "compress;dur=" in server_timing

        cached = client.get(tile_url)
        assert cached.headers["X-TIAToolbox-Cache"] == "memory"
        not_modified = client.get(
            tile_url, headers={"If-None-Match": tile.headers["ETag"]}
        )
        assert not_modified.status_code == 304

        feature_url = manifest["urls"]["feature"].replace("{featureId}", "1")
        feature = client.get(feature_url)
        assert feature.status_code == 200
        assert feature.get_json()["type"] == "Feature"
        assert feature.get_json()["id"] == "1"
        assert "canonicalId" in feature.get_json()["properties"]["_tiatoolbox"]
        feature_not_modified = client.get(
            feature_url,
            headers={"If-None-Match": feature.headers["ETag"]},
        )
        assert feature_not_modified.status_code == 304

        pick = client.get(f"{manifest['urls']['pick']}?x=20&y=20&tolerance=0")
        assert pick.status_code == 200
        assert pick.get_json() == {"featureId": 1}
        assert (
            client.get(f"{manifest['urls']['pick']}?x=0&y=0&tolerance=1").status_code
            == 204
        )
        invalid_pick = client.get(f"{manifest['urls']['pick']}?x=nan&y=20&tolerance=1")
        assert invalid_pick.status_code == 400


def test_label_tile_and_invalid_requests(viewer_app: TileServer) -> None:
    """Data tiles are immutable and invalid IDs/coordinates are rejected."""
    with viewer_app.test_client() as client:
        _, _, overlay = _load_resources(client)
        initial = overlay.get_json()["store"]
        source = viewer_app.viewer_services._sources[initial["id"]]
        assert source._lod_future is not None
        source._lod_future.result(timeout=30)
        manifest = client.get(f"/api/v1/stores/{initial['id']}").get_json()

        label_url = manifest["urls"]["labels"].format(z=1, x=0, y=0)
        label = client.get(label_url)
        assert label.status_code == 200
        assert label.content_type == "application/vnd.tiatoolbox.annotation-tile"
        assert gzip.decompress(label.data).startswith(b"TIAD")

        bad_tile = client.get(
            manifest["urls"]["tiles"].format(z=99, x=0, y=0),
        )
        assert bad_tile.status_code == 404
        assert client.get("/api/v1/stores/not-a-store").status_code == 404


def test_vector_tile_property_filter_contract(viewer_app: TileServer) -> None:
    """Filters are selective, canonicalized for caching, and safely bounded."""
    with viewer_app.test_client() as client:
        _, _, overlay = _load_resources(client)
        initial = overlay.get_json()["store"]
        source = viewer_app.viewer_services._sources[initial["id"]]
        assert source._lod_future is not None
        source._lod_future.result(timeout=30)
        manifest = client.get(f"/api/v1/stores/{initial['id']}").get_json()
        tile_url = manifest["urls"]["representations"]["polygon"].format(
            z=1,
            x=0,
            y=0,
        )

        first_filter = {
            "op": "and",
            "args": [
                {"op": "eq", "property": "type", "value": 0},
                {"op": "lt", "property": "prob", "value": 0.4},
            ],
        }
        first = client.get(_with_filter(tile_url, first_filter))
        assert first.status_code == 200
        assert first.headers["X-TIAToolbox-Cache"] == "miss"
        assert 0 < int(first.headers["X-TIAToolbox-Features"]) < 300

        equivalent_filter = {
            "args": list(reversed(first_filter["args"])),
            "op": "and",
        }
        equivalent = client.get(_with_filter(tile_url, equivalent_filter))
        assert equivalent.status_code == 200
        assert equivalent.headers["X-TIAToolbox-Cache"] == "memory"
        assert equivalent.headers["ETag"] == first.headers["ETag"]

        distinct = client.get(
            _with_filter(
                tile_url,
                {"op": "eq", "property": "type", "value": 1},
            ),
        )
        assert distinct.status_code == 200
        assert distinct.headers["X-TIAToolbox-Cache"] == "miss"
        assert distinct.headers["ETag"] != first.headers["ETag"]

        injection = client.get(
            _with_filter(
                tile_url,
                {
                    "op": "eq",
                    "property": "type",
                    "value": "0) OR 1=1 --",
                },
            ),
        )
        assert injection.status_code == 200
        assert injection.headers["X-TIAToolbox-Features"] == "0"

        invalid_json = client.get(f"{tile_url}&filter=%7B")
        assert invalid_json.status_code == 400
        assert "valid JSON" in invalid_json.get_json()["error"]

        oversized = {
            "op": "eq",
            "property": "type",
            "value": "x" * 5000,
        }
        oversized_response = client.get(_with_filter(tile_url, oversized))
        assert oversized_response.status_code == 400
        assert "maximum encoded size" in oversized_response.get_json()["error"]

        duplicated = client.get(f"{tile_url}&filter=%7B%7D&filter=%7B%7D")
        assert duplicated.status_code == 400
        assert "only once" in duplicated.get_json()["error"]

        unknown_field = client.get(f"{tile_url}&fields=not-in-the-manifest")
        assert unknown_field.status_code == 400
        assert "Unknown display properties" in unknown_field.get_json()["error"]


def test_modern_viewer_assets_are_packaged(viewer_app: TileServer) -> None:
    """The modern application is served alongside the unchanged legacy root."""
    with viewer_app.test_client() as client:
        redirect = client.get("/viewer")
        assert redirect.status_code == 308
        index = client.get("/viewer/")
        assert index.status_code == 200
        assert index.content_type == "text/html; charset=utf-8"
        assert index.headers["Cache-Control"] == "no-cache"
        assert b"TIAToolbox" in index.data
