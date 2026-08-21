"""Phase-4 HTTP lifecycle and immutable-source tests."""

from __future__ import annotations

import gzip
import hashlib
import os
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Point, Polygon

from tests.visualization._mvt_decode import decode_mvt
from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.visualization.annotation_tiles.lod import LODIndex
from tiatoolbox.visualization.annotation_tiles.source import (
    TileBudgetExceededError,
)
from tiatoolbox.visualization.resources import file_revision
from tiatoolbox.visualization.tileserver import TileServer

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from flask.testing import FlaskClient


@pytest.fixture
def phase4_app(tmp_path: Path) -> Iterator[tuple[TileServer, Path]]:
    """Return an app and its authoritative source-store path."""
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()
    Image.fromarray(np.zeros((384, 512, 3), dtype=np.uint8)).save(
        slides / "sample.tif",
    )
    store_path = overlays / "sample.db"
    store = SQLiteStore(store_path)
    store.append_many(
        [
            Annotation(
                Polygon.from_bounds(x, y, x + 12, y + 12),
                {"type": (x // 32) % 3, "prob": x / 512},
            )
            for y in range(16, 368, 24)
            for x in range(16, 496, 24)
        ],
    )
    store.close()
    app = TileServer(
        "Phase-4 viewer test",
        {},
        slide_roots=[slides],
        overlay_roots=[overlays],
        viewer_cache_dir=tmp_path / "cache",
    )
    app.config.update(TESTING=True)
    yield app, store_path
    app.viewer_services.close()


def _load_overlay(client: FlaskClient) -> dict[str, object]:
    bootstrap = client.get("/api/v1/bootstrap").get_json()
    slide_id = bootstrap["catalog"]["slides"][0]["id"]
    overlay_id = bootstrap["catalog"]["overlays"][0]["id"]
    assert (
        client.put(
            "/api/v1/session/slide",
            json={"resourceId": slide_id},
        ).status_code
        == 200
    )
    response = client.post(
        "/api/v1/session/overlays",
        json={"resourceId": overlay_id},
    )
    assert response.status_code == 201
    return response.get_json()["store"]


def test_transient_lod_tile_is_no_store_then_ready_url_changes(
    phase4_app: tuple[TileServer, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending overview cannot be cached under the eventual ready URL."""
    app, _ = phase4_app
    build_started = threading.Event()
    release_build = threading.Event()
    original_build = LODIndex.build

    def blocked_build(index: LODIndex, store: SQLiteStore) -> None:
        build_started.set()
        if not release_build.wait(timeout=20):
            pytest.fail("Timed out waiting to release the test LOD build.")
        original_build(index, store)

    monkeypatch.setattr(LODIndex, "build", blocked_build)
    try:
        with app.test_client() as client:
            initial = _load_overlay(client)
            assert build_started.wait(timeout=10)
            store_id = str(initial["id"])
            building = client.get(f"/api/v1/stores/{store_id}").get_json()
            assert building["lodStatus"] == "building"
            building_url = building["urls"]["tiles"].format(z=0, x=0, y=0)
            assert "lod=building" in building_url
            source = app.viewer_services._sources[store_id]
            assert f"tileVersion={source.tile_revision}" in building_url

            transient = client.get(building_url)
            assert transient.status_code == 200
            assert transient.headers["Cache-Control"] == "no-store"
            assert transient.headers["Retry-After"] == "1"
            assert transient.headers["X-TIAToolbox-Representation"] == "building"
            assert decode_mvt(gzip.decompress(transient.data)).features == ()

            release_build.set()
            source = app.viewer_services._sources[store_id]
            assert source._lod_future is not None
            source._lod_future.result(timeout=30)
            ready = client.get(f"/api/v1/stores/{store_id}").get_json()
            ready_url = ready["urls"]["tiles"].format(z=0, x=0, y=0)
            assert ready["lodStatus"] == "ready"
            assert "lod=ready" in ready_url
            assert ready_url != building_url

            aggregate = client.get(ready_url)
            assert "immutable" in aggregate.headers["Cache-Control"]
            assert aggregate.headers["X-TIAToolbox-Representation"] == "aggregate"
            assert decode_mvt(gzip.decompress(aggregate.data)).features
    finally:
        release_build.set()


def test_source_identity_versions_annotation_pipeline_semantics(
    phase4_app: tuple[TileServer, Path],
) -> None:
    """Changing representation semantics creates a new public source identity."""
    app, _ = phase4_app
    with app.test_client() as client:
        initial = _load_overlay(client)
        source_id = str(initial["id"])
        source = app.viewer_services._sources[source_id]
        overlay_summary = app.viewer_services.registry.catalog()["overlays"][0]
        resource = app.viewer_services.registry.get(
            str(overlay_summary["id"]),
            "annotation",
        )
        identity = (
            f"{resource.id}:{resource.revision}:"
            f"{source.matrix.width}:{source.matrix.height}:{source.matrix.tile_size}:"
        )
        current_id = hashlib.blake2b(
            f"{identity}pipeline-8".encode(),
            digest_size=12,
        ).hexdigest()
        former_id = hashlib.blake2b(
            f"{identity}pipeline-7".encode(),
            digest_size=12,
        ).hexdigest()

        assert source_id == current_id
        assert source_id != former_id
        assert f"/stores/{source_id}/revisions/" in str(
            initial["urls"]["manifest"],
        )


def test_failed_lod_url_returns_explicit_bounded_failure(
    phase4_app: tuple[TileServer, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed background index is explicit and never starts a z0 source scan."""
    app, _ = phase4_app

    def fail_build(_index: LODIndex, _store: SQLiteStore) -> None:
        msg = "deliberate LOD failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(LODIndex, "build", fail_build)
    with app.test_client() as client:
        initial = _load_overlay(client)
        store_id = str(initial["id"])
        source = app.viewer_services._sources[store_id]
        assert source._lod_future is not None
        with pytest.raises(RuntimeError, match="deliberate LOD failure"):
            source._lod_future.result(timeout=30)

        manifest = client.get(f"/api/v1/stores/{store_id}").get_json()
        assert manifest["lodStatus"] == "failed"
        tile_url = manifest["urls"]["tiles"].format(z=0, x=0, y=0)
        assert "lod=failed" in tile_url

        tile = client.get(tile_url)
        assert tile.status_code == 409
        assert tile.get_json() == {
            "error": "Overview LOD preprocessing failed; reload the overlay to retry.",
        }
        assert tile.headers["Cache-Control"] == "no-cache"


def test_uniform_tile_budget_failure_is_explicit(
    phase4_app: tuple[TileServer, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absolute limit returns an error instead of a cheaper tile family."""
    app, _ = phase4_app
    with app.test_client() as client:
        manifest = _load_overlay(client)
        source = app.viewer_services._sources[str(manifest["id"])]

        def exceed_budget(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            msg = (
                "Uniform polygon tile 1/0/0 exceeds the absolute safety limit; "
                "its representation was not changed."
            )
            raise TileBudgetExceededError(msg)

        monkeypatch.setattr(source, "vector_tile", exceed_budget)
        tile_url = str(manifest["urls"]["tiles"]).format(z=1, x=0, y=0)

        tile = client.get(tile_url)

        assert tile.status_code == 422
        assert tile.get_json()["error"].startswith(
            "Uniform polygon tile 1/0/0 exceeds the absolute safety limit",
        )
        assert tile.headers["Cache-Control"] == "no-cache"


def test_api_viewing_does_not_mutate_source_and_rejects_stale_revision(
    phase4_app: tuple[TileServer, Path],
) -> None:
    """LOD, tiles, and feature details leave the source DB byte-identical."""
    app, store_path = phase4_app
    before_bytes = store_path.read_bytes()
    before_mtime = store_path.stat().st_mtime_ns
    with app.test_client() as client:
        initial = _load_overlay(client)
        store_id = str(initial["id"])
        source = app.viewer_services._sources[store_id]
        assert source.store.con.execute("PRAGMA query_only").fetchone() == (1,)
        assert source._lod_future is not None
        source._lod_future.result(timeout=30)

        manifest = client.get(f"/api/v1/stores/{store_id}").get_json()
        tile_url = manifest["urls"]["tiles"].format(z=1, x=0, y=0)
        tile = client.get(tile_url)
        assert tile.status_code == 200
        feature_id = decode_mvt(gzip.decompress(tile.data)).features[0].feature_id
        assert feature_id is not None
        feature_url = manifest["urls"]["feature"].replace(
            "{featureId}",
            str(feature_id),
        )
        assert client.get(feature_url).status_code == 200

        stale_url = manifest["urls"]["manifest"].replace(
            str(manifest["revision"]),
            "stale-revision",
        )
        assert client.get(stale_url).status_code == 404

    assert store_path.stat().st_mtime_ns == before_mtime
    assert store_path.read_bytes() == before_bytes


def test_file_revision_changes_after_authoritative_store_edit(tmp_path: Path) -> None:
    """An authoritative store update produces a different immutable URL token."""
    path = tmp_path / "edited.db"
    store = SQLiteStore(path)
    store.append(Annotation(Polygon.from_bounds(0, 0, 8, 8), {"type": 0}))
    store.close()
    first = file_revision(path)

    store = SQLiteStore(path)
    store.append(Annotation(Polygon.from_bounds(16, 16, 24, 24), {"type": 1}))
    store.close()

    assert file_revision(path) != first


def test_sqlite_revision_is_content_structural_not_mtime(tmp_path: Path) -> None:
    """Copied SQLite bytes share a revision and committed edits cannot hide by mtime."""
    source_path = tmp_path / "source.db"
    copy_path = tmp_path / "copy.db"
    store = SQLiteStore(source_path)
    store.append(Annotation(Point(8, 8), {"type": 1}))
    store.close()
    copy_path.write_bytes(source_path.read_bytes())
    source_stat = source_path.stat()
    copy_stat = copy_path.stat()
    os.utime(
        copy_path,
        ns=(copy_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
    )

    assert file_revision(copy_path) == file_revision(source_path)
    first = file_revision(source_path)
    original_times = source_path.stat()
    store = SQLiteStore(source_path)
    store.append(Annotation(Point(16, 16), {"type": 2}))
    store.close()
    os.utime(
        source_path,
        ns=(original_times.st_atime_ns, original_times.st_mtime_ns),
    )

    assert file_revision(source_path) != first
