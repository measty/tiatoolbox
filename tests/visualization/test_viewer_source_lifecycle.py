"""Concurrency and cache-lifecycle tests for efficient viewer sources."""

from __future__ import annotations

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from tests.visualization._mvt_decode import decode_mvt
from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.visualization.annotation_tiles.cache import ByteLRUCache
from tiatoolbox.visualization.annotation_tiles.grid import TileMatrix
from tiatoolbox.visualization.annotation_tiles.lod import (
    LODBuildCancelled,
    LODIndex,
)
from tiatoolbox.visualization.annotation_tiles.source import AnnotationTileSource
from tiatoolbox.visualization.tileserver import TileServer

if TYPE_CHECKING:
    from pathlib import Path


def _create_viewer(
    tmp_path: Path,
    *,
    two_slides: bool = False,
    overlapping: bool = False,
) -> TileServer:
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()
    Image.fromarray(np.zeros((128, 192, 3), dtype=np.uint8)).save(
        slides / "first.tif",
    )
    if two_slides:
        Image.fromarray(np.full((128, 192, 3), 255, dtype=np.uint8)).save(
            slides / "second.tif",
        )
    store = SQLiteStore(overlays / "cells.db")
    if overlapping:
        store.append_many(
            (
                Annotation(
                    Polygon.from_bounds(20, 20, 100, 100),
                    {"type": "underlying"},
                ),
                Annotation(
                    Polygon.from_bounds(40, 40, 80, 80),
                    {"type": "top"},
                ),
            ),
        )
    else:
        store.append(
            Annotation(Polygon.from_bounds(20, 20, 40, 40), {"type": "cell"}),
        )
    store.close()
    Image.fromarray(np.zeros((128, 192, 3), dtype=np.uint8)).save(
        overlays / "mask.png",
    )
    app = TileServer(
        "Lifecycle test",
        {},
        slide_roots=[slides],
        overlay_roots=[overlays],
        viewer_cache_dir=tmp_path / "cache",
    )
    app.config.update(TESTING=True)
    return app


def test_overlay_mutations_validate_slide_generation(tmp_path: Path) -> None:
    """Stale clients cannot attach or remove layers from a newer slide."""
    app = _create_viewer(tmp_path)
    try:
        with app.test_client() as client:
            catalog = client.get("/api/v1/bootstrap").get_json()["catalog"]
            slide_id = catalog["slides"][0]["id"]
            overlay_id = next(
                item["id"]
                for item in catalog["overlays"]
                if item["kind"] == "annotation"
            )
            slide = client.put(
                "/api/v1/session/slide",
                json={"resourceId": slide_id},
            ).get_json()
            assert slide["slideGeneration"] == 1

            stale_add = client.post(
                "/api/v1/session/overlays",
                json={"resourceId": overlay_id, "slideGeneration": 0},
            )
            assert stale_add.status_code == 409
            added = client.post(
                "/api/v1/session/overlays",
                json={"resourceId": overlay_id, "slideGeneration": 1},
            )
            assert added.status_code == 201
            store_id = added.get_json()["store"]["id"]

            stale_remove = client.delete(
                f"/api/v1/session/overlays/{store_id}?slideGeneration=0",
            )
            assert stale_remove.status_code == 409
            assert client.get(f"/api/v1/stores/{store_id}").status_code == 200
            assert (
                client.delete(
                    f"/api/v1/session/overlays/{store_id}?slideGeneration=1",
                ).status_code
                == 204
            )
            assert client.get(f"/api/v1/stores/{store_id}").status_code == 404
    finally:
        app.viewer_services.close()


def test_server_pick_returns_top_matching_filtered_feature(tmp_path: Path) -> None:
    """Filtering out a small top polygon exposes the matching polygon beneath."""
    app = _create_viewer(tmp_path, overlapping=True)
    try:
        with app.test_client() as client:
            catalog = client.get("/api/v1/bootstrap").get_json()["catalog"]
            slide_id = catalog["slides"][0]["id"]
            overlay_id = next(
                item["id"]
                for item in catalog["overlays"]
                if item["kind"] == "annotation"
            )
            client.put(
                "/api/v1/session/slide",
                json={"resourceId": slide_id},
            )
            manifest = client.post(
                "/api/v1/session/overlays",
                json={"resourceId": overlay_id},
            ).get_json()["store"]
            pick_url = manifest["urls"]["pick"]

            unfiltered = client.get(f"{pick_url}?x=50&y=50").get_json()
            assert unfiltered == {"featureId": 2}
            query = urlencode(
                {
                    "x": 50,
                    "y": 50,
                    "filter": json.dumps(
                        {
                            "op": "eq",
                            "property": "type",
                            "value": "underlying",
                        },
                    ),
                },
            )
            filtered = client.get(f"{pick_url}?{query}")
            assert filtered.status_code == 200
            assert filtered.get_json() == {"featureId": 1}
    finally:
        app.viewer_services.close()


@pytest.mark.parametrize("overlay_kind", ["annotation", "raster-overlay"])
def test_slow_overlay_open_cannot_commit_to_superseded_slide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlay_kind: str,
) -> None:
    """Generation is checked both before and after slow annotation/raster work."""
    app = _create_viewer(tmp_path, two_slides=True)
    services = app.viewer_services
    services.max_idle_annotation_sources = 0
    session, _ = services.ensure_session(None)
    catalog = services.registry.catalog()
    slide_ids = {item["name"]: item["id"] for item in catalog["slides"]}
    overlay_id = next(
        item["id"] for item in catalog["overlays"] if item["kind"] == overlay_kind
    )
    services.select_slide(session, slide_ids["first.tif"])
    work_started = threading.Event()
    release_work = threading.Event()

    if overlay_kind == "annotation":
        original = services._open_store

        def delayed_open(resource):  # noqa: ANN001, ANN202
            work_started.set()
            assert release_work.wait(timeout=10)
            return original(resource)

        monkeypatch.setattr(services, "_open_store", delayed_open)
    else:
        original = services._open_raster_layer

        def delayed_open(resource, slide_reader):  # noqa: ANN001, ANN202
            work_started.set()
            assert release_work.wait(timeout=10)
            return original(resource, slide_reader)

        monkeypatch.setattr(services, "_open_raster_layer", delayed_open)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            obsolete = executor.submit(
                lambda: services.add_overlay(
                    session,
                    overlay_id,
                    slide_generation=1,
                ),
            )
            assert work_started.wait(timeout=10)
            services.select_slide(session, slide_ids["second.tif"])
            release_work.set()
            with pytest.raises(RuntimeError, match="superseded"):
                obsolete.result(timeout=10)
        assert session.annotation_sources == []
        assert session.raster_layers == {}
        assert not services._sources
    finally:
        release_work.set()
        services.close()


def test_idle_eviction_reopens_warm_lod_sidecar(tmp_path: Path) -> None:
    """Eviction closes process objects but deliberately preserves ready disk LOD."""
    app = _create_viewer(tmp_path)
    services = app.viewer_services
    services.max_idle_annotation_sources = 0
    session, _ = services.ensure_session(None)
    catalog = services.registry.catalog()
    slide_id = catalog["slides"][0]["id"]
    overlay_id = next(
        item["id"] for item in catalog["overlays"] if item["kind"] == "annotation"
    )
    try:
        generation = services.select_slide(session, slide_id)["slideGeneration"]
        first = services.add_overlay(
            session,
            overlay_id,
            slide_generation=generation,
        )["store"]
        first_source = services._sources[first["id"]]
        assert first_source._lod_future is not None
        first_source._lod_future.result(timeout=30)
        sidecar = first_source.lod.path

        services.remove_overlay(
            session,
            first["id"],
            slide_generation=generation,
        )
        assert first["id"] not in services._sources
        assert sidecar.is_file()

        second = services.add_overlay(
            session,
            overlay_id,
            slide_generation=generation,
        )["store"]
        second_source = services._sources[second["id"]]
        assert second_source is not first_source
        assert second_source.lod.ready
    finally:
        services.close()


def test_session_lru_eviction_detaches_sources_and_tile_state(tmp_path: Path) -> None:
    """Abandoned cookie sessions cannot retain sources beyond the global bound."""
    app = _create_viewer(tmp_path)
    services = app.viewer_services
    services.max_sessions = 1
    services.max_idle_annotation_sources = 0
    catalog = services.registry.catalog()
    slide_id = catalog["slides"][0]["id"]
    overlay_id = next(
        item["id"] for item in catalog["overlays"] if item["kind"] == "annotation"
    )
    try:
        first_session, _ = services.ensure_session("first-session")
        generation = services.select_slide(first_session, slide_id)["slideGeneration"]
        store_id = services.add_overlay(
            first_session,
            overlay_id,
            slide_generation=generation,
        )["store"]["id"]
        source = services._sources[store_id]
        assert source._lod_future is not None
        source._lod_future.result(timeout=30)

        second_session, _ = services.ensure_session("second-session")
        assert second_session.id == "second-session"
        assert list(services._sessions) == ["second-session"]
        assert "first-session" not in services.tile_server.layers
        assert "first-session" not in services.tile_server.pyramids
        assert store_id not in services._sources
    finally:
        services.close()


def test_current_session_refreshes_while_other_idle_sessions_expire(
    tmp_path: Path,
) -> None:
    """Returning to an idle tab preserves it while unrelated abandoned tabs reap."""
    app = _create_viewer(tmp_path)
    services = app.viewer_services
    services.session_idle_seconds = 1
    try:
        current, _ = services.ensure_session("current-session")
        services.ensure_session("abandoned-session")
        services._session_last_access["current-session"] -= 2
        services._session_last_access["abandoned-session"] -= 2

        refreshed, is_new = services.ensure_session("current-session")
        assert refreshed is current
        assert not is_new
        assert list(services._sessions) == ["current-session"]
    finally:
        services.close()


def test_active_sources_enforce_process_wide_hard_bound(tmp_path: Path) -> None:
    """Many sessions/stores cannot grow open SQLite handles without a limit."""
    app = _create_viewer(tmp_path)
    services = app.viewer_services
    second_store = SQLiteStore(services.registry.overlay_roots[0] / "other.db")
    second_store.append(
        Annotation(Polygon.from_bounds(80, 80, 100, 100), {"type": "other"}),
    )
    second_store.close()
    services.registry.refresh()
    services.max_annotation_sources = 1
    services.max_idle_annotation_sources = 0
    catalog = services.registry.catalog()
    slide_id = catalog["slides"][0]["id"]
    annotation_ids = [
        item["id"] for item in catalog["overlays"] if item["kind"] == "annotation"
    ]
    session, _ = services.ensure_session(None)
    try:
        generation = services.select_slide(session, slide_id)["slideGeneration"]
        services.add_overlay(
            session,
            annotation_ids[0],
            slide_generation=generation,
        )
        with pytest.raises(RuntimeError, match="source limit"):
            services.add_overlay(
                session,
                annotation_ids[1],
                slide_generation=generation,
            )
        assert len(services._sources) == 1
    finally:
        services.close()


def test_concurrent_same_overlay_open_creates_one_shared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-source creation locking avoids duplicate stores and sidecar handles."""
    app = _create_viewer(tmp_path)
    services = app.viewer_services
    session, _ = services.ensure_session(None)
    catalog = services.registry.catalog()
    slide_id = catalog["slides"][0]["id"]
    overlay_id = next(
        item["id"] for item in catalog["overlays"] if item["kind"] == "annotation"
    )
    generation = services.select_slide(session, slide_id)["slideGeneration"]
    original_open = services._open_store
    open_started = threading.Event()
    release_open = threading.Event()
    count_lock = threading.Lock()
    open_count = 0

    def delayed_open(resource):  # noqa: ANN001, ANN202
        nonlocal open_count
        with count_lock:
            open_count += 1
        open_started.set()
        assert release_open.wait(timeout=10)
        return original_open(resource)

    monkeypatch.setattr(services, "_open_store", delayed_open)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    lambda: services.add_overlay(
                        session,
                        overlay_id,
                        slide_generation=generation,
                    ),
                )
                for _ in range(4)
            ]
            assert open_started.wait(timeout=10)
            release_open.set()
            store_ids = {future.result(timeout=10)["store"]["id"] for future in futures}
        assert len(store_ids) == 1
        assert open_count == 1
        assert len(services._sources) == 1
    finally:
        release_open.set()
        services.close()


def test_cancelled_attached_lod_restarts_after_reattach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detach cancellation cannot strand a source reattached during shutdown."""
    app = _create_viewer(tmp_path)
    services = app.viewer_services
    services.max_idle_annotation_sources = 0
    session, _ = services.ensure_session(None)
    catalog = services.registry.catalog()
    slide_id = catalog["slides"][0]["id"]
    overlay_id = next(
        item["id"] for item in catalog["overlays"] if item["kind"] == "annotation"
    )
    original_build = LODIndex.build
    first_started = threading.Event()
    cancellation_seen = threading.Event()
    release_cancelled = threading.Event()
    calls = 0

    def controlled_build(
        index: LODIndex,
        store: SQLiteStore,
        cancel_event: threading.Event | None = None,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert cancel_event is not None
            assert cancel_event.wait(timeout=10)
            cancellation_seen.set()
            assert release_cancelled.wait(timeout=10)
            message = "controlled cancellation"
            raise LODBuildCancelled(message)
        original_build(index, store, cancel_event=cancel_event)

    monkeypatch.setattr(LODIndex, "build", controlled_build)
    try:
        generation = services.select_slide(session, slide_id)["slideGeneration"]
        first = services.add_overlay(
            session,
            overlay_id,
            slide_generation=generation,
        )["store"]
        assert first_started.wait(timeout=10)
        services.remove_overlay(
            session,
            first["id"],
            slide_generation=generation,
        )
        assert cancellation_seen.wait(timeout=10)

        # Reattach while the original worker is still winding down. The service
        # callback must start a replacement after that worker reports cancellation.
        services.add_overlay(
            session,
            overlay_id,
            slide_generation=generation,
        )
        release_cancelled.set()
        deadline = time.monotonic() + 10
        while not services._sources[first["id"]].lod.ready:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert calls == 2
        assert first["id"] in session.annotation_sources
    finally:
        release_cancelled.set()
        services.close()


def test_queued_lod_is_cancelled_without_partial_manifest(tmp_path: Path) -> None:
    """Obsolete work still in the executor queue is removed before it can scan."""
    store = SQLiteStore(tmp_path / "queued.db")
    store.append(
        Annotation(Polygon.from_bounds(16, 16, 80, 80), {"type": "cell"}),
    )
    source = AnnotationTileSource(
        store,
        TileMatrix(256, 256),
        store_id="queued",
        revision="revision",
        name="queued",
        cache_dir=tmp_path / "cache",
        owns_store=True,
    )
    release_worker = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release_worker.wait)
        future = source.ensure_lod(executor)
        assert future is not None
        assert source.cancel_lod() is future
        assert future.cancelled()
        assert source.lod_status == "not-built"
        assert source.lod.manifest() is None
        release_worker.set()
        blocker.result(timeout=10)
    source.close()


def test_shared_memory_cache_is_namespaced_and_not_cleared_per_source(
    tmp_path: Path,
) -> None:
    """Same coordinates in two stores cannot alias in the process-wide ByteLRU."""
    shared = ByteLRUCache(1024 * 1024)
    matrix = TileMatrix(256, 256)
    sources: list[AnnotationTileSource] = []
    for store_id, category in (("source-a", "a"), ("source-b", "b")):
        store = SQLiteStore(tmp_path / f"{store_id}.db")
        store.append(
            Annotation(Polygon.from_bounds(16, 16, 80, 80), {"type": category}),
        )
        source = AnnotationTileSource(
            store,
            matrix,
            store_id=store_id,
            revision="revision",
            name=store_id,
            cache_dir=tmp_path / "cache",
            owns_store=True,
            memory_cache=shared,
        )
        sources.append(source)

    first = sources[0].vector_tile("polygon", 0, 0, 0, fields=("type",))
    first_category = (
        decode_mvt(gzip.decompress(first.data)).features[0].properties["type"]
    )
    size_after_first = shared.size
    sources[0].close()
    assert shared.size == size_after_first > 0

    second = sources[1].vector_tile("polygon", 0, 0, 0, fields=("type",))
    second_category = (
        decode_mvt(gzip.decompress(second.data)).features[0].properties["type"]
    )
    assert (first_category, second_category) == ("a", "b")
    sources[1].close()


def test_service_close_drains_active_source_leases(tmp_path: Path) -> None:
    """Shutdown cannot close a store underneath an in-flight API operation."""
    app = _create_viewer(tmp_path)
    services = app.viewer_services
    session, _ = services.ensure_session(None)
    catalog = services.registry.catalog()
    slide_id = catalog["slides"][0]["id"]
    overlay_id = next(
        item["id"] for item in catalog["overlays"] if item["kind"] == "annotation"
    )
    generation = services.select_slide(session, slide_id)["slideGeneration"]
    store_id = services.add_overlay(
        session,
        overlay_id,
        slide_generation=generation,
    )["store"]["id"]
    source = services._sources[store_id]
    assert source._lod_future is not None
    source._lod_future.result(timeout=30)

    lease_entered = threading.Event()
    release_lease = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()

    def hold_lease() -> None:
        with services.source_lease(session, store_id):
            lease_entered.set()
            assert release_lease.wait(timeout=10)

    def close_service() -> None:
        close_started.set()
        services.close()
        close_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        lease_future = executor.submit(hold_lease)
        assert lease_entered.wait(timeout=10)
        close_future = executor.submit(close_service)
        assert close_started.wait(timeout=10)
        assert not close_finished.wait(timeout=0.1)
        release_lease.set()
        lease_future.result(timeout=10)
        close_future.result(timeout=10)
    assert close_finished.is_set()
