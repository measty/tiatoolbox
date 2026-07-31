"""Resource identity and confinement tests for the efficient viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.visualization.resources import ResourceRegistry
from tiatoolbox.visualization.tileserver import TileServer

if TYPE_CHECKING:
    from pathlib import Path


def _resource_id(registry: ResourceRegistry, group: str, name: str) -> str:
    """Return one named public resource ID from a catalog group."""
    return next(
        resource["id"]
        for resource in registry.catalog()[group]
        if resource["name"] == name
    )


def test_resource_ids_survive_registry_restart_and_isolate_roots(
    tmp_path: Path,
) -> None:
    """Stable IDs remain scoped by the canonical resource path."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "same-name.tif").write_bytes(b"same content")
    (second_root / "same-name.tif").write_bytes(b"same content")

    first = ResourceRegistry(slide_roots=[first_root])
    restarted = ResourceRegistry(slide_roots=[first_root])
    second = ResourceRegistry(slide_roots=[second_root])
    first_id = _resource_id(first, "slides", "same-name.tif")
    restarted_id = _resource_id(restarted, "slides", "same-name.tif")
    second_id = _resource_id(second, "slides", "same-name.tif")

    assert restarted_id == first_id
    assert second_id != first_id
    with pytest.raises(KeyError, match="Unknown viewer resource"):
        first.get(second_id)
    with pytest.raises(KeyError, match="Unknown viewer resource"):
        second.get(first_id)


def test_resource_ids_isolate_slide_and_overlay_roles(tmp_path: Path) -> None:
    """One TIFF exposed under both roots retains two unambiguous identities."""
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    (shared_root / "shared.tif").write_bytes(b"image")

    registry = ResourceRegistry(
        slide_roots=[shared_root],
        overlay_roots=[shared_root],
    )
    slide_id = _resource_id(registry, "slides", "shared.tif")
    overlay_id = _resource_id(registry, "overlays", "shared.tif")

    assert slide_id != overlay_id
    assert registry.get(slide_id, "slide").kind == "slide"
    assert registry.get(overlay_id, "raster-overlay").kind == "raster-overlay"


def test_resource_ids_survive_server_restart(tmp_path: Path) -> None:
    """Public catalog IDs do not depend on process-local server state."""
    slides = tmp_path / "slides"
    overlays = tmp_path / "overlays"
    slides.mkdir()
    overlays.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
        slides / "sample.tif",
    )
    store = SQLiteStore(overlays / "sample.db")
    store.close()

    catalogs = []
    stores = []
    for restart in range(2):
        app = TileServer(
            "Stable resource IDs",
            {},
            slide_roots=[slides],
            overlay_roots=[overlays],
            viewer_cache_dir=tmp_path / f"cache-{restart}",
        )
        app.config.update(TESTING=True)
        try:
            with app.test_client() as client:
                response = client.get("/api/v1/bootstrap")
                assert response.status_code == 200
                catalog = response.get_json()["catalog"]
                catalogs.append(catalog)
                slide_id = _resource_id(
                    app.viewer_services.registry, "slides", "sample.tif"
                )
                overlay_id = _resource_id(
                    app.viewer_services.registry,
                    "overlays",
                    "sample.db",
                )
                assert (
                    client.put(
                        "/api/v1/session/slide",
                        json={"resourceId": slide_id},
                    ).status_code
                    == 200
                )
                added = client.post(
                    "/api/v1/session/overlays",
                    json={"resourceId": overlay_id},
                )
                assert added.status_code == 201
                store = added.get_json()["store"]
                stores.append((store["id"], store["revision"]))
        finally:
            app.viewer_services.close()
            app.viewer_services.close()

    first = {
        (resource["kind"], resource["name"]): resource["id"]
        for group in catalogs[0].values()
        for resource in group
    }
    restarted = {
        (resource["kind"], resource["name"]): resource["id"]
        for group in catalogs[1].values()
        for resource in group
    }
    assert restarted == first
    assert stores[1] == stores[0]
