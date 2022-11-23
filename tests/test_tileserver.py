"""Tests for tileserver."""
import pathlib
import urllib
from pathlib import Path, PureWindowsPath
from typing import List, Union

import numpy as np
import pytest
from flask import session
from matplotlib import cm
from shapely.geometry import LineString, Polygon
from shapely.geometry.point import Point

from tests.test_annotation_stores import cell_polygon
from tiatoolbox.annotation.storage import Annotation, AnnotationStore, SQLiteStore
from tiatoolbox.cli.common import cli_name
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import WSIReader


def make_safe_name(name):
    """Make a name safe for use in a URL."""
    return urllib.parse.quote(str(PureWindowsPath(name)), safe="")


def setup_app(client):
    resp = s.get(f"http://{host2}:5000/tileserver/setup")
    print(f"cookies are: {resp.cookies}")
    cookies = resp.cookies
    user = resp.cookies.get("user")


@pytest.fixture(scope="session")
def cell_grid() -> List[Polygon]:
    """Generate a grid of fake cell boundary polygon annotations."""
    np.random.seed(0)
    return [
        cell_polygon(((i + 0.5) * 100, (j + 0.5) * 100)) for i, j in np.ndindex(5, 5)
    ]


@pytest.fixture(scope="session")
def points_grid(spacing=60) -> List[Point]:
    """Generate a grid of fake point annotations."""
    np.random.seed(0)
    return [Point((600 + i * spacing, 600 + j * spacing)) for i, j in np.ndindex(7, 7)]


@pytest.fixture(scope="session")
def fill_store(cell_grid, points_grid):
    """Factory fixture to fill stores with test data."""

    def _fill_store(
        store_class: AnnotationStore,
        path: Union[str, pathlib.Path],
    ):
        """Fills store with random variety of annotations."""
        store = store_class(path)

        cells = [
            Annotation(cell, {"type": "cell", "prob": np.random.rand(1)[0]})
            for cell in cell_grid
        ]
        points = [
            Annotation(point, {"type": "pt", "prob": np.random.rand(1)[0]})
            for point in points_grid
        ]
        lines = [
            Annotation(
                LineString(((x, x + 500) for x in range(100, 400, 10))),
                {"type": "line", "prob": 0.75},
            )
        ]

        annotations = cells + points + lines
        keys = store.append_many(annotations)
        return keys, store

    return _fill_store


@pytest.fixture()
def app(remote_sample, tmp_path, fill_store) -> TileServer:
    """Create a testing TileServer WSGI app."""

    # Make a low-res .jpg of the right shape to be used as
    # a low-res overlay.
    sample_svs = Path(remote_sample("svs-1-small"))
    wsi = WSIReader.open(sample_svs)
    thumb = wsi.slide_thumbnail()
    thumb_path = tmp_path / "thumb.jpg"
    imwrite(thumb_path, thumb)

    sample_store = Path(remote_sample("annotation_store_svs_1"))
    store = SQLiteStore(sample_store)
    geo_path = tmp_path / "test.geojson"
    store.to_geojson(geo_path)
    store.commit()
    store.close()

    # make tileserver with layers representing all the types
    # of things it should be able to handle
    app = TileServer(
        "Testing TileServer",
        {
            "slide": str(Path(sample_svs)),
            "tile": str(thumb_path),
            "im_array": np.zeros(wsi.slide_dimensions(1.25, "power"), dtype=np.uint8).T,
            "store_geojson": tmp_path / "test.geojson",
            "overlay": str(sample_store),
        },
    )
    app.config.from_mapping({"TESTING": True})
    # with app.test_client() as client:
    #     response = client.get("/tileserver/setup")
    #     #get the "user" cookie
    #     cookie = next(
    #         (cookie for cookie in client.cookie_jar if cookie.name == "user"),
    #         None
    #     )

    return app  # , cookie.value


def layer_get_tile(app, layer) -> None:
    """Get a single tile and check the status code and content type."""
    with app.test_client() as client:
        # with client.session_transaction() as session:
        # set a user id without going through the login route
        # session.cookies["user"] = "u"
        response = client.get(
            f"/tileserver/layer/{layer}/default/zoomify/TileGroup0/0-0-0@1x.jpg"
        )
        assert response.status_code == 200
        assert response.content_type == "image/webp"


def test_get_tile(app):
    """do test on each layer"""
    layer_get_tile(app, "slide")
    layer_get_tile(app, "tile")
    layer_get_tile(app, "im_array")
    layer_get_tile(app, "store_geojson")
    layer_get_tile(app, "overlay")


def layer_get_tile_404(app, layer) -> None:
    """Request a tile with an index."""
    with app.test_client() as client:
        response = client.get(
            f"/tileserver/layer/{layer}/default/zoomify/TileGroup0/10-0-0@1x.jpg"
        )
        assert response.status_code == 404
        assert response.get_data(as_text=True) == "Tile not found"


def test_get_tile_404(app):
    """do test on each layer"""
    layer_get_tile_404(app, "slide")
    layer_get_tile_404(app, "tile")
    layer_get_tile_404(app, "im_array")
    layer_get_tile_404(app, "store_geojson")
    layer_get_tile_404(app, "overlay")


def test_get_tile_layer_key_error(app) -> None:
    """Request a tile with an invalid layer key."""
    with app.test_client() as client:
        response = client.get(
            "/tileserver/layer/foo/default/zoomify/TileGroup0/0-0-0@1x.jpg"
        )
        assert response.status_code == 404
        assert response.get_data(as_text=True) == "Layer not found"


def test_get_index(app) -> None:
    """Get the index page and check that it is HTML."""
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"


def test_create_with_dict(sample_svs):
    """test initialising with layers dict"""
    wsi = WSIReader.open(Path(sample_svs))

    app = TileServer(
        "Testing TileServer",
        {"Test": wsi},
    )
    app.config.from_mapping({"TESTING": True})
    with app.test_client() as client:
        response = client.get(
            "/tileserver/layer/Test/default/zoomify/TileGroup0/0-0-0@1x.jpg"
        )
        assert response.status_code == 200
        assert response.content_type == "image/webp"


def test_cli_name_multiple_flag():
    """Test cli_name multiple flag."""

    @cli_name()
    def dummy_fn():
        """it's empty because its a dummy function"""

    assert "Multiple" not in dummy_fn.__click_params__[0].help

    @cli_name(multiple=True)
    def dummy_fn():
        """it's empty because its a dummy function"""

    assert "Multiple" in dummy_fn.__click_params__[0].help


def test_setup(app):
    """Test setup endpoint."""
    with app.test_client() as client:
        response = client.get("/tileserver/setup")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"


def test_color_prop(app):
    """Test endpoint to change property to color by."""
    with app.test_client() as client:
        response = client.get("/tileserver/change_color_prop/test_prop")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the color prop has been correctly set
        assert app.tia_pyramids["default"]["overlay"].renderer.score_prop == "test_prop"


def test_change_slide(app, remote_sample):
    """Test changing slide."""
    slide_path = remote_sample("svs-1-small")
    with app.test_client() as client:
        response = client.get(
            f"/tileserver/change_slide/slide/{make_safe_name(slide_path)}"
        )
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the slide has been correctly changed
        assert app.tia_pyramids["default"]["slide"].wsi.info.file_path == slide_path


def test_change_cmap(app):
    """Test changing colormap."""
    with app.test_client() as client:
        response = client.get("/tileserver/change_cmap/Reds")
        assert response.status_code == 200
        assert response.content_type == "text/html; charset=utf-8"
        # check that the colormap has been correctly changed
        assert app.tia_pyramids["default"]["overlay"].renderer.mapper(
            0.5
        ) == cm.get_cmap("Reds")(0.5)
