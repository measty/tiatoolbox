"""tests for annotation rendering using
AnnotationRenderer and AnnotationTileGenerator
"""
import pytest
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from pathlib import Path
from typing import List, Union
from shapely.geometry import Polygon, LineString
from shapely.geometry.point import Point

from tiatoolbox.tools.pyramid import AnnotationTileGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore import wsireader
from tests.test_annotation_stores import cell_polygon
from tiatoolbox.utils.env_detection import running_on_travis
from tiatoolbox.annotation.storage import (
    Annotation,
    AnnotationStore,
    SQLiteStore,
)


# Constants

GRID_SIZE = (5, 5)


@pytest.fixture(scope="session")
def cell_grid() -> List[Polygon]:
    """Generate a grid of fake cell boundary polygon annotations."""
    np.random.seed(0)
    return [
        cell_polygon(((i + 1) * 100, (j + 1) * 100)) for i, j in np.ndindex(*GRID_SIZE)
    ]


@pytest.fixture(scope="session")
def points_grid() -> List[Polygon]:
    """Generate a grid of fake point annotations."""
    np.random.seed(0)
    return [Point((600 + i * 60, 600 + j * 60)) for i, j in np.ndindex(*GRID_SIZE)]


@pytest.fixture()
def fill_store(cell_grid, points_grid):
    """Factory fixture to fill stores with test data."""

    def _fill_store(
        store_class: AnnotationStore,
        path: Union[str, Path],
    ):
        store = store_class(path)
        annotations = (
            [
                Annotation(cell, {"type": "cell", "prob": np.random.rand(1)[0]})
                for cell in cell_grid
            ]
            + [
                Annotation(point, {"type": "pt", "prob": np.random.rand(1)[0]})
                for point in points_grid
            ]
            + [
                Annotation(
                    LineString(((x, x + 500) for x in range(100, 400, 10))),
                    {"type": "line", "prob": np.random.rand(1)[0]},
                )
            ]
        )
        keys = store.append_many(annotations)
        return keys, store

    return _fill_store


def test_tile_generator_len(fill_store):
    """Test __len__ for AnnotationTileGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, ":memory:")
    dz = AnnotationTileGenerator(wsi.info, store, tile_size=256)
    assert len(dz) == (4 * 4) + (2 * 2) + 1


def test_tile_generator_iter(fill_store):
    """Test __iter__ for AnnotationTileGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, ":memory:")
    dz = AnnotationTileGenerator(wsi.info, store, tile_size=256)
    for tile in dz:
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)


@pytest.mark.skipif(running_on_travis(), reason="no display on travis.")
def test_show_generator_iter(fill_store):
    """Show tiles with example annotations (if not travis)"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, ":memory:")
    renderer = AnnotationRenderer("prob")
    dz = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    for i, tile in enumerate(dz):
        if i > 5:
            break
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)
        plt.imshow(tile)
        plt.show()


def test_correct_number_rendered(fill_store):
    """test that the expected number of annotations are rendered"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, ":memory:")
    dz = AnnotationTileGenerator(wsi.info, store, tile_size=256)

    thumb = dz.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 51  # expect 51 rendered objects


def test_correct_colour_rendered(fill_store):
    """test colour mapping"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, ":memory:")
    renderer = AnnotationRenderer(
        "type",
        {"cell": (255, 0, 0, 255), "pt": (0, 255, 0, 255), "line": (0, 0, 255, 255)},
    )
    dz = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)

    thumb = dz.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])
    assert num == 25  # expect 25 green objects
    _, num = label(np.array(thumb)[:, :, 0])
    assert num == 25  # expect 25 red objects
    _, num = label(np.array(thumb)[:, :, 2])
    assert num == 1  # expect 1 blue objects


def test_filter_by_expression(fill_store):
    """test filtering using a where expression"""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, ":memory:")
    renderer = AnnotationRenderer(where='props["type"] == "cell"')
    dz = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    thumb = dz.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])
    assert num == 25  # expect 25 cell objects