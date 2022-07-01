"""tests for annotation rendering using
AnnotationRenderer and AnnotationTileGenerator
"""
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage.measurements import label
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.geometry.point import Point
from skimage import data

from tests.test_annotation_stores import cell_polygon
from tiatoolbox.annotation.storage import Annotation, AnnotationStore, SQLiteStore
from tiatoolbox.tools.pyramid import AnnotationTileGenerator
from tiatoolbox.utils.env_detection import running_on_travis
from tiatoolbox.utils.visualization import AnnotationRenderer
from tiatoolbox.wsicore import wsireader


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


@pytest.fixture()
def fill_store(cell_grid, points_grid, spacing=60):
    """Factory fixture to fill stores with test data."""

    def _fill_store(
        store_class: AnnotationStore,
        path: Union[str, Path],
    ):
        """Fills store with random variety of annotations."""
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


def test_tile_generator_len(fill_store, tmp_path):
    """Test __len__ for AnnotationTileGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = AnnotationTileGenerator(wsi.info, store, tile_size=256)
    assert len(tg) == (4 * 4) + (2 * 2) + 1


def test_tile_generator_iter(fill_store, tmp_path):
    """Test __iter__ for AnnotationTileGenerator."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = AnnotationTileGenerator(wsi.info, store, tile_size=256)
    for tile in tg:
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)


@pytest.mark.skipif(running_on_travis(), reason="no display on travis.")
def test_show_generator_iter(fill_store, tmp_path):
    """Show tiles with example annotations (if not travis)."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer("prob")
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    for i, tile in enumerate(tg):
        if i > 5:
            break
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)
        plt.imshow(tile)
        plt.show()


def test_correct_number_rendered(fill_store, tmp_path):
    """Test that the expected number of annotations are rendered."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = AnnotationTileGenerator(wsi.info, store, tile_size=256)

    thumb = tg.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 75  # expect 75 rendered objects


def test_correct_colour_rendered(fill_store, tmp_path):
    """Test colour mapping."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(
        "type",
        {"cell": (255, 0, 0, 255), "pt": (0, 255, 0, 255), "line": (0, 0, 255, 255)},
    )
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)

    thumb = tg.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])
    assert num == 49  # expect 49 green objects
    _, num = label(np.array(thumb)[:, :, 0])
    assert num == 25  # expect 25 red objects
    _, num = label(np.array(thumb)[:, :, 2])
    assert num == 1  # expect 1 blue objects


def test_filter_by_expression(fill_store, tmp_path):
    """Test filtering using a where expression."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(where='props["type"] == "cell"')
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    thumb = tg.get_thumb_tile()
    _, num = label(np.array(thumb)[:, :, 1])
    assert num == 25  # expect 25 cell objects


def test_zoomed_out_rendering(fill_store, tmp_path):
    """Test that the expected number of annotations are rendered."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(max_scale=1)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)

    thumb = tg.get_tile(1, 0, 0)
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 25  # expect 25 boxes in top left quadrant

    thumb = tg.get_tile(1, 0, 1)
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 1  # expect 1 line in bottom left quadrant


def test_decimation(fill_store, tmp_path):
    """Test decimation."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array, mpp=(1, 1))
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(max_scale=1)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)

    thumb = tg.get_tile(1, 1, 1)
    plt.imshow(thumb)
    plt.show()
    _, num = label(np.array(thumb)[:, :, 1])  # default colour is green
    assert num == 16  # expect 16 pts in bottom right quadrant


def test_get_tile_negative_level(fill_store, tmp_path):
    """Test for IndexError on negative levels."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(max_scale=1)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    with pytest.raises(IndexError):
        tg.get_tile(-1, 0, 0)


def test_get_tile_large_level(fill_store, tmp_path):
    """Test for IndexError on too large a level."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(max_scale=1)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    with pytest.raises(IndexError):
        tg.get_tile(100, 0, 0)


def test_get_tile_large_xy(fill_store, tmp_path):
    """Test for IndexError on too large an xy index."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    renderer = AnnotationRenderer(max_scale=1)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    with pytest.raises(IndexError):
        tg.get_tile(0, 100, 100)


def test_sub_tile_levels(fill_store, tmp_path):
    """Test sub-tile level generation."""
    array = data.camera()
    wsi = wsireader.VirtualWSIReader(array)

    class MockTileGenerator(AnnotationTileGenerator):
        """Mock generator with specific subtile_level."""

        def tile_path(self, level: int, x: int, y: int) -> Path:
            """Tile path."""
            return Path(level, x, y)

        @property
        def sub_tile_level_count(self):
            return 1

    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = MockTileGenerator(wsi.info, store, tile_size=224)

    tile = tg.get_tile(0, 0, 0)
    assert tile.size == (112, 112)


def test_unknown_geometry(fill_store, tmp_path):
    """Test warning when unknown geometries are present that cannot
    be rendered.
    """
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    store.append(
        Annotation(geometry=MultiPoint([(5.0, 5.0), (10.0, 10.0)]), properties={})
    )
    store.commit()
    renderer = AnnotationRenderer(max_scale=8)
    tg = AnnotationTileGenerator(wsi.info, store, renderer, tile_size=256)
    with pytest.warns(UserWarning, match="Unknown geometry"):
        tg.get_tile(0, 0, 0)


def test_interp_pad_warning(fill_store, tmp_path):
    """Test warning when providing unused options."""
    array = np.ones((1024, 1024))
    wsi = wsireader.VirtualWSIReader(array)
    _, store = fill_store(SQLiteStore, tmp_path / "test.db")
    tg = AnnotationTileGenerator(wsi.info, store, tile_size=256)
    with pytest.warns(UserWarning, match="interpolation, pad_mode are unused"):
        tg.get_tile(0, 0, 0, pad_mode="constant")