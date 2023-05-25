import io

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pytest
import requests
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler, FunctionHandler
from bokeh.client.session import pull_session
from bokeh.events import ButtonClick, MenuItemClick
from matplotlib import colormaps
from PIL import Image

from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.visualization.bokeh_app import main

BOKEH_PATH = pkg_resources.resource_filename("tiatoolbox", "visualization/bokeh_app")


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    """Set up a temporary data directory."""
    tmp_path = tmp_path_factory.mktemp("data")
    (tmp_path / "slides").mkdir()
    (tmp_path / "overlays").mkdir()
    return {"base_path": tmp_path}


@pytest.fixture(scope="module", autouse=True)
def annotation_path(data_path):
    data_path["slide1"] = _fetch_remote_sample(
        "svs-1-small", data_path["base_path"] / "slides"
    )
    data_path["slide2"] = _fetch_remote_sample(
        "ndpi-1", data_path["base_path"] / "slides"
    )
    data_path["annotations"] = _fetch_remote_sample(
        "annotation_store_svs_1", data_path["base_path"] / "overlays"
    )
    data_path["graph"] = _fetch_remote_sample(
        "graph_svs_1", data_path["base_path"] / "overlays"
    )
    return data_path


"""Test bokeh_app."""


@pytest.fixture(scope="module", autouse=True)
def doc(data_path):
    # make a bokeh app
    # handler = DirectoryHandler(filename=BOKEH_PATH, argv=[str(tmp_path)])
    # import pdb; pdb.set_trace()
    main.config.set_sys_args(argv=["dummy_str", str(data_path["base_path"])])
    handler = FunctionHandler(main.config.setup_doc)
    app = Application(handler)
    doc = app.create_document()
    return doc


def test_roots(doc):
    # should be 2 roots, main window and controls
    assert len(doc.roots) == 2


def test_slide_select(doc, data_path):
    slide_select = doc.get_model_by_name("slide_select0")
    # check there are two available slides
    assert len(slide_select.options) == 2
    assert slide_select.options[0][0] == data_path["slide1"].name

    # select a slide and check it is loaded
    slide_select.value = ["CMU-1.ndpi"]
    assert main.UI["vstate"].slide_path == data_path["slide2"]


def test_dual_window(doc, data_path):
    control_tabs = doc.get_model_by_name("ui_layout")
    slide_wins = doc.get_model_by_name("slide_windows")
    control_tabs.active = 1
    slide_select = doc.get_model_by_name("slide_select1")
    assert len(slide_select.options) == 2
    assert slide_select.options[0][0] == data_path["slide1"].name


def test_remove_dual_window(doc, data_path):
    control_tabs = doc.get_model_by_name("ui_layout")
    slide_wins = doc.get_model_by_name("slide_windows")
    assert len(slide_wins.children) == 2
    # remove the second window
    control_tabs.tabs.pop()
    assert len(slide_wins.children) == 1

    slide_select = doc.get_model_by_name("slide_select0")
    slide_select.value = [data_path["slide1"].name]
    assert main.UI["vstate"].slide_path == data_path["slide1"]


def test_add_annotation_layer(doc, data_path):
    layer_drop = doc.get_model_by_name("layer_drop0")
    assert len(layer_drop.menu) == 2
    n_renderers = len(doc.get_model_by_name("slide_windows").children[0].renderers)
    # trigger an event to select the annotation .db file
    click = MenuItemClick(layer_drop, layer_drop.menu[0][0])
    layer_drop._trigger_event(click)
    # should be one more renderer now
    assert len(doc.get_model_by_name("slide_windows").children[0].renderers) == (
        n_renderers + 1
    )
    # we should have got the types of annotations back from the server too
    assert main.UI["vstate"].types == ["0", "1", "2", "3", "4"]


def test_cprop_input(doc):
    cprop_input = doc.get_model_by_name("cprop0")
    cmap_select = doc.get_model_by_name("cmap0")
    cprop_input.value = ["prob"]
    # as prob is continuous, cmap should be set to whatever cmap is selected
    assert main.UI["vstate"].cprop == "prob"
    assert main.UI["color_bar"].color_mapper.palette[0] == main.rgb2hex(
        colormaps[cmap_select.value](0)
    )

    cprop_input.value = ["type"]
    # as type is discrete, cmap should be a dict mapping types to colors
    assert isinstance(main.UI["vstate"].mapper, dict)
    assert list(main.UI["vstate"].mapper.keys()) == list(
        main.UI["vstate"].orig_types.values()
    )


def test_type_cmap_select(doc):
    cmap_select = doc.get_model_by_name("type_cmap0")
    vstate = main.UI["vstate"]
    user = main.UI["user"]
    cmap_select.value = ["prob", "0"]
    # set edge thicknes to 0 so the edges don't add an extra colour
    spinner = doc.get_model_by_name("edge_size0")
    spinner.value = 0
    source = main.UI["p"].renderers[main.UI["vstate"].layer_dict["overlay"]].tile_source
    url = source.url
    # replace {x}, {y}, {z} with 0, 0, 0
    url = url.replace("{x}", "1").replace("{y}", "2").replace("{z}", "2")
    im = io.BytesIO(requests.get(url).content)
    # import pdb; pdb.set_trace()
    im = np.array(Image.open(im))
    plt.imshow(im)
    plt.show()
    # check there are more than just num_types unique colors in the image,
    # as we have mapped type 0 to a continuous cmap on prob
    assert len(np.unique(im.sum(axis=2))) > 10


def test_load_graph(doc, data_path):
    layer_drop = doc.get_model_by_name("layer_drop0")
    # trigger an event to select the graph .db file
    click = MenuItemClick(layer_drop, layer_drop.menu[1][1])
    layer_drop._trigger_event(click)
    # we should have 2144 nodes in the node_source now
    assert len(main.UI["node_source"].data["x_"]) == 2144
