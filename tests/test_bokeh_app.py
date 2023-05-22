import pkg_resources
import pytest
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler, FunctionHandler
from bokeh.client.session import pull_session

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
