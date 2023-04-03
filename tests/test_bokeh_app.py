import pkg_resources
import pytest
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler, FunctionHandler
from bokeh.client.session import pull_session

from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.visualization.bokeh_app import main

BOKEH_PATH = pkg_resources.resource_filename("tiatoolbox", "visualization/bokeh_app")


def test_bokeh_app(tmp_path):
    """Test bokeh_app."""
    (tmp_path / "slides").mkdir()
    (tmp_path / "overlays").mkdir()
    wsi_path = _fetch_remote_sample("svs-1-small", tmp_path / "slides")
    wsi_path_2 = _fetch_remote_sample("ndpi-1", tmp_path / "slides")
    anns_path = _fetch_remote_sample("annotation_store_svs_1", tmp_path / "overlays")

    # make a bokeh app
    # handler = DirectoryHandler(filename=BOKEH_PATH, argv=[str(tmp_path)])
    handler = FunctionHandler(main)
    app = Application(handler)
    doc = app.create_document()
    # handler.modify_document(doc)

    # need to get doc to use tmp_path as base
    # doc = main.curdoc()
    assert len(doc.roots) == 2

    slide_select = doc.get_model_by_name("slide_select")
    assert len(slide_select.options) == 2
    assert slide_select.options[0] == "svs-1-small.svs"

    slide_select.value = ["CMU-1.ndpi"]
    assert doc.vstate.slide_path == wsi_path_2
