from tiatoolbox.data import _fetch_remote_sample
from tiatoolbox.visualization.bokeh_app import main


def test_bokeh_app(tmp_path):
    """Test bokeh_app."""
    wsi_path = _fetch_remote_sample("svs-1-small", tmp_path / "slides")
    wsi_path_2 = _fetch_remote_sample("ndpi-1", tmp_path / "slides")
    anns_path = _fetch_remote_sample("annotation_store_svs_1", tmp_path / "overlays")

    # need to get doc to use tmp_path as base
    doc = main.curdoc()
    assert len(doc.roots) == 2

    slides = main.slide_select.options  # could also access via doc
    assert len(slides) == 2
    assert slides[0] == "svs-1-small.svs"

    main.slide_select.value = ["CMU-1.ndpi"]
    assert main.vstate.slide_path == wsi_path_2
