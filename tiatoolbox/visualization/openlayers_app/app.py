from __future__ import annotations

from pathlib import Path

from tiatoolbox.visualization.tileserver import TileServer


def create_app(slide: str | None = None) -> TileServer:
    layers = {}
    if slide is not None:
        layers = {"slide": slide}
    app = TileServer(title="TIAToolbox OpenLayers", layers=layers)
    tpl = Path(__file__).parent / "templates"
    static = Path(__file__).parent / "static"
    app.jinja_loader.searchpath.insert(0, str(tpl))
    app.static_folder = str(static)
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", threaded=True)
