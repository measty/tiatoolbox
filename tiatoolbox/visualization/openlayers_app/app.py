"""OpenLayers based tile server application."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flask import send_from_directory
from flask.templating import render_template

from tiatoolbox.visualization.tileserver import TileServer


class OpenLayersServer(TileServer):
    """TileServer subclass serving an OpenLayers front-end."""

    def __init__(
        self,
        title: str,
        layers: dict[str, str] | list[str],
        slide_folder: str | None = None,
        overlay_folder: str | None = None,
    ) -> None:
        """Create an :class:`OpenLayersServer` instance."""
        self.slide_folder = Path(slide_folder) if slide_folder else None
        self.overlay_folder = Path(overlay_folder) if overlay_folder else None
        super().__init__(title=title, layers=layers)

    def index(self) -> str:  # type: ignore[override]
        """Serve the index page with slide and overlay lists."""
        session_id = self._get_session_id()
        resp = None
        if (session_id is None) or (session_id not in self.layers):
            resp = self.session_id()
            # no layers loaded yet for a new session
            layers: list[dict] = []
        else:
            layers = [
                {
                    "name": name,
                    "url": f"/tileserver/layer/{name}/default/zoomify/"
                    "{TileGroup}/{z}-{x}-{y}@1x.jpg",
                    "size": [int(x) for x in layer.info.slide_dimensions],
                    "mpp": float(np.mean(layer.info.mpp)),
                }
                for name, layer in self.layers[session_id].items()
            ]

        slide_opts: list[str] = []
        overlay_opts: list[str] = []
        if self.slide_folder:
            for ext in [
                "*.svs",
                "*.ndpi",
                "*.tiff",
                "*.tif",
                "*.mrxs",
                "*.png",
                "*.jpg",
                "*.qptiff",
            ]:
                slide_opts.extend(sorted(map(str, self.slide_folder.glob(ext))))
        if self.overlay_folder:
            for ext in [
                "*.db",
                "*.dat",
                "*.geojson",
                "*.png",
                "*.jpg",
                "*.json",
                "*.tiff",
                "*.pkl",
                "*.mrxs",
                "*.ndpi",
                "*.svs",
                "*.tif",
                "*.npy",
                "*.mha",
            ]:
                overlay_opts.extend(sorted(map(str, self.overlay_folder.glob(ext))))

        rendered = render_template(
            "index.html",
            title=self.title,
            layers=json.dumps(layers),
            slide_options=slide_opts,
            overlay_options=overlay_opts,
        )
        if resp is None:
            return rendered
        resp.set_data(rendered)
        return resp


def create_app(
    slide: str | None = None,
    slide_folder: str | None = None,
    overlay_folder: str | None = None,
) -> TileServer:
    """Return a ready-to-run :class:`TileServer` instance."""
    layers: dict[str, str] = {}
    if slide is not None:
        layers = {"slide": slide}
    app = OpenLayersServer(
        title="TIAToolbox OpenLayers",
        layers=layers,
        slide_folder=slide_folder,
        overlay_folder=overlay_folder,
    )
    tpl = Path(__file__).parent / "templates"
    static = Path(__file__).parent / "static"
    app.jinja_loader.searchpath.insert(0, str(tpl))
    app.add_url_rule(
        "/main.js",
        endpoint="openlayers_main_js",
        view_func=lambda: send_from_directory(static, "main.js"),
    )
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", threaded=True)  # noqa: S104
