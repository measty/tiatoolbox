"""Simple Flask WSGI apps to display tiles as slippery maps."""
import io
import json
from pathlib import Path
from typing import Dict
from flask_cors import cross_origin

import numpy as np
import matplotlib.cm as cm
from flask import Flask, Response, send_file
from flask.templating import render_template

from tiatoolbox import data
from tiatoolbox.tools.pyramid import AnnotationTileGenerator, ZoomifyGenerator
from tiatoolbox.wsicore.wsireader import WSIReader


class TileServer(Flask):
    """A Flask app to display Zoomify tiles as a slippery map.

    Args:
        title (str):
            The title of the tile server, displayed in the browser as
            the page title.
        layers (Dict[str, WSIReader]):
            A dictionary mapping layer names to :obj:`WSIReader` objects
            to display.

    Examples:
        >>> from tiatoolbox.wsiscore.wsireader import WSIReader
        >>> from tiatoolbox.visualization.tileserver import TileServer
        >>> wsi = WSIReader.open("CMU-1.svs")
        >>> app = TileServer(
        ...     title="Testing TileServer",
        ...     layers={
        ...         "My SVS": wsi,
        ...     },
        ... )
        >>> app.run()
    """

    def __init__(self, title: str, layers: Dict[str, WSIReader], state = None) -> None:
        super().__init__(
            __name__,
            template_folder=data._local_sample_path(
                Path("visualization") / "templates"
            ),
            static_url_path="",
            static_folder=data._local_sample_path(Path("visualization") / "static"),
        )
        self.tia_title = title
        self.tia_layers = layers
        self.tia_pyramids = {}
        self.state=state
        for key, layer in self.tia_layers.items():
            if isinstance(layer, WSIReader):
                self.tia_pyramids[key] = ZoomifyGenerator(layer)
            else:
                self.tia_pyramids[key] = layer  # its an AnnotationTileGenerator

        self.route(
            "/layer/<layer>/zoomify/TileGroup<int:tile_group>/"
            "<int:z>-<int:x>-<int:y>.jpg"
        )(
            self.zoomify,
        )
        self.route("/")(self.index)
        self.route("/changepredicate/<pred>")(self.change_pred)
        self.route("/changeprop/<prop>")(self.change_prop)
        self.route("/changeslide/<layer>/<layer_path>")(self.change_slide)
        self.route("/changecmap/<cmap>")(self.change_mapper)
        self.route("/loadannotations/<file_path>")(self.load_annotations)

    def zoomify(
        self, layer: str, tile_group: int, z: int, x: int, y: int  # skipcq: PYL-w0613
    ) -> Response:
        """Serve a Zoomify tile for a particular layer.

        Note that this should not be called directly, but will be called
        automatically by the Flask framework when a client requests a
        tile at the registered URL.

        Args:
            layer (str):
                The layer name.
            tile_group (int):
                The tile group. Currently unused.
            z (int):
                The zoom level.
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.

        Returns:
            Response:
                The tile image response.

        """
        try:
            pyramid = self.tia_pyramids[layer]
        except KeyError:
            return Response("Layer not found", status=404)
        try:
            tile_image = pyramid.get_tile(level=z, x=x, y=y)
        except IndexError:
            return Response("Tile not found", status=404)
        image_io = io.BytesIO()
        tile_image.save(image_io, format="webp")
        image_io.seek(0)
        return send_file(image_io, mimetype="image/webp")

    def index(self) -> Response:
        """Serve the index page.

        Returns:
            Response: The index page.

        """
        layers = [
            {
                "name": name,
                "url": f"/layer/{name}/zoomify/{{TileGroup}}/{{z}}-{{x}}-{{y}}.jpg",
                "size": [int(x) for x in layer.info.slide_dimensions],
                "mpp": float(np.mean(layer.info.mpp)),
            }
            for name, layer in self.tia_layers.items()    
        ]

        return render_template(
            "index.html", title=self.tia_title, layers=json.dumps(layers)
        )

    #@cross_origin()
    def change_pred(self, pred):
        for layer in self.tia_pyramids.values():
            if isinstance(layer, AnnotationTileGenerator):
                print(pred)
                if pred=="None":
                    pred=None
                layer.renderer.where=pred

        return self.index()

    def change_prop(self, prop):
        for layer in self.tia_pyramids.values():
            if isinstance(layer, AnnotationTileGenerator):
                print(prop)
                if prop=="None":
                    prop=None
                layer.renderer.score_prop=prop

        return self.index()

    def change_slide(self, layer, layer_path):
        layer_path='\\'.join(layer_path.split('-*-'))
        print(layer_path)

        self.tia_layers[layer]=WSIReader.open(Path(layer_path))
        self.tia_pyramids[layer]=ZoomifyGenerator(self.tia_layers[layer])

        return self.index()

    def change_mapper(self, cmap):
        if cmap[0]=='{':
            cmap=eval(cmap)

        if cmap is None:
            cmapp = cm.get_cmap("jet")
        elif isinstance(cmap, str):
            cmapp = cm.get_cmap(cmap)
        elif isinstance(cmap, dict):
            cmapp = lambda x: cmap[x]

        for layer in self.tia_pyramids.values():
            if isinstance(layer, AnnotationTileGenerator):
                print(cmap)
                if cmap=="None":
                    cmap=None
                layer.renderer.mapper=cmapp

        return self.index()

    def load_annotations(self, file_path):
        file_path='\\'.join(file_path.split('-*-'))
        print(file_path)

        for layer in self.tia_pyramids.values():
            if isinstance(layer, AnnotationTileGenerator):
                layer.store.add_from(file_path)
                self.state.types=layer.store.query_property("props['type']",[0,0,*self.state.dims],distinct=True)

        return self.index()

