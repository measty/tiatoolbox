"""Simple Flask WSGI apps to display tiles as slippery maps."""
import io
import json
import os
import pickle
import secrets
import urllib
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.cm as cm
import numpy as np
from flask import Flask, Response, request, send_file, make_response
from flask.templating import render_template
from PIL import Image

from tiatoolbox import data
from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.tools.pyramid import AnnotationTileGenerator, ZoomifyGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer, colourise_image
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, VirtualWSIReader, WSIReader


class TileServer(Flask):
    """A Flask app to display Zoomify tiles as a slippery map.

    Args:
        title (str):
            The title of the tile server, displayed in the browser as
            the page title.
        layers (Dict[str, WSIReader | str] | List[WSIReader | str]):
            A dictionary mapping layer names to image paths or
            :obj:`WSIReader` objects to display. May also be a list,
            in which case generic names 'layer-1', 'layer-2' etc.
            will be used.
            If layer is a single-channel low-res overlay, it will be
            colourized using the 'viridis' colourmap

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

    def __init__(
        self,
        title: str,
        layers: Union[Dict[str, Union[WSIReader, str]], List[Union[WSIReader, str]]],
        state: Dict = None,
    ) -> None:
        super().__init__(
            __name__,
            template_folder=data._local_sample_path(
                Path("visualization") / "templates"
            ),
            static_url_path="",
            static_folder=data._local_sample_path(Path("visualization") / "static"),
        )
        self.tia_title = title
        self.tia_layers = {}
        self.tia_pyramids = {}
        self.slide_mpps = {}
        self.renderers = {}

        # Generic layer names if none provided.
        if isinstance(layers, list):
            layers = {f"layer-{i}": p for i, p in enumerate(layers)}
        # Set up the layer dict.
        meta = None
        for i, key in enumerate(layers):
            layer = layers[key]

            if isinstance(layer, (str, Path)):
                layer_path = Path(layer)
                if layer_path.suffix in [".jpg", ".png"]:
                    # Assume its a low-res heatmap.
                    layer = Image.open(layer_path)
                    layer = np.array(layer)
                else:
                    layer = WSIReader.open(layer_path)

            if isinstance(layer, np.ndarray):
                # Make into rgb if single channel.
                layer = colourise_image(layer)
                layer = VirtualWSIReader(layer, info=meta)

            self.tia_layers[key] = layer

            if isinstance(layer, WSIReader):
                self.tia_pyramids[key] = ZoomifyGenerator(layer)
            else:
                self.tia_pyramids[key] = layer  # its an AnnotationTileGenerator

            if i == 0:
                meta = layer.info

        self.route(
            "/tileserver/layer/<layer>/<user>/zoomify/TileGroup<int:tile_group>/"
            "<int:z>-<int:x>-<int:y>@<int:res>x.jpg"
        )(
            self.zoomify,
        )
        self.route("/")(self.index)
        self.route("/tileserver/setup")(self.setup)
        self.route("/tileserver/changepredicate/<pred>")(self.change_pred)
        self.route("/tileserver/changeprop/<prop>")(self.change_prop)
        self.route("/tileserver/changeslide/<layer>/<layer_path>")(self.change_slide)
        self.route("/tileserver/changecmap/<cmap>")(self.change_mapper)
        self.route("/tileserver/loadannotations/<file_path>/<float:model_mpp>")(self.load_annotations)
        self.route("/tileserver/changeoverlay/<overlay_path>")(self.change_overlay)
        self.route("/tileserver/commit/<save_path>")(self.commit_db)
        self.route("/tileserver/updaterenderer/<prop>/<val>")(self.update_renderer)
        self.route("/tileserver/updatewhere", methods=["POST"])(self.update_where)
        self.route("/tileserver/changesecondarycmap/<type>/<prop>/<cmap>")(self.change_secondary_cmap)
        self.route("/tileserver/getprops")(self.get_properties)
        self.route("/tileserver/reset")(self.reset)

    def zoomify(
        self,
        layer: str,
        user: str,
        tile_group: int,
        z: int,
        x: int,
        y: int,  # skipcq: PYL-w0613
        res: int,
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
        #user=request.cookies.get('user')
        try:
            pyramid = self.tia_pyramids[user][layer]
        except KeyError:
            return Response("Layer not found", status=404)
        try:
            tile_image = pyramid.get_tile(level=z, x=x, y=y, res=res)
        except IndexError:
            return Response("Tile not found", status=404)
        image_io = io.BytesIO()
        tile_image.save(image_io, format="webp")
        image_io.seek(0)
        return send_file(image_io, mimetype="image/webp")

    def update_types(self, SQ):
        types = SQ.pquery("props['type']")
        if None in types:
            types.remove(None)
        #if len(types) == 0:
            #return None
        return tuple(types)

    def get_pyramid(self, user):
        return self.tia_pyramids[user][user]

    @staticmethod
    def decode_safe_name(name):
        return Path(urllib.parse.unquote(name).replace("\\", os.sep))

    def index(self) -> Response:
        """Serve the index page.

        Returns:
            Response: The index page.

        """
        layers = [
            {
                "name": name,
                "url": f"/layer/{name}/zoomify/{{TileGroup}}/{{z}}-{{x}}-{{y}}.jpg",
                "size": [int(x) for x in reader.info.slide_dimensions],
                "mpp": float(np.mean(reader.info.mpp)),
            }
            for name, reader in self.tia_layers.items()
        ]
        return render_template(
            "index.html", title=self.tia_title, layers=json.dumps(layers)
        )

    # @cross_origin()
    def change_pred(self, pred):
        user=request.cookies.get('user')
        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                print(pred)
                if pred == "None":
                    pred = None
                layer.renderer.where = pred

        return "done"

    def change_prop(self, prop):
        user=request.cookies.get('user')
        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                print(prop)
                if prop == "None":
                    prop = None
                layer.renderer.score_prop = prop

        return "done"

    def setup(self):
        #respond with a random cookie 
        resp = make_response("done")
        user = secrets.token_urlsafe(16)
        resp.set_cookie("user", user)
        self.renderers[user] = AnnotationRenderer(
            "type",
            {"class1": (1, 0, 0, 1), "class2": (0, 0, 1, 1), "class3": (0, 1, 0, 1)},
            thickness=-1,
            edge_thickness=2,
            zoomed_out_strat="scale",
            max_scale=8,
        )
        return resp

    def reset(self):
        user=request.cookies.get('user')
        self.tia_layers[user] = {}
        self.tia_pyramids[user] = {}
        self.slide_mpps[user] = None
        return "done"

    def change_slide(self, layer, layer_path):
        user=request.cookies.get('user')
        # layer_path='\\'.join(layer_path.split('-*-'))
        layer_path = self.decode_safe_name(layer_path)
        print(layer_path)

        """self.tia_layers[user][layer]=WSIReader.open(Path(layer_path))
        self.tia_pyramids[user][layer]=ZoomifyGenerator(self.tia_layers[user][layer])
        for layer in self.tia_layers[user].keys():
            if layer!='slide':
                del self.tia_pyramids[user][layer]
                del self.tia_layers[user][layer]"""

        self.tia_layers[user] = {layer: WSIReader.open(Path(layer_path))}
        self.tia_pyramids[user] = {layer: ZoomifyGenerator(self.tia_layers[user][layer])}
        self.slide_mpps[user] = self.tia_layers[user][layer].info.mpp

        return layer

    def change_mapper(self, cmap):
        user=request.cookies.get('user')
        if cmap[0] == "{":
            cmap = eval(cmap)

        if cmap is None:
            cmapp = cm.get_cmap("jet")
        elif isinstance(cmap, str):
            cmapp = cm.get_cmap(cmap)
        elif isinstance(cmap, dict):
            cmapp = lambda x: cmap[x]

        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                print(cmap)
                if cmap == "None":
                    cmap = None
                layer.renderer.mapper = cmapp

        return "done"

    def change_secondary_cmap(self, type, prop, cmap):
        user=request.cookies.get('user')
        if cmap[0] == "{":
            cmap = eval(cmap)

        if cmap is None:
            cmapp = cm.get_cmap("jet")
        elif isinstance(cmap, str):
            cmapp = cm.get_cmap(cmap)
        elif isinstance(cmap, dict):
            cmapp = lambda x: cmap[x]

        if prop == "None":
            cmap_dict = None
        else:
            cmap_dict={'type': type, 'score_prop': prop, 'mapper': cmapp}

        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                print(cmap)
                #if cmapp == "None":
                 #   cmapp = None
                layer.renderer.secondary_cmap = cmap_dict

        return "done"


    def update_renderer(self, prop, val):
        user=request.cookies.get('user')
        val = json.loads(val)
        if val == "None" or val == "null":
            val = None
        self.renderers[user].__setattr__(prop, val)

        return "done"

    def update_where(self):
        user=request.cookies.get('user')
        get_types = json.loads(request.form["types"])
        filter_val = json.loads(request.form["filter"])

        if filter_val == "None":

            def pred(props):
                return props["type"] in get_types

        else:

            def pred(props):
                return eval(filter_val) and props["type"] in get_types

        self.renderers[user].where = pred
        return "done"

    def load_annotations(self, file_path, model_mpp):
        # file_path='\\'.join(file_path.split('-*-'))
        user=request.cookies.get('user')
        file_path = self.decode_safe_name(file_path)
        print(file_path)

        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                layer.store.add_from(
                    file_path,
                    saved_res=model_mpp,
                    slide_res=self.slide_mpps[user],
                )
                types = self.update_types(layer.store)
                return json.dumps(types)

        SQ = SQLiteStore(auto_commit=False)
        SQ.add_from(file_path, saved_res=model_mpp, slide_res=self.slide_mpps[user][0])
        self.tia_pyramids[user]["overlay"] = AnnotationTileGenerator(
            self.tia_layers[user]["slide"].info, SQ, self.renderers[user]
        )
        self.tia_layers[user]["overlay"] = self.tia_pyramids[user]["overlay"]
        types = self.update_types(SQ)
        print(types)
        return json.dumps(types)  # "overlay"

    def change_overlay(self, overlay_path):
        user=request.cookies.get('user')
        print(f'User is: {user}')
        # overlay_path='\\'.join(overlay_path.split('-*-'))
        overlay_path = self.decode_safe_name(overlay_path)
        print(overlay_path)
        if overlay_path.suffix == ".geojson":
            SQ = SQLiteStore.from_geojson(overlay_path)
        elif overlay_path.suffix == ".dat":
            SQ = SQLiteStore(auto_commit=False)
            SQ.add_from(overlay_path, slide_res=self.slide_mpps[user])
        elif overlay_path.suffix in [".jpg", ".png", ".tiff"]:
            layer = f"layer{len(self.tia_pyramids[user][user])}"
            if overlay_path.suffix == ".tiff":
                self.tia_layers[user][layer] = OpenSlideWSIReader(
                    overlay_path, mpp=self.tia_layers[user]["slide"].info.mpp[0]
                )
            else:
                self.tia_layers[user][layer] = VirtualWSIReader(
                    Path(overlay_path), info=self.tia_layers[user]["slide"].info
                )
            self.tia_pyramids[user][layer] = ZoomifyGenerator(self.tia_layers[user][layer])
            return json.dumps(layer)
        else:
            SQ = SQLiteStore(overlay_path, auto_commit=False)

        for key, layer in self.tia_pyramids[user].items():
            if isinstance(layer, AnnotationTileGenerator):
                layer.store = SQ
                types = self.update_types(SQ)
                return json.dumps(types)
        self.tia_pyramids[user]["overlay"] = AnnotationTileGenerator(
            self.tia_layers[user]["slide"].info, SQ, self.renderers[user]
        )
        self.tia_layers[user]["overlay"] = self.tia_pyramids[user]["overlay"]
        types = self.update_types(SQ)
        return json.dumps(types)

    def get_properties(self, type=None):
        #get all properties present in the store
        user=request.cookies.get('user')
        where = None
        if type is not None:
            where = f'props["type"]="{type}"',
        ann_props = self.tia_pyramids[user]['overlay'].store.pquery(
            select = "*",
            where = where,
            unique = False,
            )
        props = []
        for prop_dict in ann_props.values():
            props.extend(list(prop_dict.keys()))
        return json.dumps(list(set(props)))

    def commit_db(self, save_path):
        user=request.cookies.get('user')
        save_path = self.decode_safe_name(save_path)
        print(save_path)
        for key, layer in self.tia_pyramids[user].items():
            if isinstance(layer, AnnotationTileGenerator):
                if layer.store.path.suffix == ".db":
                    print("db committed")
                    layer.store.commit()
                else:
                    layer.store.commit()
                    layer.store.dump(str(save_path))
        return "done"
