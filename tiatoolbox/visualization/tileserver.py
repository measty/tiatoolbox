"""Simple Flask WSGI apps to display tiles as slippery maps."""
import ast
import copy
import gc
import io
import json
import os
import secrets
import tempfile
import threading
import time
import urllib
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.cm as cm
import numpy as np
from flask import Flask, Response, make_response, request, send_file
from flask.templating import render_template
from matplotlib import colormaps
from PIL import Image

import tiatoolbox.visualization.bokeh_app.postproc_defs as postproc
from tiatoolbox import data
from tiatoolbox.annotation.storage import AnnotationStore, SQLiteStore
from tiatoolbox.tools.pyramid import AnnotationTileGenerator, ZoomifyGenerator
from tiatoolbox.utils.misc import add_from_dat, store_from_dat
from tiatoolbox.utils.visualization import AnnotationRenderer, colourise_image
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, VirtualWSIReader, WSIReader


class TileServer(Flask):
    """A Flask app to display Zoomify tiles as a slippery map.

    Args:
        title (str):
            The title of the tile server, displayed in the browser as
            the page title.
        layers (Dict[str, WSIReader | str] | List[WSIReader | str]):
            A dictionary mapping layer names to image paths, annotation paths,
            or :obj:`WSIReader` objects to display. The dictionary should have
            a 'slide' key which is the base slide for the visualization.
            May also be a list, in which case generic names 'slide', 'layer-1',
            'layer-2' etc. will be used. First entry in list will be assumed to
            be the base slide. If a layer is a single-channel low-res overlay,
            it will be colourized using the 'viridis' colourmap.

    Examples:
        >>> from tiatoolbox.wsicore.wsireader import WSIReader
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
        renderer: AnnotationRenderer = None,
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
        self.renderer = renderer
        self.overlap = 0
        if renderer is None:  # pragma: no branch
            self.renderer = AnnotationRenderer(
                score_prop="type",
                thickness=-1,
                edge_thickness=1,
                zoomed_out_strat="scale",
                max_scale=8,
                blur_radius=0,
            )
        self.slide_mpps = {}
        self.renderers = {}
        self.overlaps = {}

        # Generic layer names if none provided.
        if isinstance(layers, list):
            layers_dict = {"slide": layers[0]}
            for i, p in enumerate(layers[1:]):
                layers_dict[f"layer-{i+1}"] = p
            layers = layers_dict
        # Set up the layer dict.
        meta = None
        # if layers provided directly, not using with app,
        # so just set default user
        self.default_user = len(layers) > 0
        if self.default_user:
            self.tia_layers["default"] = {}
            self.tia_pyramids["default"] = {}
            self.renderers["default"] = copy.deepcopy(self.renderer)
        for i, (key, layer) in enumerate(layers.items()):
            layer = self._get_layer_as_wsireader(layer, meta)

            self.tia_layers["default"][key] = layer

            if isinstance(layer, WSIReader):
                self.tia_pyramids["default"][key] = ZoomifyGenerator(layer)
            else:
                self.tia_pyramids["default"][
                    key
                ] = layer  # it's an AnnotationTileGenerator

            if i == 0:
                meta = layer.info  # base slide info
                self.slide_mpps["default"] = meta.mpp

        self.route(
            "/tileserver/layer/<layer>/<user>/zoomify/TileGroup<int:tile_group>/"
            "<int:z>-<int:x>-<int:y>@<int:res>x.jpg"
        )(
            self.zoomify,
        )
        self.route("/")(self.index)
        self.route("/tileserver/get_user")(self.get_user)
        self.route("/tileserver/change_color_prop/<prop>", methods=["PUT"])(
            self.change_prop
        )
        self.route("/tileserver/change_slide/<slide_path>", methods=["PUT"])(
            self.change_slide
        )
        self.route("/tileserver/change_cmap/<cmap>", methods=["PUT"])(
            self.change_mapper
        )
        self.route(
            "/tileserver/load_annotations/<file_path>/<float:model_mpp>",
            methods=["PUT"],
        )(self.load_annotations)
        self.route("/tileserver/change_overlay/<overlay_path>", methods=["PUT"])(
            self.change_overlay
        )
        self.route("/tileserver/commit/<save_path>")(self.commit_db)
        self.route("/tileserver/update_renderer/<prop>/<val>", methods=["PUT"])(
            self.update_renderer
        )
        self.route("/tileserver/build_cmap/<mix_type>/<cmap>/<cdict>", methods=["PUT"])(
            self.build_cmap
        )
        self.route("/tileserver/add_post_proc", methods=["POST"])(self.add_post_proc)
        self.route(
            "/tileserver/change_demux/<stain>/<float:wed>/<float:weh>/<float:wdh>/<float:wde>",
            methods=["PUT"],
        )(self.change_demux)
        self.route("/tileserver/reset/<user>")(self.reset)
        self.route(
            "/tileserver/change_secondary_cmap/<type_id>/<prop>/<cmap>", methods=["PUT"]
        )(self.change_secondary_cmap)
        self.route("/tileserver/get_prop_names/<ann_type>")(self.get_properties)
        self.route("/tileserver/get_prop_values/<prop>/<ann_type>")(
            self.get_property_values
        )

    def _get_user(self):
        """Get the user from the request.

        Returns:
            str: The user name.

        """
        if self.default_user:
            return "default"
        return request.cookies.get("user")

    @staticmethod
    def _get_cmap(cmap):
        """Get the colourmap from the string sent."""
        if cmap[0] == "{":
            cmap = ast.literal_eval(cmap)

        if cmap == "None":
            return None
        if isinstance(cmap, str):
            return colormaps[cmap]

        def cmapp(x):
            """Dictionary colormap callable wrapper"""
            return cmap[x]

        return cmapp

    def _get_layer_as_wsireader(self, layer, meta):
        """Gets appropriate image provider for layer.

        Args:
            layer (str | ndarray | WSIReader):
                A reference to an image or annotations to be displayed.
            meta (WSIMeta):
                The metadata of the base slide.

        Returns:
            WSIReader or AnnotationTileGenerator:
                The appropriate image source for the layer.

        """
        if isinstance(layer, (str, Path)):
            layer_path = Path(layer)
            if layer_path.suffix in [".jpg", ".png"]:
                # Assume it's a low-res heatmap.
                layer = np.array(Image.open(layer_path))
            elif layer_path.suffix == ".db":
                # Assume it's an annotation store.
                layer = AnnotationTileGenerator(
                    meta,
                    SQLiteStore(layer_path),
                    self.renderers["default"],
                    overlap=self.overlap,
                )
            elif layer_path.suffix == ".geojson":
                # Assume annotations in geojson format
                layer = AnnotationTileGenerator(
                    meta,
                    SQLiteStore.from_geojson(layer_path),
                    self.renderers["default"],
                    overlap=self.overlap,
                )
            else:
                # Assume it's a WSI.
                return WSIReader.open(layer_path)

        if isinstance(layer, np.ndarray):
            # Make into rgb if single channel.
            layer = colourise_image(layer)
            return VirtualWSIReader(layer, info=meta)

        if isinstance(layer, AnnotationStore):
            layer = AnnotationTileGenerator(
                meta,
                layer,
                self.renderers["default"],
                overlap=self.overlap,
            )

        return layer

    def zoomify(
        self,
        layer: str,
        user: str,
        tile_group: int,  # skipcq: PYL-w0613
        z: int,
        x: int,
        y: int,
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
            flask.Response:
                The tile image response.

        """
        try:
            pyramid = self.tia_pyramids[user][layer]
            interpolation = (
                "nearest"
                if isinstance(self.tia_layers[user][layer], VirtualWSIReader)
                else "optimise"
            )
            if isinstance(pyramid, AnnotationTileGenerator):
                interpolation = None
        except KeyError:
            return Response("Layer not found", status=404)
        try:
            tile_image = pyramid.get_tile(
                level=z, x=x, y=y, res=res, interpolation=interpolation
            )
        except IndexError:
            return Response("Tile not found", status=404)
        image_io = io.BytesIO()
        tile_image.save(image_io, format="webp")
        image_io.seek(0)
        return send_file(image_io, mimetype="image/webp")

    @staticmethod
    def update_types(sq: SQLiteStore):
        """Get the available types from the store."""
        types = sq.pquery("props['type']")
        types = sorted([t for t in types if t is not None])
        return tuple(types)

    @staticmethod
    def decode_safe_name(name):
        """Decode a url-safe name."""
        return Path(urllib.parse.unquote(name).replace("\\", os.sep))

    def get_ann_layer(self, user):
        """Get the annotation layer for a user."""
        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                return layer
        raise ValueError("No annotation layer found.")

    def index(self) -> Response:
        """Serve the index page.

        Returns:
            flask.Response:
                The index page.

        """
        user = self._get_user()
        layers = [
            {
                "name": name,
                "url": f"/tileserver/layer/{name}/default/zoomify/"
                + "{TileGroup}/{z}-{x}-{y}@1x.jpg",
                "size": [int(x) for x in layer.info.slide_dimensions],
                "mpp": float(np.mean(layer.info.mpp)),
            }
            for name, layer in self.tia_layers[user].items()
        ]

        return render_template(
            "index.html", title=self.tia_title, layers=json.dumps(layers)
        )

    def change_prop(self, prop):
        """Change the property to colour annotations by."""
        user = self._get_user()
        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                if prop == "None":
                    prop = None
                layer.renderer.score_prop = prop

        return "done"

    def get_user(self):
        """Setup a user."""
        # respond with a random cookie
        resp = make_response("done")
        if self.default_user:
            user = "default"
        else:
            user = secrets.token_urlsafe(16)
        resp.set_cookie("user", user, httponly=True)
        self.renderers[user] = copy.deepcopy(self.renderer)
        self.overlaps[user] = 0
        self.tia_layers[user] = {}
        self.tia_pyramids[user] = {}
        return resp

    def reset(self, user):
        """Reset the tileserver."""
        # user = self._get_user()
        del self.tia_layers[user]
        del self.tia_pyramids[user]
        del self.slide_mpps[user]
        del self.renderers[user]
        del self.overlaps[user]
        return "done"

    def change_slide(self, slide_path):
        """Change the slide."""
        user = self._get_user()
        slide_path = self.decode_safe_name(slide_path)

        self.tia_layers[user] = {"slide": WSIReader.open(Path(slide_path))}
        self.tia_pyramids.pop(user, None)
        self.tia_pyramids[user] = {
            "slide": ZoomifyGenerator(self.tia_layers[user]["slide"], tile_size=256)
        }
        if self.tia_layers[user]["slide"].info.mpp is None:
            self.tia_layers[user]["slide"].info.mpp = [1, 1]
        self.slide_mpps[user] = self.tia_layers[user]["slide"].info.mpp

        return "done"

    def change_mapper(self, cmap):
        """Change the colour mapper for the overlay."""
        user = self._get_user()
        cmapp = self._get_cmap(cmap)

        self.renderers[user].mapper = cmapp
        self.renderers[user].function_mapper = None

        return "done"

    def change_secondary_cmap(self, type_id, prop, cmap):
        """Change the type-specific colour mapper for the overlay."""
        user = self._get_user()
        cmapp = self._get_cmap(cmap)

        cmap_dict = {"type": json.loads(type_id), "score_prop": prop, "mapper": cmapp}
        self.renderers[user].secondary_cmap = cmap_dict

        return "done"

    def build_cmap(self, cmap, mix_type, cdict):
        """build a color mapper function from a dictionary."""
        user = self._get_user()
        cdict = json.loads(cdict)
        props_list = list(cdict.keys())
        color_matrix = np.array([cdict[prop] for prop in props_list])

        def mapper_fn(props, props_list, color_matrix, mix_type, cmap):
            prop_vals = np.array([props[prop] for prop in props_list])
            # prop_vals[prop_vals < 0.3] = 0
            # prop_vals = prop_vals ** 3 * (np.sum(prop_vals) / np.sum(prop_vals ** 3))
            if mix_type == "pow":
                prop_vals = prop_vals**3 * (
                    np.max(prop_vals) / (np.max(prop_vals**3) + 0.00001)
                )
            elif mix_type == "max":
                prop_vals[prop_vals < np.max(prop_vals)] = 0
            elif mix_type == "softm":
                # calculate softmax
                mv = np.max(prop_vals)
                prop_vals = np.exp(prop_vals)
                prop_vals = prop_vals / np.sum(prop_vals)
                prop_vals = prop_vals * mv / np.max(prop_vals)
            elif mix_type == "lin":
                prop_vals[prop_vals < 0.1] = 0
            # calculate weighted average
            if mix_type in ["pow", "max", "lin", "softm"]:
                color = np.matmul(prop_vals, color_matrix) * 255
                if np.max(color) > 255:
                    color = color * (255 / np.max(color))
                # import pdb; pdb.set_trace()
            elif mix_type == "avg":
                color = np.array(cmap(np.mean(prop_vals))) * 255
            else:
                color = (
                    np.array(cmap(np.prod(prop_vals))) * 255 * (1.15 ** len(prop_vals))
                )
            return (int(color[0]), int(color[1]), int(color[2]), 255)

        mapper_fn = partial(
            mapper_fn,
            props_list=props_list,
            color_matrix=color_matrix,
            mix_type=mix_type,
            cmap=cm.get_cmap(cmap),
        )
        # set renderer mapper
        self.renderers[user].function_mapper = mapper_fn
        return "done"

    def update_renderer(self, prop, val):
        """Update a property in the renderer."""
        user = self._get_user()
        val = json.loads(val)
        if val == "None" or val == "null":
            val = None
        self.renderers[user].__setattr__(prop, val)
        self.renderers[user].__setattr__(prop, val)
        if prop == "blur_radius":
            self.overlaps[user] = int(1.5 * val)
            self.get_ann_layer(user).overlap = self.overlaps[user]
        return "done"

    def update_where(self):
        """Update the where callable in the renderer."""
        user = self._get_user()
        get_types = json.loads(request.form["types"])
        filter_val = json.loads(request.form["filter"])

        if filter_val == "None":
            if len(get_types) == 0:
                pred = None
            else:

                def pred(props):
                    return props["type"] in get_types

        else:
            if len(get_types) == 0:

                def pred(props):
                    return eval(filter_val)

            else:

                def pred(props):
                    return eval(filter_val) and props["type"] in get_types

        self.renderers[user].where = pred
        return "done"

    def load_annotations(self, file_path, model_mpp):
        """Load annotations from a dat file.

        Adds to an existing store if one is already present,
        otherwise creates a new store.

        """
        user = self._get_user()
        file_path = self.decode_safe_name(file_path)

        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                add_from_dat(
                    layer.store,
                    file_path,
                    np.array(model_mpp) / np.array(self.slide_mpps[user]),
                )
                types = self.update_types(layer.store)
                return json.dumps(types)

        sq = store_from_dat(
            file_path, np.array(model_mpp) / np.array(self.slide_mpps[user])
        )
        tmp_path = Path(tempfile.gettempdir()) / "temp.db"
        sq.dump(tmp_path)
        sq = SQLiteStore(tmp_path)
        self.tia_pyramids[user]["overlay"] = AnnotationTileGenerator(
            self.tia_layers[user]["slide"].info,
            sq,
            self.renderers[user],
            overlap=self.overlaps[user],
        )
        self.tia_layers[user]["overlay"] = sq
        types = self.update_types(sq)
        return json.dumps(types)

    def change_overlay(self, overlay_path):
        """Change the overlay.

        If the path points to some annotations, the current overlay
        is replaced with the new one. If the path points to an image,
        it is added as a new layer.

        """
        user = self._get_user()
        overlay_path = self.decode_safe_name(overlay_path)

        if overlay_path.suffix in [".jpg", ".png", ".tiff", ".svs", ".ndpi", ".mrxs"]:
            layer = f"layer{len(self.tia_pyramids[user])}"
            if overlay_path.suffix == ".tiff":
                self.tia_layers[user][layer] = OpenSlideWSIReader(
                    overlay_path, mpp=self.tia_layers[user]["slide"].info.mpp[0]
                )
            elif overlay_path.suffix in [".jpg", ".png"]:
                self.tia_layers[user][layer] = VirtualWSIReader(
                    Path(overlay_path), info=self.tia_layers[user]["slide"].info
                )
            else:
                self.tia_layers[user][layer] = WSIReader.open(overlay_path)
            self.tia_pyramids[user][layer] = ZoomifyGenerator(
                self.tia_layers[user][layer]
            )
            return json.dumps(layer)
        if overlay_path.suffix == ".geojson":
            sq = SQLiteStore(f"file:{overlay_path.stem}?mode=memory&cache=shared")
            sq.add_from_geojson(overlay_path)
        elif overlay_path.suffix == ".dat":
            sq = SQLiteStore(f"file:{overlay_path.stem}?mode=memory&cache=shared")
            add_from_dat(sq, overlay_path)
            sq.create_index("area", '"area"')
        else:
            sq = SQLiteStore(overlay_path, auto_commit=False)

        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                layer.store = sq
                print(f"loaded {len(sq)} annotations")
                types = self.update_types(sq)
                sq = None
                return json.dumps(types)
        self.tia_pyramids[user]["overlay"] = AnnotationTileGenerator(
            self.tia_layers[user]["slide"].info,
            sq,
            self.renderers[user],
            overlap=self.overlaps[user],
        )
        print(f'loaded {len(self.tia_pyramids[user]["overlay"].store)} annotations')
        self.tia_layers[user]["overlay"] = sq
        types = self.update_types(sq)
        sq = None
        return json.dumps(types)

    def get_properties(self, ann_type):
        """Get all the properties of the annotations in the store."""
        user = self._get_user()
        where = None
        if ann_type != "all":
            where = f'props["type"]=={ann_type}'
        ann_props = self.get_ann_layer(user).store.pquery(
            select="*",
            where=where,
            unique=False,
        )
        props = []
        for prop_dict in ann_props.values():
            props.extend(list(prop_dict.keys()))
        return json.dumps(list(set(props)))

    def get_property_values(self, prop, ann_type):
        """Get all the values of a property in the store."""
        user = self._get_user()
        where = None
        if ann_type != "all":
            where = f'props["type"]=={ann_type}'
        if "overlay" not in self.tia_pyramids[user]:
            return json.dumps([])
        ann_props = self.get_ann_layer(user).store.pquery(
            select=f"props['{prop}']",
            where=where,
            unique=True,
        )
        return json.dumps(list(ann_props))

    def add_post_proc(self):
        user = self._get_user()
        post_processor = json.loads(request.form["name"])
        kwargs = json.loads(request.form["kwargs"])
        pp_class = getattr(postproc, post_processor)
        self.tia_pyramids[user][json.loads(request.form["layer"])].post_proc = pp_class(
            **kwargs
        )
        return "done"

    def change_demux(self, stain, wed, weh, wdh, wde):
        user = self._get_user()
        self.tia_pyramids[user]["slide"].post_proc.stains = stain
        self.tia_pyramids[user]["slide"].post_proc.coupling_coeffs = {
            "wed": wed,
            "weh": weh,
            "wdh": wdh,
            "wde": wde,
        }
        return "done"

    def commit_db(self, save_path):
        """Commit changes to the current store.

        If the store is not already associated with a .db file,
        the save_path is used to create a new .db file.

        """
        user = self._get_user()
        save_path = self.decode_safe_name(save_path)
        for layer in self.tia_pyramids[user].values():
            if isinstance(layer, AnnotationTileGenerator):
                if layer.store.path.suffix == ".db" and layer.store.path.stem != "temp":
                    print("db committed")
                    layer.store.commit()
                else:
                    layer.store.commit()
                    layer.store.dump(str(save_path))
                    print(f"db saved to {save_path}")
                return "done"
        return "nothing to save"
