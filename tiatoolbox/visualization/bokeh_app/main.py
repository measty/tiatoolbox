import inspect
import json
import operator
import os
import pickle
import sys
import urllib
from cmath import pi
from pathlib import Path, PureWindowsPath
from shutil import rmtree
from threading import Thread
from typing import Callable

import matplotlib.cm as cm
import numpy as np
import requests
import torch

# Bokeh basics
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import (
    BasicTickFormatter,
    BoxEditTool,
    Button,
    CheckboxButtonGroup,
    Circle,
    ColorBar,
    ColorPicker,
    ColumnDataSource,
    Dropdown,
    FuncTickFormatter,
    GraphRenderer,
    HoverTool,
    LinearColorMapper,
    MultiChoice,
    Panel,
    PointDrawTool,
    RadioGroup,
    Slider,
    Spinner,
    StaticLayoutProvider,
    TapTool,
    TextInput,
    Toggle,
)
from bokeh.models.tiles import WMTSTileSource
from bokeh.plotting import figure
from bokeh.util import token
from flask_cors import CORS
from requests.adapters import HTTPAdapter, Retry

from tiatoolbox.annotation.dsl import SQL_GLOBALS, SQLTriplet
from tiatoolbox.models.architecture.nuclick import NuClick
from tiatoolbox.models.engine.interactive_segmentor import (
    InteractiveSegmentor,
    IOInteractiveSegmentorConfig,
)
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.tools.pyramid import ZoomifyGenerator
from tiatoolbox.utils.visualization import AnnotationRenderer, random_colors
from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.visualization.ui_utils import get_level_by_extent
from tiatoolbox.wsicore.wsireader import WSIReader

is_deployed = False
# global active
active = 0
rand_id = token.generate_session_id()
print(f"rand id is: {rand_id}")

if is_deployed:
    host = os.environ.get("HOST")
    host2 = os.environ.get("HOST2")
    port = os.environ.get("PORT")
else:
    host = "127.0.0.1"
    host2 = "127.0.0.1"
    port = "5000"


class DummyWidget:
    def __init__(self, item):
        self.item = item


# Define helper functions


def make_ts(route):
    sf = 2 ** (vstate[active].num_zoom_levels - 9)
    ts = WMTSTileSource(
        name="WSI provider",
        url=route,
        attribution="",
        snap_to_zoom=False,
        min_zoom=0,
        max_zoom=vstate[active].num_zoom_levels - 1,
    )
    ts.tile_size = 256
    ts.initial_resolution = (
        40211.5 * sf * (2 / (100 * pi))  # * (256/512)
    )  # 156543.03392804097    40030 great circ
    ts.x_origin_offset = 0  # 5000000
    # ts.y_origin_offset=-2500000
    ts.y_origin_offset = sf * 10294144.78 * (2 / (100 * pi))
    ts.wrap_around = False
    # ts.max_zoom=10
    # ts.min_zoom=10
    return ts


def to_float_rgb(rgb):
    """Helper to convert from int to float rgb(a) tuple"""
    return tuple(v / 255 for v in rgb)


def to_int_rgb(rgb):
    """Helper to convert from float to int rgb(a) tuple"""
    return tuple(int(v * 255) for v in rgb)


def name2type(name):
    try:
        return int(name)
    except:
        return f'"{name}"'


def name2type_key(name):
    try:
        return int(name)
    except:
        return f"{name}"


def hex2rgb(hex_val):
    return tuple(int(hex_val[i : i + 2], 16) / 255 for i in (1, 3, 5))


def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*to_int_rgb(rgb))


def make_color_seq_from_cmap(cmap):
    if cmap is None:
        return [
            rgb2hex((1.0, 1.0, 1.0)),
            rgb2hex((1.0, 1.0, 1.0)),
        ]  # no colors if using dict
    colors = []
    for v in np.linspace(0, 1, 20):
        colors.append(rgb2hex(cmap(v)))
    return colors


def make_safe_name(name):
    return urllib.parse.quote(str(PureWindowsPath(name)), safe="")


def update_mapper():
    if vstate[active].types is not None:
        type_cmap_select.options = [str(t) for t in vstate[active].types]
        colors = random_colors(len(vstate[active].types))
        vstate[active].mapper = {
            key: (*color, 1) for key, color in zip(vstate[active].types, colors)
        }
        renderer.mapper = lambda x: vstate[active].mapper[x]
        update_renderer("mapper", vstate[active].mapper)


def update_renderer(prop, value):
    if prop == "mapper":
        if value == "dict" or isinstance(value, dict):
            value = vstate[active].mapper  # put the mapper dict back
            color_bar.color_mapper.palette = make_color_seq_from_cmap(None)
        else:
            color_bar.color_mapper.palette = make_color_seq_from_cmap(
                cm.get_cmap(value)
            )
            color_bar.visible = True
        return s.get(f"http://{host2}:5000/tileserver/changecmap/{value}")
    return s.get(
        f"http://{host2}:5000/tileserver/updaterenderer/{prop}/{json.dumps(value)}"
    )


def add_post_proc(layer, fn: Callable, kwargs: dict = None):
    """add a post-processing function to a layer"""
    s.post(
        f"http://{host2}:5000/tileserver/addpostproc/{layer}/{fn.__name__}",
        data={
            "fn_source": json.dumps(inspect.getsource(fn)),
            "kwargs": json.dumps(kwargs),
        },
    )


def build_predicate():
    """Builds a predicate function from the currently selected types,
    and the filter input.
    """
    preds = [
        eval(f'props["type"]=={name2type(l.label)}', SQL_GLOBALS, {})
        for l in box_column.children
        if l.active
    ]
    if len(preds) == len(box_column.children):
        preds = []
    combo = None
    if len(preds) > 0:
        combo = preds[0]
        for pred in preds[1:]:
            combo = SQLTriplet(combo, operator.or_, pred)
    if filter_input.value != "None":
        combo = SQLTriplet(
            eval(filter_input.value, SQL_GLOBALS, {}), operator.and_, combo
        )

    vstate[active].renderer.where = combo
    update_renderer("where", combo)
    return combo


def build_predicate_callable():
    get_types = [name2type_key(l.label) for l in box_column.children if l.active]
    if len(get_types) == len(box_column.children) or len(get_types) == 0:
        if filter_input.value == "None":
            vstate[active].renderer.where = None
            update_renderer("where", "None")
            return None

    if filter_input.value == "None":
        if len(get_types) == 0:
            pred = None
        else:

            def pred(props):
                return props["type"] in get_types

    else:
        if len(get_types) == 0:

            def pred(props):
                return eval(filter_input.value)

        else:

            def pred(props):
                return eval(filter_input.value) and props["type"] in get_types

    vstate[active].renderer.where = pred
    # update_renderer("where", json.dumps(pred))
    s.post(
        f"http://{host2}:5000/tileserver/updatewhere",
        data={"types": json.dumps(get_types), "filter": json.dumps(filter_input.value)},
    )
    return pred


def make_plot_win(p=None):
    if p is None:
        return figure(
            x_range=(0, vstate[active].dims[0]),
            y_range=(0, -vstate[active].dims[1]),
            x_axis_type="linear",
            y_axis_type="linear",
            width=1700,
            height=1000,
            # max_width=1700,
            # max_height=1000,
            # width_policy="max",
            # height_policy="max",
            # tooltips=TOOLTIPS,
            tools="pan,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
            output_backend="canvas",
            hidpi=True,
            match_aspect=False,
            # lod_factor=100,
            # lod_interval=500,
            # lod_threshold=10,
            # lod_timeout=200,
            sizing_mode="stretch_both",
            # name="slide_window",
        )
    return figure(
        x_range=p.x_range,
        y_range=p.y_range,
        x_axis_type="linear",
        y_axis_type="linear",
        width=1700,
        height=1000,
        # max_width=1700,
        # max_height=1000,
        # width_policy="max",
        # height_policy="max",
        # tooltips=TOOLTIPS,
        tools="pan,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
        output_backend="canvas",
        hidpi=True,
        match_aspect=False,
        # lod_factor=100,
        # lod_interval=500,
        # lod_threshold=10,
        # lod_timeout=200,
        sizing_mode="stretch_both",
        # name="slide_window",
    )


# ps[active].axis.visible = False


def initialise_slide():
    vstate[active].mpp = wsi[0].info.mpp
    if vstate[active].mpp is None:
        vstate[active].mpp = [1, 1]
    vstate[active].dims = wsi[0].info.slide_dimensions
    vstate[active].types = []
    vstate[active].props = []
    pad = int(np.mean(vstate[active].dims) / 10)
    plot_size = np.array([ps[active].width, ps[active].height])
    aspect_ratio = plot_size[0] / plot_size[1]
    large_dim = np.argmax(np.array(vstate[active].dims) / plot_size)
    import pdb

    pdb.set_trace()
    vstate[active].micron_formatter.args["mpp"] = vstate[active].mpp[0]
    if large_dim == 1:
        ps[active].x_range.start = (
            -0.5 * (vstate[active].dims[1] * aspect_ratio - vstate[active].dims[0])
            - aspect_ratio * pad
        )
        ps[active].x_range.end = (
            vstate[active].dims[1] * aspect_ratio
            - 0.5 * (vstate[active].dims[1] * aspect_ratio - vstate[active].dims[0])
            + aspect_ratio * pad
        )
        ps[active].y_range.start = -vstate[active].dims[1] - pad
        ps[active].y_range.end = pad
        # ps[active].x_range.min_interval = ?
    else:
        ps[active].x_range.start = -aspect_ratio * pad
        ps[active].x_range.end = vstate[active].dims[0] + pad * aspect_ratio
        ps[active].y_range.start = (
            -vstate[active].dims[0] / aspect_ratio
            + 0.5 * (vstate[active].dims[0] / aspect_ratio - vstate[active].dims[1])
            - pad
        )
        ps[active].y_range.end = (
            0.5 * (vstate[active].dims[0] / aspect_ratio - vstate[active].dims[1]) + pad
        )

    # ps[active].x_range.bounds = (ps[active].x_range.start - 2 * pad, ps[active].x_range.end + 2 * pad)
    # ps[active].y_range.bounds = (ps[active].y_range.start - 2 * pad, ps[active].y_range.end + 2 * pad)
    # ps[active]._trigger_event()

    z = ZoomifyGenerator(wsi[0], tile_size=256)
    vstate[active].num_zoom_levels = z.level_count
    print(f"nzoom_levs: {vstate[active].num_zoom_levels}")
    zlev = get_level_by_extent((0, ps[active].y_range.start, ps[active].x_range.end, 0))
    print(f"initial_zoom: {zlev}")
    print(wsi[0].info.as_dict())


def initialise_overlay():
    vstate[active].colors = list(vstate[active].mapper.values())
    vstate[active].types = [
        str(t) for t in vstate[active].types
    ]  # vstate[active].mapper.keys()]
    now_active = {b.label: b.active for b in box_column.children}
    print(vstate[active].types)
    print(now_active)
    for t in vstate[active].types:
        if str(t) not in now_active.keys():
            box_column.children.append(
                Toggle(
                    label=str(t),
                    active=True,
                    width=130,
                    height=30,
                    max_width=130,
                    sizing_mode="stretch_width",
                )
            )
            box_column.children[-1].on_click(layer_select_cb)
            try:
                color_column.children.append(
                    ColorPicker(
                        color=to_int_rgb(vstate[active].mapper[t][0:3]),
                        name=str(t),
                        width=60,
                        max_width=60,
                        height=30,
                        sizing_mode="stretch_width",
                    )
                )
            except KeyError:
                color_column.children.append(
                    ColorPicker(
                        color=to_int_rgb(vstate[active].mapper[int(t)][0:3]),
                        name=str(t),
                        width=60,
                        max_width=60,
                        sizing_mode="stretch_width",
                    )
                )
            color_column.children[-1].on_change(
                "color", bind_cb_obj(color_column.children[-1], color_input_cb)
            )

    for b in box_column.children.copy():
        if (
            b.label not in vstate[active].types
            and b.label not in vstate[active].layer_dict.keys()
        ):
            print(f"removing {b.label}")
            box_column.children.remove(b)
    for c in color_column.children.copy():
        if c.name not in vstate[active].types and "slider" not in c.name:
            color_column.children.remove(c)

    build_predicate_callable()


def add_layer(lname):
    box_column.children.append(
        Toggle(
            label=lname,
            active=True,
            width=130,
            max_width=130,
            sizing_mode="stretch_width",
        )
    )
    box_column.children[-1].on_click(
        bind_cb_obj_tog(box_column.children[-1], fixed_layer_select_cb)
    )
    color_column.children.append(
        Slider(
            start=0,
            end=1,
            value=0.5,
            step=0.01,
            title=lname,
            width=100,
            max_width=90,
            sizing_mode="stretch_width",
            name=f"{lname}_slider",
        )
    )
    color_column.children[-1].on_change(
        "value", bind_cb_obj(color_column.children[-1], layer_slider_cb)
    )

    # layer_boxes=[Toggle(label=t, active=a, width=100) for t,a in now_active.items()]
    # lcolors=[ColorPicker(color=col[0:3], name=t, width=60) for col, t in zip(vstate[active].colors, vstate[active].types)]


class TileGroup:
    def __init__(self):
        self.group = 1

    def get_grp(self):
        self.group = self.group + 1
        return self.group


tg = TileGroup()


def change_tiles(layer_name="overlay"):

    grp = tg.get_grp()

    if layer_name == "graph" and layer_name not in vstate[active].layer_dict.keys():
        ps[active].renderers.append(graph)
        vstate[active].layer_dict[layer_name] = (
            len(ps[active].renderers) + len(p_linked.renderers) - 1
        )
        for layer_key in vstate[active].layer_dict.keys():
            if layer_key in ["rect", "pts", "graph"]:
                continue
            grp = tg.get_grp()
            ts = make_ts(
                f"http://{host}:{port}/tileserver/layer/{layer_key}/{user}/zoomify/TileGroup{grp}"
                + r"/{z}-{x}-{y}@2x.jpg",
            )
            ps[active].renderers[vstate[active].layer_dict[layer_key]].tile_source = ts

        return
    import pdb

    pdb.set_trace()
    ts = make_ts(
        f"http://{host}:{port}/tileserver/layer/{layer_name}/{user}/zoomify/TileGroup{grp}"
        + r"/{z}-{x}-{y}@2x.jpg",
    )
    if layer_name in vstate[active].layer_dict:
        ps[active].renderers[vstate[active].layer_dict[layer_name]].tile_source = ts
    else:
        ps[active].add_tile(
            ts,
            smoothing=True,
            alpha=overlay_alpha.value,
            level="image",
            render_parents=False,
        )

        for layer_key in vstate[active].layer_dict.keys():
            if layer_key in ["rect", "pts", "graph"]:
                continue
            grp = tg.get_grp()
            ts = make_ts(
                f"http://{host}:{port}/tileserver/layer/{layer_key}/{user}/zoomify/TileGroup{grp}"
                + r"/{z}-{x}-{y}@2x.jpg",
            )
            ps[active].renderers[vstate[active].layer_dict[layer_key]].tile_source = ts
        vstate[active].layer_dict[layer_name] = len(ps[active].renderers) - 1

    print(vstate[active].layer_dict)
    print(ps[active].renderers)


class ViewerState:
    def __init__(self):
        self.dims = [30000, 20000]
        self.mpp = None
        self.mapper = {}
        self.colors = list(self.mapper.values())
        self.types = list(self.mapper.keys())
        self.layer_dict = {"slide": 0, "rect": 1, "pts": 2}
        self.renderer = []
        self.num_zoom_levels = 0
        self.slide_path = None
        self.update_state = 0
        self.model_mpp = 0
        self.micron_formatter = None
        self.current_model = "hovernet"
        self.props = []
        self.props_old = []


vstate = {}
vstate[0] = ViewerState()
vstate[1] = ViewerState()

# base_folder = r"E:\TTB_vis_folder"
base_folder = "/app_data"
if len(sys.argv) > 1 and sys.argv[1] != "None":
    base_folder = Path(sys.argv[1])
    slide_folder = base_folder.joinpath("slides")
    overlay_folder = base_folder.joinpath("overlays")
if len(sys.argv) == 3:
    slide_folder = Path(sys.argv[1])
    overlay_folder = Path(sys.argv[2])

# vstate[active].slide_path = r"E:\\TTB_vis_folder\\slides\\TCGA-SC-A6LN-01Z-00-DX1.svs"
# vstate[active].slide_path=Path(r'/tiatoolbox/app_data/slides/TCGA-SC-A6LN-01Z-00-DX1.svs')

# set initial slide to first one in base folder
slide_list = []
for ext in ["*.svs", "*ndpi", "*.tiff", "*.mrxs", "*.jpg"]:  # ,'*.png','*.jpg']:
    slide_list.extend(list(slide_folder.glob(ext)))
    slide_list.extend(list(slide_folder.glob(str(Path("*") / ext))))
vstate[active].slide_path = slide_list[0]

renderer = AnnotationRenderer(
    "type",
    {"class1": (1, 0, 0, 1), "class2": (0, 0, 1, 1), "class3": (0, 1, 0, 1)},
    thickness=-1,
    edge_thickness=1,
    zoomed_out_strat="scale",
    max_scale=8,
)
vstate[active].renderer = renderer

wsi = [WSIReader.open(vstate[active].slide_path)]
vstate[active].dims = wsi[0].info.slide_dimensions
vstate[active].mpp = wsi[0].info.mpp
if vstate[active].mpp is None:
    vstate[active].mpp = [1, 1]


def run_app():

    app = TileServer(
        title="Testing TileServer",
        layers={
            # "slide": wsi[0],
        },
    )
    CORS(app, send_wildcard=True)
    app.run(host="127.0.0.1", threaded=False)


# start tile server
if not is_deployed:
    proc = Thread(target=run_app, daemon=True)
    proc.start()


# set up main window
vstate[active].micron_formatter = FuncTickFormatter(
    args={"mpp": 0.1},
    code="""
    return Math.round(tick*mpp)
    """,
)

do_feats = False

p = make_plot_win()
p_linked = make_plot_win(p)
ps = [p, p_linked]
slide_wins = row(
    children=ps,
    name="slide_window",
    sizing_mode="stretch_both",
)
initialise_slide()

s = requests.Session()

retries = Retry(
    total=5,
    backoff_factor=0.1,
)
# status_forcelist=[ 500, 502, 503, 504 ])
s.mount("http://", HTTPAdapter(max_retries=retries))

resp = s.get(f"http://{host2}:5000/tileserver/setup")
print(f"cookies are: {resp.cookies}")
cookies = resp.cookies
user = resp.cookies.get("user")

ts1 = make_ts(
    f"http://{host}:{port}/tileserver/layer/slide/{user}/zoomify/TileGroup1"
    + r"/{z}-{x}-{y}@2x.jpg",
)
print(ps[active].renderers)
print(ps[active].y_range)
ps[active].add_tile(ts1, smoothing=True, level="image", render_parents=False)
ps[active - 1].add_tile(ts1, smoothing=True, level="image", render_parents=False)
print(ps[active].y_range)
print(f"max zoom is: {ps[active].renderers[0].tile_source.max_zoom}")

ps[active].grid.grid_line_color = None
ps[active - 1].grid.grid_line_color = None
box_source = ColumnDataSource({"x": [], "y": [], "width": [], "height": []})
pt_source = ColumnDataSource({"x": [], "y": []})
r = ps[active].rect("x", "y", "width", "height", source=box_source, fill_alpha=0)
c = ps[active].circle("x", "y", source=pt_source, color="red", size=5)
hover = HoverTool()
ps[active].add_tools(BoxEditTool(renderers=[r], num_objects=1))
ps[active].add_tools(PointDrawTool(renderers=[c]))
ps[active].add_tools(TapTool(), hover)
tslist = []

ps[active].renderers[0].tile_source.max_zoom = 10

node_source = ColumnDataSource({"index": [], "node_color": []})
edge_source = ColumnDataSource({"start": [], "end": []})
graph = GraphRenderer()  # level='overlay')
graph.node_renderer.data_source = node_source
graph.edge_renderer.data_source = edge_source
graph.node_renderer.glyph = Circle(
    radius=50, radius_units="data", fill_color="node_color"
)


# Define UI elements
slide_alpha = Slider(
    title="Slide Alpha",
    start=0,
    end=1,
    step=0.05,
    value=1.0,
    width=200,
    max_width=200,
    sizing_mode="stretch_width",
)

overlay_alpha = Slider(
    title="Overlay Alpha",
    start=0,
    end=1,
    step=0.05,
    value=0.75,
    width=200,
    height=40,
    max_width=200,
    sizing_mode="stretch_width",
)

pt_size_spinner = Spinner(
    title="Pt. Size:",
    low=0,
    high=10,
    step=1,
    value=1,
    width=60,
    max_width=60,
    sizing_mode="stretch_width",
)

color_bar = ColorBar(
    color_mapper=LinearColorMapper(make_color_seq_from_cmap(cm.get_cmap("viridis"))),
    label_standoff=12,
)
# ps[active].add_layout(color_bar, 'below')
slide_toggle = Toggle(
    label="Slide",
    button_type="success",
    width=90,
    max_width=90,
    sizing_mode="stretch_width",
)
overlay_toggle = Toggle(
    label="Overlay",
    button_type="success",
    width=90,
    max_width=90,
    height=40,
    sizing_mode="stretch_width",
)
filter_input = TextInput(
    value="None", title="Filter:", max_width=300, sizing_mode="stretch_width"
)
# cprop_input = TextInput(
#    value="type", title="CProp:", max_width=300, sizing_mode="stretch_width"
# )
cprop_input = MultiChoice(
    title="Colour by:",
    max_items=1,
    options=["*"],
    search_option_limit=5000,
    sizing_mode="stretch_width",
    max_width=300,
)
active_win_buttons = RadioGroup(labels=["Left", "Right"], active=0)
slide_select = MultiChoice(
    title="Select Slide:",
    max_items=1,
    options=["*"],
    search_option_limit=5000,
    max_width=300,
    sizing_mode="stretch_width",
)
cmmenu = [
    ("jet", "jet"),
    ("coolwarm", "coolwarm"),
    ("viridis", "viridis"),
    ("dict", "dict"),
]
cmap_drop = Dropdown(
    label="Colourmap",
    button_type="warning",
    menu=cmmenu,
    width=100,
    max_width=100,
    height=45,
    sizing_mode="stretch_width",
)
blur_spinner = Spinner(
    title="Blur:",
    low=0,
    high=10,
    step=1,
    value=0,
    width=80,
    max_width=80,
    sizing_mode="stretch_width",
)
scale_spinner = Spinner(
    title="max_scale:",
    low=0,
    high=540,
    step=8,
    value=16,
    width=80,
    max_width=80,
    sizing_mode="stretch_width",
)
to_model_button = Button(
    label="Go",
    button_type="success",
    width=60,
    max_width=60,
    sizing_mode="stretch_width",
)
model_drop = Dropdown(
    label="Choose Model",
    button_type="warning",
    menu=["hovernet", "nuclick"],
    width=120,
    max_width=120,
    sizing_mode="stretch_width",
)
type_cmap_select = MultiChoice(
    title="Colour type by property:",
    max_items=2,
    options=["*"],
    search_option_limit=5000,
    sizing_mode="stretch_width",
    max_width=300,
)
swap_button = Button(
    label="Swap feat/importance",
    button_type="success",
    width=140,
    max_width=140,
    sizing_mode="stretch_width",
    height=40,
)
layer_boxes = [
    Toggle(label=t, active=True, width=100, max_width=100, sizing_mode="stretch_width")
    for t in vstate[active].types
]
lcolors = [
    ColorPicker(color=col[0:3], width=60, max_width=60, sizing_mode="stretch_width")
    for col in vstate[active].colors
]
layer_folder_input = TextInput(
    value=str(overlay_folder),
    title="Overlay Folder:",
    max_width=300,
    sizing_mode="stretch_width",
)
layer_drop = Dropdown(
    label="Add Overlay",
    button_type="warning",
    menu=[None],
    max_width=300,
    sizing_mode="stretch_width",
)
opt_buttons = CheckboxButtonGroup(
    labels=["Filled", "Microns", "Grid"],
    active=[0],
    max_width=300,
    sizing_mode="stretch_width",
)
save_button = Button(
    label="Save", button_type="success", max_width=90, sizing_mode="stretch_width"
)


# Define UI callbacks
def slide_toggle_cb(attr):
    if ps[active].renderers[0].alpha == 0:
        ps[active].renderers[0].alpha = slide_alpha.value
    else:
        ps[active].renderers[0].alpha = 0.0


def node_select_cb(attr, old, new):
    # only used for old slidegraph clustering
    print(f"selected is: {new}")
    vstate[active].mapper = {new[0]: (1, 0, 0, 1)}
    vstate[active].renderer.mapper = lambda x: vstate[active].mapper[x]
    update_renderer("mapper", vstate[active].mapper)
    vstate[active].update_state = 1


def overlay_toggle_cb(attr):
    for i in range(3, len(ps[active].renderers)):
        if isinstance(ps[active].renderers[i], GraphRenderer):
            # set_graph_alpha(ps[active].renderers[i], new)
            continue
        if ps[active].renderers[i].alpha == 0:
            ps[active].renderers[i].alpha = overlay_alpha.value
        else:
            ps[active].renderers[i].alpha = 0.0


def folder_input_cb(attr, old, new):
    populate_slide_list(slide_folder, new)


def populate_layer_list(slide_name, overlay_path: Path):
    file_list = []
    for ext in [
        "*.db",
        "*.dat",
        "*.geojson",
        "*.png",
        "*.jpg",
        "*.pkl",
        "*.tiff",
    ]:  # and '*.tiff'?
        file_list.extend(list(overlay_path.glob(str(Path("*") / ext))))
        file_list.extend(list(overlay_path.glob(ext)))
    file_list = [(str(p), str(p)) for p in sorted(file_list) if slide_name in str(p)]
    layer_drop.menu = file_list


def populate_slide_list(slide_folder, search_txt=None):
    file_list = []
    len_slidepath = len(slide_folder.parts)
    for ext in ["*.svs", "*ndpi", "*.tiff", "*.mrxs", "*.jpg"]:
        file_list.extend(list(Path(slide_folder).glob(str(Path("*") / ext))))
        file_list.extend(list(Path(slide_folder).glob(ext)))
    if search_txt is None:
        file_list = [
            (str(Path(*p.parts[len_slidepath:])), str(Path(*p.parts[len_slidepath:])))
            for p in sorted(file_list)
        ]
    else:
        file_list = [
            (str(Path(*p.parts[len_slidepath:])), str(Path(*p.parts[len_slidepath:])))
            for p in sorted(file_list)
            if search_txt in str(p)
        ]

    slide_select.options = file_list


def layer_folder_input_cb(attr, old, new):
    # unused at the moment
    file_list = []
    for ext in ["*.db", "*.dat", "*.geojson", "*.png", "*.jpg", ".tiff"]:
        file_list.extend(list(Path(new).glob("*\\" + ext)))
    file_list = [(str(p), str(p)) for p in sorted(file_list)]
    layer_drop.menu = file_list
    return file_list


def filter_input_cb(attr, old, new):
    """Change predicate to be used to filter annotations"""
    # s.get(f"http://{host2}:5000/tileserver/changepredicate/{new}")
    build_predicate_callable()
    vstate[active].update_state = 1


def cprop_input_cb(attr, old, new):
    """Change property to colour by"""
    s.get(f"http://{host2}:5000/tileserver/changeprop/{new[0]}")
    vstate[active].update_state = 1


def set_graph_alpha(g_renderer, value):
    # set all components of graph to given alpha value
    g_renderer.node_renderer.glyph.fill_alpha = value
    g_renderer.node_renderer.glyph.line_alpha = value
    g_renderer.edge_renderer.glyph.line_alpha = value


def slide_alpha_cb(attr, old, new):
    print("meep")
    ps[active].renderers[0].alpha = new
    # ps[active].renderers[0].tile_source.max_zoom=7
    # ps[active].renderers[1].tile_source.max_zoom=7


def overlay_alpha_cb(attr, old, new):
    print("meep")
    for i in range(3, len(ps[active].renderers)):
        if isinstance(ps[active].renderers[i], GraphRenderer):
            # set_graph_alpha(ps[active].renderers[i], new)
            pass
        else:
            ps[active].renderers[i].alpha = new


def pt_size_cb(attr, old, new):
    update_renderer("edge_thickness", new)
    graph.node_renderer.glyph.radius = 20 * new
    vstate[active].update_state = 1


def opt_buttons_cb(attr, old, new):
    old_thickness = vstate[active].renderer.thickness
    if 0 in new:
        vstate[active].renderer.thickness = -1
        update_renderer("thickness", -1)
    else:
        vstate[active].renderer.thickness = 1
        update_renderer("thickness", 1)
    if old_thickness != vstate[active].renderer.thickness:
        vstate[active].update_state = 1
    if 1 in new:
        ps[active].xaxis[0].formatter = vstate[active].micron_formatter
        ps[active].yaxis[0].formatter = vstate[active].micron_formatter
    else:
        ps[active].xaxis[0].formatter = BasicTickFormatter()
        ps[active].yaxis[0].formatter = BasicTickFormatter()
    if 2 in new:
        ps[active].ygrid.grid_line_color = "gray"
        ps[active].xgrid.grid_line_color = "gray"
        ps[active].ygrid.grid_line_alpha = 0.6
        ps[active].xgrid.grid_line_alpha = 0.6
    else:
        ps[active].ygrid.grid_line_alpha = 0
        ps[active].xgrid.grid_line_alpha = 0
    print(ps[active].ygrid)
    print(ps[active].grid)


def cmap_drop_cb(attr):
    update_renderer("mapper", attr.item)
    # change_tiles('overlay')
    vstate[active].update_state = 1


def blur_spinner_cb(attr, old, new):
    update_renderer("blur_radius", new)
    vstate[active].update_state = 1


def scale_spinner_cb(attr, old, new):
    update_renderer("max_scale", new)
    vstate[active].update_state = 1


def slide_select_cb(attr, old, new):
    print(active)
    """setup the newly chosen slide"""
    if len(new) == 0:
        return
    slide_path = Path(slide_folder) / Path(new[0])
    if active == 1:
        layer_drop_cb(DummyWidget(slide_path))
        return
    pt_source.data = {"x": [], "y": []}
    box_source.data = {"x": [], "y": [], "width": [], "height": []}
    if len(ps[active].renderers) > 3:
        for r in ps[active].renderers[3:].copy():
            ps[active].renderers.remove(r)
    vstate[active].layer_dict = {"slide": 0, "rect": 1, "pts": 2}
    vstate[active].slide_path = slide_path
    """
    for c in color_column.children.copy():
        if "_slider" in c.name:
            color_column.children.remove(c)
    for b in box_column.children.copy():
        if "layer" in b.label or "graph" in b.label:
            box_column.children.remove(b)
    """
    color_column.children = []
    box_column.children = []
    print(ps[active].renderers)
    print(slide_path)
    populate_layer_list(slide_path.stem, overlay_folder)
    wsi[0] = WSIReader.open(slide_path)
    initialise_slide()
    # fname='-*-'.join(attr.item.split('\\'))
    fname = make_safe_name(str(slide_path))
    print(fname)
    print(vstate[active].mpp)
    s.get(f"http://{host2}:5000/tileserver/changeslide/slide/{fname}")
    change_tiles("slide")
    # if len(ps[active].renderers)==1:
    # r=ps[active].rect('x', 'y', 'width', 'height', source=box_source, fill_alpha=0)
    # ps[active].add_tools(BoxEditTool(renderers=[r], num_objects=1))
    # ps[active].x_range.bounds=MinMaxBounds(0,vstate[active].dims[0])
    # ps[active].y_range.bounds=(0,-vstate[active].dims[1])


def layer_drop_cb(attr):
    """setup the newly chosen overlay"""
    if Path(attr.item).suffix == ".pkl":
        # its a graph
        do_feats = False
        with open(attr.item, "rb") as f:
            graph_dict = pickle.load(f)
        node_cm = cm.get_cmap("viridis")
        num_nodes = graph_dict["coordinates"].shape[0]
        if "score" in graph_dict:
            node_source.data = {
                "index": list(range(num_nodes)),
                "node_color": [rgb2hex(node_cm(v)) for v in graph_dict["score"]],
            }
        else:
            # default to green
            node_source.data = {
                "index": list(range(num_nodes)),
                "node_color": [rgb2hex((0, 1, 0))] * num_nodes,
            }
        edge_source.data = {
            "start": graph_dict["edge_index"][0, :],
            "end": graph_dict["edge_index"][1, :],
        }

        graph_layout = dict(
            zip(
                node_source.data["index"],
                [
                    # (x / (4 * vstate[active].mpp[0]), -y / (4 * vstate[active].mpp[1]))
                    (x, -y)
                    for x, y in graph_dict["coordinates"]
                ],
            )
        )
        graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        add_layer("graph")
        change_tiles("graph")

        # add additional data to graph datasource
        for key in graph_dict:
            if key == "feat_names":
                graph_feat_names = graph_dict[key]
                do_feats = True
            try:
                if (
                    key in ["edge_index", "coordinates"]
                    or len(graph_dict[key]) != num_nodes
                ):
                    continue
            except TypeError:
                continue  # not arraylike, cant add to node data
            node_source.data[key] = graph_dict[key]

        if do_feats:
            for i in range(graph_dict["feats"].shape[1]):
                if i > 9:
                    break  # more than 10 wont really fit, ignore rest
                node_source.data[graph_feat_names[i]] = graph_dict["feats"][:, i]

            TOOLTIPS = [
                ("Index", "$index"),
                ("(x,y)", "($x, $y)"),
            ]
            TOOLTIPS.extend(
                [
                    (graph_feat_names[i], f"@{graph_feat_names[i]}")
                    for i in range(np.minimum(graph_dict["feats"].shape[1], 9))
                ]
            )
            hover.tooltips = TOOLTIPS

        return

    # fname='-*-'.join(attr.item.split('\\'))
    fname = make_safe_name(attr.item)
    resp = s.get(f"http://{host2}:5000/tileserver/changeoverlay/{fname}")
    resp = json.loads(resp.text)

    if Path(attr.item).suffix in [".db", ".dat", ".geojson"]:
        vstate[active].types = resp
        props = s.get(f"http://{host2}:5000/tileserver/getprops")
        vstate[active].props = json.loads(props.text)
        # type_cmap_select.options = vstate[active].props
        cprop_input.options = vstate[active].props
        cprop_input.options.append("None")
        if not vstate[active].props == vstate[active].props_old:
            update_mapper()
            vstate[active].props_old = vstate[active].props
        initialise_overlay()
        change_tiles("overlay")
    else:
        add_layer(resp)
        change_tiles(resp)


def layer_select_cb(attr):
    build_predicate_callable()
    # change_tiles('overlay')
    vstate[active].update_state = 1


def fixed_layer_select_cb(obj, attr):
    print(vstate[active].layer_dict)
    key = vstate[active].layer_dict[obj.label]
    if obj.label == "graph":
        if ps[active].renderers[key].node_renderer.glyph.fill_alpha == 0:
            ps[active].renderers[
                key
            ].node_renderer.glyph.fill_alpha = overlay_alpha.value
            ps[active].renderers[
                key
            ].node_renderer.glyph.line_alpha = overlay_alpha.value
            ps[active].renderers[
                key
            ].edge_renderer.glyph.line_alpha = overlay_alpha.value
        else:
            ps[active].renderers[key].node_renderer.glyph.fill_alpha = 0.0
            ps[active].renderers[key].node_renderer.glyph.line_alpha = 0.0
            ps[active].renderers[key].edge_renderer.glyph.line_alpha = 0.0
    else:
        if ps[active].renderers[key].alpha == 0:
            ps[active].renderers[key].alpha = overlay_alpha.value
        else:
            ps[active].renderers[key].alpha = 0.0


def layer_slider_cb(obj, attr, old, new):
    if isinstance(
        ps[active].renderers[vstate[active].layer_dict[obj.name.split("_")[0]]],
        GraphRenderer,
    ):
        set_graph_alpha(
            ps[active].renderers[vstate[active].layer_dict[obj.name.split("_")[0]]], new
        )
    else:
        ps[active].renderers[
            vstate[active].layer_dict[obj.name.split("_")[0]]
        ].alpha = new


def color_input_cb(obj, attr, old, new):
    print(new)
    vstate[active].mapper[name2type_key(obj.name)] = (*hex2rgb(new), 1)
    if vstate[active].renderer.score_prop == "type":
        vstate[active].renderer.mapper = lambda x: vstate[active].mapper[x]
        update_renderer("mapper", vstate[active].mapper)
    # change_tiles('overlay')
    vstate[active].update_state = 1


def bind_cb_obj(cb_obj, cb):
    def wrapped(attr, old, new):
        cb(cb_obj, attr, old, new)

    return wrapped


def bind_cb_obj_tog(cb_obj, cb):
    def wrapped(attr):
        cb(cb_obj, attr)

    return wrapped


def swap_cb(attr):
    val = type_cmap_select.value
    if len(val) == 0:
        return
    if "_exp" in val[0]:
        type_cmap_select.value = [val[0][:-4]]
    else:
        type_cmap_select.value = [val[0] + "_exp"]
    type_cmap_cb(None, None, type_cmap_select.value)


def model_drop_cb(attr):
    vstate[active].current_model = attr.item


def to_model_cb(attr):
    if vstate[active].current_model == "hovernet":
        segment_on_box(attr)
    elif vstate[active].current_model == "nuclick":
        nuclick_on_pts(attr)
    else:
        print("unknown model")


def type_cmap_cb(attr, old, new):
    if len(new) == 0:
        type_cmap_select.options = vstate[active].types
        s.get(
            f"http://{host2}:5000/tileserver/changesecondarycmap/{'None'}/{'None'}/viridis"
        )
        vstate[active].update_state = 1
        return
    if len(new) == 1:
        # find out what still has to be selected
        if new[0] in vstate[active].types:
            type_cmap_select.options = vstate[active].props + [new[0]]
        elif new[0] in vstate[active].props:
            type_cmap_select.options = vstate[active].types + [new[0]]
    else:
        # both selected, update the renderer
        if new[1] in vstate[active].types:
            type_cmap_select.value = [new[1], new[0]]
            return
        s.get(
            f"http://{host2}:5000/tileserver/changesecondarycmap/{new[0]}/{new[1]}/viridis"
        )

        color_bar.color_mapper.palette = make_color_seq_from_cmap(
            cm.get_cmap("viridis")
        )
        color_bar.visible = True
        vstate[active].update_state = 1


def save_cb(attr):
    save_path = make_safe_name(
        str(overlay_folder / (vstate[active].slide_path.stem + "_saved_anns.db"))
    )
    s.get(f"http://{host2}:5000/tileserver/commit/{save_path}")


def active_win_cb(attr, old, new):
    global active
    active = new


# run NucleusInstanceSegmentor on a region of wsi defined by the box in box_source
def segment_on_box(attr):
    print(vstate[active].types)
    # thumb=wsi[0].slide_thumbnail(resolution=8, units='mpp')
    thumb = wsi[0].slide_thumbnail()
    # conv_mpp=wsi.convert_resolution_units(1.25, 'power', 'mpp')[0]
    conv_mpp = vstate[active].dims[0] / thumb.shape[1]
    print(f'box tl: {box_source.data["x"][0]}, {box_source.data["y"][0]}')
    x = round((box_source.data["x"][0] - 0.5 * box_source.data["width"][0]) / conv_mpp)
    y = -round(
        (box_source.data["y"][0] + 0.5 * box_source.data["height"][0]) / conv_mpp
    )
    width = round(box_source.data["width"][0] / conv_mpp)
    height = round(box_source.data["height"][0] / conv_mpp)
    print(x, y, width, height)

    # img_tile=wsi.read_rect((x,y),(width,height))
    mask = np.zeros((thumb.shape[0], thumb.shape[1]))
    mask[y : y + height, x : x + width] = 1

    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=6,
        num_postproc_workers=12,
        batch_size=24,
    )

    vstate[active].model_mpp = inst_segmentor.ioconfig.save_resolution["resolution"]
    tile_output = inst_segmentor.predict(
        [vstate[active].slide_path],
        [mask],
        save_dir="sample_tile_results/",
        mode="wsi",
        # resolution=vstate[active].mpp,
        # units='mpp',
        on_gpu=True,
        crash_on_exception=True,
    )

    # fname='-*-'.join('.\\sample_tile_results\\0.dat'.split('\\'))
    fname = make_safe_name(".\\sample_tile_results\\0.dat")
    print(fname)
    resp = s.get(
        f"http://{host2}:5000/tileserver/loadannotations/{fname}/{vstate[active].model_mpp}"
    )
    vstate[active].types = json.loads(resp.text)
    update_mapper()
    # type_drop.menu=[(str(t),str(t)) for t in vstate[active].types]
    rmtree(r"./sample_tile_results")
    initialise_overlay()
    change_tiles("overlay")

    return tile_output


# run nuclick on user selected points in pt_source
def nuclick_on_pts(attr):
    x = np.round(np.array(pt_source.data["x"]))
    y = -np.round(np.array(pt_source.data["y"]))

    model = NuClick(5, 1)
    pretrained_weights = r"C:\Users\meast\app_data\NuClick_Nuclick_40xAll.pth"
    saved_state_dict = torch.load(pretrained_weights, map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)
    vstate[active].model_mpp = 0.25
    ioconf = IOInteractiveSegmentorConfig(
        input_resolutions=[{"resolution": 0.25, "units": "mpp"}], patch_size=(128, 128)
    )
    inst_segmentor = InteractiveSegmentor(
        num_loader_workers=0,
        batch_size=16,
        model=model,
    )

    points = np.vstack([x, y]).T
    points = points / (
        ioconf.input_resolutions[0]["resolution"] / vstate[active].mpp[0]
    )
    print(points.shape)
    nuclick_output = inst_segmentor.predict(
        [vstate[active].slide_path],
        [points],
        ioconfig=ioconf,
        save_dir="/app_data/sample_tile_results/",
        patch_size=(128, 128),
        resolution=0.25,
        units="mpp",
        on_gpu=False,
        save_output=True,
    )

    # fname='-*-'.join('.\\sample_tile_results\\0.dat'.split('\\'))
    fname = make_safe_name("\\app_data\\sample_tile_results\\0.dat")
    print(fname)
    resp = s.get(
        f"http://{host2}:5000/tileserver/loadannotations/{fname}/{vstate[active].model_mpp}"
    )
    print(resp.text)
    vstate[active].types = json.loads(resp.text)
    update_mapper()
    rmtree(Path(r"/app_data/sample_tile_results"))
    initialise_overlay()
    change_tiles("overlay")


# associate callback functions to the widgets
slide_alpha.on_change("value", slide_alpha_cb)
overlay_alpha.on_change("value", overlay_alpha_cb)
pt_size_spinner.on_change("value", pt_size_cb)
slide_select.on_change("value", slide_select_cb)
save_button.on_click(save_cb)
cmap_drop.on_click(cmap_drop_cb)
blur_spinner.on_change("value", blur_spinner_cb)
scale_spinner.on_change("value", scale_spinner_cb)
to_model_button.on_click(to_model_cb)
model_drop.on_click(model_drop_cb)
layer_drop.on_click(layer_drop_cb)
opt_buttons.on_change("active", opt_buttons_cb)
slide_toggle.on_click(slide_toggle_cb)
overlay_toggle.on_click(overlay_toggle_cb)
filter_input.on_change("value", filter_input_cb)
cprop_input.on_change("value", cprop_input_cb)
node_source.selected.on_change("indices", node_select_cb)
type_cmap_select.on_change("value", type_cmap_cb)
swap_button.on_click(swap_cb)
active_win_buttons.on_change("active", active_win_cb)

populate_slide_list(slide_folder)
populate_layer_list(Path(vstate[active].slide_path).stem, overlay_folder)

box_column = column(children=layer_boxes)
color_column = column(children=lcolors)

# open up first slide in list
slide_select_cb(None, None, new=[slide_list[0]])

ui_layout = column(
    [
        active_win_buttons,
        slide_select,
        layer_drop,
        row([slide_toggle, slide_alpha]),
        row([overlay_toggle, overlay_alpha, pt_size_spinner]),
        filter_input,
        cprop_input,
        # cmap_drop,
        row([cmap_drop, blur_spinner, scale_spinner]),
        type_cmap_select,
        opt_buttons,
        row([to_model_button, model_drop, save_button]),
        row(children=[box_column, color_column]),
    ],
    name="ui_layout",
    sizing_mode="stretch_both",
)


def cleanup_session(session_context):
    # If present, this function executes when the server closes a session.
    sys.exit()


def update():
    if vstate[active].update_state == 2:
        change_tiles("overlay")
        vstate[active].update_state = 0
    if vstate[active].update_state == 1:
        vstate[active].update_state = 2


curdoc().add_periodic_callback(update, 220)
curdoc().add_root(slide_wins)
curdoc().add_root(ui_layout)
