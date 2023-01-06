import json
import operator
import os
import pickle
import sys
import tempfile
import urllib
from cmath import pi
from pathlib import Path, PureWindowsPath
from shutil import rmtree
from threading import Thread

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
    CheckboxGroup,
    Circle,
    ColorBar,
    ColorPicker,
    ColumnDataSource,
    Dropdown,
    FuncTickFormatter,
    GraphRenderer,
    HoverTool,
    Line,
    LinearColorMapper,
    MultiChoice,
    PointDrawTool,
    RadioButtonGroup,
    Segment,
    Select,
    Slider,
    Spinner,
    StaticLayoutProvider,
    TabPanel,
    Tabs,
    TapTool,
    TextInput,
    Toggle,
)
from bokeh.models.tiles import WMTSTileSource
from bokeh.plotting import figure
from bokeh.util import token
from flask_cors import CORS
from requests.adapters import HTTPAdapter, Retry
from sklearn.neighbors import KernelDensity

from tiatoolbox.annotation.dsl import SQL_GLOBALS, SQLTriplet
from tiatoolbox.models.architecture import fetch_pretrained_weights
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

is_deployed = True
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


class DummyAttr:
    def __init__(self, val):
        self.item = val


# Define helper functions


def make_ts(route):
    sf = 2 ** (vstate.num_zoom_levels - 9)
    ts = WMTSTileSource(
        name="WSI provider",
        url=route,
        attribution="",
        snap_to_zoom=False,
        min_zoom=0,
        max_zoom=vstate.num_zoom_levels - 1,
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


def make_color_dict(types):
    colors = random_colors(len(types))
    return {key: (*color, 1) for key, color in zip(types, colors)}


def set_alpha_glyph(glyph, alpha):
    """Sets both fill and line alpha for a glyph"""
    glyph.fill_alpha = alpha
    glyph.line_alpha = alpha


def get_mapper_for_prop(prop, enforce_dict=False):
    # find out the unique values of the chosen property
    print(prop)
    resp = s.get(f"http://{host2}:5000/tileserver/get_prop_values/{prop}")
    prop_vals = json.loads(resp.text)
    # guess what cmap should be
    if len(prop_vals) > 10 and not enforce_dict:
        cmap = "viridis" if cmap_select.value == "Dict" else cmap_select.value
    else:
        cmap = make_color_dict(prop_vals)
    return cmap


def update_mapper():
    if vstate.types is not None:
        type_cmap_select.options = [str(t) for t in vstate.types]
        if len(node_source.data["x_"]) > 0:
            type_cmap_select.options.append("graph_overlay")
        vstate.mapper = make_color_dict(vstate.types)
        renderer.mapper = lambda x: vstate.mapper[x]
        update_renderer("mapper", vstate.mapper)


def update_renderer(prop, value):
    if prop == "mapper":
        if value == "dict":
            if cprop_input.value == "type":
                value = vstate.mapper  # put the type mapper dict back
            else:
                value = get_mapper_for_prop(cprop_input.value[0], enforce_dict=True)
            color_bar.color_mapper.palette = make_color_seq_from_cmap(None)
        if not isinstance(value, dict):
            color_bar.color_mapper.palette = make_color_seq_from_cmap(
                cm.get_cmap(value)
            )
            color_bar.visible = True
        return s.put(f"http://{host2}:5000/tileserver/change_cmap/{value}")
    return s.put(
        f"http://{host2}:5000/tileserver/update_renderer/{prop}/{json.dumps(value)}"
    )


def build_predicate():
    """Builds a predicate function from the currently selected types,
    and the filter input.
    """
    preds = [
        f'props["type"]=={name2type(l.label)}'
        for l in box_column.children
        if l.active and l.label in vstate.types
    ]
    if len(preds) == len(box_column.children):
        preds = []
    combo = "None"
    if len(preds) > 0:
        combo = "(" + ") | (".join(preds) + ")"
    if filter_input.value not in ["None", ""]:
        if combo == "None":
            combo = filter_input.value
        else:
            combo = "(" + combo + ") & (" + filter_input.value + ")"

    vstate.renderer.where = combo
    print(combo)
    update_renderer("where", combo)
    return combo


def build_predicate_callable():
    get_types = [
        name2type_key(l.label)
        for l in box_column.children
        if l.active and l.label in vstate.types
    ]
    if len(get_types) == len(box_column.children) or len(get_types) == 0:
        if filter_input.value == "None" or filter_input.value == "":
            vstate.renderer.where = None
            update_renderer("where", "None")
            return None

    if filter_input.value == "None" or filter_input.value == "":
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

    vstate.renderer.where = pred
    # update_renderer("where", json.dumps(pred))
    s.post(
        f"http://{host2}:5000/tileserver/update_where",
        data={"types": json.dumps(get_types), "filter": json.dumps(filter_input.value)},
    )
    return pred


def initialise_slide():
    vstate.mpp = wsi[0].info.mpp
    if vstate.mpp is None:
        vstate.mpp = [1, 1]
    vstate.dims = wsi[0].info.slide_dimensions
    vstate.types = []
    vstate.props = []
    pad = int(np.mean(vstate.dims) / 10)
    plot_size = np.array([p.width, p.height])
    aspect_ratio = plot_size[0] / plot_size[1]
    large_dim = np.argmax(np.array(vstate.dims) / plot_size)

    vstate.micron_formatter.args["mpp"] = vstate.mpp[0]
    if large_dim == 1:
        p.x_range.start = (
            -0.5 * (vstate.dims[1] * aspect_ratio - vstate.dims[0]) - aspect_ratio * pad
        )
        p.x_range.end = (
            vstate.dims[1] * aspect_ratio
            - 0.5 * (vstate.dims[1] * aspect_ratio - vstate.dims[0])
            + aspect_ratio * pad
        )
        p.y_range.start = -vstate.dims[1] - pad
        p.y_range.end = pad
        # p.x_range.min_interval = ?
    else:
        p.x_range.start = -aspect_ratio * pad
        p.x_range.end = vstate.dims[0] + pad * aspect_ratio
        p.y_range.start = (
            -vstate.dims[0] / aspect_ratio
            + 0.5 * (vstate.dims[0] / aspect_ratio - vstate.dims[1])
            - pad
        )
        p.y_range.end = 0.5 * (vstate.dims[0] / aspect_ratio - vstate.dims[1]) + pad

    # p.x_range.bounds = (p.x_range.start - 2 * pad, p.x_range.end + 2 * pad)
    # p.y_range.bounds = (p.y_range.start - 2 * pad, p.y_range.end + 2 * pad)
    # p._trigger_event()

    z = ZoomifyGenerator(wsi[0], tile_size=256)
    vstate.num_zoom_levels = z.level_count
    print(f"nzoom_levs: {vstate.num_zoom_levels}")
    zlev = get_level_by_extent((0, p.y_range.start, p.x_range.end, 0))
    print(f"initial_zoom: {zlev}")
    print(wsi[0].info.as_dict())


def initialise_overlay():
    vstate.colors = list(vstate.mapper.values())
    vstate.types = [str(t) for t in vstate.types]  # vstate.mapper.keys()]
    now_active = {b.label: b.active for b in box_column.children}
    print(vstate.types)
    print(now_active)
    for t in vstate.types:
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
                        color=to_int_rgb(vstate.mapper[t][0:3]),
                        name=str(t),
                        width=60,
                        min_width=60,
                        max_width=70,
                        height=30,
                        sizing_mode="stretch_width",
                    )
                )
            except KeyError:
                color_column.children.append(
                    ColorPicker(
                        color=to_int_rgb(vstate.mapper[int(t)][0:3]),
                        name=str(t),
                        width=60,
                        height=30,
                        min_width=60,
                        max_width=70,
                        sizing_mode="stretch_width",
                    )
                )
            color_column.children[-1].on_change(
                "color", bind_cb_obj(color_column.children[-1], color_input_cb)
            )

    for b in box_column.children.copy():
        if b.label not in vstate.types and b.label not in vstate.layer_dict.keys():
            print(f"removing {b.label}")
            box_column.children.remove(b)
    for c in color_column.children.copy():
        if c.name not in vstate.types and "slider" not in c.name:
            color_column.children.remove(c)

    build_predicate()


def add_layer(lname):
    box_column.children.append(
        Toggle(
            label=lname,
            active=True,
            width=130,
            height=40,
            max_width=160,
            sizing_mode="stretch_width",
        )
    )
    if lname == "nodes":
        box_column.children[-1].active = (
            p.renderers[vstate.layer_dict[lname]].glyph.line_alpha > 0
        )
    if lname == "edges":
        box_column.children[-1].active = p.renderers[vstate.layer_dict[lname]].visible
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
            height=40,
            width=100,
            max_width=190,
            sizing_mode="stretch_width",
            name=f"{lname}_slider",
        )
    )
    color_column.children[-1].on_change(
        "value", bind_cb_obj(color_column.children[-1], layer_slider_cb)
    )

    # layer_boxes=[Toggle(label=t, active=a, width=100) for t,a in now_active.items()]
    # lcolors=[ColorPicker(color=col[0:3], name=t, width=60) for col, t in zip(vstate.colors, vstate.types)]


class TileGroup:
    def __init__(self):
        self.group = 1

    def get_grp(self):
        self.group = self.group + 1
        return self.group


tg = TileGroup()


def change_tiles(layer_name="overlay"):

    grp = tg.get_grp()
    # if grp == 7:
    # import pdb; pdb.set_trace()

    if layer_name == "graph" and layer_name not in vstate.layer_dict.keys():

        # for layer_key in vstate.layer_dict.keys():
        #     if layer_key in ["rect", "pts", "nodes", "edges"]:
        #         continue
        #     grp = tg.get_grp()
        #     ts = make_ts(
        #         f"http://{host}:{port}/tileserver/layer/{layer_key}/{user}/zoomify/TileGroup{grp}"
        #         + r"/{z}-{x}-{y}@{vstate.res}x.jpg",
        #     )
        #     p.renderers[vstate.layer_dict[layer_key]].tile_source = ts
        return

    ts = make_ts(
        f"http://{host}:{port}/tileserver/layer/{layer_name}/{user}/zoomify/TileGroup{grp}"
        + r"/{z}-{x}-{y}"
        + f"@{vstate.res}x.jpg",
    )
    if layer_name in vstate.layer_dict:
        p.renderers[vstate.layer_dict[layer_name]].tile_source = ts
    else:
        p.add_tile(
            ts,
            smoothing=True,
            alpha=overlay_alpha.value,
            level="image",
            render_parents=False,
        )
        for layer_key in vstate.layer_dict.keys():
            if layer_key in ["rect", "pts", "nodes", "edges"]:
                continue
            grp = tg.get_grp()
            # if grp == 7:
            #    import pdb; pdb.set_trace()
            ts = make_ts(
                f"http://{host}:{port}/tileserver/layer/{layer_key}/{user}/zoomify/TileGroup{grp}"
                + r"/{z}-{x}-{y}"
                + f"@{vstate.res}x.jpg",
            )
            # p.renderers[vstate.layer_dict[layer_key]].tile_source = ts
        vstate.layer_dict[layer_name] = len(p.renderers) - 1

    print(vstate.layer_dict)
    print(p.renderers)


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
        self.graph = []
        self.res = 2


vstate = ViewerState()

# base_folder = r"E:\TTB_vis_folder"
base_folder = "/app_data"
if len(sys.argv) > 1 and sys.argv[1] != "None":
    base_folder = Path(sys.argv[1])
    slide_folder = base_folder.joinpath("slides")
    overlay_folder = base_folder.joinpath("overlays")
if len(sys.argv) == 3:
    slide_folder = Path(sys.argv[1])
    overlay_folder = Path(sys.argv[2])

# vstate.slide_path = r"E:\\TTB_vis_folder\\slides\\TCGA-SC-A6LN-01Z-00-DX1.svs"
# vstate.slide_path=Path(r'/tiatoolbox/app_data/slides/TCGA-SC-A6LN-01Z-00-DX1.svs')

# set initial slide to first one in base folder
slide_list = []
for ext in ["*.svs", "*ndpi", "*.tiff", "*.mrxs", "*.png", "*.jpg"]:
    slide_list.extend(list(slide_folder.glob(ext)))
    slide_list.extend(list(slide_folder.glob(str(Path("*") / ext))))
vstate.slide_path = slide_list[0]

renderer = AnnotationRenderer(
    "type",
    {"class1": (1, 0, 0, 1), "class2": (0, 0, 1, 1), "class3": (0, 1, 0, 1)},
    thickness=-1,
    edge_thickness=1,
    zoomed_out_strat="scale",
    max_scale=8,
)
vstate.renderer = renderer

wsi = [WSIReader.open(vstate.slide_path)]
vstate.dims = wsi[0].info.slide_dimensions
vstate.mpp = wsi[0].info.mpp
if vstate.mpp is None:
    vstate.mpp = [1, 1]


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
vstate.micron_formatter = FuncTickFormatter(
    args={"mpp": 0.1},
    code="""
    return Math.round(tick*mpp)
    """,
)

do_feats = False

p_hist = figure(
    width=400,
    height=190,
    x_range=(-0.025, 1.025),
    y_range=(0, 1),
    # sizing_mode="scale_both",
    toolbar_location=None,
    title="Mesogram",
)
line_ds = ColumnDataSource(data=dict(x=np.linspace(0, 1, 100), y=np.zeros(100)))
p_hist.line(x="x", y="y", source=line_ds, line_width=3)

p_bar = figure(
    width=280,
    height=160,
    x_range=[str(i) for i in range(10)],
    y_range=(0, 1),
    # name="plot",
    tools="hover",
    tooltips=[
        ("Index", "$index"),
        ("(x,y)", "(@name, @y)"),
    ],
)
bar_ds = ColumnDataSource(
    data=dict(
        x=[str(i) for i in range(10)], y=np.zeros(10), name=[str(i) for i in range(10)]
    )
)
p_bar.vbar(bottom=0, x="x", top="y", width=0.5, source=bar_ds)
p_bar.xaxis.major_label_orientation = "vertical"


p = figure(
    x_range=(0, vstate.dims[0]),
    y_range=(0, -vstate.dims[1]),
    x_axis_type="linear",
    y_axis_type="linear",
    width=1500,
    height=1000,
    # max_width=1700,
    # max_height=1000,
    # width_policy="max",
    # height_policy="max",
    # tooltips=TOOLTIPS,
    tools="pan,wheel_zoom,reset,save",
    active_scroll="wheel_zoom",
    output_backend="webgl",
    hidpi=True,
    match_aspect=False,
    lod_factor=200000,
    # lod_interval=500,
    # lod_threshold=500,
    # lod_timeout=200,
    sizing_mode="scale_both",
    name="slide_window",
)
# p.axis.visible = False
initialise_slide()
p.toolbar.tools[1].zoom_on_axis = False

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

ts = make_ts(
    f"http://{host}:{port}/tileserver/layer/slide/{user}/zoomify/TileGroup1"
    + r"/{z}-{x}-{y}"
    + f"@{vstate.res}x.jpg",
)
print(p.renderers)
print(p.y_range)
p.add_tile(ts, smoothing=True, level="image", render_parents=False)
print(p.y_range)
print(f"max zoom is: {p.renderers[0].tile_source.max_zoom}")

p.grid.grid_line_color = None
box_source = ColumnDataSource({"x": [], "y": [], "width": [], "height": []})
pt_source = ColumnDataSource({"x": [], "y": []})
r = p.rect("x", "y", "width", "height", source=box_source, fill_alpha=0)
c = p.circle("x", "y", source=pt_source, color="red", size=5)
p.add_tools(BoxEditTool(renderers=[r], num_objects=1))
p.add_tools(PointDrawTool(renderers=[c]))
p.add_tools(TapTool())
p.toolbar.active_inspect = None
tslist = []

p.renderers[0].tile_source.max_zoom = 10

node_source = ColumnDataSource({"x_": [], "y_": [], "node_color_": []})
edge_source = ColumnDataSource({"x0_": [], "y0_": [], "x1_": [], "y1_": []})
vstate.graph_node = Circle(x="x_", y="y_", fill_color="node_color_", size=5)
vstate.graph_edge = Segment(x0="x0_", y0="y0_", x1="x1_", y1="y1_")
p.add_glyph(node_source, vstate.graph_node)
set_alpha_glyph(p.renderers[-1].glyph, 0.0)
p.add_glyph(edge_source, vstate.graph_edge)
p.renderers[-1].visible = False
vstate.layer_dict["nodes"] = len(p.renderers) - 2
vstate.layer_dict["edges"] = len(p.renderers) - 1
hover = HoverTool(renderers=[p.renderers[-2]])
p.add_tools(hover)

# Define UI elements
options_button = Button(label="Options", button_type="primary")

OPTIONS = [
    "normalize props",
]
options_check = CheckboxGroup(labels=OPTIONS, active=[])
active_options = []

res_switch = RadioButtonGroup(labels=["1x", "2x"], active=1)

slide_alpha = Slider(
    title="Slide Alpha",
    start=0,
    end=1,
    step=0.05,
    value=1.0,
    width=200,
    # max_width=200,
    sizing_mode="stretch_width",
)

overlay_alpha = Slider(
    title="Overlay Alpha",
    start=0,
    end=1,
    step=0.05,
    value=0.75,
    width=200,
    # max_width=200,
    sizing_mode="stretch_width",
)

edge_size_spinner = Spinner(
    title="Edge thickness:",
    low=0,
    high=10,
    step=1,
    value=0,
    width=60,
    # max_width=60,
    height=50,
    sizing_mode="stretch_width",
)

pt_size_spinner = Spinner(
    title="Pt. Size:",
    low=0,
    high=20,
    step=1,
    value=4,
    width=60,
    # max_width=60,
    height=50,
    sizing_mode="stretch_width",
)

color_bar = ColorBar(
    color_mapper=LinearColorMapper(make_color_seq_from_cmap(cm.get_cmap("viridis"))),
    label_standoff=12,
)
# p.add_layout(color_bar, 'below')
slide_toggle = Toggle(
    label="Slide",
    button_type="success",
    width=90,
    # max_width=90,
    sizing_mode="stretch_width",
)
overlay_toggle = Toggle(
    label="Overlay",
    button_type="success",
    width=90,
    # max_width=90,
    sizing_mode="stretch_width",
)
filter_input = TextInput(value="None", title="Filter:", sizing_mode="stretch_width")
# cprop_input = TextInput(
#    value="type", title="CProp:", max_width=300, sizing_mode="stretch_width"
# )
cprop_input = MultiChoice(
    title="Colour by:",
    max_items=1,
    options=["*"],
    search_option_limit=5000,
    sizing_mode="stretch_width",
    # max_width=300,
)
slide_select = MultiChoice(
    title="Select Slide:",
    max_items=1,
    options=["*"],
    search_option_limit=5000,
    # max_width=300,
    sizing_mode="stretch_width",
)
cmmenu = [
    ("jet", "jet"),
    ("coolwarm", "coolwarm"),
    ("viridis", "viridis"),
    ("dict", "dict"),
]
cmap_select = Select(
    title="Cmap",
    options=cmmenu,
    width=60,
    value="coolwarm",
    # max_width=60,
    height=45,
    sizing_mode="stretch_width",
)
blur_spinner = Spinner(
    title="Blur:",
    low=0,
    high=10,
    step=1,
    value=0,
    width=60,
    height=50,
    # max_width=60,
    sizing_mode="stretch_width",
)
scale_spinner = Spinner(
    title="max scale:",
    low=0,
    high=540,
    step=8,
    value=16,
    width=60,
    # max_width=60,
    height=50,
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
    # max_width=300,
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
    for t in vstate.types
]
lcolors = [
    ColorPicker(color=col[0:3], width=60, max_width=60, sizing_mode="stretch_width")
    for col in vstate.colors
]
layer_folder_input = TextInput(
    value=str(overlay_folder),
    title="Overlay Folder:",
    # max_width=300,
    sizing_mode="stretch_width",
)
layer_drop = Dropdown(
    label="Add Overlay",
    button_type="warning",
    menu=[None],
    # max_width=300,
    sizing_mode="stretch_width",
)
opt_buttons = CheckboxButtonGroup(
    labels=["Filled", "Microns", "Grid"],
    active=[0],
    # max_width=300,
    sizing_mode="stretch_width",
)
save_button = Button(
    label="Save", button_type="success", max_width=90, sizing_mode="stretch_width"
)


# Define UI callbacks
def res_switch_cb(attr, old, new):
    if new == 0:
        vstate.res = 1
    elif new == 1:
        vstate.res = 2
    else:
        raise ValueError("Invalid resolution")
    vstate.update_state = 1


def options_check_cb(attr, old, new):
    # incomplete - will need to save prop 2,98 percentiles
    # and use them to normalize
    # global active_options
    # changed_option = set(attr.active).union(set(active_options)).difference(set(attr.active).intersection(set(active_options)))
    changed_option = (
        set(new).union(set(old)).difference(set(new).intersection(set(old)))
    )
    # active_options = attr.active
    if "normalize props" in changed_option:
        # update renderer with norm fn
        f"http://{host2}:5000/tileserver/update_renderer/score_fn/what?"


def slide_toggle_cb(attr):
    if p.renderers[0].alpha == 0:
        p.renderers[0].alpha = slide_alpha.value
    else:
        p.renderers[0].alpha = 0.0


def node_select_cb(attr, old, new):
    # do something on node select if desired
    pass


def overlay_toggle_cb(attr):
    for i in range(5, len(p.renderers)):
        if isinstance(p.renderers[i], GraphRenderer):
            # set_graph_alpha(p.renderers[i], new)
            continue
        if p.renderers[i].alpha == 0:
            p.renderers[i].alpha = overlay_alpha.value
        else:
            p.renderers[i].alpha = 0.0


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
    for ext in ["*.svs", "*ndpi", "*.tiff", "*.mrxs", "*.jpg", "*.png"]:
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
    # s.get(f"http://{host2}:5000/tileserver/change_predicate/{new}")
    build_predicate()
    vstate.update_state = 1


def cprop_input_cb(attr, old, new):
    """Change property to colour by"""
    if len(new) == 0:
        return
    if new[0] == "type":
        cmap = vstate.mapper
    else:
        cmap = get_mapper_for_prop(new[0])
    update_renderer("mapper", cmap)
    s.put(f"http://{host2}:5000/tileserver/change_color_prop/{new[0]}")
    vstate.update_state = 1


def set_graph_alpha(g_renderer, value):
    # set all components of graph to given alpha value
    g_renderer.node_renderer.glyph.fill_alpha = value
    g_renderer.node_renderer.glyph.line_alpha = value
    g_renderer.edge_renderer.glyph.line_alpha = value


def slide_alpha_cb(attr, old, new):
    p.renderers[0].alpha = new
    # p.renderers[0].tile_source.max_zoom=7
    # p.renderers[1].tile_source.max_zoom=7


def overlay_alpha_cb(attr, old, new):
    for i in range(5, len(p.renderers)):
        if isinstance(p.renderers[i], GraphRenderer):
            # set_graph_alpha(p.renderers[i], new)
            pass
        else:
            p.renderers[i].alpha = new


def pt_size_cb(attr, old, new):
    vstate.graph_node.size = 2 * new


def edge_size_cb(attr, old, new):
    update_renderer("edge_thickness", new)
    vstate.update_state = 1


def opt_buttons_cb(attr, old, new):
    old_thickness = vstate.renderer.thickness
    if 0 in new:
        vstate.renderer.thickness = -1
        update_renderer("thickness", -1)
    else:
        vstate.renderer.thickness = 1
        update_renderer("thickness", 1)
    if old_thickness != vstate.renderer.thickness:
        vstate.update_state = 1
    if 1 in new:
        p.xaxis[0].formatter = vstate.micron_formatter
        p.yaxis[0].formatter = vstate.micron_formatter
    else:
        p.xaxis[0].formatter = BasicTickFormatter()
        p.yaxis[0].formatter = BasicTickFormatter()
    if 2 in new:
        p.ygrid.grid_line_color = "gray"
        p.xgrid.grid_line_color = "gray"
        p.ygrid.grid_line_alpha = 0.6
        p.xgrid.grid_line_alpha = 0.6
    else:
        p.ygrid.grid_line_alpha = 0
        p.xgrid.grid_line_alpha = 0
    print(p.ygrid)
    print(p.grid)


def cmap_select_cb(attr, old, new):
    update_renderer("mapper", new)
    # change_tiles('overlay')
    vstate.update_state = 1


def blur_spinner_cb(attr, old, new):
    update_renderer("blur_radius", new)
    vstate.update_state = 1


def scale_spinner_cb(attr, old, new):
    update_renderer("max_scale", new)
    vstate.update_state = 1


def slide_select_cb(attr, old, new):
    """setup the newly chosen slide"""
    if len(new) == 0:
        return
    slide_path = Path(slide_folder) / Path(new[0])
    # reset the data sources for glyph overlays
    pt_source.data = {"x": [], "y": []}
    box_source.data = {"x": [], "y": [], "width": [], "height": []}
    node_source.data = {"x_": [], "y_": [], "node_color_": []}
    edge_source.data = {"x0_": [], "y0_": [], "x1_": [], "y1_": []}
    hover.tooltips = None
    if len(p.renderers) > 5:
        for r in p.renderers[5:].copy():
            p.renderers.remove(r)
    vstate.layer_dict = {"slide": 0, "rect": 1, "pts": 2, "nodes": 3, "edges": 4}
    vstate.slide_path = slide_path
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
    print(p.renderers)
    print(slide_path)
    populate_layer_list(slide_path.stem, overlay_folder)
    wsi[0] = WSIReader.open(slide_path)
    initialise_slide()
    # fname='-*-'.join(attr.item.split('\\'))
    fname = make_safe_name(str(slide_path))
    print(fname)
    print(vstate.mpp)
    s.put(f"http://{host2}:5000/tileserver/change_slide/slide/{fname}")
    change_tiles("slide")
    # if len(p.renderers)==1:
    # r=p.rect('x', 'y', 'width', 'height', source=box_source, fill_alpha=0)
    # p.add_tools(BoxEditTool(renderers=[r], num_objects=1))
    # p.x_range.bounds=MinMaxBounds(0,vstate.dims[0])
    # p.y_range.bounds=(0,-vstate.dims[1])

    # load the overlay and graph automatically for demo purposes
    dummy_attr = DummyAttr(overlay_folder / slide_path.with_suffix(".db").name)
    layer_drop_cb(dummy_attr)
    dummy_attr = DummyAttr(overlay_folder / slide_path.with_suffix(".pkl").name)
    layer_drop_cb(dummy_attr)
    cprop_input_cb(None, None, ["score"])


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
                "x_": graph_dict["coordinates"][:, 0],
                "y_": -graph_dict["coordinates"][:, 1],
                "node_color_": [rgb2hex(node_cm(v)) for v in graph_dict["score"]],
            }
        else:
            # default to green
            node_source.data = {
                "x_": graph_dict["coordinates"][:, 0],
                "y_": -graph_dict["coordinates"][:, 1],
                "node_color_": [rgb2hex((0, 1, 0))] * num_nodes,
            }
        edge_source.data = {
            "x0_": [
                graph_dict["coordinates"][i, 0] for i in graph_dict["edge_index"][0, :]
            ],
            "y0_": [
                -graph_dict["coordinates"][i, 1] for i in graph_dict["edge_index"][0, :]
            ],
            "x1_": [
                graph_dict["coordinates"][i, 0] for i in graph_dict["edge_index"][1, :]
            ],
            "y1_": [
                -graph_dict["coordinates"][i, 1] for i in graph_dict["edge_index"][1, :]
            ],
        }
        # edge_source.data = {
        #     "xs": [[graph_dict["coordinates"][inds[0], 0], graph_dict["coordinates"][inds[1], 0]] for inds in graph_dict["edge_index"].T],
        #     "ys": [[-graph_dict["coordinates"][inds[0], 1], -graph_dict["coordinates"][inds[1], 1]] for inds in graph_dict["edge_index"].T],
        # }

        # graph_layout = dict(
        #     zip(
        #         node_source.data["index"],
        #         [
        #             # (x / (4 * vstate.mpp[0]), -y / (4 * vstate.mpp[1]))
        #             (x, -y)
        #             for x, y in graph_dict["coordinates"]
        #         ],
        #     )
        # )
        add_layer("edges")
        add_layer("nodes")
        change_tiles("graph")
        if "graph_overlay" not in type_cmap_select.options:
            type_cmap_select.options = type_cmap_select.options + ["graph_overlay"]

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

        # make a density plot of the scores
        cell_scores = graph_dict["score"]
        # hist,_ = np.histogram(cell_scores, bins=100)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(
            cell_scores.reshape(-1, 1)
        )
        x = np.linspace(0, 1, 100)
        # print(kde.score_samples(x.reshape(-1, 1)))
        # import pdb; pdb.set_trace()
        dens = np.exp(kde.score_samples(x.reshape(-1, 1)))
        line_ds.data["y"] = dens
        p_hist.y_range.end = 1.1 * np.max(dens)

        bar_ds.data["y"] = graph_dict["top_scores"]
        bar_ds.data["name"] = graph_dict["top_feats"]
        p_bar.y_range.end = 1.1 * np.max(graph_dict["top_scores"])
        # set x-ticks to be the feature names
        # p_bar.x_range.factors = graph_dict["top_feats"]

        return

    # fname='-*-'.join(attr.item.split('\\'))
    fname = make_safe_name(attr.item)
    resp = s.put(f"http://{host2}:5000/tileserver/change_overlay/{fname}")
    resp = json.loads(resp.text)

    if Path(attr.item).suffix in [".db", ".dat", ".geojson"]:
        vstate.types = resp
        props = s.get(f"http://{host2}:5000/tileserver/get_prop_names")
        vstate.props = json.loads(props.text)
        # type_cmap_select.options = vstate.props
        cprop_input.options = vstate.props
        cprop_input.options.append("None")
        if not vstate.props == vstate.props_old:
            update_mapper()
            vstate.props_old = vstate.props
        initialise_overlay()
        change_tiles("overlay")
    else:
        add_layer(resp)
        change_tiles(resp)


def layer_select_cb(attr):
    build_predicate()
    # change_tiles('overlay')
    vstate.update_state = 1


def fixed_layer_select_cb(obj, attr):
    print(vstate.layer_dict)
    key = vstate.layer_dict[obj.label]
    if obj.label == "edges":
        if not p.renderers[key].visible:  # line_alpha == 0:
            p.renderers[key].visible = True  # line_alpha = overlay_alpha.value
            # p.renderers[key].node_renderer.glyph.line_alpha = overlay_alpha.value
            # p.renderers[key].edge_renderer.glyph.line_alpha = overlay_alpha.value
        else:
            p.renderers[key].visible = False  # line_alpha = 0.0
            # p.renderers[key].node_renderer.glyph.line_alpha = 0.0
            # p.renderers[key].edge_renderer.glyph.line_alpha = 0.0
    elif obj.label == "nodes":
        if p.renderers[key].glyph.fill_alpha == 0:
            p.renderers[key].glyph.fill_alpha = overlay_alpha.value
            p.renderers[key].glyph.line_alpha = overlay_alpha.value
            # p.renderers[key].edge_renderer.glyph.line_alpha = overlay_alpha.value
        else:
            p.renderers[key].glyph.fill_alpha = 0.0
            p.renderers[key].glyph.line_alpha = 0.0
    else:
        if p.renderers[key].alpha == 0:
            p.renderers[key].alpha = overlay_alpha.value
        else:
            p.renderers[key].alpha = 0.0


def layer_slider_cb(obj, attr, old, new):
    if obj.name.split("_")[0] == "nodes":
        set_alpha_glyph(
            p.renderers[vstate.layer_dict[obj.name.split("_")[0]]].glyph, new
        )
    elif obj.name.split("_")[0] == "edges":
        p.renderers[vstate.layer_dict[obj.name.split("_")[0]]].glyph.line_alpha = new
    else:
        p.renderers[vstate.layer_dict[obj.name.split("_")[0]]].alpha = new


def color_input_cb(obj, attr, old, new):
    print(new)
    vstate.mapper[name2type_key(obj.name)] = (*hex2rgb(new), 1)
    if vstate.renderer.score_prop == "type":
        vstate.renderer.mapper = lambda x: vstate.mapper[x]
        update_renderer("mapper", vstate.mapper)
    # change_tiles('overlay')
    vstate.update_state = 1


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
    vstate.current_model = attr.item


def to_model_cb(attr):
    if vstate.current_model == "hovernet":
        segment_on_box(attr)
    elif vstate.current_model == "nuclick":
        nuclick_on_pts(attr)
    else:
        print("unknown model")


def type_cmap_cb(attr, old, new):
    if len(new) == 0:
        type_cmap_select.options = vstate.types + ["graph_overlay"]
        s.put(
            f"http://{host2}:5000/tileserver/change_secondary_cmap/{'None'}/{'None'}/viridis"
        )
        vstate.update_state = 1
        return
    if len(new) == 1:
        # find out what still has to be selected
        if new[0] in vstate.types + ["graph_overlay"]:
            if new[0] == "graph_overlay":
                type_cmap_select.options = [
                    key
                    for key in node_source.data.keys()
                    if key not in ["x_", "y_", "node_color_"]
                ] + [new[0]]
            else:
                type_cmap_select.options = vstate.props + [new[0]]
        elif new[0] in vstate.props:
            type_cmap_select.options = vstate.types + [new[0]] + ["graph_overlay"]
    else:
        # both selected, update the renderer
        if new[1] in vstate.types:
            # make sure the type is the first one
            type_cmap_select.value = [new[1], new[0]]
            return
        if new[0] == "graph_overlay":
            # adjust the node color in source if prop exists
            if new[1] in node_source.data:
                node_cm = cm.get_cmap("viridis")
                node_source.data["node_color_"] = [
                    rgb2hex(node_cm(v)) for v in node_source.data[new[1]]
                ]
            return
        cmap = get_mapper_for_prop(new[1])  # separate cmap select ?
        s.put(
            f"http://{host2}:5000/tileserver/change_secondary_cmap/{new[0]}/{new[1]}/{cmap}"
        )

        color_bar.color_mapper.palette = make_color_seq_from_cmap(
            cm.get_cmap("viridis")
        )
        color_bar.visible = True
        vstate.update_state = 1


def save_cb(attr):
    save_path = make_safe_name(
        str(overlay_folder / (vstate.slide_path.stem + "_saved_anns.db"))
    )
    s.get(f"http://{host2}:5000/tileserver/commit/{save_path}")


# run NucleusInstanceSegmentor on a region of wsi defined by the box in box_source
def segment_on_box(attr):
    print(vstate.types)
    # thumb=wsi[0].slide_thumbnail(resolution=8, units='mpp')
    thumb = wsi[0].slide_thumbnail()
    # conv_mpp=wsi.convert_resolution_units(1.25, 'power', 'mpp')[0]
    conv_mpp = vstate.dims[0] / thumb.shape[1]
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

    vstate.model_mpp = inst_segmentor.ioconfig.save_resolution["resolution"]
    tile_output = inst_segmentor.predict(
        [vstate.slide_path],
        [mask],
        save_dir="sample_tile_results/",
        mode="wsi",
        # resolution=vstate.mpp,
        # units='mpp',
        on_gpu=True,
        crash_on_exception=True,
    )

    # fname='-*-'.join('.\\sample_tile_results\\0.dat'.split('\\'))
    fname = make_safe_name(".\\sample_tile_results\\0.dat")
    print(fname)
    resp = s.put(
        f"http://{host2}:5000/tileserver/load_annotations/{fname}/{vstate.model_mpp}"
    )
    vstate.types = json.loads(resp.text)

    # update the props options if needed
    props = s.get(f"http://{host2}:5000/tileserver/get_prop_names")
    vstate.props = json.loads(props.text)
    # type_cmap_select.options = vstate.props
    cprop_input.options = vstate.props
    if not vstate.props == vstate.props_old:
        update_mapper()
        vstate.props_old = vstate.props

    # update_mapper()
    # type_drop.menu=[(str(t),str(t)) for t in vstate.types]
    rmtree(r"./sample_tile_results")
    initialise_overlay()
    change_tiles("overlay")

    return tile_output


# run nuclick on user selected points in pt_source
def nuclick_on_pts(attr):
    x = np.round(np.array(pt_source.data["x"]))
    y = -np.round(np.array(pt_source.data["y"]))

    model = NuClick(5, 1)
    # pretrained_weights = r"C:\Users\meast\app_data\NuClick_Nuclick_40xAll.pth"
    fetch_pretrained_weights(
        "nuclick_original-pannuke", r"./nuclick_weights.pth", overwrite=False
    )
    saved_state_dict = torch.load(r"./nuclick_weights.pth", map_location="cpu")
    model.load_state_dict(saved_state_dict, strict=True)
    vstate.model_mpp = 0.25
    ioconf = IOInteractiveSegmentorConfig(
        input_resolutions=[{"resolution": 0.25, "units": "mpp"}], patch_size=(128, 128)
    )
    inst_segmentor = InteractiveSegmentor(
        num_loader_workers=0,
        batch_size=16,
        model=model,
    )

    points = np.vstack([x, y]).T
    points = points / (ioconf.input_resolutions[0]["resolution"] / vstate.mpp[0])
    print(points.shape)
    nuclick_output = inst_segmentor.predict(
        [vstate.slide_path],
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
    resp = s.put(
        f"http://{host2}:5000/tileserver/load_annotations/{fname}/{vstate.model_mpp}"
    )
    print(resp.text)
    vstate.types = json.loads(resp.text)
    update_mapper()
    rmtree(Path(r"/app_data/sample_tile_results"))
    initialise_overlay()
    change_tiles("overlay")


# associate callback functions to the widgets
slide_alpha.on_change("value", slide_alpha_cb)
overlay_alpha.on_change("value", overlay_alpha_cb)
res_switch.on_change("active", res_switch_cb)
pt_size_spinner.on_change("value", pt_size_cb)
edge_size_spinner.on_change("value", edge_size_cb)
slide_select.on_change("value", slide_select_cb)
save_button.on_click(save_cb)
cmap_select.on_change("value", cmap_select_cb)
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
options_check.on_change("active", options_check_cb)

populate_slide_list(slide_folder)
populate_layer_list(Path(vstate.slide_path).stem, overlay_folder)

box_column = column(children=layer_boxes, sizing_mode="stretch_width")
color_column = column(children=lcolors, sizing_mode="stretch_width")

# open up first slide in list
slide_select_cb(None, None, new=[slide_list[0]])

ui_layout = column(
    [
        slide_select,
        layer_drop,
        row([slide_toggle, slide_alpha], sizing_mode="stretch_width"),
        row([overlay_toggle, overlay_alpha], sizing_mode="stretch_width"),
        # filter_input,
        cprop_input,
        cmap_select,
        # type_cmap_select,
        # row([to_model_button, model_drop, save_button], sizing_mode="stretch_width"),
        row(children=[box_column, color_column], sizing_mode="stretch_width"),
        p_hist,
        # p_bar,
    ],
    sizing_mode="stretch_width",
)

extra_options = column([opt_buttons, pt_size_spinner, edge_size_spinner, res_switch])

control_tabs = Tabs(
    tabs=[
        TabPanel(child=ui_layout, title="Main"),
        TabPanel(child=extra_options, title="More Opts"),
    ],
    name="ui_layout",
    sizing_mode="stretch_width",
)


def cleanup_session(session_context):
    # If present, this function executes when the server closes a session.
    sys.exit()


def update():
    if vstate.update_state == 2:
        if "overlay" in vstate.layer_dict:
            change_tiles("overlay")
        vstate.update_state = 0
    if vstate.update_state == 1:
        vstate.update_state = 2


curdoc().add_periodic_callback(update, 220)
curdoc().add_root(p)
# curdoc().add_root(ui_layout)
curdoc().add_root(control_tabs)
# curdoc().add_root(opts_column)
curdoc().title = "Tiatoolbox Visualization Tool"
