"""Main module for the tiatoolbox visualization bokeh app."""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import sys
import tempfile
import time
import urllib
from cmath import pi
from pathlib import Path, PureWindowsPath
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Callable, SupportsFloat

import cv2
import numpy as np
import ollama
import pandas as pd
import requests
import torch
from bokeh.events import ButtonClick, DoubleTap, MenuItemClick
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTickFormatter,
    BoxEditTool,
    Button,
    CheckboxButtonGroup,
    Circle,
    ColorBar,
    ColorPicker,
    Column,
    ColumnDataSource,
    CustomAction,
    CustomJS,
    CustomJSTickFormatter,
    DataTable,
    Div,
    Dropdown,
    FreehandDrawTool,
    Glyph,
    HoverTool,
    HTMLTemplateFormatter,
    InlineStyleSheet,
    LinearColorMapper,
    Model,
    MultiChoice,
    NumericInput,
    PasswordInput,
    PointDrawTool,
    RadioButtonGroup,
    RangeSlider,
    Row,
    Segment,
    Select,
    Slider,
    Spinner,
    StringEditor,
    TableColumn,
    TabPanel,
    Tabs,
    TapTool,
    TextAreaInput,
    TextInput,
    Toggle,
    Tooltip,
)
from bokeh.models.dom import HTML
from bokeh.models.tiles import WMTSTileSource
from bokeh.plotting import figure
from bokeh.util import token
from matplotlib import colormaps
from openai import OpenAI
from PIL import Image
from requests.adapters import HTTPAdapter, Retry

# GitHub actions seems unable to find TIAToolbox unless this is here
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from tiatoolbox import logger
from tiatoolbox.models.engine.nucleus_instance_segmentor import (
    NucleusInstanceSegmentor,
)
from tiatoolbox.tools.pyramid import ZoomifyGenerator
from tiatoolbox.utils.visualization import random_colors
from tiatoolbox.visualization.ui_utils import get_level_by_extent
from tiatoolbox.wsicore.wsireader import WSIReader

if TYPE_CHECKING:  # pragma: no cover
    from bokeh.document import Document

rng = np.random.default_rng()

# Define some constants
MAX_CAT = 10
FILLED = 0
MICRON_FORMATTER = 1
GRIDLINES = 2
MAX_FEATS = 20
N_PERMANENT_RENDERERS = 6
NO_UPDATE = 0
PENDING_UPDATE = 1
DO_UPDATE = 2

default_cm = "viridis"  # any valid matplotlib colormap string


# ---- GPT stuff ----
class GPTInterface:
    """Class to handle GPT requests."""

    def __init__(self: GPTInterface, model_name: str = "llama3.2-vision:11b") -> None:
        """Initialise the class."""
        self.set_client(model_name)

        # will store short history of prompts and responses
        self.gpt_images = []
        self.gpt_prompts = []
        self.gpt_responses = []
        self.max_history = 5

        # some default gpt prompts
        self.sys_prompt = "You are an expert pathologist tasked with providing clear, concise answers to questions about provided H&E stained histological images, from advanced Pathology Students who are learning to accurately assess tissue samples and identify medically relevant features. Provided responses will be used for educational and research purposes only and so do not constitute medical advice."
        # prompt if just a plain region is sent
        self.prompt_no_ann = "Provide a concise assessment of this image for the student. Comment on noteworthy histological features and structures, and any abnormalities present."
        # promt if a region with a user-drawn annotation is sent
        self.prompt_ann = "Provide a concise assessment of this image for the student. Comment on noteworthy histological features and structures (paying particular attention to the regions indicated by green annotations), and any abnormalities present."

    def set_client(self: GPTInterface, model_name) -> None:
        # client for openai reqs
        self.model_name = model_name
        if self.model_name == "gpt-4o":
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key is None:
                logger.warning(
                    "OPENAI_API_KEY not set, GPT-Vision will not work. Add as a system environment variable on your machine, or in .env file.",
                )
                self.client = None
            else:
                # we have an api key, so set up the client
                self.client = OpenAI(api_key=self.api_key)
        else:
            avail_models = ollama.list()
            print(f"ollama models available: {avail_models}")
            if self.model_name not in avail_models:
                print(f"attempting to pull model {self.model_name}")
                try:
                    ollama.pull(self.model_name)
                except Exception as e:
                    print(f"Error: {e} when pulling the model.")
            self.api_key = "local"
            self.client = None

    def update_api_key(self: GPTInterface, api_key: str) -> None:
        """Update the api key."""
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def add_img(self: GPTInterface, img: np.array) -> None:
        """Add an image to the history."""
        if len(self.gpt_images) >= self.max_history:
            self.gpt_images.pop(0)
        self.gpt_images.append(img)

    def send(self: GPTInterface, prompt: str) -> Any:
        im_size = self.gpt_images[-1].size
        base64_image = encode_image(self.gpt_images[-1])

        # Prepare the messages to send
        if self.model_name == "gpt-4o":
            messages = [
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        prompt,
                        {"image": base64_image},
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image],
                }
            ]

        print(f"sending image size: {im_size} to {self.model_name}")
        # send a message containing the image
        prompt_input.value = "Sent to model. Waiting for response - this typically takes < 10 seconds, but may take longer if server is busy. It may occasionally fail."

        # Send the messages to the model
        tries = 0
        response_text = "Failed to get response from the model."
        responses = None
        while tries < 3:
            try:
                if self.model_name == "gpt-4o":
                    responses = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=500,
                    )
                    # extract the text from the response
                    if responses is not None:
                        response_text = responses.choices[0].message.content
                else:
                    responses = ollama.chat(model=self.model_name, messages=messages)
                    # Collect the response text
                    response_text = responses["message"]["content"]
                break
            except Exception as e:
                print(
                    f"Error: {e} when sending to the model. Trying again (try {tries + 1} / 3)"
                )
                tries += 1
                time.sleep(1)
        else:
            response_text = "Failed to get response after 3 tries."

        print("response received.")

        # add the prompt and response to the history
        if len(self.gpt_prompts) >= self.max_history:
            self.gpt_prompts.pop(0)
            self.gpt_responses.pop(0)
        self.gpt_prompts.append(prompt)
        self.gpt_responses.append(response_text)

        return response_text

    def save_history(self: GPTInterface, path: Path, last_n=1) -> None:
        """Save the last n prompts, responses and images in the history
        to a file.

        """
        # make folder if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        # files saved with form gpt_prompts_i.pkl - find the next i
        i = 0
        while (path / f"gpt_prompts_{i}.pkl").exists():
            i += 1
        path = path / f"gpt_prompts_{i}.pkl"
        # save the last n prompts, responses and images
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "prompts": self.gpt_prompts[-last_n:],
                    "responses": self.gpt_responses[-last_n:],
                    "images": self.gpt_images[-last_n:],
                },
                f,
            )
        print(f"saved last {last_n} prompts, responses and images to {path}")


gpt_interface = GPTInterface()
# -------------------

# stylesheets to format some things better

# Stylesheet for the help tooltips
help_ss = InlineStyleSheet(
    css="""
        :host(.help_tt) {
            width:200px;
            white-space: wrap;
            padding-top: 3px;
            padding-bottom: 3px;
            margin-top: 3px;
            margin-bottom: 3px;
        }
        """,
)


# Define helper functions/classes
# region
class DummyAttr:
    """Dummy class to enable triggering a callback independently of a widget."""

    def __init__(self: DummyAttr, val: Any) -> None:  # noqa: ANN401
        """Initialize the class."""
        self.item = val


class UIWrapper:
    """Wrapper class to access ui elements."""

    def __init__(self: UIWrapper) -> None:
        """Initialize the class."""
        self.active = 0

    def __getitem__(self: UIWrapper, key: str) -> Any:  # noqa: ANN401
        """Gets ui element for the active window."""
        return win_dicts[self.active][key]


def encode_image(tile_image: Image.Image) -> str:
    """Encode an image as a base64 string."""
    image_io = io.BytesIO()
    tile_image.save(image_io, format="jpeg")
    image_io.seek(0)
    return base64.b64encode(image_io.read()).decode("utf-8")


def format_info(info: dict[str, Any]) -> str:
    """Format the slide info for display."""
    slide_name = info.pop("file_path").name
    info_str = f"<b>Slide Name: {slide_name}</b><br>"
    for k, v in info.items():
        info_str += f"{k}: {v}<br>"
    # if there is metadata, add it
    if metadata is not None:
        try:
            row = metadata.loc[slide_name]
            info_str += "<br><b>Metadata:</b><br>"
            for k, v in row.items():
                info_str += f"{k}: {v}<br>"
        except KeyError:
            info_str += "<br><b>No metadata found.</b><br>"
    return info_str


def get_channel_info() -> dict[str, tuple[int, int, int]]:
    """Get the colors for the channels."""
    resp = UI["s"].get(f"http://{host2}:5000/tileserver/channels")
    try:
        resp = json.loads(resp.text)
        return resp.get("channels", {}), resp.get("active", [])
    except json.JSONDecodeError:
        return {}, []


def set_channel_info(
    colors: dict[str, tuple[int, int, int]], active_channels: list
) -> None:
    """Set the colors for the channels."""
    UI["s"].put(
        f"http://{host2}:5000/tileserver/channels",
        data={"channels": json.dumps(colors), "active": json.dumps(active_channels)},
    )


def create_channel_color_ui():
    channel_source = ColumnDataSource(
        data={
            "channels": [],
            "dummy": [],
        }
    )
    color_source = ColumnDataSource(
        data={
            "colors": [],
            "dummy": [],
        }
    )

    color_formatter = HTMLTemplateFormatter(
        template='<div style="background-color: <%= value %>; color: <%= value %>; border: 1px solid #ddd;"><%= value %></div>'
    )

    channel_table = DataTable(
        source=channel_source,
        columns=[
            TableColumn(
                field="channels",
                title="Channel",
                editor=StringEditor(),
                sortable=False,
                width=200,
            )
        ],
        editable=True,
        width=200,
        height=260,
        selectable="checkbox",
        autosize_mode="none",
        fit_columns=True,
    )
    color_table = DataTable(
        source=color_source,
        columns=[
            TableColumn(
                field="colors",
                title="Color",
                formatter=color_formatter,
                editor=StringEditor(),
                sortable=False,
                width=130,
            )
        ],
        editable=True,
        width=130,
        height=260,
        selectable=True,
        autosize_mode="none",
        index_position=None,
        fit_columns=True,
    )

    color_picker = ColorPicker(title="Channel Color", width=100)

    def update_selected_color(attr, old, new):
        selected = color_source.selected.indices
        if selected:
            color_source.patch({"colors": [(selected[0], new)]})

    color_picker.on_change("color", update_selected_color)

    apply_button = Button(
        label="Apply Changes", button_type="success", margin=(20, 5, 5, 5)
    )

    def apply_changes() -> None:
        """Apply the changes to the image."""
        colors = dict(zip(channel_source.data["channels"], color_source.data["colors"]))
        active_channels = channel_source.selected.indices

        set_channel_info({ch: hex2rgb(colors[ch]) for ch in colors}, active_channels)
        change_tiles("slide")

    apply_button.on_click(apply_changes)

    def update_color_picker(attr, old, new):
        if new:
            selected_color = color_source.data["colors"][new[0]]
            color_picker.color = selected_color
        else:
            color_picker.color = None

    color_source.selected.on_change("indices", update_color_picker)

    enhance_slider = Slider(
        start=0.1,
        end=10,
        value=1,
        step=0.1,
        title="Enhance",
        width=200,
    )

    def enhance_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001 # skipcq: PYL-W0613
        """Enhance slider callback."""
        UI["s"].put(
            f"http://{host2}:5000/tileserver/enhance",
            data={"val": json.dumps(new)},
        )
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["slide"])

    enhance_slider.on_change("value", enhance_cb)

    instructions = Div(
        text="""
        <p>Instructions:</p>
        <ul>
            <li>Select active channels using the checkboxes</li>
            <li>Select a channel color and change using the picker below the table</li>
            <li>Click 'Apply Changes' to update the image</li>
        </ul>
    """
    )

    return column(
        instructions,
        column(
            row(channel_table, color_table),
            row(color_picker, apply_button),
            enhance_slider,
        ),
    )


def populate_table() -> None:
    """Populate the channel color table."""
    # Access the ColumnDataSource from the UI dictionary
    tables = UI["channel_select"].children[1].children[0].children
    colors, active_channels = get_channel_info()

    if colors is not None:
        tables[0].source.data = {
            "channels": list(colors.keys()),
            "dummy": list(colors.keys()),
        }
        tables[1].source.data = {
            "colors": [rgb2hex(color) for color in colors.values()],
            "dummy": list(colors.keys()),
        }
        tables[0].source.selected.indices = active_channels


def get_view_bounds(
    dims: tuple[float, float],
    plot_size: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Helper to get the current view bounds.

    Estimate a reasonable initial view bounds based on the image
    dimensions and the size of the viewing plot.

    Args:
        dims: The dimensions of the image.
        plot_size: The size of the plot.

    Returns:
        The view bounds.

    """
    pad = int(np.mean(dims) / 10)
    aspect_ratio = plot_size[0] / plot_size[1]
    large_dim = np.argmax(np.array(dims) / plot_size)

    if large_dim == 1:
        x_range_start = -0.5 * (dims[1] * aspect_ratio - dims[0]) - aspect_ratio * pad
        x_range_end = (
            dims[1] * aspect_ratio
            - 0.5 * (dims[1] * aspect_ratio - dims[0])
            + aspect_ratio * pad
        )
        y_range_start = -dims[1] - pad
        y_range_end = pad
    else:
        x_range_start = -aspect_ratio * pad
        x_range_end = dims[0] + pad * aspect_ratio
        y_range_start = (
            -dims[0] / aspect_ratio + 0.5 * (dims[0] / aspect_ratio - dims[1]) - pad
        )
        y_range_end = 0.5 * (dims[0] / aspect_ratio - dims[1]) + pad
    return x_range_start, x_range_end, y_range_start, y_range_end


def to_num(x: str | SupportsFloat) -> int | float | None:
    """Convert a str representation of a number to a numerical value."""
    if not isinstance(x, str):
        return x
    if x == "None":
        return None
    try:
        return int(x)
    except ValueError:
        return float(x)


def get_from_config(keys: list[str], default: Any = None) -> Any:  # noqa: ANN401
    """Helper to get a value from a config dict.

    Check dict values for nested keys.
    The default value is returned if the key is not found.

    Args:
        keys: The nested keys to look for. e.g ["a", "b"] will look for
            config["a"]["b"].
        default: The default value to return if the entry is not found.

    """
    c_dict = doc_config.config
    for k in keys:
        if k in c_dict:
            c_dict = c_dict[k]
        else:
            return default
    return c_dict


def make_ts(route: str, z_levels: int, init_z: int = 4) -> WMTSTileSource:
    """Helper to make a tile source."""
    sf = 2 ** (z_levels - init_z - 5)
    ts = WMTSTileSource(
        name="WSI provider",
        url=route,
        attribution="",
        snap_to_zoom=False,
        min_zoom=0,
        max_zoom=z_levels - 1,
    )
    ts.tile_size = 256
    ts.initial_resolution = 40211.5 * sf * (2 / (100 * pi))
    ts.x_origin_offset = 0
    ts.y_origin_offset = sf * 10294144.78 * (2 / (100 * pi))
    ts.wrap_around = False
    return ts


def to_float_rgb(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Helper to convert from int to float rgb(a) tuple."""
    return tuple(v / 255 for v in rgb)


def to_int_rgb(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """Helper to convert from float to int rgb(a) tuple."""
    return tuple(int(v * 255) for v in rgb)


def name2type(name: str) -> Any:  # noqa: ANN401
    """Helper to get original type from stringified version."""
    name = UI["vstate"].orig_types[name]
    if isinstance(name, str):
        return f'"{name}"'
    return name


def hex2rgb(hex_val: str) -> tuple[float, float, float]:
    """Covert hex rgb string to float rgb(a) tuple."""
    return tuple(int(hex_val[i : i + 2], 16) / 255 for i in (1, 3, 5))


def rgb2hex(rgb: tuple[float, float, float]) -> str:
    """Covert float rgb(a) tuple to hex string."""
    int_rgb = to_int_rgb(rgb)
    return f"#{int_rgb[0]:02x}{int_rgb[1]:02x}{int_rgb[2]:02x}"


def make_color_seq_from_cmap(cmap: str | None = None) -> list[str]:
    """Helper to make a color sequence from a colormap."""
    if cmap is None:
        return [
            rgb2hex((1.0, 1.0, 1.0)),
            rgb2hex((1.0, 1.0, 1.0)),
        ]  # no colors if using dict
    return [rgb2hex(cmap(v)) for v in np.linspace(0, 1, 50)]


def make_safe_name(name: str) -> str:
    """Helper to make a name safe for use in a URL."""
    return urllib.parse.quote(str(PureWindowsPath(name)), safe="")


def make_color_dict(types: list[str]) -> dict[str, tuple[float, float, float]]:
    """Helper to make a color dict from a list of types."""
    colors = random_colors(len(types), bright=True)
    # Grab colors out of doc_config["color_dict"] if possible, otherwise use random
    type_colors = {}
    for i, t in enumerate(types):
        if t in UI["vstate"].mapper:
            # Keep existing color
            type_colors[t] = UI["vstate"].mapper[t]
        elif str(t) in get_from_config(["color_dict"], {}):
            # Grab color from config if possible
            type_colors[t] = to_float_rgb(doc_config["color_dict"][str(t)])
        else:
            # Otherwise use random
            type_colors[t] = (*colors[i], 1)
    return type_colors


def set_alpha_glyph(glyph: Glyph, alpha: float) -> None:
    """Set the fill and line alpha for a glyph."""
    glyph.fill_alpha = alpha
    glyph.line_alpha = alpha


def get_mapper_for_prop(
    prop: str,
    mapper_type: str = "auto",
    slider: Optional[Slider] = None,
) -> str | dict[str, str]:
    """Helper to get appropriate mapper for a property."""
    if prop == "type":
        UI["vstate"].is_categorical = True
        return UI["vstate"].mapper
    # Find out the unique values of the chosen property
    resp = UI["s"].get(f"http://{host2}:5000/tileserver/prop_values/{prop}/all")
    prop_vals = json.loads(resp.text)
    # If auto, guess what cmap should be
    if (
        (len(prop_vals) > MAX_CAT or len(prop_vals) == 0)
        and mapper_type == "auto"
        or mapper_type == "continuous"
    ):
        # use a continuous mapper
        cmap = (
            default_cm if UI["cmap_select"].value == "dict" else UI["cmap_select"].value
        )
        UI["vstate"].is_categorical = False
        if slider is not None:
            # update passed sliders min and max values
            max_val = max(prop_vals)
            min_val = min(prop_vals)
            UI["vstate"].max_val = max_val
            UI["vstate"].min_val = min_val
            slider.start = min_val
            slider.end = max_val
            if len(UI["range_checkbox"].active) == 0:
                # if not using fixed range, adjust to new vals
                if slider.value == (min_val, max_val):
                    # need to explicitly trigger the callback
                    range_slider_cb(None, None, (min_val, max_val))
                else:
                    # set the new val and the cb will trigger on change
                    slider.value = (min_val, max_val)

    else:
        cmap = make_color_dict(prop_vals)
    return cmap


def update_mapper() -> None:
    """Helper to update the color mapper."""
    update_renderer("mapper", UI["vstate"].mapper)


def update_renderer(prop: str, value: Any) -> None:  # noqa: ANN401
    """Helper to update a renderer property."""
    if prop == "mapper":
        if value == "dict":
            value = get_mapper_for_prop(
                UI["cprop_input"].value[0],
                mapper_type="dict",
            )
            UI["color_bar"].color_mapper.palette = make_color_seq_from_cmap(None)
        if not isinstance(value, dict):
            UI["color_bar"].color_mapper.palette = make_color_seq_from_cmap(
                colormaps[value],
            )
            UI["color_bar"].visible = True
        if isinstance(value, dict):
            # Send keys and values separately so types are preserved
            value = {"keys": list(value.keys()), "values": list(value.values())}
        UI["s"].put(
            f"http://{host2}:5000/tileserver/cmap",
            data={"cmap": json.dumps(value)},
        )
        return
    UI["s"].put(
        f"http://{host2}:5000/tileserver/renderer/{prop}",
        data={"val": json.dumps(value)},
    )


def build_predicate() -> str:
    """Builds a predicate string.

    Builds the appropriate predicate string from the currently selected types,
    and the filter input.

    """
    preds = [
        f'props["type"]=={name2type(layer.label)}'
        for layer in UI["type_column"].children
        if layer.active and layer.label in UI["vstate"].types
    ]
    combo = "None"
    if len(preds) == len(UI["vstate"].types):
        preds = []
    elif len(preds) == 0:
        preds = ['props["type"]=="None"']
    if len(preds) > 0:
        combo = "(" + ") | (".join(preds) + ")"
    if UI["filter_input"].value not in ["None", ""]:
        if combo == "None":
            combo = UI["filter_input"].value
        else:
            combo = "(" + combo + ") & (" + UI["filter_input"].value + ")"

    update_renderer("where", combo)
    return combo


def initialise_slide() -> None:
    """Initialise the newly selected slide."""
    # Get some slide info
    UI["vstate"].mpp = UI["vstate"].wsi.info.mpp
    if UI["vstate"].mpp is None:
        UI["vstate"].mpp = [1, 1]
    UI["vstate"].dims = UI["vstate"].wsi.info.slide_dimensions
    slide_name = UI["vstate"].wsi.info.file_path.stem
    UI["vstate"].types = []
    UI["vstate"].props = []
    plot_size = np.array([UI["p"].width, UI["p"].height])

    # Set up initial view window
    UI["vstate"].micron_formatter.args["mpp"] = UI["vstate"].mpp[0]
    if slide_name in get_from_config(["initial_views"], {}):
        lims = doc_config["initial_views"][slide_name]
        UI["p"].x_range.start = lims[0]
        UI["p"].x_range.end = lims[2]
        UI["p"].y_range.start = -lims[3]
        UI["p"].y_range.end = -lims[1]
    # If two windows open and new slide is 'related' to the other, use the same view
    elif len(win_dicts) == 2 and (  # noqa: PLR2004
        win_dicts[0]["vstate"].slide_path.stem in win_dicts[1]["vstate"].slide_path.stem
        or win_dicts[1]["vstate"].slide_path.stem
        in win_dicts[0]["vstate"].slide_path.stem
        or win_dicts[1]["vstate"].dims == win_dicts[0]["vstate"].dims
    ):  # PLR2004
        # View should already be correct, pass
        pass
    else:
        x_start, x_end, y_start, y_end = get_view_bounds(UI["vstate"].dims, plot_size)
        UI["p"].x_range.start = x_start
        UI["p"].x_range.end = x_end
        UI["p"].y_range.start = y_start
        UI["p"].y_range.end = y_end

    init_z = get_level_by_extent((0, UI["p"].y_range.start, UI["p"].x_range.end, 0))
    UI["vstate"].init_z = init_z
    logger.warning("slide info: %s", UI["vstate"].wsi.info.as_dict(), stacklevel=2)
    slide_info.text = format_info(UI["vstate"].wsi.info.as_dict())


def initialise_overlay() -> None:
    """Initialise the newly selected overlay."""
    UI["vstate"].colors = list(UI["vstate"].mapper.values())
    now_active = {b.label: b.active for b in UI["type_column"].children}
    # Add type toggles for any that weren't already there
    for t in sorted(UI["vstate"].types):
        if str(t) not in now_active:
            UI["type_column"].children.append(
                Toggle(
                    label=str(t),
                    active=True,
                    width=130,
                    height=30,
                    max_width=130,
                    sizing_mode="stretch_width",
                ),
            )
            UI["type_column"].children[-1].on_click(layer_select_cb)
            try:
                UI["color_column"].children.append(
                    ColorPicker(
                        color=to_int_rgb(UI["vstate"].mapper[t][0:3]),
                        name=str(t),
                        width=60,
                        min_width=60,
                        max_width=70,
                        height=30,
                        sizing_mode="stretch_width",
                    ),
                )
            except KeyError:
                UI["color_column"].children.append(
                    ColorPicker(
                        color=to_int_rgb(UI["vstate"].mapper[to_num(t)][0:3]),
                        name=str(t),
                        width=60,
                        height=30,
                        min_width=60,
                        max_width=70,
                        sizing_mode="stretch_width",
                    ),
                )
            UI["color_column"].children[-1].on_change(
                "color",
                bind_cb_obj(UI["color_column"].children[-1], color_input_cb),
            )

    # Remove any that are no longer in the overlay
    for b in UI["type_column"].children.copy():
        if b.label not in UI["vstate"].types and b.label not in UI["vstate"].layer_dict:
            UI["type_column"].children.remove(b)
    for c in UI["color_column"].children.copy():
        if c.name not in UI["vstate"].types and "slider" not in c.name:
            UI["color_column"].children.remove(c)

    build_predicate()


def add_layer(lname: str) -> None:
    """Add a new layer to the visualization."""
    UI["type_column"].children.append(
        Toggle(
            label=lname,
            active=True,
            width=130,
            height=40,
            max_width=130,
            sizing_mode="stretch_width",
        ),
    )
    if lname == "nodes":
        UI["type_column"].children[-1].active = (
            UI["p"].renderers[UI["vstate"].layer_dict[lname]].glyph.line_alpha > 0
        )
    if lname == "edges":
        UI["type_column"].children[-1].active = (
            UI["p"].renderers[UI["vstate"].layer_dict[lname]].visible
        )
    UI["type_column"].children[-1].on_click(
        bind_cb_obj_tog(UI["type_column"].children[-1], fixed_layer_select_cb),
    )
    UI["color_column"].children.append(
        Slider(
            start=0,
            end=1,
            value=0.75,
            step=0.01,
            title=lname,
            height=40,
            width=100,
            max_width=90,
            sizing_mode="stretch_width",
            name=f"{lname}_slider",
        ),
    )
    UI["color_column"].children[-1].on_change(
        "value",
        bind_cb_obj(UI["color_column"].children[-1], layer_slider_cb),
    )


class TileGroup:
    """Class to keep track of the current tile group."""

    def __init__(self: TileGroup) -> None:
        """Initialise the tile group."""
        self.group = 1

    def get_grp(self: TileGroup) -> int:
        """Get the current tile group."""
        self.group = self.group + 1
        return self.group


class ColorCycler:
    """Class to cycle through a list of colors."""

    def __init__(self: ColorCycler, colors: list[str] | None = None) -> None:
        """Initialise the color cycler."""
        if colors is None:
            colors = ["red", "blue", "lime", "yellow", "cyan", "magenta", "orange"]
        self.colors = colors
        self.index = -1

    def get_next(self: ColorCycler) -> str:
        """Get the next color in the list."""
        self.index = (self.index + 1) % len(self.colors)
        return self.colors[self.index]

    def get_random(self: ColorCycler) -> str:
        """Get a random color from the list."""
        return str(rng.choice(self.colors))

    @staticmethod
    def generate_random() -> str:
        """Generate a new random color."""
        return rgb2hex(rng.choice(256, 3) / 255)


def change_tiles(layer_name: str = "overlay") -> None:
    """Update tilesources.

    If a layer is updated/added, will update the tilesource to ensure
    that the new layer is displayed.

    """
    grp = tg.get_grp()

    if layer_name == "graph" and layer_name not in UI["vstate"].layer_dict:
        return

    ts = make_ts(
        f"//{host}:{port}/tileserver/layer/{layer_name}/{UI['user']}/"
        f"zoomify/TileGroup{grp}"
        r"/{z}-{x}-{y}"
        f"@{UI['vstate'].res}x.jpg",
        UI["vstate"].num_zoom_levels,
    )
    if layer_name in UI["vstate"].layer_dict:
        UI["p"].renderers[UI["vstate"].layer_dict[layer_name]].tile_source = ts
    else:
        UI["p"].add_tile(
            ts,
            smoothing=True,
            alpha=UI["overlay_alpha"].value,
            level="image",
            render_parents=False,
        )
        for layer_key in UI["vstate"].layer_dict:
            if layer_key in ["rect", "pts", "line", "nodes", "edges"]:
                continue
            grp = tg.get_grp()
            ts = make_ts(
                f"//{host}:{port}/tileserver/layer/{layer_key}/{UI['user']}/"
                f"zoomify/TileGroup{grp}"
                r"/{z}-{x}-{y}"
                f"@{UI['vstate'].res}x.jpg",
                UI["vstate"].num_zoom_levels,
            )
            UI["p"].renderers[UI["vstate"].layer_dict[layer_key]].tile_source = ts
        UI["vstate"].layer_dict[layer_name] = len(UI["p"].renderers) - 1

    logger.info("current layers: %s", UI["vstate"].layer_dict)


class ViewerState:
    """Class to keep track of the current state of the viewer."""

    def __init__(self: ViewerState, slide_path: str | Path) -> None:
        """Initialise the viewer state."""
        self.wsi = WSIReader.open(slide_path)
        self.slide_path = slide_path
        self.mpp = self.wsi.info.mpp
        if self.mpp is None:
            self.mpp = [1, 1]
        self.dims = self.wsi.info.slide_dimensions
        self.mapper = {}
        self.colors = list(self.mapper.values())
        self.cprop = None
        self.init_z = None
        self.types = list(self.mapper.keys())
        self.layer_dict = {"slide": 0, "rect": 1, "pts": 2, "line": 3}
        self.update_state = 0
        self.thickness = -1
        self.model_mpp = 0
        self.init = True
        self.micron_formatter = CustomJSTickFormatter(
            args={"mpp": 0.1},
            code="""
                return Math.round(tick*mpp)
                """,
        )
        self.current_model = "hovernet"
        self.props = []
        self.props_old = []
        self.to_update = set()
        self.graph = []
        self.res = 2
        self.max_val = 1
        self.min_val = 0
        self.is_categorical = True

    def __setattr__(
        self: ViewerState,
        __name: str,
        __value: Any,  # noqa: ANN401
    ) -> None:
        """Set an attribute of the viewer state."""
        if __name == "types":
            self.__dict__["mapper"] = make_color_dict(__value)
            self.__dict__["colors"] = list(self.mapper.values())
            if self.cprop == "type":
                update_mapper()
            # We will standardise the types to strings, keep dict of originals
            self.__dict__["orig_types"] = {str(x): x for x in __value}
            __value = [str(x) for x in __value]

        if __name == "wsi":
            z = ZoomifyGenerator(__value, tile_size=256)
            self.__dict__["num_zoom_levels"] = z.level_count

        self.__dict__[__name] = __value


# endregion


# Define UI callbacks
# region
def res_switch_cb(attr: str, old: int, new: int) -> None:  # noqa: ARG001
    """Callback to switch between resolutions."""
    if new == 0:
        UI["vstate"].res = 1
    else:
        UI["vstate"].res = 2
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay", "slide"])


def slide_toggle_cb(attr: str) -> None:  # noqa: ARG001
    """Callback to toggle the slide on/off."""
    if UI["p"].renderers[0].alpha == 0:
        UI["p"].renderers[0].alpha = UI["slide_alpha"].value
    else:
        UI["p"].renderers[0].alpha = 0.0


def node_select_cb(attr: str, old: int, new: int) -> None:
    """Placeholder callback to do something on node selection."""
    # Do something on node select if desired


def overlay_toggle_cb(attr: str) -> None:  # noqa: ARG001
    """Callback to toggle the overlay on/off."""
    for i in range(N_PERMANENT_RENDERERS, len(UI["p"].renderers)):
        if UI["p"].renderers[i].alpha == 0:
            UI["p"].renderers[i].alpha = UI["overlay_alpha"].value
        else:
            UI["p"].renderers[i].alpha = 0.0


def populate_layer_list(slide_name: str, overlay_path: Path) -> None:
    """Populate the layer list with the available overlays."""
    file_list = []
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
        file_list.extend(list(overlay_path.glob(ext)))
    file_list = [(str(p), str(p)) for p in sorted(file_list) if slide_name in str(p)]
    UI["layer_drop"].menu = file_list


def populate_slide_list(slide_folder: Path, search_txt: str | None = None) -> None:
    """Populate the slide list with the available slides."""
    file_list = []
    len_slidepath = len(slide_folder.parts)
    for ext in [
        "*.svs",
        "*.ndpi",
        "*.tiff",
        "*.mrxs",
        "*.jpg",
        "*.png",
        "*.tif",
        "*.qptiff",
    ]:
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
    # filter by selected cohort if in config
    cohorts = get_from_config(["cohorts"], {})
    if len(cohorts) > 0:
        if len(cohorts) == 1:
            cats = iter(cohorts.keys())
        else:
            cats = UI["cohort_select"].value
        cohort_slides = []
        for c in cats:
            cohort_slides.extend(cohorts[c])
        if len(cohort_slides) > 0:
            file_list = [p for p in file_list if Path(p[0]).name in cohort_slides]

    UI["slide_select"].options = file_list


def filter_input_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001
    """Change predicate to be used to filter annotations."""
    build_predicate()
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def cprop_input_cb(attr: str, old: str, new: list[str]) -> None:  # noqa: ARG001
    """Change property to color by."""
    if len(new) == 0:
        return
    UI["vstate"].cprop = new[0]
    cmap = (
        UI["vstate"].mapper
        if new[0] == "type"
        else get_mapper_for_prop(new[0], slider=UI["range_slider"])
    )
    update_renderer("mapper", cmap)
    UI["s"].put(
        f"http://{host2}:5000/tileserver/color_prop",
        data={"prop": json.dumps(new[0])},
    )
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def slide_alpha_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the alpha of the slide."""
    UI["p"].renderers[0].alpha = new


def overlay_alpha_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the alpha of all overlay layers."""
    for i in range(N_PERMANENT_RENDERERS, len(UI["p"].renderers)):
        UI["p"].renderers[i].alpha = new


def pt_size_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the size of the points."""
    UI["vstate"].graph_node.radius = 2 * new


def edge_size_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the size of the edges."""
    update_renderer("edge_thickness", new)
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def opt_buttons_cb(attr: str, old: list[int], new: list[int]) -> None:  # noqa: ARG001
    """Callback to handle options changes in the ui widget."""
    old_thickness = UI["vstate"].thickness
    if FILLED in new:
        UI["vstate"].thickness = -1
        update_renderer("thickness", -1)
    else:
        UI["vstate"].thickness = 1
        update_renderer("thickness", 1)
    if old_thickness != UI["vstate"].thickness:
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])
    if MICRON_FORMATTER in new:
        UI["p"].xaxis[0].formatter = UI["vstate"].micron_formatter
        UI["p"].yaxis[0].formatter = UI["vstate"].micron_formatter
    else:
        UI["p"].xaxis[0].formatter = BasicTickFormatter()
        UI["p"].yaxis[0].formatter = BasicTickFormatter()
    if GRIDLINES in new:
        UI["p"].ygrid.grid_line_color = "gray"
        UI["p"].xgrid.grid_line_color = "gray"
        UI["p"].ygrid.grid_line_alpha = 0.6
        UI["p"].xgrid.grid_line_alpha = 0.6
    else:
        UI["p"].ygrid.grid_line_alpha = 0
        UI["p"].xgrid.grid_line_alpha = 0


def cmap_select_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001
    """Callback to change the color map."""
    if not (UI["vstate"].is_categorical and new != "dict"):
        update_renderer("mapper", new)
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])


def blur_spinner_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the blur radius."""
    update_renderer("blur_radius", new)
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def scale_spinner_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the max scale.

    This defines a scale above which small annotations are
    no longer diplayed.

    """
    update_renderer("max_scale", new)
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def slide_select_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001
    """Set up the newly chosen slide."""
    if len(new) == 0:
        return
    slide_path = Path(doc_config["slide_folder"]) / Path(new[0])
    # Reset the data sources for glyph overlays
    UI["pt_source"].data = {"x": [], "y": []}
    UI["box_source"].data = {"x": [], "y": [], "width": [], "height": []}
    UI["ml_source"].data = {"xs": [], "ys": []}
    UI["node_source"].data = {"x_": [], "y_": [], "node_color_": []}
    UI["edge_source"].data = {"x0_": [], "y0_": [], "x1_": [], "y1_": []}
    UI["hover"].tooltips = None
    if len(UI["p"].renderers) > N_PERMANENT_RENDERERS:
        for r in UI["p"].renderers[N_PERMANENT_RENDERERS:].copy():
            UI["p"].renderers.remove(r)
    UI["vstate"].layer_dict = {
        "slide": 0,
        "rect": 1,
        "pts": 2,
        "line": 3,
        "nodes": 4,
        "edges": 5,
    }
    UI["vstate"].slide_path = slide_path
    UI["color_column"].children = []
    UI["type_column"].children = []
    logger.warning("loading %s", slide_path, stacklevel=2)
    populate_layer_list(slide_path.stem, doc_config["overlay_folder"])
    UI["vstate"].wsi = WSIReader.open(slide_path)
    initialise_slide()
    fname = make_safe_name(str(slide_path))
    UI["s"].put(f"http://{host2}:5000/tileserver/slide", data={"slide_path": fname})
    change_tiles("slide")
    populate_table()

    # Load the overlay and graph automatically if set in config
    if doc_config["auto_load"]:
        for f in UI["layer_drop"].menu:
            dummy_attr = DummyAttr(f[0])
            layer_drop_cb(dummy_attr)


def cohort_select_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001
    """Callback to change the cohort selection."""
    populate_slide_list(Path(doc_config["slide_folder"]))


def range_slider_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001
    """Callback to change the range of the color mapper."""
    if UI["vstate"].cprop != "type" and UI["cmap_select"].value != "dict":
        UI["s"].put(
            f"http://{host2}:5000/tileserver/prop_range",
            data={"range": json.dumps(new)},
        )
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])
    UI["color_bar"].color_mapper.low = new[0]
    UI["color_bar"].color_mapper.high = new[1]


def range_checkbox_cb(attr: str, old: list, new: list) -> None:  # noqa: ARG001
    """Callback to toggle range slider endpoints behaviour.

    Toggles if range slider endpoints are fixed or adaptively set to
      the min/max of the data.

    """
    if len(new) == 1:
        # its on, fix range to user specified values
        UI["range_slider"].start = UI["range_min"].value
        UI["range_slider"].end = UI["range_max"].value
        UI["range_slider"].value = (UI["range_min"].value, UI["range_max"].value)
    else:
        # its off, set range to min/max of data
        UI["range_slider"].start = UI["vstate"].min_val
        UI["range_slider"].end = UI["vstate"].max_val
        UI["range_slider"].value = (UI["vstate"].min_val, UI["vstate"].max_val)


def range_min_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the minimum of the range slider."""
    UI["range_slider"].start = new
    if UI["range_slider"].value[0] < new:
        UI["range_slider"].value = (new, UI["range_slider"].value[1])


def range_max_cb(attr: str, old: float, new: float) -> None:  # noqa: ARG001
    """Callback to change the maximum of the range slider."""
    UI["range_slider"].end = new
    if UI["range_slider"].value[1] > new:
        UI["range_slider"].value = (UI["range_slider"].value[0], new)


def handle_graph_layer(attr: MenuItemClick) -> None:  # skipcq: PY-R1000
    """Handle adding a graph layer."""
    do_feats = False
    with Path(attr.item).open("rb") as f:
        if Path(attr.item).suffix == ".json":
            graph_dict = json.load(f)
        else:
            graph_dict = pickle.load(f)
    # convert the values to numpy arrays
    for k, v in graph_dict.items():
        if isinstance(v, list):
            graph_dict[k] = np.array(v)
    node_cm = colormaps[default_cm]
    num_nodes = graph_dict["coordinates"].shape[0]
    if "score" in graph_dict:
        UI["node_source"].data = {
            "x_": graph_dict["coordinates"][:, 0],
            "y_": -graph_dict["coordinates"][:, 1],
            "node_color_": [rgb2hex(node_cm(to_num(v))) for v in graph_dict["score"]],
        }
    else:
        # Default to green
        UI["node_source"].data = {
            "x_": graph_dict["coordinates"][:, 0],
            "y_": -graph_dict["coordinates"][:, 1],
            "node_color_": [rgb2hex((0, 1, 0))] * num_nodes,
        }
    UI["edge_source"].data = {
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
    add_layer("edges")
    add_layer("nodes")
    change_tiles("graph")
    if "graph_overlay" not in UI["type_cmap_select"].options:
        UI["type_cmap_select"].options = [
            *UI["type_cmap_select"].options,
            "graph_overlay",
        ]

    # Add additional data to graph datasource
    for key in graph_dict:
        if key == "feat_names":
            graph_feat_names = graph_dict[key]
            do_feats = True
        elif (
            key not in ["edge_index", "coordinates"]
            and hasattr(graph_dict[key], "__len__")
            and len(graph_dict[key]) == num_nodes
        ):
            # Valid form to add to node data
            UI["node_source"].data[key] = graph_dict[key]

    if do_feats:
        # Set up the node hover tooltips to show feats
        for i in range(min(graph_dict["feats"].shape[1], MAX_FEATS)):
            # more than 10 wont really fit in hover, ignore rest
            UI["node_source"].data[graph_feat_names[i].replace(" ", "_")] = graph_dict[
                "feats"
            ][:, i]

        tooltips = [
            ("Index", "$index"),
            ("(x,y)", "($x, $y)"),
        ]
        tooltips.extend(
            [
                (graph_feat_names[i], f"@{graph_feat_names[i].replace(' ', '_')}")
                for i in range(np.minimum(graph_dict["feats"].shape[1], 9))
            ],
        )
        UI["hover"].tooltips = tooltips


def update_ui_on_new_annotations(ann_types: list[str]) -> None:
    """Update the UI when new annotations are added."""
    UI["vstate"].types = ann_types
    props = UI["s"].get(f"http://{host2}:5000/tileserver/prop_names/all")
    UI["vstate"].props = json.loads(props.text)
    # Update the color type by prop menu
    UI["type_cmap_select"].options = list(UI["vstate"].types)
    if len(UI["node_source"].data["x_"]) > 0:
        UI["type_cmap_select"].options.append("graph_overlay")
    # Update the color type by prop menu
    UI["type_cmap_select"].options = list(UI["vstate"].types)
    if len(UI["node_source"].data["x_"]) > 0:
        UI["type_cmap_select"].options.append("graph_overlay")
    UI["cprop_input"].options = UI["vstate"].props
    UI["cprop_input"].options.append("None")
    if UI["vstate"].props != UI["vstate"].props_old:
        # If color by prop no longer exists, reset to type
        if (
            len(UI["cprop_input"].value) == 0
            or UI["cprop_input"].value[0] not in UI["vstate"].props
        ):
            UI["cprop_input"].value = ["type"]
        UI["vstate"].props_old = UI["vstate"].props
    else:
        # trigger the color by callbacks to update the mapper
        # based on updated property values
        cprop_input_cb(None, None, UI["cprop_input"].value)
        type_cmap_cb(None, None, UI["type_cmap_select"].value)
        # cmap = get_mapper_for_prop(UI["cprop_input"].value[0])
        # update_renderer("mapper", cmap)

    initialise_overlay()
    change_tiles("overlay")


def layer_drop_cb(attr: MenuItemClick) -> None:
    """Setup the newly chosen overlay."""
    # eventually depreciate pkl in  favour of json
    if Path(attr.item).suffix in [".json", ".pkl"]:
        # its a graph
        handle_graph_layer(attr)
        return

    # Otherwise it's a tile-based overlay of some form
    fname = make_safe_name(attr.item)
    resp = UI["s"].put(
        f"http://{host2}:5000/tileserver/overlay",
        data={"overlay_path": fname},
    )
    resp = json.loads(resp.text)

    if Path(attr.item).suffix in [".db", ".dat", ".geojson"]:
        update_ui_on_new_annotations(resp)
    else:
        if resp != "slide":
            add_layer(resp)
        change_tiles(resp)


def layer_select_cb(attr: ButtonClick) -> None:  # noqa: ARG001
    """Callback to handle toggling specific annotation types on and off."""
    build_predicate()
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def fixed_layer_select_cb(obj: Button, attr: ButtonClick) -> None:  # noqa: ARG001
    """Callback to handle toggling non-annotation layers on and off."""
    key = UI["vstate"].layer_dict[obj.label]
    if obj.label == "edges":
        if not UI["p"].renderers[key].visible:
            UI["p"].renderers[key].visible = True
        else:
            UI["p"].renderers[key].visible = False
    elif obj.label == "nodes":
        if UI["p"].renderers[key].glyph.fill_alpha == 0:
            UI["p"].renderers[key].glyph.fill_alpha = UI["overlay_alpha"].value
            UI["p"].renderers[key].glyph.line_alpha = UI["overlay_alpha"].value
        else:
            UI["p"].renderers[key].glyph.fill_alpha = 0.0
            UI["p"].renderers[key].glyph.line_alpha = 0.0
    elif UI["p"].renderers[key].alpha == 0:
        UI["p"].renderers[key].alpha = float(obj.name)
    else:
        obj.name = str(UI["p"].renderers[key].alpha)  # save old alpha
        UI["p"].renderers[key].alpha = 0.0


def layer_slider_cb(
    obj: Slider,
    attr: str,  # noqa: ARG001
    old: float,  # noqa: ARG001
    new: float,
) -> None:
    """Callback to handle changing the alpha of a layer."""
    if obj.name.split("_")[0] == "nodes":
        set_alpha_glyph(
            UI["p"].renderers[UI["vstate"].layer_dict[obj.name.split("_")[0]]].glyph,
            new,
        )
    elif obj.name.split("_")[0] == "edges":
        UI["p"].renderers[
            UI["vstate"].layer_dict[obj.name.split("_")[0]]
        ].glyph.line_alpha = new
    else:
        UI["p"].renderers[UI["vstate"].layer_dict[obj.name.split("_")[0]]].alpha = new


def color_input_cb(
    obj: ColorPicker,
    attr: str,  # noqa: ARG001
    old: str,  # noqa: ARG001
    new: str,
) -> None:
    """Callback to handle changing the color of an annotation type."""
    UI["vstate"].mapper[UI["vstate"].orig_types[obj.name]] = (*hex2rgb(new), 1)
    if UI["vstate"].cprop == "type":
        update_renderer("mapper", UI["vstate"].mapper)
    UI["vstate"].update_state = 1
    UI["vstate"].to_update.update(["overlay"])


def bind_cb_obj(cb_obj: Model, cb: Callable[[Model, str, Any, Any], None]) -> Callable:
    """Wrapper to bind a callback to a bokeh object."""

    def wrapped(attr: str, old: Any, new: Any) -> None:  # noqa: ANN401
        """Wrapper function."""
        cb(cb_obj, attr, old, new)

    return wrapped


def bind_cb_obj_tog(cb_obj: Model, cb: Callable[[Model, Any], None]) -> Callable:
    """Wrapper to bind a callback to a bokeh toggle object."""

    def wrapped(attr: ButtonClick) -> None:
        """Wrapper function."""
        cb(cb_obj, attr)

    return wrapped


def model_drop_cb(attr: str, old: str, new: str) -> None:  # noqa: ARG001
    """Callback to handle model selection."""
    UI["vstate"].current_model = new


def to_model_cb(attr: ButtonClick) -> None:  # noqa: ARG001
    """Callback to run currently selected model."""
    if UI["vstate"].current_model == "hovernet":
        segment_on_box()
    # add other models here
    elif UI["vstate"].current_model == "gpt-vision":
        gpt_inference()
    else:  # pragma: no cover
        logger.warning("unknown model")


def type_cmap_cb(attr: str, old: list[str], new: list[str]) -> None:  # noqa: ARG001
    """Callback to handle changing a type-specific color property."""
    if len(new) == 0:
        # Remove type-specific coloring
        UI["type_cmap_select"].options = [*UI["vstate"].types, "graph_overlay"]
        UI["s"].put(
            f"http://{host2}:5000/tileserver/secondary_cmap",
            data={
                "type_id": json.dumps("None"),
                "prop": "None",
                "cmap": json.dumps(default_cm),
            },
        )
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])
        return
    if len(new) == 1:
        # Find out what still has to be selected
        if new[0] in [*UI["vstate"].types, "graph_overlay"]:
            if new[0] == "graph_overlay":
                UI["type_cmap_select"].options = [
                    key
                    for key in UI["node_source"].data
                    if key not in ["x_", "y_", "node_color_"]
                ] + [new[0]]
            else:
                UI["type_cmap_select"].options = [*UI["vstate"].props, new[0]]
        else:
            UI["type_cmap_select"].options = [
                *UI["vstate"].types,
                new[0],
                "graph_overlay",
            ]
    else:
        # Both are selected, update the renderer
        if new[1] in UI["vstate"].types:
            # Make sure the type is the first one
            UI["type_cmap_select"].value = [new[1], new[0]]
            return
        if new[0] == "graph_overlay":
            # Adjust the node color in source if prop exists
            if new[1] in UI["node_source"].data:
                node_cm = colormaps[default_cm]
                # UI["node_source"].data["node_color_"] = [
                #     rgb2hex(node_cm((to_num(v) - minv) / (maxv - minv)))
                # for v in UI["node_source"].data[new[1]]]
                UI["node_source"].data["node_color_"] = [
                    rgb2hex(node_cm(to_num(v))) for v in UI["node_source"].data[new[1]]
                ]
            return
        cmap = get_mapper_for_prop(new[1])  # separate cmap select ?
        UI["s"].put(
            f"http://{host2}:5000/tileserver/secondary_cmap",
            data={
                "type_id": json.dumps(UI["vstate"].orig_types.get(new[0], new[0])),
                "prop": new[1],
                "cmap": json.dumps(cmap),
            },
        )

        UI["color_bar"].color_mapper.palette = make_color_seq_from_cmap(
            colormaps[default_cm],
        )
        UI["color_bar"].visible = True
        UI["vstate"].update_state = 1
        UI["vstate"].to_update.update(["overlay"])


def save_cb(attr: ButtonClick) -> None:  # noqa: ARG001
    """Callback to handle saving annotations."""
    save_path = make_safe_name(
        str(
            doc_config["overlay_folder"]
            / (UI["vstate"].slide_path.stem + "_saved_anns.db"),
        ),
    )
    UI["s"].post(
        f"http://{host2}:5000/tileserver/commit",
        data={"save_path": save_path},
    )


def tap_event_cb(event: DoubleTap) -> None:
    """Callback to handle double tap events to inspect annotations."""
    resp = UI["s"].get(f"http://{host2}:5000/tileserver/tap_query/{event.x}/{-event.y}")
    data_dict = json.loads(resp.text)

    popup_table.source.data = {
        "property": list(data_dict.keys()),
        "value": list(data_dict.values()),
    }


def gpt_inference() -> None:
    """Callback to send selected region of the slide to gpt-vision.

    Will use openai API to send image ROI to gpt asssistant instucted to provide
    accurate descriptions of provided histology images, and will display the
    generated text in a popup window.

    """
    # reset image plot size
    im_fig.height = 350
    im_fig.width = 350
    im_fig.x_range.end = 100
    im_fig.y_range.end = 100
    # also reset image source
    im_fig.image_rgba(image=[], x=0, y=0, dw=100, dh=100)

    if len(UI["box_source"].data["x"]) > 0:
        x = round(
            UI["box_source"].data["x"][0] - 0.5 * UI["box_source"].data["width"][0],
        )
        y = -round(
            UI["box_source"].data["y"][0] + 0.5 * UI["box_source"].data["height"][0],
        )
        width = round(UI["box_source"].data["width"][0])
        height = round(UI["box_source"].data["height"][0])

    else:
        # find the box that contains all the lines in ml_source
        xs = UI["ml_source"].data["xs"]
        ys = UI["ml_source"].data["ys"]
        x = round(min([min(x) for x in xs]))
        y = -round(max([max(y) for y in ys]))
        width = round(max([max(x) for x in xs]) - x)
        height = -round(min([min(y) for y in ys]) + y)

        if max(width, height) > 1300:
            # we will resize the region to 1500px max by reading at lower res
            scale = 1300 / max(width, height)
        else:
            scale = 1.0
        # pad the box a bit
        x -= int(100 / scale)
        y -= int(100 / scale)
        width += int(200 / scale)
        height += int(200 / scale)

    if len(UI["ml_source"].data["xs"]) > 0:
        prompt = gpt_interface.prompt_ann
    else:
        # we've drawn nothing, so use the no annotation prompt
        prompt = gpt_interface.prompt_no_ann

    if max(width, height) > 1500:
        # we will resize the region to 1500px max by reading at lower res
        scale = 1500 / max(width, height)
    else:
        scale = 1.0

    print(
        f"preparing region x: {x}, y: {y}, width: {width}, height: {height} to gpt-vision.",
    )

    if scale < 1.0:
        read_res = UI["vstate"].wsi.info.mpp / scale

        region = UI["vstate"].wsi.read_bounds(
            (x, y, x + width, y + height),
            read_res,
            "mpp",
        )
    else:
        # just read at baseline
        region = UI["vstate"].wsi.read_bounds(
            (x, y, x + width, y + height),
        )
    xs = UI["ml_source"].data["xs"]
    ys = UI["ml_source"].data["ys"]
    # use opencv to draw the polylines on the region
    for i in range(len(xs)):
        cv2.polylines(
            region,
            np.array(
                [(np.array([xs[i], -np.array(ys[i])]).T - np.array([x, y])) * scale],
                dtype=np.int32,
            ),
            False,
            (0, 255, 0),
            3,
        )
    # convert image to base64
    img_array = np.array(Image.fromarray(region).convert("RGBA"), dtype=np.uint8)
    img_array = img_array.view(dtype=np.uint32).reshape(img_array.shape[:-1])
    img_array = np.flipud(
        img_array,
    )  # Flip the image array vertically because Bokeh origin is at the bottom left
    # img_array[:, :, [0, 1, 2, 3]] = img_array[:, :, [2, 1, 0, 3]]  # Swap bytes from BGRA to RGBA

    # Display the image in the Bokeh figure
    print(f"image size to be sent is: {img_array.shape}")
    aspect_ratio = img_array.shape[1] / img_array.shape[0]
    if aspect_ratio > 1:
        im_fig.height = int(350 / aspect_ratio)
        im_fig.y_range.end = 100 / aspect_ratio
        im_fig.image_rgba(image=[img_array], x=0, y=0, dw=100, dh=100 / aspect_ratio)
    else:
        im_fig.width = int(350 * aspect_ratio)
        im_fig.x_range.end = 100 * aspect_ratio
        im_fig.image_rgba(image=[img_array], x=0, y=0, dw=100 * aspect_ratio, dh=100)
    prompt_input.value = prompt
    # dialog.visible = True
    gpt_interface.add_img(Image.fromarray(region))


def segment_on_box() -> None:
    """Callback to run hovernet on a region of the slide.

    Will run NucleusInstanceSegmentor on selected region of wsi defined
    by the box in box_source.

    """
    # Make a mask defining the box
    thumb = UI["vstate"].wsi.slide_thumbnail()
    conv_mpp = UI["vstate"].dims[0] / thumb.shape[1]
    msg = f'box tl: {UI["box_source"].data["x"][0]}, {UI["box_source"].data["y"][0]}'
    logger.info(msg)
    x = round(
        (UI["box_source"].data["x"][0] - 0.5 * UI["box_source"].data["width"][0])
        / conv_mpp,
    )
    y = -round(
        (UI["box_source"].data["y"][0] + 0.5 * UI["box_source"].data["height"][0])
        / conv_mpp,
    )
    width = round(UI["box_source"].data["width"][0] / conv_mpp)
    height = round(UI["box_source"].data["height"][0] / conv_mpp)

    mask = np.zeros((thumb.shape[0], thumb.shape[1]), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 1

    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        num_loader_workers=4,
        num_postproc_workers=8,
        batch_size=24,
    )
    tmp_save_dir = Path(tempfile.mkdtemp())
    tmp_mask_dir = Path(tempfile.mkdtemp())
    Image.fromarray(mask).save(tmp_mask_dir / "mask.png")

    # Run hovernet inside the box
    UI["vstate"].model_mpp = inst_segmentor.ioconfig.save_resolution["resolution"]
    inst_segmentor.predict(
        [UI["vstate"].slide_path],
        [tmp_mask_dir / "mask.png"],
        save_dir=tmp_save_dir / "hover_out",
        mode="wsi",
        on_gpu=torch.cuda.is_available(),
        crash_on_exception=True,
    )

    fname = make_safe_name(tmp_save_dir / "hover_out" / "0.dat")
    resp = UI["s"].put(
        f"http://{host2}:5000/tileserver/annotations",
        data={"file_path": fname, "model_mpp": json.dumps(UI["vstate"].model_mpp)},
    )
    ann_types = json.loads(resp.text)
    update_ui_on_new_annotations(ann_types)

    # Clean up temp files
    rmtree(tmp_save_dir)
    rmtree(tmp_mask_dir)


# endregion

# Set up main window
slide_wins = row(
    children=[],
    name="slide_windows",
    sizing_mode="stretch_both",
)
# and the controls
control_tabs = Tabs(tabs=[], name="ui_layout")
# Slide info div
slide_info = Div(
    text="",
    name="description",
    width=800,
    height=200,
    sizing_mode="stretch_width",
)


def gather_ui_elements(  # noqa: PLR0915
    vstate: ViewerState,
    win_num: int,
) -> tuple[Column, Column, dict]:
    """Gather all the ui elements into a dict.

    Defines and gathers the main UI elements for a window, excluding any
    elements that have been deactivated in the config file.

    Args:
        vstate: the ViewerState object for the window
        win_num: the window number (0 or 1)

    Returns:
        A tuple containing the layouts for the main and extra options tabs of the UI,
        and a dict containing all the UI elements for ease of acess.

    """
    # Define all the various widgets
    res_switch = RadioButtonGroup(labels=["1x", "2x"], active=1, name=f"res{win_num}")

    slide_alpha = Slider(
        title="Slide Alpha",
        start=0,
        end=1,
        step=0.05,
        value=1.0,
        width=200,
        sizing_mode="stretch_width",
        name=f"slide_alpha{win_num}",
    )

    overlay_alpha = Slider(
        title="Overlay Alpha",
        start=0,
        end=1,
        step=0.05,
        value=0.75,
        width=200,
        sizing_mode="stretch_width",
        name=f"overlay_alpha{win_num}",
    )

    edge_size_spinner = Spinner(
        title="Edge thickness:",
        low=0,
        high=10,
        step=1,
        value=1,
        width=60,
        height=50,
        sizing_mode="stretch_width",
        name=f"edge_size{win_num}",
    )

    pt_size_spinner = Spinner(
        title="Pt. Size:",
        low=0,
        high=20,
        step=1,
        value=4,
        width=60,
        height=50,
        sizing_mode="stretch_width",
        name=f"pt_size{win_num}",
    )

    slide_toggle = Toggle(
        label="Slide",
        active=True,
        button_type="success",
        width=90,
        sizing_mode="stretch_width",
        name=f"slide_toggle{win_num}",
    )
    overlay_toggle = Toggle(
        label="Overlay",
        active=True,
        button_type="success",
        width=90,
        sizing_mode="stretch_width",
        name=f"overlay_toggle{win_num}",
    )
    filter_tooltip = Tooltip(
        content=HTML(
            """Enter a filter string that is a valid string for AnnotationStore
              'where' argument. It will be used to filter the annotations displayed.
                <br>E.g: props['prob']>0.5
            """,
        ),
        position="right",
        css_classes=["help_tt"],
        stylesheets=[help_ss],
    )
    filter_input = TextInput(
        value="None",
        title="Filter:",
        sizing_mode="stretch_width",
        name=f"filter{win_num}",
        description=filter_tooltip,
    )
    cprop_tooltip = Tooltip(
        content="Choose a property to color annotations by",
        position="right",
    )
    cprop_input = MultiChoice(
        title="color by:",
        max_items=1,
        options=[get_from_config(["default_cprop"], "type")],
        value=[get_from_config(["default_cprop"], "type")],
        search_option_limit=5000,
        sizing_mode="stretch_width",
        name=f"cprop{win_num}",
        description=cprop_tooltip,
    )
    cohort_select = MultiChoice(
        title="Select Cohort:",
        max_items=3,
        options=list(get_from_config(["cohorts"], {}).keys()),
        value=list(get_from_config(["cohorts"], {"*": "*"}).keys())[0:1],
        search_option_limit=5000,
        sizing_mode="stretch_width",
        name=f"cohort_select{win_num}",
    )
    slide_tt = Tooltip(
        content=HTML(
            """Select a slide. Overlays whose filenames contain the slide stem
            will be available below.""",
        ),
        position="right",
        css_classes=["help_tt"],
        stylesheets=[help_ss],
    )
    slide_select = MultiChoice(
        title="Select Slide:",
        max_items=1,
        options=[get_from_config(["first_slide"], "*")],
        value=[get_from_config(["first_slide"], "*")],
        search_option_limit=5000,
        sizing_mode="stretch_width",
        name=f"slide_select{win_num}",
        description=slide_tt,
    )
    cmmenu = [
        ("jet", "jet"),
        ("coolwarm", "coolwarm"),
        ("viridis", "viridis"),
        ("dict", "dict"),
    ]
    cmap_tooltip = Tooltip(
        content=HTML(
            """Choose a colormap. If the property being colored by is categorical,
            dict should be used.""",
        ),
        position="right",
        css_classes=["help_tt"],
        stylesheets=[help_ss],
    )
    cmap_select = Select(
        title="Cmap",
        options=cmmenu,
        width=60,
        value=get_from_config(["UI_settings", "mapper"], "coolwarm"),
        height=45,
        sizing_mode="stretch_width",
        name=f"cmap{win_num}",
        description=cmap_tooltip,
    )
    blur_spinner = Spinner(
        title="Blur:",
        low=0,
        high=20,
        step=1,
        value=get_from_config(["UI_settings", "blur_radius"], 0),
        width=60,
        height=50,
        sizing_mode="stretch_width",
        name=f"blur{win_num}",
    )
    scale_tt = Tooltip(
        content=HTML(
            """Controls scale at which small annotations are no longer shown. Smaller
            values -> small objects will only appear when zoomed in.""",
        ),
        position="right",
        css_classes=["help_tt"],
        stylesheets=[help_ss],
    )
    scale_spinner = Spinner(
        title="max scale:",
        low=0,
        high=540,
        step=8,
        value=get_from_config(["UI_settings", "max_scale"], 16),
        width=60,
        height=50,
        sizing_mode="stretch_width",
        name=f"scale{win_num}",
        description=scale_tt,
    )
    to_model_button = Button(
        label="Run",
        button_type="success",
        width=80,
        max_width=90,
        height=35,
        sizing_mode="stretch_width",
        name=f"to_model{win_num}",
    )
    model_tt = Tooltip(
        content=HTML("""Must select a region before running model"""),
        position="right",
        css_classes=["help_tt"],
        stylesheets=[help_ss],
    )
    model_drop = Select(
        title="choose model:",
        options=["hovernet", "gpt-vision"],
        height=25,
        width=120,
        max_width=120,
        sizing_mode="stretch_width",
        name=f"model_drop{win_num}",
        description=model_tt,
    )
    save_button = Button(
        label="Save",
        button_type="success",
        max_width=90,
        width=80,
        height=35,
        sizing_mode="stretch_width",
        name=f"save_button{win_num}",
    )
    type_cprop_tt = Tooltip(
        content=HTML(
            """Select a type of object, and a property to color by. Objects of
            selected type will be colored by the selected property.
            This will override the global 'color by' property for that type.""",
        ),
        position="right",
        css_classes=["help_tt"],
        stylesheets=[help_ss],
    )
    type_cmap_select = MultiChoice(
        title="color type by property:",
        max_items=2,
        options=["*"],
        search_option_limit=5000,
        sizing_mode="stretch_width",
        name=f"type_cmap{win_num}",
        description=type_cprop_tt,
    )
    layer_boxes = [
        Toggle(
            label=t,
            active=True,
            width=100,
            max_width=100,
            sizing_mode="stretch_width",
        )
        for t in vstate.types
    ]
    lcolors = [
        ColorPicker(color=col[0:3], width=60, max_width=60, sizing_mode="stretch_width")
        for col in vstate.colors
    ]
    layer_drop = Dropdown(
        label="Add Overlay",
        button_type="warning",
        menu=[None],
        sizing_mode="stretch_width",
        name=f"layer_drop{win_num}",
    )
    opt_buttons = CheckboxButtonGroup(
        labels=["Filled", "Microns", "Grid"],
        active=[0],
        sizing_mode="stretch_width",
        name=f"opt_buttons{win_num}",
    )
    range_checkbox = CheckboxButtonGroup(
        labels=["Fixed Range:"],
        active=[0],
        max_width=100,
        sizing_mode="stretch_width",
        name=f"range_checkbox{win_num}",
    )
    range_min = NumericInput(
        value=0,
        max_width=60,
        sizing_mode="stretch_width",
        name=f"range_min{win_num}",
        mode="float",
    )
    range_max = NumericInput(
        value=1,
        max_width=60,
        sizing_mode="stretch_width",
        name=f"range_max{win_num}",
        mode="float",
    )
    range_slider = RangeSlider(
        start=0,
        end=1,
        value=(0, 1),
        step=0.05,
        title="prop range",
        width=200,
        sizing_mode="stretch_width",
        name=f"range_slider{win_num}",
    )

    # Associate callback functions to the widgets
    slide_alpha.on_change("value", slide_alpha_cb)
    overlay_alpha.on_change("value", overlay_alpha_cb)
    res_switch.on_change("active", res_switch_cb)
    pt_size_spinner.on_change("value", pt_size_cb)
    edge_size_spinner.on_change("value", edge_size_cb)
    slide_select.on_change("value", slide_select_cb)
    cohort_select.on_change("value", cohort_select_cb)
    save_button.on_click(save_cb)
    cmap_select.on_change("value", cmap_select_cb)
    blur_spinner.on_change("value", blur_spinner_cb)
    scale_spinner.on_change("value", scale_spinner_cb)
    to_model_button.on_click(to_model_cb)
    js_popup_code = """
        if (model_drop.value == 'gpt-vision') {
            var gpt_popup = document.getElementById('gpt-popup');
            gpt_popup.classList.remove('hidden');
        }
    """
    callback = CustomJS(args={"model_drop": model_drop}, code=js_popup_code)
    to_model_button.js_on_click(callback)
    model_drop.on_change("value", model_drop_cb)
    layer_drop.on_click(layer_drop_cb)
    opt_buttons.on_change("active", opt_buttons_cb)
    slide_toggle.on_click(slide_toggle_cb)
    overlay_toggle.on_click(overlay_toggle_cb)
    filter_input.on_change("value", filter_input_cb)
    cprop_input.on_change("value", cprop_input_cb)
    type_cmap_select.on_change("value", type_cmap_cb)
    range_slider.on_change("value", range_slider_cb)
    range_checkbox.on_change("active", range_checkbox_cb)
    range_min.on_change("value", range_min_cb)
    range_max.on_change("value", range_max_cb)

    # Create some layouts
    type_column = column(children=layer_boxes, name=f"type_column{win_num}")
    color_column = column(
        children=lcolors,
        sizing_mode="stretch_width",
        name=f"color_column{win_num}",
    )

    slide_row = row([slide_toggle, slide_alpha], sizing_mode="stretch_width")
    overlay_row = row([overlay_toggle, overlay_alpha], sizing_mode="stretch_width")
    cmap_row = row(
        [cmap_select, scale_spinner, blur_spinner],
        sizing_mode="stretch_width",
    )
    model_row = row(
        [to_model_button, save_button, model_drop],
        sizing_mode="stretch_width",
    )
    type_select_row = row(
        children=[type_column, color_column],
        sizing_mode="stretch_width",
    )
    range_row = row(
        [range_checkbox, range_min, range_max],
        sizing_mode="stretch_width",
        name=f"range_row{win_num}",
    )

    # Make element dictionaries
    ui_elements_1 = dict(
        zip(
            [
                "cohort_select",
                "slide_select",
                "layer_drop",
                "slide_row",
                "overlay_row",
                "filter_input",
                "cprop_input",
                "cmap_row",
                "type_cmap_select",
                "model_row",
                "type_select_row",
            ],
            [
                cohort_select,
                slide_select,
                layer_drop,
                slide_row,
                overlay_row,
                filter_input,
                cprop_input,
                cmap_row,
                type_cmap_select,
                model_row,
                type_select_row,
            ],
        ),
    )
    if "ui_elements_1" in doc_config:
        # Only add the elements specified in config file
        ui_layout = column(
            [
                ui_elements_1[el]
                for el in ui_elements_1
                if get_from_config(["ui_elements_1", el], 1) == 1
            ],
            sizing_mode="stretch_width",
        )
    else:
        ui_layout = column(
            list(ui_elements_1.values()),
            sizing_mode="stretch_width",
        )
    if len(get_from_config(["cohorts"], {})) < 2:
        # if no cohorts, or only one, dont need cohort select
        ui_elements_1.pop("cohort_select")

    # Elements in the secondary controls tab
    ui_elements_2 = dict(
        zip(
            [
                "opt_buttons",
                "pt_size_spinner",
                "edge_size_spinner",
                "res_switch",
                "range_row",
                "range_slider",
                "channel_select",
            ],
            [
                opt_buttons,
                pt_size_spinner,
                edge_size_spinner,
                res_switch,
                range_row,
                range_slider,
                create_channel_color_ui(),
            ],
        ),
    )
    if "ui_elements_2" in doc_config:
        # Only add the elements specified in config file
        extra_options = column(
            [
                ui_elements_2[el]
                for el in ui_elements_2
                if get_from_config(["ui_elements_2", el], 1) == 1
            ],
        )
    else:
        extra_options = column(
            list(ui_elements_2.values()),
        )
    # Put everything together
    elements_dict = {
        **ui_elements_1,
        **ui_elements_2,
        "color_column": color_column,
        "type_column": type_column,
        "overlay_alpha": overlay_alpha,
        "cmap_select": cmap_select,
        "slide_alpha": slide_alpha,
        "range_min": range_min,
        "range_max": range_max,
        "range_checkbox": range_checkbox,
    }

    return ui_layout, extra_options, elements_dict


def make_window(vstate: ViewerState) -> dict:  # noqa: PLR0915
    """Make a new window for a slide.

    Creates a new window for the slide, including all the UI elements and
    the main viewing window.

    Args:
        vstate: the ViewerState object for the window
    Returns:
        A dict containing the UI elements and other elements associated with the
        window that we may need to reference, for ease of access.

    """
    win_num = str(len(windows))
    if len(windows) == 1:
        slide_wins.children[0].width = 800
        p = figure(
            x_range=slide_wins.children[0].x_range,
            y_range=slide_wins.children[0].y_range,
            x_axis_type="linear",
            y_axis_type="linear",
            width=800,
            height=1000,
            tools=tool_str,
            active_scroll="wheel_zoom",
            output_backend="webgl",
            hidpi=True,
            match_aspect=False,
            lod_factor=200000,
            sizing_mode="stretch_both",
            name=f"slide_window{win_num}",
        )
        init_z = first_z[0]
    else:
        p = figure(
            x_range=(0, vstate.dims[0]),
            y_range=(0, vstate.dims[1]),
            x_axis_type="linear",
            y_axis_type="linear",
            width=1700,
            height=1000,
            tools=tool_str,
            active_scroll="wheel_zoom",
            output_backend="webgl",
            hidpi=True,
            match_aspect=False,
            lod_factor=200000,
            sizing_mode="stretch_both",
            name=f"slide_window{win_num}",
        )
        init_z = get_level_by_extent((0, p.y_range.start, p.x_range.end, 0))
        first_z[0] = init_z
    p.axis.visible = False
    p.toolbar.tools[1].zoom_on_axis = False

    # Tap query popup callbacks
    js_popup_code = """
        var popupContent = document.getElementById('props-popup');
        if (popupContent.classList.contains('hidden')) {
            popupContent.classList.remove('hidden');
            }
    """
    p.on_event(DoubleTap, tap_event_cb)
    p.js_on_event(DoubleTap, CustomJS(code=js_popup_code))

    # Set up a session for communicating with tile server
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))

    resp = s.get(f"http://{host2}:5000/tileserver/session_id")
    user = resp.cookies.get("session_id")
    if curdoc().session_context:
        curdoc().session_context.request.arguments["user"] = user

    # Set up the main slide window
    vstate.init_z = init_z
    ts1 = make_ts(
        f"//{host}:{port}/tileserver/layer/slide/{user}/zoomify/TileGroup1"
        r"/{z}-{x}-{y}"
        f"@{vstate.res}x.jpg",
        vstate.num_zoom_levels,
    )
    p.add_tile(ts1, smoothing=True, level="image", render_parents=True)

    p.grid.grid_line_color = None
    box_source = ColumnDataSource({"x": [], "y": [], "width": [], "height": []})
    pt_source = ColumnDataSource({"x": [], "y": []})
    ml_source = ColumnDataSource({"xs": [], "ys": []})
    r = p.rect(
        "x",
        "y",
        "width",
        "height",
        source=box_source,
        fill_alpha=0,
        line_width=3,
    )
    c = p.circle(
        "x", "y", source=pt_source, color="red", radius=3, radius_units="screen"
    )
    ml = p.multi_line("xs", "ys", source=ml_source, color="green", line_width=3)
    p.add_tools(BoxEditTool(renderers=[r], num_objects=1))
    p.add_tools(PointDrawTool(renderers=[c]))
    p.add_tools(FreehandDrawTool(renderers=[ml], num_objects=5))
    # p.add_tools(LassoSelectTool())
    p.add_tools(TapTool())
    clear_code = """
                box_source.clear()
                pt_source.clear()
                ml_source.clear()
                """
    p.add_tools(
        CustomAction(
            callback=CustomJS(
                args={
                    "box_source": box_source,
                    "pt_source": pt_source,
                    "ml_source": ml_source,
                },
                code=clear_code,
            ),
            description="Clear",
        ),
    )

    if get_from_config(["opts", "hover_on"], 0) == 0:
        p.toolbar.active_inspect = None

    p.renderers[0].tile_source.max_zoom = 10

    # Add graph stuff
    node_source = ColumnDataSource({"x_": [], "y_": [], "node_color_": []})
    edge_source = ColumnDataSource({"x0_": [], "y0_": [], "x1_": [], "y1_": []})
    vstate.graph_node = Circle(
        x="x_",
        y="y_",
        fill_color="node_color_",
        radius=3,
        radius_units="screen",
    )
    vstate.graph_edge = Segment(x0="x0_", y0="y0_", x1="x1_", y1="y1_")
    p.add_glyph(node_source, vstate.graph_node)
    node_source.selected.on_change("indices", node_select_cb)
    if not get_from_config(["opts", "nodes_on"], default=True):
        p.renderers[-1].glyph.fill_alpha = 0
        p.renderers[-1].glyph.line_alpha = 0
    p.add_glyph(edge_source, vstate.graph_edge)
    if not get_from_config(["opts", "edges_on"], default=False):
        p.renderers[-1].visible = False
    vstate.layer_dict["nodes"] = len(p.renderers) - 2
    vstate.layer_dict["edges"] = len(p.renderers) - 1
    hover = HoverTool(renderers=[p.renderers[-2]])
    p.add_tools(hover)

    color_bar = ColorBar(
        color_mapper=LinearColorMapper(
            make_color_seq_from_cmap(colormaps[default_cm]),
        ),
        label_standoff=12,
    )
    color_bar.color_mapper.low = 0
    color_bar.color_mapper.high = 1
    if get_from_config(["opts", "colorbar_on"], 1) == 1:
        p.add_layout(color_bar, "below")
    vstate.cprop = get_from_config(["default_cprop"], "type")

    # Define UI elements
    ui_layout, extra_options, elements_dict = gather_ui_elements(vstate, win_num)

    if len(windows) == 0:
        # Setting up the first window
        controls.append(
            TabPanel(
                child=Tabs(
                    tabs=[
                        TabPanel(child=ui_layout, title="Main"),
                        TabPanel(child=extra_options, title="More Opts"),
                    ],
                ),
                title="window 1",
            ),
        )
        controls.append(TabPanel(child=Div(), title="window 2"))
        windows.append(p)
    else:
        # Setting up a dual window
        control_tabs.tabs[1] = TabPanel(
            child=Tabs(
                tabs=[
                    TabPanel(child=ui_layout, title="Main"),
                    TabPanel(child=extra_options, title="More Opts"),
                ],
            ),
            title="window 2",
            closable=True,
        )
        slide_wins.children.append(p)

    # Return a dictionary collecting all the things related to window
    return {
        **elements_dict,
        "p": p,
        "vstate": vstate,
        "s": s,
        "box_source": box_source,
        "pt_source": pt_source,
        "ml_source": ml_source,
        "node_source": node_source,
        "edge_source": edge_source,
        "hover": hover,
        "user": user,
        "color_bar": color_bar,
    }


# Main ui containers
UI = UIWrapper()
windows = []
controls = []
win_dicts = []

# Popup for annotation viewing on double click
popup_div = Div(
    width=300,
    height=300,
    name="popup_div",
    text="test popup",
)
template_str = r"""<% if (typeof value === 'number' || !isNaN(parseFloat(value)))
                    { %> <%= parseFloat(value).toFixed(3) %> <% }
                    else { %> <%= value %> <% } %>"""
formatter = HTMLTemplateFormatter(
    template=template_str,
)
popup_table = DataTable(
    source=ColumnDataSource({"property": [], "value": []}),
    columns=[
        TableColumn(field="property", title="Property"),
        TableColumn(
            field="value",
            title="Value",
            formatter=formatter,
        ),
    ],
    index_position=None,
    width=300,
    height=300,
    name="popup_window",
)


def submit_cb(event: ButtonClick) -> None:
    """Callback to submit prompt to gpt-vision via api."""
    if gpt_interface.api_key is None:
        # no api key found
        prompt_input.value = (
            "No API key found. Please set the OPENAI_API_KEY environment variable."
        )
        return
    prompt = prompt_input.value

    text = gpt_interface.send(prompt)

    # update the popup
    prompt_input.value = text


def gpt_save_cb(event: ButtonClick) -> None:
    """Callback to save the last image sent to gpt-vision to disk,
    together with the prompt and response.

    """
    gpt_interface.save_history(get_from_config(["overlay_folder"]) / "gpt_prompts", 1)


def model_choice_cb(attr: str, old: str, new: str) -> None:
    """Callback to change the model used by gpt-vision."""
    gpt_interface.set_client(new)


def api_input_cb(attr: str, old: str, new: str) -> None:
    """Callback to update api key when input changes."""
    gpt_interface.update_api_key(new)


im_fig = figure(
    x_range=(0, 100),
    y_range=(0, 100),
    height=350,
    width=350,
    toolbar_location=None,
    name="im_fig",
)
im_fig.axis.visible = False
api_key_input = PasswordInput(
    placeholder="enter api key..",
    title="OpenAI API Key:",
    width=300,
    height=50,
    sizing_mode="stretch_width",
)
prompt_input = TextAreaInput(value=".", rows=7, width=350, height=250)
submit_button = Button(label="Submit", button_type="success")
gpt_save_button = Button(label="Save", button_type="success")
model_choice = Select(
    title="Choose Model",
    options=["llama3.2-vision:11b", "gpt-4o"],
)
close_button = Button(label="Close", button_type="success")
js_popup_code = """
    var popupContent = document.getElementById('gpt-popup');
    popupContent.classList.add('hidden');
"""

close_button.js_on_event(ButtonClick, CustomJS(code=js_popup_code))
submit_button.on_click(submit_cb)
gpt_save_button.on_click(gpt_save_cb)
model_choice.on_change("value", model_choice_cb)
api_key_input.on_change("value", api_input_cb)

dialog_content = Column(
    children=[
        api_key_input,
        im_fig,
        prompt_input,
        Row(children=[model_choice, submit_button, gpt_save_button, close_button]),
    ],
    name="dialog",
)
# dialog = Dialog(visible=True, closable=True, content=im_fig, name="dialog", draggable=True)

# some setup

color_cycler = ColorCycler()
tg = TileGroup()
tool_str = "pan,wheel_zoom,save,copy,fullscreen"
req_args = []
do_doc = False
if curdoc().session_context is not None:
    req_args = curdoc().session_context.request.arguments
    do_doc = True

is_deployed = False
rand_id = token.generate_session_id()
first_z = [1]

# set hosts and ports
if is_deployed:
    host = os.environ.get("HOST")
    host2 = os.environ.get("HOST2")
    port = os.environ.get("PORT")
else:
    host = "127.0.0.1"
    host2 = "127.0.0.1"
    port = "5000"


def update() -> None:
    """Callback to ensure tiles are updated when needed."""
    if UI["vstate"].update_state == DO_UPDATE:
        for layer in UI["vstate"].to_update:
            if layer in UI["vstate"].layer_dict:
                change_tiles(layer)
        UI["vstate"].update_state = NO_UPDATE
        UI["vstate"].to_update = set()
    if UI["vstate"].update_state == PENDING_UPDATE:
        UI["vstate"].update_state = DO_UPDATE


def check_folders() -> None:
    """Update the list of available slides and overlays."""
    populate_slide_list(doc_config["slide_folder"])
    populate_layer_list(
        Path(UI["vstate"].slide_path).stem,
        doc_config["overlay_folder"],
    )


def control_tabs_cb(attr: str, old: int, new: int) -> None:  # noqa: ARG001
    """Callback to handle selecting active window."""
    if new == 1 and len(slide_wins.children) == 1:
        # Make new window
        win_dicts.append(make_window(ViewerState(win_dicts[0]["vstate"].slide_path)))
        win_dicts[1]["vstate"].thickness = win_dicts[0]["vstate"].thickness
        bounds = get_view_bounds(
            UI["vstate"].dims,
            np.array([UI["p"].width, UI["p"].height]),
        )
        UI.active = new
        setup_config_ui_settings(doc_config)
        win_dicts[0]["vstate"].init_z = get_level_by_extent(
            (0, bounds[2], bounds[1], 0),
        )
        UI["vstate"].init = False
    else:
        UI.active = new
        slide_info.text = format_info(UI["vstate"].wsi.info.as_dict())


def control_tabs_remove_cb(
    attr: str,  # noqa: ARG001
    old: list[int],  # noqa: ARG001
    new: list[int],
) -> None:
    """Callback to handle removing a window."""
    if len(new) == 1:
        # Remove the second window
        slide_wins.children.pop()
        slide_wins.children[0].width = 1700  # set back to original size
        control_tabs.tabs.append(TabPanel(child=Div(), title="window 2"))
        win_dicts.pop()
        UI.active = 0


def setup_config_ui_settings(config: dict) -> None:
    """Set up the UI settings from the config file.

    Args:
        config: a dictionary of configuration options

    """
    if "UI_settings" in config:
        for k in config["UI_settings"]:
            update_renderer(k, config["UI_settings"][k])
        if "default_cprop" in config and config["default_cprop"] is not None:
            UI["s"].put(
                f"http://{host2}:5000/tileserver/color_prop",
                data={"prop": json.dumps(config["default_cprop"])},
            )
    # Open up initial slide
    if "default_type_cprop" in config:
        UI["type_cmap_select"].value = list(
            doc_config["default_type_cprop"].values(),
        )
    populate_slide_list(config["slide_folder"])
    UI["slide_select"].value = [str(UI["vstate"].slide_path.name)]
    slide_select_cb(None, None, new=[UI["vstate"].slide_path.name])
    populate_layer_list(
        Path(UI["vstate"].slide_path).stem,
        doc_config["overlay_folder"],
    )


class DocConfig:
    """class to configure and set up a document."""

    def __init__(self: DocConfig) -> None:
        """Initialise the class."""
        self.config = {
            "color_dict": {},
            "initial_views": {},
            "default_cprop": "type",
            "demo_name": "TIAvis",
            "base_folder": Path("/app_data"),
            "slide_folder": Path("/app_data").joinpath("slides"),
            "overlay_folder": Path("/app_data").joinpath("overlays"),
        }
        self.sys_args = None

    def __getitem__(self: DocConfig, key: str) -> Any:  # noqa: ANN401
        """Get an item from the config."""
        return self.config[key]

    def __contains__(self: DocConfig, key: str) -> bool:
        """Check if a key is in the config."""
        return key in self.config

    def set_sys_args(self: DocConfig, argv: list[str]) -> None:
        """Set the system arguments."""
        self.sys_args = argv

    def get_config(self: DocConfig) -> None:
        """Get config info from config.json and/or request args."""
        sys_args = self.sys_args
        if len(sys_args) == 2 and sys_args[1] != "None":  # noqa: PLR2004
            # Only base folder given
            base_folder = Path(sys_args[1])
            if "demo" in req_args:
                self.config["demo_name"] = str(req_args["demo"][0], "utf-8")
                base_folder = base_folder.joinpath(str(req_args["demo"][0], "utf-8"))
            sys_args[1] = base_folder.joinpath("slides")
            sys_args.append(base_folder.joinpath("overlays"))

        slide_folder = Path(sys_args[1])
        base_folder = slide_folder.parent
        overlay_folder = Path(sys_args[2])

        # Load a color_dict and/or slide initial view windows from a json file
        config_file = list(overlay_folder.glob("*config.json"))
        config = self.config
        if len(config_file) > 0:
            config_file = config_file[0]
            with config_file.open() as f:
                config = json.load(f)
                logger.info("loaded config: %s", config)

        config["base_folder"] = base_folder
        config["slide_folder"] = slide_folder
        config["overlay_folder"] = overlay_folder
        config["demo_name"] = self.config["demo_name"]
        if "initial_views" not in config:
            config["initial_views"] = {}

        # Get any extra info from query url
        if "slide" in req_args:
            config["first_slide"] = str(req_args["slide"][0], "utf-8")
            if "window" in req_args:
                config["initial_views"][Path(config["first_slide"]).stem] = [
                    int(s) for s in str(req_args["window"][0], "utf-8")[1:-1].split(",")
                ]
        self.config = config
        self.config["auto_load"] = get_from_config(["auto_load"], 0) == 1

    def setup_doc(self: DocConfig, base_doc: Document) -> tuple[Row, Tabs]:
        """Set up the document.

        Args:
            base_doc: the document to set up
        Returns:
            A tuple containing a layout of the main slide window(s), and
            the controls tab.

        """
        # Set initial slide to first one in base folder
        slide_list = []
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
            slide_list.extend(list(doc_config["slide_folder"].glob(ext)))
            slide_list.extend(
                list(doc_config["slide_folder"].glob(str(Path("*") / ext))),
            )
        first_slide_path = slide_list[0]
        if "first_slide" in self.config:
            first_slide_path = self.config["slide_folder"] / self.config["first_slide"]

        # Make initial window
        win_dicts.append(make_window(ViewerState(first_slide_path)))
        # Set up any initial ui settings from config file
        setup_config_ui_settings(self.config)
        UI["vstate"].init = False

        # Set up main window
        slide_wins.children = windows
        control_tabs.tabs = controls

        control_tabs.on_change("active", control_tabs_cb)
        control_tabs.on_change("tabs", control_tabs_remove_cb)

        # Add the window and controls etc to the document
        base_doc.template_variables["demo_name"] = doc_config["demo_name"]
        base_doc.template_variables["allow_upload"] = (
            get_from_config(
                ["allow_upload"],
                0,
            )
            == 1
        )
        base_doc.add_periodic_callback(update, 220)
        base_doc.add_periodic_callback(check_folders, 5000)
        base_doc.add_root(slide_wins)
        base_doc.add_root(control_tabs)
        base_doc.add_root(popup_table)
        base_doc.add_root(dialog_content)
        base_doc.add_root(slide_info)
        base_doc.title = "Tiatoolbox Visualization Tool"
        base_doc.template_variables["slide_folder"] = make_safe_name(
            doc_config["slide_folder"],
        )
        base_doc.template_variables["overlay_folder"] = make_safe_name(
            doc_config["overlay_folder"],
        )
        return slide_wins, control_tabs


doc_config = DocConfig()
if do_doc:
    # Set up the document
    doc_config.set_sys_args(sys.argv)
    doc_config.get_config()
    # see if there's a metadata .csv file in slides folder
    metadata_file = list(doc_config["slide_folder"].glob("*.csv"))
    if len(metadata_file) > 0:
        metadata_file = metadata_file[0]
        with metadata_file.open() as f:
            metadata = pd.read_csv(f)
            # must have an 'Image File' column to associate with slides
            if "Image File" in metadata.columns:
                metadata = metadata.set_index("Image File")
            else:
                # can't use it so set to None
                metadata = None
    else:
        # no metadata file
        metadata = None
    doc = curdoc()
    slide_wins, control_tabs = doc_config.setup_doc(doc)
