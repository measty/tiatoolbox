"""Visualization package for tiatoolbox."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.visualization import tileserver as tileserver
    from tiatoolbox.visualization.tileserver import TileServer

__all__ = ["TileServer", "tileserver"]


def __getattr__(name: str):
    """Lazily import heavy visualization modules on demand."""
    if name == "TileServer":
        return importlib.import_module(
            "tiatoolbox.visualization.tileserver"
        ).TileServer
    if name == "tileserver":
        return importlib.import_module("tiatoolbox.visualization.tileserver")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
