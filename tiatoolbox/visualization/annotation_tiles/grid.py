"""Coordinate and tile-grid helpers shared by all viewer representations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TileMatrix:
    """A Zoomify-compatible tile matrix in baseline slide pixels.

    Coordinates use a top-left origin with positive X to the right and positive Y
    down. Level zero is the overview and ``max_zoom`` is baseline resolution.

    Args:
        width: Baseline slide width in pixels.
        height: Baseline slide height in pixels.
        tile_size: Display tile edge length in pixels.
    """

    width: int
    height: int
    tile_size: int = 256

    def __post_init__(self) -> None:
        """Validate matrix dimensions."""
        if self.width <= 0 or self.height <= 0:
            msg = "Tile matrix dimensions must be positive."
            raise ValueError(msg)
        if self.tile_size <= 0:
            msg = "Tile size must be positive."
            raise ValueError(msg)

    @property
    def max_zoom(self) -> int:
        """Return the baseline-resolution zoom level."""
        ratio = max(self.width, self.height) / self.tile_size
        return max(0, math.ceil(math.log2(ratio))) if ratio > 1 else 0

    @property
    def resolutions(self) -> tuple[int, ...]:
        """Return source-pixel units per display pixel from overview to detail."""
        return tuple(self.downsample(z) for z in range(self.max_zoom + 1))

    @property
    def map_extent(self) -> tuple[float, float, float, float]:
        """Return the OpenLayers pixel-projection extent."""
        return (0.0, -float(self.height), float(self.width), 0.0)

    def downsample(self, z: int) -> int:
        """Return the baseline downsample for zoom ``z``."""
        self._validate_z(z)
        return 2 ** (self.max_zoom - z)

    def level_dimensions(self, z: int) -> tuple[int, int]:
        """Return the pixel dimensions at zoom ``z``."""
        downsample = self.downsample(z)
        return (
            math.ceil(self.width / downsample),
            math.ceil(self.height / downsample),
        )

    def grid_size(self, z: int) -> tuple[int, int]:
        """Return the number of tiles in X and Y at zoom ``z``."""
        level_width, level_height = self.level_dimensions(z)
        return (
            math.ceil(level_width / self.tile_size),
            math.ceil(level_height / self.tile_size),
        )

    def tile_bounds(
        self,
        z: int,
        x: int,
        y: int,
        *,
        buffer_pixels: float = 0,
        clip_to_slide: bool = False,
    ) -> tuple[float, float, float, float]:
        """Return tile bounds in baseline slide pixels.

        Args:
            z: Zoom level.
            x: Tile column.
            y: Tile row, increasing downwards.
            buffer_pixels: Display-pixel buffer added around each edge.
            clip_to_slide: Clip returned bounds to the slide extent.
        """
        self.validate_tile(z, x, y)
        downsample = self.downsample(z)
        span = self.tile_size * downsample
        buffer_source = float(buffer_pixels) * downsample
        bounds = (
            (x * span) - buffer_source,
            (y * span) - buffer_source,
            ((x + 1) * span) + buffer_source,
            ((y + 1) * span) + buffer_source,
        )
        if not clip_to_slide:
            return bounds
        return (
            max(0.0, bounds[0]),
            max(0.0, bounds[1]),
            min(float(self.width), bounds[2]),
            min(float(self.height), bounds[3]),
        )

    def validate_tile(self, z: int, x: int, y: int) -> None:
        """Raise ``IndexError`` when a tile coordinate is outside the matrix."""
        self._validate_z(z)
        width, height = self.grid_size(z)
        if x < 0 or y < 0 or x >= width or y >= height:
            msg = f"Tile ({z}, {x}, {y}) is outside grid {width}x{height}."
            raise IndexError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the serialisable tile-grid contract used by the frontend."""
        return {
            "coordinateSystem": "baseline-pixels-top-left",
            "width": self.width,
            "height": self.height,
            "tileSize": self.tile_size,
            "maxZoom": self.max_zoom,
            "resolutions": list(self.resolutions),
            "mapExtent": list(self.map_extent),
        }

    def _validate_z(self, z: int) -> None:
        if z < 0 or z > self.max_zoom:
            msg = f"Zoom {z} is outside 0..{self.max_zoom}."
            raise IndexError(msg)
