import Projection from "ol/proj/Projection.js";
import TileGrid from "ol/tilegrid/TileGrid.js";

import type { SlideManifest } from "../api/types";

export function slideExtent(
  slide: Pick<SlideManifest, "width" | "height">,
): readonly [number, number, number, number] {
  return [0, -slide.height, slide.width, 0];
}

export function slideToMap(
  coordinate: readonly [number, number],
): [number, number] {
  return [coordinate[0], -coordinate[1]];
}

export function mapToSlide(
  coordinate: readonly [number, number],
): [number, number] {
  return [coordinate[0], -coordinate[1]];
}

export function downsampleAtZoom(maxZoom: number, zoom: number): number {
  return 2 ** (maxZoom - zoom);
}

export function createSlideProjection(slide: SlideManifest): Projection {
  const mpp = slide.mpp ? (slide.mpp[0] + slide.mpp[1]) / 2 : undefined;
  return new Projection({
    code: `TIATOOLBOX:SLIDE:${slide.id}:${slide.revision ?? "current"}`,
    units: "pixels",
    extent: [...slideExtent(slide)],
    ...(mpp === undefined ? {} : { metersPerUnit: mpp * 1e-6 }),
  });
}

export function createSlideTileGrid(slide: SlideManifest): TileGrid {
  return new TileGrid({
    extent: [...slideExtent(slide)],
    origin: [0, 0],
    tileSize: slide.tileSize,
    resolutions: slide.resolutions,
  });
}

export function transformGeoJsonGeometryToMap(
  geometry: Record<string, unknown>,
): Record<string, unknown> {
  return {
    ...geometry,
    coordinates: transformCoordinates(geometry.coordinates),
  };
}

function transformCoordinates(value: unknown): unknown {
  if (!Array.isArray(value)) return value;
  if (
    value.length >= 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number"
  ) {
    return [value[0], -value[1], ...value.slice(2)];
  }
  return value.map(transformCoordinates);
}
