import { describe, expect, it } from "vitest";

import type { SlideManifest } from "../src/api/types";
import {
  createSlideTileGrid,
  downsampleAtZoom,
  mapToSlide,
  slideExtent,
  slideToMap,
  transformGeoJsonGeometryToMap,
} from "../src/map/projection";

const slide: SlideManifest = {
  id: "slide",
  name: "Slide",
  width: 1024,
  height: 512,
  mpp: [0.25, 0.25],
  tileSize: 256,
  maxZoom: 2,
  resolutions: [4, 2, 1],
  mapExtent: [0, -512, 1024, 0],
  rasterTileUrlTemplate: "/tiles/{z}/{x}/{y}",
  associatedOverlays: [],
};

describe("WSI pixel projection", () => {
  it("maps top-left slide coordinates to the OpenLayers fourth quadrant", () => {
    expect(slideExtent(slide)).toEqual([0, -512, 1024, 0]);
    expect(slideToMap([100, 200])).toEqual([100, -200]);
    expect(mapToSlide([100, -200])).toEqual([100, 200]);
  });

  it("keeps z=maxZoom at baseline resolution", () => {
    expect(downsampleAtZoom(9, 0)).toBe(512);
    expect(downsampleAtZoom(9, 9)).toBe(1);
    expect(createSlideTileGrid(slide).getResolutions()).toEqual([4, 2, 1]);
  });

  it("flips every y coordinate in an exact GeoJSON geometry", () => {
    expect(
      transformGeoJsonGeometryToMap({
        type: "Polygon",
        coordinates: [
          [
            [0, 10],
            [20, 30],
            [0, 10],
          ],
        ],
      }),
    ).toEqual({
      type: "Polygon",
      coordinates: [
        [
          [0, -10],
          [20, -30],
          [0, -10],
        ],
      ],
    });
  });
});
