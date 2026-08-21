import { describe, expect, it } from "vitest";

import Feature from "ol/Feature.js";
import Point from "ol/geom/Point.js";
import Polygon from "ol/geom/Polygon.js";
import { convertStyleToShaders } from "ol/render/webgl/VectorStyleRenderer.js";
import CircleStyle from "ol/style/Circle.js";
import TileGrid from "ol/tilegrid/TileGrid.js";

import type { StoreManifest } from "../src/api/types";
import type { LayerPresentation } from "../src/domain/style-spec";
import {
  compileWebGlStyle,
  createCanvasStyleFunction,
} from "../src/renderers/openlayers/style-compiler";

const presentation: LayerPresentation = {
  visible: true,
  opacity: 1,
  overviewMode: "aggregate",
  fillOpacity: 0.6,
  strokeColor: "#111827",
  strokeWidth: 1,
  pointRadius: 3,
  colorBy: {
    mode: "categorical",
    property: "type",
    categories: [
      { value: 0, color: "#e11d48", visible: true },
      { value: 1, color: "#2563eb", visible: false },
    ],
  },
  rangeFilter: { property: "prob", min: 0.25, max: 0.9 },
};

const store: StoreManifest = {
  id: "cells",
  name: "Cells",
  revision: "1",
  count: 2,
  bounds: [0, 0, 10, 10],
  overlaps: false,
  geometryTypes: { Polygon: 2 },
  properties: [
    { name: "type", kind: "categorical", values: [0, 1] },
    { name: "prob", kind: "numeric", min: 0, max: 1 },
  ],
  representations: [{
    kind: "auto",
    minZoom: 0,
    maxZoom: 9,
    urlTemplate: "/tiles/{z}/{x}/{y}",
    policy: {
      scope: "store-zoom",
      ranges: [
        { minZoom: 0, maxZoom: 3, representation: "aggregate" },
        { minZoom: 4, maxZoom: 6, representation: "centroid" },
        { minZoom: 7, maxZoom: 9, representation: "polygon" },
      ],
    },
  }],
  tileUrlTemplates: { auto: "/tiles/{z}/{x}/{y}" },
  featureUrlTemplate: "/features/{fid}",
};

describe("OpenLayers WebGL style compiler", () => {
  it("produces expressions accepted by OpenLayers 10.9", () => {
    const compiled = compileWebGlStyle(presentation, store);
    const shaders = convertStyleToShaders(compiled.rules, compiled.variables);
    expect(shaders.length).toBeGreaterThan(0);
    expect(shaders.every((entry) => entry.builder !== undefined)).toBe(true);
    const shaderSource = shaders.flatMap((entry) => [
      entry.builder.getFillFragmentShader(),
      entry.builder.getStrokeFragmentShader(),
      entry.builder.getSymbolFragmentShader(),
    ]).filter((source): source is string => source !== null).join("\n");
    expect(shaderSource).toContain("prop_tiatoolbox_aggregate");
    expect(shaderSource).not.toContain("prop__");
  });

  it("accepts continuous colour interpolation and dynamic range variables", () => {
    const compiled = compileWebGlStyle({
      ...presentation,
      colorBy: {
        mode: "numeric",
        property: "prob",
        numeric: { domain: [0, 1], colors: ["#2563eb", "#f59e0b"] },
      },
    }, store);
    expect(() => convertStyleToShaders(compiled.rules, compiled.variables)).not.toThrow();
  });

  it("reads normalized feature colours as a WebGL color attribute", () => {
    const compiled = compileWebGlStyle({
      ...presentation,
      colorBy: {
        mode: "direct",
        property: "color",
        fallback: "#9ca3af",
      },
      rangeFilter: undefined,
    }, store);
    expect(compiled.rules[0]?.style["fill-color"]).toEqual([
      "*",
      ["get", "color"],
      ["color", 255, 255, 255, ["var", "fillOpacity"]],
    ]);
    expect(() => convertStyleToShaders(compiled.rules, compiled.variables)).not.toThrow();
  });

  it("separates aggregate and centroid points from polygon outlines", () => {
    const compiled = compileWebGlStyle(presentation, store);
    expect(compiled.rules).toHaveLength(3);
    expect(compiled.rules[0]?.style["stroke-width"]).toEqual([
      "var",
      "strokeWidth",
    ]);
    expect(compiled.rules[1]?.style["circle-stroke-width"]).toBeUndefined();
    expect(compiled.rules[2]?.style["circle-stroke-width"]).toBeUndefined();
    expect(compiled.rules[1]?.style["circle-radius"]).toBe(3);
    expect(compiled.rules[2]?.style["circle-radius"]).toEqual([
      "interpolate",
      ["linear"],
      ["zoom"],
      4,
      ["*", ["var", "pointRadius"], 0.5],
      6,
      ["var", "pointRadius"],
    ]);
    expect(JSON.stringify(compiled.rules[1]?.filter)).toContain(
      "tiatoolbox_aggregate",
    );
  });

  it("hides only aggregate primitives through a style variable", () => {
    const shown = compileWebGlStyle(presentation, store);
    const hidden = compileWebGlStyle(
      { ...presentation, overviewMode: "hidden" },
      store,
    );
    expect(shown.variables.showAggregates).toBe(1);
    expect(hidden.variables.showAggregates).toBe(0);
    expect(hidden.structureKey).toBe(shown.structureKey);
  });
});

describe("OpenLayers Canvas point styles", () => {
  const tileGrid = new TileGrid({
    origin: [0, 0],
    resolutions: [512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
    tileSize: 256,
  });

  it("scales plain centroid dots across the centroid zoom range", () => {
    const style = createCanvasStyleFunction(presentation, store, tileGrid);
    const centroid = new Feature({ geometry: new Point([1, 1]), type: 0, prob: 0.5 });
    const low = style(centroid, tileGrid.getResolution(4)!)!;
    const high = style(centroid, tileGrid.getResolution(6)!)!;
    expect((low.getImage() as CircleStyle).getRadius()).toBe(1.5);
    expect((high.getImage() as CircleStyle).getRadius()).toBe(3);
    expect((high.getImage() as CircleStyle).getStroke()).toBeNull();
  });

  it("hides aggregates while retaining promoted polygons and their outlines", () => {
    const hidden = createCanvasStyleFunction(
      { ...presentation, overviewMode: "hidden" },
      store,
      tileGrid,
    );
    const aggregate = new Feature({
      geometry: new Point([1, 1]),
      type: 0,
      prob: 0.5,
      tiatoolbox_aggregate: true,
    });
    const promoted = new Feature({
      geometry: new Polygon([[[0, 0], [1, 0], [1, 1], [0, 0]]]),
      type: 0,
      prob: 0.5,
    });
    expect(hidden(aggregate, tileGrid.getResolution(3)!)).toBeUndefined();
    expect(hidden(promoted, tileGrid.getResolution(3)!)?.getStroke()?.getWidth())
      .toBe(1);
  });
});
