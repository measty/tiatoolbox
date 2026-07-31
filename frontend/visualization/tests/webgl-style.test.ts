import { describe, expect, it } from "vitest";

import { convertStyleToShaders } from "ol/render/webgl/VectorStyleRenderer.js";

import type { LayerPresentation } from "../src/domain/style-spec";
import { compileWebGlStyle } from "../src/renderers/openlayers/style-compiler";

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

describe("OpenLayers WebGL style compiler", () => {
  it("produces expressions accepted by OpenLayers 10.9", () => {
    const compiled = compileWebGlStyle(presentation);
    const shaders = convertStyleToShaders(compiled.rules, compiled.variables);
    expect(shaders.length).toBeGreaterThan(0);
    expect(shaders.every((entry) => entry.builder !== undefined)).toBe(true);
  });

  it("accepts continuous colour interpolation and dynamic range variables", () => {
    const compiled = compileWebGlStyle({
      ...presentation,
      colorBy: {
        mode: "numeric",
        property: "prob",
        numeric: { domain: [0, 1], colors: ["#2563eb", "#f59e0b"] },
      },
    });
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
    });
    expect(compiled.rules[0]?.style["fill-color"]).toEqual([
      "*",
      ["get", "color"],
      ["color", 255, 255, 255, ["var", "fillOpacity"]],
    ]);
    expect(() => convertStyleToShaders(compiled.rules, compiled.variables)).not.toThrow();
  });
});
