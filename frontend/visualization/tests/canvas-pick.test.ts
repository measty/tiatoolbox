import { describe, expect, it } from "vitest";

import Feature from "ol/Feature.js";
import Point from "ol/geom/Point.js";

import type { LayerPresentation } from "../src/domain/style-spec";
import { localFeaturePickResult } from "../src/renderers/openlayers/canvas-mvt-renderer";

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

function feature(
  id: string | number | undefined,
  properties: Record<string, unknown>,
): Feature {
  const result = new Feature({ geometry: new Point([1, 1]), ...properties });
  if (id !== undefined) result.setId(id);
  return result;
}

describe("Canvas local annotation picking", () => {
  it("accepts a visible feature matching all presentation filters", () => {
    expect(
      localFeaturePickResult(feature(7, { type: 0, prob: 0.5 }), presentation),
    ).toMatchObject({ fid: 7, properties: { type: 0, prob: 0.5 } });
  });

  it("skips hidden categories and values outside the numeric range", () => {
    expect(
      localFeaturePickResult(feature(8, { type: 1, prob: 0.5 }), presentation),
    ).toBeNull();
    expect(
      localFeaturePickResult(feature(9, { type: 0, prob: 0.1 }), presentation),
    ).toBeNull();
  });

  it("skips suppressed aggregates and non-authoritative id-less features", () => {
    const aggregate = feature(10, {
      type: 0,
      prob: 0.5,
      tiatoolbox_aggregate: true,
    });
    expect(
      localFeaturePickResult(aggregate, {
        ...presentation,
        overviewMode: "hidden",
      }),
    ).toBeNull();
    expect(
      localFeaturePickResult(feature(undefined, { type: 0, prob: 0.5 }), presentation),
    ).toBeNull();
  });
});
