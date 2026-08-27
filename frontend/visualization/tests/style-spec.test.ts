import { describe, expect, it } from "vitest";

import type { StoreManifest } from "../src/api/types";
import {
  DIRECT_COLOR_OPTION,
  aggregateOverviewMaxZoom,
  annotationLayerMinZoom,
  annotationRepresentationSummary,
  centroidRepresentationRange,
  defaultPresentation,
  matchesPresentation,
  presentationAllowsPicking,
  presentationForProperty,
  presentationForStore,
  presentationPropertyFilter,
  refreshProvisionalPresentation,
  requiredTileProperties,
  zoomRepresentationAt,
  zoomRepresentationPolicyRanges,
} from "../src/domain/style-spec";
import {
  colorForValue,
  normaliseFeatureColor,
} from "../src/renderers/openlayers/style-compiler";

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
  representations: [],
  tileUrlTemplates: { auto: "/tiles/{z}/{x}/{y}" },
  featureUrlTemplate: "/features/{fid}",
};

const colorStore: StoreManifest = {
  ...store,
  properties: [...store.properties, { name: "color", kind: "text" }],
};

describe("portable annotation presentation", () => {
  it("starts with a categorical palette when one is available", () => {
    const presentation = defaultPresentation(store);
    expect(presentation.overviewMode).toBe("aggregate");
    expect(presentation.colorBy.mode).toBe("categorical");
    expect(colorForValue(presentation.colorBy, { type: 1 })).toMatch(/^#/);
  });

  it("keeps the layer active when aggregate primitives are suppressed", () => {
    const overviewStore: StoreManifest = {
      ...store,
      representations: [
        {
          kind: "aggregate",
          minZoom: 0,
          maxZoom: 3,
          urlTemplate: "/aggregate/{z}/{x}/{y}",
        },
      ],
    };
    const presentation = defaultPresentation(overviewStore);
    expect(annotationLayerMinZoom(overviewStore, presentation)).toBe(-Infinity);
    expect(
      annotationLayerMinZoom(overviewStore, {
        ...presentation,
        overviewMode: "hidden",
      }),
    ).toBe(-Infinity);
    expect(
      annotationLayerMinZoom(store, {
        ...presentation,
        overviewMode: "hidden",
      }),
    ).toBe(-Infinity);
  });

  it("describes and resolves only the authoritative uniform zoom policy", () => {
    const policyStore: StoreManifest = {
      ...store,
      representations: [
        {
          kind: "auto",
          minZoom: 0,
          maxZoom: 9,
          urlTemplate: "/auto/{z}/{x}/{y}",
          policy: {
            scope: "store-zoom",
            geometryPromotion: {
              metric: "projected-area",
              minimumPixelsSquared: 36,
              maximumZoom: 6,
              representation: "polygon",
            },
            ranges: [
              { minZoom: 0, maxZoom: 3, representation: "aggregate" },
              { minZoom: 4, maxZoom: 6, representation: "centroid" },
              { minZoom: 7, maxZoom: 9, representation: "polygon" },
            ],
          },
        },
      ],
    };

    expect(zoomRepresentationPolicyRanges(policyStore).map((range) =>
      range.representation,
    )).toEqual(["aggregate", "centroid", "polygon"]);
    expect(zoomRepresentationAt(policyStore, 3)).toBe("aggregate");
    expect(zoomRepresentationAt(policyStore, 4)).toBe("centroid");
    expect(zoomRepresentationAt(policyStore, 7)).toBe("polygon");
    expect(annotationRepresentationSummary(policyStore)).toBe(
      "Rendering schedule: aggregates + visible structures z0-3, "
        + "centroids + visible structures z4-6, polygons z7-9.",
    );
    expect(aggregateOverviewMaxZoom(policyStore)).toBe(3);
    expect(centroidRepresentationRange(policyStore)).toEqual({
      minZoom: 4,
      maxZoom: 6,
      representation: "centroid",
    });

    const overlapping: StoreManifest = {
      ...policyStore,
      representations: [{
        ...policyStore.representations[0]!,
        policy: {
          scope: "store-zoom",
          ranges: [
            { minZoom: 0, maxZoom: 4, representation: "aggregate" },
            { minZoom: 4, maxZoom: 9, representation: "centroid" },
          ],
        },
      }],
    };
    expect(zoomRepresentationPolicyRanges(overlapping)).toEqual([]);
    expect(annotationRepresentationSummary(overlapping)).toBeUndefined();
  });

  it("filters hidden categories without a server request", () => {
    const presentation = defaultPresentation(store);
    if (presentation.colorBy.mode !== "categorical") throw new Error("test setup");
    presentation.colorBy.categories[0]!.visible = false;
    expect(matchesPresentation({ type: 0 }, presentation)).toBe(false);
    expect(matchesPresentation({ type: 1 }, presentation)).toBe(true);
    expect(matchesPresentation({ type: 2 }, presentation)).toBe(false);
  });

  it("compiles categorical and numeric presentation filters for server picking", () => {
    const presentation = defaultPresentation(store);
    if (presentation.colorBy.mode !== "categorical") throw new Error("test setup");
    presentation.colorBy.categories[0]!.visible = false;
    presentation.rangeFilter = { property: "prob", min: 0.25, max: 0.9 };

    expect(presentationPropertyFilter(presentation)).toEqual({
      matchesNothing: false,
      filter: {
        op: "and",
        args: [
          { op: "in", property: "type", values: [1] },
          { op: "gte", property: "prob", value: 0.25 },
          { op: "lte", property: "prob", value: 0.9 },
        ],
      },
    });
    expect(presentationAllowsPicking(presentation)).toBe(true);
    expect(presentationAllowsPicking({ ...presentation, opacity: 0 })).toBe(false);

    presentation.colorBy.categories[1]!.visible = false;
    expect(presentationPropertyFilter(presentation)).toEqual({ matchesNothing: true });
    expect(presentationAllowsPicking(presentation)).toBe(false);
  });

  it("interpolates numeric colours", () => {
    expect(
      colorForValue(
        {
          mode: "numeric",
          property: "prob",
          numeric: { domain: [0, 1], colors: ["#000000", "#ffffff"] },
        },
        { prob: 0.5 },
      ),
    ).toBe("#808080");
  });

  it("normalises legacy per-feature colour payloads for Canvas", () => {
    expect(normaliseFeatureColor("#0F8")).toBe("#00ff88");
    expect(normaliseFeatureColor("#112233cc")).toBe("#112233");
    expect(normaliseFeatureColor("rgb(255, 0, 128)")).toBe("#ff0080");
    expect(normaliseFeatureColor("rgb(100% 50% 0%)")).toBe("#ff8000");
    expect(normaliseFeatureColor([1, 0, 0.5])).toBe("#ff0080");
    expect(normaliseFeatureColor([255, 128, 0])).toBe("#ff8000");
    expect(normaliseFeatureColor("[0, 0.5, 1]")).toBe("#0080ff");
    expect(normaliseFeatureColor("url(javascript:bad)", "#123456")).toBe(
      "#123456",
    );
    expect(normaliseFeatureColor([256, 0, 0])).toBe("#9ca3af");

    expect(
      colorForValue(
        { mode: "direct", property: "color", fallback: "#9ca3af" },
        { color: "rgb(12, 34, 56)" },
      ),
    ).toBe("#0c2238");
  });

  it("selects legacy feature colours without retaining a numeric filter", () => {
    const current = {
      ...defaultPresentation(store),
      rangeFilter: { property: "prob", min: 0.2, max: 0.8 },
    };
    const presentation = presentationForProperty(
      colorStore,
      current,
      DIRECT_COLOR_OPTION,
    );
    expect(presentation.colorBy).toEqual({
      mode: "direct",
      property: "color",
      fallback: "#9ca3af",
    });
    expect(presentation.rangeFilter).toBeUndefined();
    expect(requiredTileProperties(presentation)).toEqual(["color"]);
  });

  it("rejects direct feature colour styling for stores without color", () => {
    const current = defaultPresentation(store);
    expect(
      presentationForProperty(store, current, DIRECT_COLOR_OPTION),
    ).toBe(current);

    const imported = {
      ...current,
      opacity: 0.2,
      colorBy: {
        mode: "direct" as const,
        property: "color" as const,
        fallback: "#9ca3af",
      },
    };
    expect(presentationForStore(store, imported)).toEqual(
      defaultPresentation(store),
    );
    expect(presentationForStore(colorStore, imported)).toBe(imported);
  });

  it("requests only properties used by tile styling and filtering", () => {
    const presentation = defaultPresentation(store);
    expect(requiredTileProperties(presentation)).toEqual(["type"]);
    const numeric = {
      ...presentation,
      colorBy: {
        mode: "numeric" as const,
        property: "prob",
        numeric: {
          domain: [0, 1] as const,
          colors: ["#000000", "#ffffff"] as const,
        },
      },
      rangeFilter: { property: "prob", min: 0.1, max: 0.9 },
    };
    expect(requiredTileProperties(numeric)).toEqual(["prob"]);
  });

  it("upgrades only an untouched provisional data style", () => {
    const provisionalStore: StoreManifest = {
      ...store,
      properties: [],
      lodStatus: "building",
    };
    const provisional = {
      ...defaultPresentation(provisionalStore),
      opacity: 0.4,
    };
    const refreshed = refreshProvisionalPresentation(
      provisional,
      provisionalStore,
      { ...store, lodStatus: "ready" },
    );
    expect(refreshed.opacity).toBe(0.4);
    expect(refreshed.colorBy.mode).toBe("categorical");

    const intentional = {
      ...provisional,
      colorBy: { mode: "constant" as const, color: "#123456" },
    };
    expect(
      refreshProvisionalPresentation(
        intentional,
        provisionalStore,
        { ...store, lodStatus: "ready" },
      ),
    ).toBe(intentional);

    // A ready manifest can win the state-update race before its provisional
    // presentation is reconciled. The untouched constant default must still
    // be recognisable and upgraded in that ordering.
    const readyStore = { ...store, lodStatus: "ready" };
    expect(
      refreshProvisionalPresentation(provisional, readyStore, readyStore)
        .colorBy.mode,
    ).toBe("categorical");
  });
});
