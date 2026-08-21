import { describe, expect, it } from "vitest";

import {
  normaliseBootstrap,
  normaliseFeatureDetail,
  normaliseSlide,
  normaliseStore,
} from "../src/api/normalise";

describe("API manifest normalisation", () => {
  it("normalises the session-oriented bootstrap catalogue", () => {
    const bootstrap = normaliseBootstrap({
      apiVersion: "1.0",
      title: "Test viewer",
      catalog: {
        slides: [{ id: "slide-a", name: "Slide A", kind: "slide" }],
        overlays: [
          { id: "overlay-a", name: "Cells.db", kind: "annotation" },
        ],
      },
      session: {
        id: "session-a",
        slide: null,
        annotationStores: [],
        rasterLayers: [],
      },
      capabilities: { linkedViews: true },
    });
    expect(bootstrap.apiVersion).toBe("1.0");
    expect(bootstrap.slides[0]?.id).toBe("slide-a");
    expect(bootstrap.overlays[0]?.id).toBe("overlay-a");
    expect(bootstrap.session.id).toBe("session-a");
    expect(bootstrap.capabilities.linkedViews).toBe(true);
  });

  it("retains compatibility with the flat prototype bootstrap", () => {
    const bootstrap = normaliseBootstrap({
      api_version: "v1",
      slides: [{ id: "slide-a", name: "Slide A" }],
      stores: ["store-a"],
      default_store_ids: ["store-a"],
    });
    expect(bootstrap.stores).toEqual([{ id: "store-a", name: "store-a" }]);
    expect(bootstrap.defaultStoreIds).toEqual(["store-a"]);
  });

  it("normalises the concrete WSI tile matrix", () => {
    const slide = normaliseSlide({
      id: "slide-a",
      name: "Slide A",
      dimensions: [106_496, 85_248],
      mpp: [0.227, 0.228],
      tileMatrix: {
        tileSize: 256,
        maxZoom: 9,
        resolutions: [512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        mapExtent: [0, -85_248, 106_496, 0],
      },
      tileUrl:
        "/tileserver/layer/slide/session/zoomify/TileGroup{TileGroup}/{z}-{x}-{y}@1x.jpg",
      associatedOverlays: [
        { id: "cells", name: "Cells.db", kind: "annotation" },
      ],
    });
    expect(slide.width).toBe(106_496);
    expect(slide.height).toBe(85_248);
    expect(slide.maxZoom).toBe(9);
    expect(slide.resolutions.at(-1)).toBe(1);
    expect(slide.mapExtent).toEqual([0, -85_248, 106_496, 0]);
    expect(slide.associatedOverlays[0]?.id).toBe("cells");
  });

  it("normalises revisioned store metadata and canonical URL templates", () => {
    const store = normaliseStore({
      id: "cells",
      name: "Cells",
      revision: "sha256:abc",
      bounds: [0, 0, 1000, 1000],
      featureCount: 627_761,
      geometryTypes: { Polygon: 627_761 },
      lodStatus: "ready",
      properties: {
        type: {
          kind: "categorical",
          categories: [
            { value: 0, count: 2 },
            { value: 1, count: 3 },
            { value: 2, count: 4 },
          ],
        },
        prob: { kind: "numeric", numeric: { min: 0, max: 1 } },
      },
      urls: {
        tiles:
          "/api/v1/stores/cells/revisions/sha256%3Aabc/tiles/auto/{z}/{x}/{y}.mvt?lod=ready",
        representations: {
          aggregate:
            "/api/v1/stores/cells/revisions/sha256%3Aabc/tiles/aggregate/{z}/{x}/{y}.mvt?lod=ready",
          polygon:
            "/api/v1/stores/cells/revisions/sha256%3Aabc/tiles/polygon/{z}/{x}/{y}.mvt?lod=ready",
        },
        feature:
          "/api/v1/stores/cells/revisions/sha256%3Aabc/features/{featureId}",
      },
      representations: {
        auto: {
          format: "mvt",
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
        aggregate: { format: "mvt", maxZoom: 3 },
        polygon: { format: "mvt", maxZoom: 9 },
      },
    });
    expect(store.count).toBe(627_761);
    expect(store.properties.map((property) => property.name)).toEqual([
      "type",
      "prob",
    ]);
    expect(store.properties[0]?.values).toEqual([0, 1, 2]);
    expect(store.properties[1]).toMatchObject({ kind: "numeric", min: 0, max: 1 });
    expect(store.tileUrlTemplates.auto).toContain("lod=ready");
    expect(store.representations.find(({ kind }) => kind === "aggregate"))
      .toMatchObject({ minZoom: 0, maxZoom: 3 });
    expect(store.representations.find(({ kind }) => kind === "auto")?.policy)
      .toEqual({
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
      });
    expect(store.featureUrlTemplate).toContain("{featureId}");
    expect(store.geometryTypes.Polygon).toBe(627_761);
    expect(store.lodStatus).toBe("ready");
  });

  it("normalises an exact GeoJSON feature and its canonical ID", () => {
    const feature = normaliseFeatureDetail({
      type: "Feature",
      id: "42",
      geometry: { type: "Point", coordinates: [12, 34] },
      properties: {
        type: 2,
        prob: 0.9,
        _tiatoolbox: { canonicalId: "aa-bb" },
      },
    });
    expect(feature.fid).toBe("42");
    expect(feature.geometry?.type).toBe("Point");
    expect(feature.properties.prob).toBe(0.9);
    expect(feature.uuid).toBe("aa-bb");
  });
});
