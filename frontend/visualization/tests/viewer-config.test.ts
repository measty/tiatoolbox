import { describe, expect, it } from "vitest";

import type { LayerPresentation } from "../src/domain/style-spec";
import {
  createViewerConfig,
  normaliseViewerConfig,
  parseViewerConfig,
  serialiseViewerConfig,
  type ViewerConfigCatalog,
} from "../src/domain/viewer-config";

const catalog: ViewerConfigCatalog = {
  slideResourceIds: ["slide-a"],
  overlayKinds: { cells: "annotation", heatmap: "raster" },
};

const annotationPresentation: LayerPresentation = {
  visible: true,
  opacity: 0.8,
  overviewMode: "hidden",
  fillOpacity: 0.5,
  strokeColor: "#112233",
  strokeWidth: 1,
  pointRadius: 4,
  colorBy: {
    mode: "categorical",
    property: "type",
    categories: [
      { value: 1, color: "#ff0000", visible: false },
      { value: 2, color: "#00ff00", visible: true },
    ],
  },
  rangeFilter: { property: "prob", min: 0.2, max: 0.9 },
};

describe("versioned viewer configuration", () => {
  it("round-trips independent view state and client filters", () => {
    const config = createViewerConfig({
      slideResourceId: "slide-a",
      overlayResourceIds: ["cells", "heatmap"],
      renderer: "webgl",
      compare: { enabled: true, linked: false },
      views: {
        primary: [
          {
            kind: "annotation",
            resourceId: "cells",
            presentation: annotationPresentation,
          },
          {
            kind: "raster",
            resourceId: "heatmap",
            presentation: { visible: true, opacity: 0.4 },
          },
        ],
        secondary: [
          {
            kind: "annotation",
            resourceId: "cells",
            presentation: {
              ...annotationPresentation,
              opacity: 0.3,
              colorBy: {
                mode: "direct",
                property: "color",
                fallback: "#abcdef",
              },
            },
          },
        ],
      },
    });

    expect(parseViewerConfig(serialiseViewerConfig(config), catalog)).toEqual(
      config,
    );
  });

  it("locks imported direct colour mode to the legacy color property", () => {
    const config = normaliseViewerConfig(
      {
        schema: "tiatoolbox.viewer",
        version: 1,
        slideResourceId: "slide-a",
        overlayResourceIds: ["cells"],
        renderer: "webgl",
        compare: { enabled: false, linked: true },
        views: {
          primary: [
            {
              kind: "annotation",
              resourceId: "cells",
              presentation: {
                colorBy: {
                  mode: "direct",
                  property: "unexpected-property",
                  fallback: "javascript:bad",
                },
              },
            },
          ],
          secondary: [],
        },
      },
      catalog,
    );

    expect(config.views.primary[0]?.presentation).toMatchObject({
      colorBy: {
        mode: "direct",
        property: "color",
        fallback: "#9ca3af",
      },
    });
  });

  it("sanitises presentation bounds, duplicate IDs, and renderer values", () => {
    const config = normaliseViewerConfig(
      {
        schema: "tiatoolbox.viewer",
        version: 1,
        slideResourceId: "slide-a",
        overlayResourceIds: ["cells", "cells", "heatmap"],
        renderer: "unsafe-renderer",
        compare: { enabled: true, linked: false },
        views: {
          primary: [
            {
              kind: "annotation",
              resourceId: "cells",
              presentation: {
                opacity: 9,
                fillOpacity: -4,
                strokeColor: "javascript:bad",
                colorBy: {
                  mode: "numeric",
                  property: "prob",
                  numeric: {
                    domain: [10, -2],
                    colors: ["#ABCDEF", "invalid"],
                  },
                },
                rangeFilter: { property: "prob", min: 8, max: 2 },
              },
            },
          ],
          secondary: [
            {
              kind: "raster",
              resourceId: "heatmap",
              presentation: { visible: false, opacity: -1 },
            },
          ],
        },
      },
      catalog,
    );

    expect(config.overlayResourceIds).toEqual(["cells", "heatmap"]);
    expect(config.renderer).toBe("auto");
    expect(config.views.primary[0]?.presentation).toMatchObject({
      opacity: 1,
      overviewMode: "aggregate",
      fillOpacity: 0,
      strokeColor: "#111827",
      rangeFilter: { min: 2, max: 8 },
      colorBy: {
        numeric: { domain: [-2, 10], colors: ["#abcdef", "#f59e0b"] },
      },
    });
    expect(config.views.secondary[0]?.presentation).toEqual({
      visible: false,
      opacity: 0,
    });
  });

  it("rejects unsupported versions and non-catalog resource IDs", () => {
    expect(() =>
      normaliseViewerConfig(
        {
          schema: "tiatoolbox.viewer",
          version: 2,
          slideResourceId: "slide-a",
        },
        catalog,
      ),
    ).toThrow(/version/);
    expect(() =>
      normaliseViewerConfig(
        {
          schema: "tiatoolbox.viewer",
          version: 1,
          slideResourceId: "C:\\private\\slide.svs",
          overlayResourceIds: [],
          views: { primary: [], secondary: [] },
        },
        catalog,
      ),
    ).toThrow(/catalog/);
  });
});
