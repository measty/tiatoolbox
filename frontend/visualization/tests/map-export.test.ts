import { describe, expect, it } from "vitest";

import { canvasCompositionTransform } from "../src/map/map-export";

describe("map canvas composition", () => {
  it("preserves an OpenLayers 2D CSS matrix", () => {
    expect(
      canvasCompositionTransform({
        transform: "matrix(0.5, 0, 0, 0.5, 12, -7)",
        canvasWidth: 2000,
        canvasHeight: 1000,
      }),
    ).toEqual([0.5, 0, 0, 0.5, 12, -7]);
  });

  it("extracts the affine portion of a CSS matrix3d", () => {
    expect(
      canvasCompositionTransform({
        transform:
          "matrix3d(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 40, 50, 0, 1)",
        canvasWidth: 100,
        canvasHeight: 100,
      }),
    ).toEqual([2, 0, 0, 3, 40, 50]);
  });

  it("falls back to CSS-to-backing-store scaling", () => {
    expect(
      canvasCompositionTransform({
        transform: "none",
        canvasWidth: 1600,
        canvasHeight: 1200,
        cssWidth: 800,
        cssHeight: 600,
      }),
    ).toEqual([0.5, 0, 0, 0.5, 0, 0]);
  });
});
