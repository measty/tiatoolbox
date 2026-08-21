import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { StoreManifest } from "../src/api/types";
import { LayerPanel } from "../src/components/LayerPanel";
import { defaultPresentation } from "../src/domain/style-spec";

const store: StoreManifest = {
  id: "cells",
  name: "Cells",
  revision: "1",
  count: 2,
  bounds: [0, 0, 10, 10],
  overlaps: false,
  geometryTypes: { Polygon: 2 },
  properties: [],
  lodStatus: "building",
  representations: [],
  tileUrlTemplates: { auto: "/tiles/{z}/{x}/{y}" },
  featureUrlTemplate: "/features/{fid}",
};

describe("annotation layer metadata discovery", () => {
  it("labels the provisional colour selector while property summaries build", () => {
    const markup = renderToStaticMarkup(
      <LayerPanel
        store={store}
        presentation={defaultPresentation(store)}
        onChange={() => undefined}
        onRemove={() => undefined}
      />,
    );

    expect(markup).toContain("Preparing colour options…");
    expect(markup).toMatch(/<select[^>]*disabled=""[^>]*>/);
  });

  it("enables discovered properties once the store is ready", () => {
    const readyStore: StoreManifest = {
      ...store,
      lodStatus: "ready",
      properties: [
        { name: "type", kind: "categorical", values: [0, 1] },
        { name: "prob", kind: "numeric", min: 0, max: 1 },
      ],
    };
    const markup = renderToStaticMarkup(
      <LayerPanel
        store={readyStore}
        presentation={defaultPresentation(readyStore)}
        onChange={() => undefined}
        onRemove={() => undefined}
      />,
    );

    expect(markup).not.toContain("Preparing colour options…");
    expect(markup).toContain('<option value="type" selected="">type</option>');
    expect(markup).toContain('<option value="prob">prob</option>');
  });
});
