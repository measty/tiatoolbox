import { describe, expect, it, vi } from "vitest";

import MixedGeometryBatch from "ol/render/webgl/MixedGeometryBatch.js";
import VectorStyleRenderer from "ol/render/webgl/VectorStyleRenderer.js";
import { create, makeInverse, type Transform } from "ol/transform.js";

describe("OpenLayers WebGL empty-tile backport", () => {
  it("keeps and safely renders the mask transform for an empty MVT batch", async () => {
    const transform: Transform = [2, 0, 0, 2, 10, 20];
    const renderer = Object.create(
      VectorStyleRenderer.prototype,
    ) as VectorStyleRenderer;

    const buffers = await renderer.generateBuffers(
      new MixedGeometryBatch(),
      transform,
    );

    expect(buffers).not.toBeNull();
    if (!buffers) throw new Error("OpenLayers empty-batch patch is not applied.");
    expect(buffers.polygonBuffers).toBeNull();
    expect(buffers.lineStringBuffers).toBeNull();
    expect(buffers.pointBuffers).toBeNull();
    expect(buffers.invertVerticesTransform).toEqual(
      makeInverse(create(), transform),
    );

    const renderInternal = vi.fn();
    Object.assign(renderer as unknown as Record<string, unknown>, {
      renderPasses_: [
        {
          fillRenderPass: {},
          strokeRenderPass: {},
          symbolRenderPass: {},
        },
      ],
      renderInternal_: renderInternal,
    });
    expect(() =>
      renderer.render(
        buffers,
        {} as Parameters<VectorStyleRenderer["render"]>[1],
        () => {},
      ),
    ).not.toThrow();
    expect(renderInternal).not.toHaveBeenCalled();
  });
});
