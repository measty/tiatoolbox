import MixedGeometryBatch from "ol/render/webgl/MixedGeometryBatch.js";
import VectorStyleRenderer from "ol/render/webgl/VectorStyleRenderer.js";
import { create, makeInverse } from "ol/transform.js";

const transform = [2, 0, 0, 2, 10, 20];
const renderer = Object.create(VectorStyleRenderer.prototype);
const expectedInverse = makeInverse(create(), transform);

try {
  const buffers = await renderer.generateBuffers(
    new MixedGeometryBatch(),
    transform,
  );
  const patched =
    buffers !== null &&
    buffers.polygonBuffers === null &&
    buffers.lineStringBuffers === null &&
    buffers.pointBuffers === null &&
    buffers.invertVerticesTransform.length === expectedInverse.length &&
    buffers.invertVerticesTransform.every(
      (value, index) => value === expectedInverse[index],
    );

  if (!patched) {
    throw new Error("the empty-batch buffer contract is missing");
  }

  renderer.renderPasses_ = [
    {
      fillRenderPass: {},
      strokeRenderPass: {},
      symbolRenderPass: {},
    },
  ];
  renderer.renderInternal_ = () => {
    throw new Error("an empty geometry buffer reached a WebGL render pass");
  };
  renderer.render(buffers, {}, () => {});
  console.log("Verified OpenLayers empty-vector-tile WebGL patch.");
} catch (error) {
  console.error(
    "OpenLayers 10.9.0 is missing TIAToolbox's required WebGL patch. " +
      "Run `npm install` (without --ignore-scripts) or `npx patch-package`, " +
      "then retry.",
  );
  console.error(error);
  process.exit(1);
}
