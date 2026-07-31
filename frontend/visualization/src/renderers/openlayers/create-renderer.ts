import type {
  AnnotationLayerHandle,
  AnnotationRendererContext,
  RendererPreference,
} from "../annotation-renderer";
import { resolvedRendererKind } from "../annotation-renderer";
import { CanvasMvtRenderer } from "./canvas-mvt-renderer";
import { WebGlMvtRenderer } from "./webgl-mvt-renderer";

export function createRenderer(
  preference: RendererPreference,
  context: AnnotationRendererContext,
): AnnotationLayerHandle {
  if (resolvedRendererKind(preference) === "webgl") {
    try {
      return new WebGlMvtRenderer(context);
    } catch (error) {
      console.warn("WebGL annotation renderer unavailable; using Canvas.", error);
    }
  }
  return new CanvasMvtRenderer(context);
}
