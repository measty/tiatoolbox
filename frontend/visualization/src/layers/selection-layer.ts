import GeoJSON from "ol/format/GeoJSON.js";
import VectorLayer from "ol/layer/Vector.js";
import VectorSource from "ol/source/Vector.js";
import Fill from "ol/style/Fill.js";
import Stroke from "ol/style/Stroke.js";
import Style from "ol/style/Style.js";

import type { FeatureDetail } from "../api/types";
import { transformGeoJsonGeometryToMap } from "../map/projection";

export function createSelectionLayer(): VectorLayer {
  return new VectorLayer({
    source: new VectorSource({ wrapX: false }),
    updateWhileAnimating: true,
    updateWhileInteracting: true,
    style: new Style({
      fill: new Fill({ color: "rgba(255, 255, 255, 0.15)" }),
      stroke: new Stroke({ color: "#ffffff", width: 3 }),
      zIndex: 10_000,
    }),
    properties: { role: "selection" },
    zIndex: 10_000,
  });
}

export function updateSelectionLayer(
  layer: VectorLayer,
  detail: FeatureDetail | null,
): void {
  const source = layer.getSource();
  source?.clear(true);
  if (!source || !detail?.geometry) return;
  const geometry = transformGeoJsonGeometryToMap(detail.geometry);
  const feature = new GeoJSON().readFeature({
    type: "Feature",
    id: detail.fid,
    geometry,
    properties: detail.properties,
  });
  source.addFeature(feature);
}
