import type { StoreManifest } from "../../api/types";
import {
  presentationAllowsPicking,
  presentationPropertyFilter,
  type LayerPresentation,
} from "../../domain/style-spec";
import type { PickResult } from "../annotation-renderer";
import type { TileRequestManager } from "./tile-request-manager";

const MAX_PICK_RESOLUTION = 16;

export async function pickFeatureOnServer(
  store: StoreManifest,
  coordinate: readonly [number, number] | undefined,
  resolution: number | undefined,
  presentation: LayerPresentation,
  requests: TileRequestManager,
): Promise<PickResult | null> {
  if (
    !store.pickUrl ||
    !coordinate ||
    !presentationAllowsPicking(presentation) ||
    resolution === undefined ||
    !Number.isFinite(resolution) ||
    resolution > MAX_PICK_RESOLUTION
  ) {
    return null;
  }
  const separator = store.pickUrl.includes("?") ? "&" : "?";
  const query = new URLSearchParams({
    x: String(coordinate[0]),
    y: String(coordinate[1]),
    tolerance: String(Math.min(64, Math.max(1, resolution * 4))),
  });
  const propertyFilter = presentationPropertyFilter(presentation).filter;
  if (propertyFilter) query.set("filter", JSON.stringify(propertyFilter));
  const url = `${store.pickUrl}${separator}${query}`;
  const requestKey = `pick:${store.id}`;
  const signal = requests.start(requestKey);
  try {
    const response = await fetch.call(globalThis, url, {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      signal,
    });
    if (!requests.isCurrent(requestKey, signal)) return null;
    if (response.status === 204) return null;
    if (!response.ok) {
      throw new Error(`Feature pick failed (${response.status}).`);
    }
    const document = (await response.json()) as { featureId?: unknown };
    if (!requests.isCurrent(requestKey, signal)) return null;
    if (
      typeof document.featureId !== "number" &&
      typeof document.featureId !== "string"
    ) {
      throw new TypeError("Feature pick response has no featureId.");
    }
    return { fid: document.featureId, properties: {} };
  } finally {
    requests.finish(requestKey, signal);
  }
}
