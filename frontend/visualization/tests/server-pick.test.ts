import { afterEach, describe, expect, it, vi } from "vitest";

import type { StoreManifest } from "../src/api/types";
import type { LayerPresentation } from "../src/domain/style-spec";
import { pickFeatureOnServer } from "../src/renderers/openlayers/server-pick";
import { TileRequestManager } from "../src/renderers/openlayers/tile-request-manager";

const store = {
  id: "store",
  pickUrl: "/api/v1/stores/store/revisions/rev/features/pick",
} as StoreManifest;

const presentation: LayerPresentation = {
  visible: true,
  opacity: 1,
  overviewMode: "aggregate",
  fillOpacity: 0.5,
  strokeColor: "#111827",
  strokeWidth: 1,
  pointRadius: 3,
  colorBy: { mode: "constant", color: "#e11d48" },
};

afterEach(() => vi.unstubAllGlobals());

describe("server feature-pick fallback", () => {
  it("resolves an authoritative feature near a close-detail click", async () => {
    const fetchImplementation = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify({ featureId: 42 }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchImplementation);

    await expect(
      pickFeatureOnServer(
        store,
        [120, 80],
        2,
        presentation,
        new TileRequestManager(),
      ),
    ).resolves.toEqual({ fid: 42, properties: {} });

    expect(fetchImplementation).toHaveBeenCalledOnce();
    expect(String(fetchImplementation.mock.calls[0]?.[0])).toContain(
      "x=120&y=80&tolerance=8",
    );
  });

  it("does not issue a broad server query from an overview", async () => {
    const fetchImplementation = vi.fn();
    vi.stubGlobal("fetch", fetchImplementation);
    await expect(
      pickFeatureOnServer(
        store,
        [120, 80],
        32,
        presentation,
        new TileRequestManager(),
      ),
    ).resolves.toBeNull();
    expect(fetchImplementation).not.toHaveBeenCalled();
  });

  it("ignores a response superseded while fetch is pending", async () => {
    let resolveFetch!: (response: Response) => void;
    const fetchImplementation = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
    );
    vi.stubGlobal("fetch", fetchImplementation);
    const requests = new TileRequestManager();

    const pick = pickFeatureOnServer(store, [120, 80], 2, presentation, requests);
    const replacement = requests.start("pick:store");
    const response = new Response(JSON.stringify({ featureId: 42 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
    const json = vi.spyOn(response, "json");
    resolveFetch(response);

    await expect(pick).resolves.toBeNull();
    expect(json).not.toHaveBeenCalled();
    expect(requests.isCurrent("pick:store", replacement)).toBe(true);
    requests.abortAll();
  });

  it("ignores a response superseded while JSON is pending", async () => {
    let resolveJson!: (document: { featureId: number }) => void;
    const json = vi.fn(
      () =>
        new Promise<{ featureId: number }>((resolve) => {
          resolveJson = resolve;
        }),
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, status: 200, json }) as unknown as Response),
    );
    const requests = new TileRequestManager();

    const pick = pickFeatureOnServer(store, [120, 80], 2, presentation, requests);
    await vi.waitFor(() => expect(json).toHaveBeenCalledOnce());
    const replacement = requests.start("pick:store");
    resolveJson({ featureId: 42 });

    await expect(pick).resolves.toBeNull();
    expect(requests.isCurrent("pick:store", replacement)).toBe(true);
    requests.abortAll();
  });

  it("sends categorical and numeric presentation filters to the picker", async () => {
    const fetchImplementation = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal("fetch", fetchImplementation);
    const filtered: LayerPresentation = {
      ...presentation,
      colorBy: {
        mode: "categorical",
        property: "type",
        categories: [
          { value: 0, color: "#e11d48", visible: false },
          { value: 1, color: "#2563eb", visible: true },
        ],
      },
      rangeFilter: { property: "prob", min: 0.25, max: 0.9 },
    };

    await pickFeatureOnServer(
      store,
      [120, 80],
      2,
      filtered,
      new TileRequestManager(),
    );

    const url = new URL(String(fetchImplementation.mock.calls[0]?.[0]), "http://test");
    expect(JSON.parse(url.searchParams.get("filter") ?? "null")).toEqual({
      op: "and",
      args: [
        { op: "in", property: "type", values: [1] },
        { op: "gte", property: "prob", value: 0.25 },
        { op: "lte", property: "prob", value: 0.9 },
      ],
    });
  });

  it("does not pick hidden layers or presentations that match nothing", async () => {
    const fetchImplementation = vi.fn();
    vi.stubGlobal("fetch", fetchImplementation);
    const noCategories: LayerPresentation = {
      ...presentation,
      colorBy: { mode: "categorical", property: "type", categories: [] },
    };

    await expect(
      pickFeatureOnServer(
        store,
        [120, 80],
        2,
        { ...presentation, visible: false },
        new TileRequestManager(),
      ),
    ).resolves.toBeNull();
    await expect(
      pickFeatureOnServer(
        store,
        [120, 80],
        2,
        noCategories,
        new TileRequestManager(),
      ),
    ).resolves.toBeNull();
    expect(fetchImplementation).not.toHaveBeenCalled();
  });
});
