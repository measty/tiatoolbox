import { afterEach, describe, expect, it, vi } from "vitest";

import type { StoreManifest } from "../src/api/types";
import { pickFeatureOnServer } from "../src/renderers/openlayers/server-pick";
import { TileRequestManager } from "../src/renderers/openlayers/tile-request-manager";

const store = {
  id: "store",
  pickUrl: "/api/v1/stores/store/revisions/rev/features/pick",
} as StoreManifest;

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
      pickFeatureOnServer(store, [120, 80], 2, new TileRequestManager()),
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
      pickFeatureOnServer(store, [120, 80], 32, new TileRequestManager()),
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

    const pick = pickFeatureOnServer(store, [120, 80], 2, requests);
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

    const pick = pickFeatureOnServer(store, [120, 80], 2, requests);
    await vi.waitFor(() => expect(json).toHaveBeenCalledOnce());
    const replacement = requests.start("pick:store");
    resolveJson({ featureId: 42 });

    await expect(pick).resolves.toBeNull();
    expect(requests.isCurrent("pick:store", replacement)).toBe(true);
    requests.abortAll();
  });
});
