import {
  normaliseAddedOverlay,
  normaliseBootstrap,
  normaliseFeatureDetail,
  normaliseSlide,
  normaliseStore,
} from "./normalise";
import type {
  AddedOverlay,
  BootstrapManifest,
  FeatureDetail,
  FeatureId,
  SlideManifest,
  StoreManifest,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly url: string,
  ) {
    super(message);
  }
}

export class ApiClient {
  private readonly controllers = new Map<string, AbortController>();

  constructor(private readonly fetchImplementation: typeof fetch = fetch) {}

  async bootstrap(): Promise<BootstrapManifest> {
    return normaliseBootstrap(
      await this.json("/api/v1/bootstrap", { scope: "bootstrap" }),
    );
  }

  async slide(id: string): Promise<SlideManifest> {
    return normaliseSlide(
      await this.json(`/api/v1/slides/${encodeURIComponent(id)}`, {
        scope: "slide",
      }),
    );
  }

  async selectSlide(resourceId: string): Promise<SlideManifest> {
    return normaliseSlide(
      await this.json("/api/v1/session/slide", {
        scope: "slide",
        method: "PUT",
        body: { resourceId },
      }),
    );
  }

  async addOverlay(
    resourceId: string,
    slideGeneration?: number,
  ): Promise<AddedOverlay> {
    return normaliseAddedOverlay(
      await this.json("/api/v1/session/overlays", {
        scope: `add-overlay:${resourceId}`,
        method: "POST",
        body: {
          resourceId,
          ...(slideGeneration === undefined ? {} : { slideGeneration }),
        },
      }),
    );
  }

  async removeOverlay(layerId: string, slideGeneration?: number): Promise<void> {
    const generationQuery = slideGeneration === undefined
      ? ""
      : `?slideGeneration=${encodeURIComponent(String(slideGeneration))}`;
    await this.json(
      `/api/v1/session/overlays/${encodeURIComponent(layerId)}${generationQuery}`,
      {
        scope: `remove-overlay:${layerId}`,
        method: "DELETE",
      },
    );
  }

  async store(id: string): Promise<StoreManifest> {
    return normaliseStore(
      await this.json(`/api/v1/stores/${encodeURIComponent(id)}`, {
        scope: `store:${id}`,
      }),
    );
  }

  async feature(
    store: StoreManifest,
    fid: FeatureId,
  ): Promise<FeatureDetail> {
    const url = expandUrlTemplate(store.featureUrlTemplate, { fid });
    const expandedUrl = expandUrlTemplate(url, { featureId: fid });
    return normaliseFeatureDetail(
      await this.json(expandedUrl, { scope: "feature-detail" }),
    );
  }

  abort(scope: string): void {
    this.controllers.get(scope)?.abort();
    this.controllers.delete(scope);
  }

  abortPrefix(prefix: string): void {
    for (const scope of [...this.controllers.keys()]) {
      if (scope.startsWith(prefix)) this.abort(scope);
    }
  }

  dispose(): void {
    for (const controller of this.controllers.values()) controller.abort();
    this.controllers.clear();
  }

  private async json(
    url: string,
    options: {
      scope: string;
      method?: "GET" | "POST" | "PUT" | "DELETE";
      body?: Record<string, unknown>;
    },
  ): Promise<unknown> {
    this.abort(options.scope);
    const controller = new AbortController();
    this.controllers.set(options.scope, controller);
    try {
      // Some WebView implementations require the native fetch receiver to be
      // Window/globalThis. Calling a fetch stored as an object property can
      // otherwise fail with "Illegal invocation" before any request is sent.
      const response = await this.fetchImplementation.call(globalThis, url, {
        signal: controller.signal,
        method: options.method ?? "GET",
        credentials: "same-origin",
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
        },
        ...(options.body ? { body: JSON.stringify(options.body) } : {}),
      });
      if (!response.ok) {
        throw new ApiError(
          `Request failed with ${response.status} ${response.statusText}.`,
          response.status,
          url,
        );
      }
      return response.status === 204 ? undefined : await response.json();
    } finally {
      if (this.controllers.get(options.scope) === controller) {
        this.controllers.delete(options.scope);
      }
    }
  }
}

export function expandUrlTemplate(
  template: string,
  values: Record<string, string | number>,
): string {
  return template.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (token, key: string) =>
    values[key] === undefined ? token : encodeURIComponent(String(values[key])),
  );
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
