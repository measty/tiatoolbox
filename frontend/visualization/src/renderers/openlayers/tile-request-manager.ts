export interface RequestLifecycle {
  owner?: object;
  onAbandon?: () => void;
}

interface ActiveRequest extends RequestLifecycle {
  key: string;
  controller: AbortController;
}

export class TileRequestManager {
  private readonly active = new Map<string, ActiveRequest>();
  private readonly owners = new WeakMap<object, ActiveRequest>();

  start(key: string, lifecycle: RequestLifecycle = {}): AbortSignal {
    const previousForKey = this.active.get(key);
    const previousForOwner = lifecycle.owner
      ? this.owners.get(lifecycle.owner)
      : undefined;
    const request: ActiveRequest = {
      key,
      controller: new AbortController(),
      ...lifecycle,
    };

    // Publish the replacement before abandoning prior work. When OpenLayers
    // reuses the same VectorTile object, this transfers ownership atomically
    // and prevents the old loader from setting ERROR on the new generation.
    this.active.set(key, request);
    if (request.owner) this.owners.set(request.owner, request);

    const superseded = new Set(
      [previousForKey, previousForOwner].filter(
        (item): item is ActiveRequest => item !== undefined && item !== request,
      ),
    );
    for (const previous of superseded) {
      if (this.active.get(previous.key) === previous) {
        this.active.delete(previous.key);
      }
      this.abandon(previous);
    }
    return request.controller.signal;
  }

  isCurrent(key: string, signal: AbortSignal): boolean {
    return this.active.get(key)?.controller.signal === signal;
  }

  finish(key: string, signal: AbortSignal): void {
    const current = this.active.get(key);
    if (current?.controller.signal !== signal) return;
    this.active.delete(key);
    if (current.owner && this.owners.get(current.owner) === current) {
      this.owners.delete(current.owner);
    }
  }

  abort(key: string): void {
    const current = this.active.get(key);
    if (!current) return;
    this.active.delete(key);
    this.abandon(current);
  }

  abortAll(): void {
    const active = [...this.active.values()];
    this.active.clear();
    for (const request of active) this.abandon(request);
  }

  get size(): number {
    return this.active.size;
  }

  private abandon(request: ActiveRequest): void {
    request.controller.abort();
    if (request.owner && this.owners.get(request.owner) !== request) {
      return;
    }
    try {
      request.onAbandon?.();
    } finally {
      if (request.owner && this.owners.get(request.owner) === request) {
        this.owners.delete(request.owner);
      }
    }
  }
}
