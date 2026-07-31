import type { EventsKey } from "ol/events.js";
import { unByKey } from "ol/Observable.js";
import type View from "ol/View.js";

export interface ViewCoordinateTransform {
  toCanonical(coordinate: readonly number[]): [number, number];
  fromCanonical(coordinate: readonly [number, number]): [number, number];
}

interface LinkedView {
  view: View;
  transform: ViewCoordinateTransform;
  listeners: EventsKey[];
}

interface CanonicalViewState {
  center: [number, number];
  resolution: number;
  rotation: number;
}

export const slideViewTransform: ViewCoordinateTransform = {
  toCanonical: (coordinate) => [coordinate[0] ?? 0, -(coordinate[1] ?? 0)],
  fromCanonical: (coordinate) => [coordinate[0], -coordinate[1]],
};

export class ViewLinkController {
  private readonly views = new Map<string, LinkedView>();
  private applying = false;
  private enabled = true;
  private lastState: CanonicalViewState | null = null;

  register(
    id: string,
    view: View,
    transform: ViewCoordinateTransform = slideViewTransform,
  ): () => void {
    this.unregister(id);
    const sync = () => this.handleChange(id);
    const entry: LinkedView = {
      view,
      transform,
      listeners: [
        view.on("change:center", sync),
        view.on("change:resolution", sync),
        view.on("change:rotation", sync),
      ],
    };
    this.views.set(id, entry);
    if (this.enabled && this.lastState) this.apply(entry, this.lastState);
    else this.capture(entry);
    return () => this.unregister(id);
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled) {
      const first = this.views.values().next().value as LinkedView | undefined;
      if (first) {
        this.capture(first);
        if (this.lastState) {
          for (const entry of this.views.values()) this.apply(entry, this.lastState);
        }
      }
    }
  }

  dispose(): void {
    for (const entry of this.views.values()) unByKey(entry.listeners);
    this.views.clear();
  }

  private unregister(id: string): void {
    const entry = this.views.get(id);
    if (!entry) return;
    unByKey(entry.listeners);
    this.views.delete(id);
  }

  private handleChange(sourceId: string): void {
    if (this.applying) return;
    const source = this.views.get(sourceId);
    if (!source) return;
    this.capture(source);
    if (!this.enabled || !this.lastState) return;
    this.applying = true;
    try {
      for (const [id, target] of this.views) {
        if (id !== sourceId) this.apply(target, this.lastState);
      }
    } finally {
      this.applying = false;
    }
  }

  private capture(entry: LinkedView): void {
    const center = entry.view.getCenter();
    const resolution = entry.view.getResolution();
    if (!center || resolution === undefined) return;
    this.lastState = {
      center: entry.transform.toCanonical(center),
      resolution,
      rotation: entry.view.getRotation(),
    };
  }

  private apply(entry: LinkedView, state: CanonicalViewState): void {
    const wasApplying = this.applying;
    this.applying = true;
    try {
      entry.view.setCenter(entry.transform.fromCanonical(state.center));
      entry.view.setResolution(state.resolution);
      entry.view.setRotation(state.rotation);
    } finally {
      this.applying = wasApplying;
    }
  }
}
