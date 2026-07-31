import type OlMap from "ol/Map.js";
import { unByKey } from "ol/Observable.js";

export type AffineTransform = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
];

export interface CanvasTransformInput {
  transform?: string;
  canvasWidth: number;
  canvasHeight: number;
  cssWidth?: number;
  cssHeight?: number;
}

/** Convert an OpenLayers canvas CSS transform into a 2D draw transform. */
export function canvasCompositionTransform({
  transform,
  canvasWidth,
  canvasHeight,
  cssWidth,
  cssHeight,
}: CanvasTransformInput): AffineTransform {
  const parsed = transform ? parseCssTransform(transform) : undefined;
  if (parsed) return parsed;
  const scaleX =
    cssWidth && canvasWidth > 0 ? cssWidth / canvasWidth : 1;
  const scaleY =
    cssHeight && canvasHeight > 0 ? cssHeight / canvasHeight : 1;
  return [scaleX, 0, 0, scaleY, 0, 0];
}

export async function renderMapToPng(map: OlMap): Promise<Blob> {
  await waitForCompleteRender(map);
  const canvas = composeMapCanvases(map);
  return canvasToPng(canvas);
}

export function composeMapCanvases(map: OlMap): HTMLCanvasElement {
  const size = map.getSize();
  const width = size?.[0] ?? 0;
  const height = size?.[1] ?? 0;
  if (width <= 0 || height <= 0) {
    throw new Error("The map has no drawable size.");
  }
  const output = document.createElement("canvas");
  output.width = width;
  output.height = height;
  const context = output.getContext("2d");
  if (!context) throw new Error("A 2D canvas is required for PNG export.");

  const viewport = map.getViewport();
  const layersRoot = Array.from(viewport.children).find(
    (element): element is HTMLElement =>
      element instanceof HTMLElement && element.classList.contains("ol-layers"),
  );
  if (!layersRoot) throw new Error("OpenLayers has no rendered layer container.");

  context.fillStyle = firstOpaqueBackground(viewport) ?? "#d1d5db";
  context.fillRect(0, 0, output.width, output.height);

  const canvases = Array.from(layersRoot.querySelectorAll("canvas"))
    .map((canvas, index) => ({
      canvas,
      index,
      opacity: effectiveOpacity(canvas, layersRoot),
      zIndex: effectiveZIndex(canvas, layersRoot),
    }))
    .filter(
      ({ canvas, opacity }) =>
        canvas.width > 0 && canvas.height > 0 && opacity > 0,
    )
    .sort(
      (left, right) =>
        left.zIndex - right.zIndex || left.index - right.index,
    );

  for (const { canvas, opacity } of canvases) {
    const computed = window.getComputedStyle(canvas);
    const transform = canvasCompositionTransform({
      transform:
        canvas.style.transform && canvas.style.transform !== "none"
          ? canvas.style.transform
          : computed.transform,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      cssWidth:
        finiteCssPixels(canvas.style.width) || canvas.clientWidth || canvas.width,
      cssHeight:
        finiteCssPixels(canvas.style.height) || canvas.clientHeight || canvas.height,
    });
    context.save();
    context.globalAlpha = opacity;
    context.setTransform(...transform);
    const background = canvasBackground(canvas);
    if (background) {
      context.fillStyle = background;
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    context.drawImage(canvas, 0, 0);
    context.restore();
  }
  return output;
}

function parseCssTransform(transform: string): AffineTransform | undefined {
  if (!transform || transform === "none") return undefined;
  const matrix = /^matrix\(([^)]+)\)$/.exec(transform.trim());
  if (matrix?.[1]) {
    const values = numericList(matrix[1]);
    if (values.length === 6) return values as unknown as AffineTransform;
  }
  const matrix3d = /^matrix3d\(([^)]+)\)$/.exec(transform.trim());
  if (matrix3d?.[1]) {
    const values = numericList(matrix3d[1]);
    if (values.length === 16) {
      return [
        values[0]!,
        values[1]!,
        values[4]!,
        values[5]!,
        values[12]!,
        values[13]!,
      ];
    }
  }
  return undefined;
}

function numericList(value: string): number[] {
  const values = value.split(",").map((entry) => Number(entry.trim()));
  return values.every(Number.isFinite) ? values : [];
}

function finiteCssPixels(value: string): number | undefined {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function effectiveOpacity(canvas: HTMLCanvasElement, root: HTMLElement): number {
  let opacity = 1;
  let element: HTMLElement | null = canvas;
  while (element && element !== root) {
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden") return 0;
    const value = Number(style.opacity);
    if (Number.isFinite(value)) opacity *= value;
    element = element.parentElement;
  }
  return opacity;
}

function effectiveZIndex(canvas: HTMLCanvasElement, root: HTMLElement): number {
  let element: HTMLElement | null = canvas;
  while (element && element !== root) {
    const value = Number.parseFloat(window.getComputedStyle(element).zIndex);
    if (Number.isFinite(value)) return value;
    element = element.parentElement;
  }
  return 0;
}

function canvasBackground(canvas: HTMLCanvasElement): string | undefined {
  const candidates = [canvas, canvas.parentElement].filter(
    (element): element is HTMLElement => element instanceof HTMLElement,
  );
  for (const element of candidates) {
    const color = window.getComputedStyle(element).backgroundColor;
    if (!isTransparent(color)) return color;
  }
  return undefined;
}

function firstOpaqueBackground(element: HTMLElement): string | undefined {
  let current: HTMLElement | null = element;
  while (current) {
    const color = window.getComputedStyle(current).backgroundColor;
    if (!isTransparent(color)) return color;
    current = current.parentElement;
  }
  return undefined;
}

function isTransparent(color: string): boolean {
  return (
    !color ||
    color === "transparent" ||
    color === "rgba(0, 0, 0, 0)" ||
    color === "rgba(0,0,0,0)"
  );
}

function waitForCompleteRender(map: OlMap): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      unByKey(key);
      reject(new Error("The visible map did not finish rendering within 30 seconds."));
    }, 30_000);
    const key = map.once("rendercomplete", () => {
      window.clearTimeout(timeout);
      resolve();
    });
    map.renderSync();
  });
}

function canvasToPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    try {
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("The browser could not encode the map as PNG."));
      }, "image/png");
    } catch (error) {
      reject(error);
    }
  });
}
