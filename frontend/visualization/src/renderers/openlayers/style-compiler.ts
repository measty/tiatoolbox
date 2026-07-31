import type { FeatureLike } from "ol/Feature.js";
import CircleStyle from "ol/style/Circle.js";
import Fill from "ol/style/Fill.js";
import Stroke from "ol/style/Stroke.js";
import Style from "ol/style/Style.js";

import type { ColorBy, LayerPresentation } from "../../domain/style-spec";
import { matchesPresentation } from "../../domain/style-spec";

type Expression = unknown[];
type WebGlRule = { filter?: Expression; style: Record<string, unknown> };

export interface CompiledWebGlStyle {
  rules: WebGlRule[];
  variables: Record<string, number>;
  structureKey: string;
}

export function createCanvasStyleFunction(
  presentation: LayerPresentation,
): (feature: FeatureLike) => Style | undefined {
  const cache = new Map<string, Style>();
  return (feature) => {
    const properties = feature.getProperties();
    if (!matchesPresentation(properties, presentation)) return undefined;
    const color = colorForValue(presentation.colorBy, properties);
    const geometryType = feature.getGeometry()?.getType() ?? "Polygon";
    const cacheKey = `${geometryType}:${color}`;
    let style = cache.get(cacheKey);
    if (!style) {
      const fill = new Fill({ color: colorWithAlpha(color, presentation.fillOpacity) });
      const stroke = new Stroke({
        color: presentation.strokeColor,
        width: presentation.strokeWidth,
      });
      style = geometryType.includes("Point")
        ? new Style({
            image: new CircleStyle({
              radius: presentation.pointRadius,
              fill,
              stroke,
            }),
          })
        : new Style({ fill, stroke });
      cache.set(cacheKey, style);
    }
    return style;
  };
}

export function compileWebGlStyle(
  presentation: LayerPresentation,
): CompiledWebGlStyle {
  const featureFilter = webGlFilter(presentation);
  const color = webGlColor(presentation.colorBy);
  const polygonFilter: Expression = [
    "all",
    featureFilter,
    [
      "any",
      ["==", ["geometry-type"], "Polygon"],
      ["==", ["geometry-type"], "MultiPolygon"],
    ],
  ];
  const pointFilter: Expression = [
    "all",
    featureFilter,
    [
      "any",
      ["==", ["geometry-type"], "Point"],
      ["==", ["geometry-type"], "MultiPoint"],
    ],
  ];
  const rules: WebGlRule[] = [
    {
      filter: polygonFilter,
      style: {
        "fill-color": color,
        "stroke-color": presentation.strokeColor,
        "stroke-width": ["var", "strokeWidth"],
      },
    },
    {
      filter: pointFilter,
      style: {
        "circle-radius": ["var", "pointRadius"],
        "circle-fill-color": color,
        "circle-stroke-color": presentation.strokeColor,
        "circle-stroke-width": ["var", "strokeWidth"],
      },
    },
  ];
  return {
    rules,
    variables: {
      fillOpacity: presentation.fillOpacity,
      strokeWidth: presentation.strokeWidth,
      pointRadius: presentation.pointRadius,
      rangeMin: presentation.rangeFilter?.min ?? -Number.MAX_VALUE,
      rangeMax: presentation.rangeFilter?.max ?? Number.MAX_VALUE,
    },
    structureKey: JSON.stringify({
      colorBy: presentation.colorBy,
      strokeColor: presentation.strokeColor,
      rangeProperty: presentation.rangeFilter?.property,
      hasRange: Boolean(presentation.rangeFilter),
    }),
  };
}

export function colorForValue(
  colorBy: ColorBy,
  properties: Record<string, unknown>,
): string {
  if (colorBy.mode === "constant") return colorBy.color;
  if (colorBy.mode === "direct") {
    return normaliseFeatureColor(
      properties[colorBy.property],
      colorBy.fallback,
    );
  }
  if (colorBy.mode === "categorical") {
    const value = properties[colorBy.property];
    return (
      colorBy.categories.find((category) => String(category.value) === String(value))
        ?.color ?? "#9ca3af"
    );
  }
  const value = Number(properties[colorBy.property]);
  const [min, max] = colorBy.numeric.domain;
  const amount = max === min ? 0 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  return interpolateHex(colorBy.numeric.colors[0], colorBy.numeric.colors[1], amount);
}

function webGlFilter(presentation: LayerPresentation): Expression {
  const parts: Expression[] = [];
  if (presentation.colorBy.mode === "categorical") {
    const visible = presentation.colorBy.categories.filter((entry) => entry.visible);
    if (visible.length === 0) parts.push(["==", 1, 0]);
    else {
      const predicates: Expression[] = visible.map((entry) => [
          "==",
          ["get", presentation.colorBy.mode === "categorical" ? presentation.colorBy.property : ""],
          entry.value,
        ]);
      parts.push(predicates.length === 1 ? predicates[0]! : ["any", ...predicates]);
    }
  }
  if (presentation.rangeFilter) {
    parts.push([
      "all",
      [">=", ["get", presentation.rangeFilter.property], ["var", "rangeMin"]],
      ["<=", ["get", presentation.rangeFilter.property], ["var", "rangeMax"]],
    ]);
  }
  if (parts.length === 0) return ["==", 1, 1];
  return parts.length === 1 ? parts[0]! : ["all", ...parts];
}

function webGlColor(colorBy: ColorBy): Expression {
  if (colorBy.mode === "constant") return colorExpression(colorBy.color);
  if (colorBy.mode === "direct") {
    return [
      "*",
      ["get", colorBy.property],
      ["color", 255, 255, 255, ["var", "fillOpacity"]],
    ];
  }
  if (colorBy.mode === "categorical") {
    return [
      "match",
      ["get", colorBy.property],
      ...colorBy.categories.flatMap((category) => [
        category.value,
        colorExpression(category.color),
      ]),
      colorExpression("#9ca3af"),
    ];
  }
  return [
    "interpolate",
    ["linear"],
    ["get", colorBy.property],
    colorBy.numeric.domain[0],
    colorExpression(colorBy.numeric.colors[0]),
    colorBy.numeric.domain[1],
    colorExpression(colorBy.numeric.colors[1]),
  ];
}

function colorExpression(color: string): Expression {
  const [r, g, b] = parseHex(color);
  return ["color", r, g, b, ["var", "fillOpacity"]];
}

function colorWithAlpha(color: string, alpha: number): string {
  const [r, g, b] = parseHex(color);
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`;
}

function parseHex(color: string): [number, number, number] {
  const normalised = normaliseFeatureColor(color);
  const value = Number.parseInt(normalised.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/**
 * Convert legacy annotation `color` payloads to a safe CSS colour.
 *
 * SQLite stores historically contain both CSS strings and RGB lists. Lists
 * may arrive from MVT as JSON strings, so Canvas rendering accepts that form
 * too. Alpha channels are deliberately ignored: layer fill opacity remains
 * the single, predictable opacity control in both renderers.
 */
export function normaliseFeatureColor(
  value: unknown,
  fallback = "#9ca3af",
): string {
  return (
    parseFeatureColor(value) ??
    parseFeatureColor(fallback) ??
    "#9ca3af"
  );
}

function parseFeatureColor(value: unknown): string | undefined {
  if (Array.isArray(value)) return rgbArrayToHex(value);
  if (typeof value !== "string") return undefined;

  const input = value.trim();
  if (input.length === 0 || input.length > 128) return undefined;
  const hex = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.exec(input)?.[1];
  if (hex) {
    const rgb = hex.length <= 4
      ? [...hex.slice(0, 3)].map((channel) => `${channel}${channel}`).join("")
      : hex.slice(0, 6);
    return `#${rgb.toLowerCase()}`;
  }

  if (input.startsWith("[")) {
    try {
      return rgbArrayToHex(JSON.parse(input));
    } catch {
      return undefined;
    }
  }

  return cssRgbToHex(input);
}

function rgbArrayToHex(value: unknown): string | undefined {
  if (!Array.isArray(value) || value.length !== 3) return undefined;
  const channels = value.map((channel) =>
    typeof channel === "number" && Number.isFinite(channel) ? channel : Number.NaN,
  );
  if (channels.some((channel) => !Number.isFinite(channel))) return undefined;

  const unitScale = channels.every((channel) => channel >= 0 && channel <= 1);
  if (
    !unitScale &&
    channels.some((channel) => channel < 0 || channel > 255)
  ) {
    return undefined;
  }
  return channelsToHex(
    channels.map((channel) => Math.round(channel * (unitScale ? 255 : 1))),
  );
}

function cssRgbToHex(input: string): string | undefined {
  const match = /^(rgba?)\((.*)\)$/i.exec(input);
  if (!match?.[1] || match[2] === undefined) return undefined;

  const functionName = match[1].toLowerCase();
  const slashParts = match[2].split("/");
  if (slashParts.length > 2) return undefined;
  let channelTokens = slashParts[0]?.includes(",")
    ? slashParts[0].split(",").map((token) => token.trim())
    : slashParts[0]?.trim().split(/\s+/) ?? [];
  let alpha = slashParts[1]?.trim();
  if (functionName === "rgba" && alpha === undefined && channelTokens.length === 4) {
    alpha = channelTokens[3];
    channelTokens = channelTokens.slice(0, 3);
  }
  if (channelTokens.length !== 3 || channelTokens.some((token) => token === "")) {
    return undefined;
  }
  if (alpha !== undefined && !validCssAlpha(alpha)) return undefined;

  const channels = channelTokens.map(cssRgbChannel);
  if (channels.some((channel) => channel === undefined)) return undefined;
  return channelsToHex(channels as number[]);
}

function cssRgbChannel(value: string): number | undefined {
  const percentage = value.endsWith("%");
  const number = Number(percentage ? value.slice(0, -1) : value);
  const maximum = percentage ? 100 : 255;
  if (!Number.isFinite(number) || number < 0 || number > maximum) return undefined;
  return Math.round(percentage ? (number / 100) * 255 : number);
}

function validCssAlpha(value: string): boolean {
  const percentage = value.endsWith("%");
  const number = Number(percentage ? value.slice(0, -1) : value);
  return Number.isFinite(number) && number >= 0 && number <= (percentage ? 100 : 1);
}

function channelsToHex(channels: readonly number[]): string {
  return `#${channels
    .slice(0, 3)
    .map((channel) => channel.toString(16).padStart(2, "0"))
    .join("")}`;
}

function interpolateHex(start: string, end: string, amount: number): string {
  const a = parseHex(start);
  const b = parseHex(end);
  const channel = (index: number) =>
    Math.round((a[index] ?? 0) + ((b[index] ?? 0) - (a[index] ?? 0)) * amount);
  return `#${[channel(0), channel(1), channel(2)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("")}`;
}
