(function () {
  "use strict";

  const config = JSON.parse(
    document.getElementById("viewer-config").textContent,
  );

  const elements = {
    addOverlay: document.getElementById("add-overlay"),
    annotationOpacity: document.getElementById("annotation-opacity"),
    annotationPath: document.getElementById("annotation-path"),
    annotationSection: document.getElementById("annotation-section"),
    applyFilter: document.getElementById("apply-filter"),
    clearOverlays: document.getElementById("clear-overlays"),
    colorProperty: document.getElementById("color-property"),
    emptyState: document.getElementById("empty-state"),
    featureDetails: document.getElementById("feature-details"),
    filterExpression: document.getElementById("filter-expression"),
    legend: document.getElementById("legend"),
    map: document.getElementById("map"),
    overlayCount: document.getElementById("overlay-count"),
    overlayOpacity: document.getElementById("overlay-opacity"),
    overlaySelect: document.getElementById("overlay-select"),
    resetFilter: document.getElementById("reset-filter"),
    slideCount: document.getElementById("slide-count"),
    slideInfo: document.getElementById("slide-info"),
    slideSelect: document.getElementById("slide-select"),
    statusText: document.getElementById("status-text"),
  };

  const palette = [
    "#f97316",
    "#0ea5e9",
    "#14b8a6",
    "#e11d48",
    "#8b5cf6",
    "#22c55e",
    "#f59e0b",
    "#64748b",
    "#dc2626",
    "#2563eb",
  ];

  const state = {
    annotationFilter: "",
    annotationLayer: null,
    annotationMeta: null,
    annotationPropertySummary: null,
    annotationSource: null,
    baseLayerMeta: null,
    categoryColorCache: {},
    map: null,
    overviewControl: null,
    project: config.project || {},
    rasterLayers: [],
    slideInfo: null,
    currentOverlays: [],
  };

  async function init() {
    bindEvents();

    if (!config.default_session_id) {
      await fetch("/tileserver/session_id", {
        credentials: "same-origin",
      });
    }

    populateSlideSelect(state.project.slides || []);

    if ((state.project.slides || []).length > 0) {
      const defaultSlide = state.project.default_slide || state.project.slides[0].path;
      elements.slideSelect.value = defaultSlide;
      await loadSlide(defaultSlide);
      return;
    }

    if ((config.initial_layers || []).length > 0) {
      renderLayers(config.initial_layers, { fit: true });
      renderSlideInfoFromLayer(config.initial_layers.find((layer) => layer.kind === "slide"));
      await refreshAnnotationControls();
      setStatus("Viewer ready.");
      return;
    }

    toggleEmptyState("No slides are configured for this viewer.");
    setStatus("No slides configured.");
  }

  function bindEvents() {
    elements.slideSelect.addEventListener("change", async (event) => {
      if (!event.target.value) {
        return;
      }
      await loadSlide(event.target.value);
    });

    elements.addOverlay.addEventListener("click", async () => {
      const overlayPath = elements.overlaySelect.value;
      if (!overlayPath) {
        return;
      }
      setStatus("Adding overlay…");
      await putForm("/tileserver/overlay", {
        overlay_path: overlayPath,
      });
      await syncLayers({ fit: false });
      setStatus("Overlay added.");
    });

    elements.clearOverlays.addEventListener("click", async () => {
      setStatus("Clearing overlays…");
      await putForm("/tileserver/clear_overlays");
      await syncLayers({ fit: false });
      setStatus("Overlays cleared.");
    });

    elements.applyFilter.addEventListener("click", async () => {
      state.annotationFilter = elements.filterExpression.value.trim();
      await syncLayers({ fit: false });
      await updateLegend();
      setStatus("Annotation filter applied.");
    });

    elements.resetFilter.addEventListener("click", async () => {
      elements.filterExpression.value = "";
      state.annotationFilter = "";
      await syncLayers({ fit: false });
      await updateLegend();
      setStatus("Annotation filter cleared.");
    });

    elements.colorProperty.addEventListener("change", async () => {
      state.annotationPropertySummary = null;
      if (state.annotationLayer) {
        state.annotationLayer.changed();
      }
      await updateLegend();
      setStatus("Annotation colors updated.");
    });

    elements.overlayOpacity.addEventListener("input", () => {
      state.rasterLayers.forEach((layer) => {
        layer.setOpacity(Number(elements.overlayOpacity.value));
      });
    });

    elements.annotationOpacity.addEventListener("input", () => {
      if (state.annotationLayer) {
        state.annotationLayer.changed();
      }
    });
  }

  async function loadSlide(slidePath) {
    setStatus("Loading slide…");
    await putForm("/tileserver/slide", {
      slide_path: slidePath,
    });
    await refreshOverlayList(slidePath);
    await syncLayers({ fit: true });

    if (state.project.auto_load) {
      for (const overlay of state.currentOverlays) {
        await putForm("/tileserver/overlay", {
          overlay_path: overlay.path,
        });
      }
      await syncLayers({ fit: false });
    }

    setStatus("Slide loaded.");
  }

  async function refreshOverlayList(slidePath) {
    const url = new URL("/tileserver/project/overlays", window.location.origin);
    url.searchParams.set("slide_path", slidePath);
    state.currentOverlays = await fetchJson(url);

    elements.overlaySelect.innerHTML = "";
    elements.overlayCount.textContent = `${state.currentOverlays.length} available`;
    elements.addOverlay.disabled = state.currentOverlays.length === 0;

    if (state.currentOverlays.length === 0) {
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = "No matching overlays";
      elements.overlaySelect.appendChild(emptyOption);
      return;
    }

    state.currentOverlays.forEach((overlay) => {
      const option = document.createElement("option");
      option.value = overlay.path;
      option.textContent = `${overlay.label} (${overlay.kind})`;
      elements.overlaySelect.appendChild(option);
    });
  }

  async function syncLayers(options) {
    const layers = await fetchJson("/tileserver/layers");
    renderLayers(layers, options);

    const slideInfo = await fetchJson("/tileserver/slide");
    state.slideInfo = slideInfo;
    renderSlideInfo(slideInfo);
    await refreshAnnotationControls();
  }

  function renderLayers(layerMetadata, options) {
    const slideLayerMeta = layerMetadata.find((layer) => layer.kind === "slide");
    if (!slideLayerMeta) {
      toggleEmptyState("No slide is currently loaded.");
      return;
    }

    const viewState =
      state.map && !options.fit
        ? {
            center: state.map.getView().getCenter(),
            resolution: state.map.getView().getResolution(),
          }
        : null;

    const baseLayer = createTileLayer(slideLayerMeta, 1);
    const rasterLayerMetadata = layerMetadata.filter((layer) => layer.kind === "raster");
    const rasterLayers = rasterLayerMetadata.map((layer) =>
      createTileLayer(layer, Number(elements.overlayOpacity.value)),
    );
    const annotationLayerMeta = layerMetadata.find((layer) => layer.kind === "annotation");
    const olLayers = [baseLayer, ...rasterLayers];
    let annotationLayer = null;

    if (annotationLayerMeta) {
      annotationLayer = createAnnotationLayer(
        annotationLayerMeta,
        baseLayer.getSource(),
        slideLayerMeta,
      );
      olLayers.push(annotationLayer);
    }

    const shouldRecreateMap =
      !state.map ||
      !state.baseLayerMeta ||
      state.baseLayerMeta.path !== slideLayerMeta.path ||
      JSON.stringify(state.baseLayerMeta.size) !== JSON.stringify(slideLayerMeta.size);

    if (shouldRecreateMap) {
      createMap(olLayers, baseLayer, slideLayerMeta, viewState, options.fit);
    } else {
      const collection = state.map.getLayers();
      collection.clear();
      olLayers.forEach((layer) => collection.push(layer));
      if (viewState) {
        state.map.getView().setCenter(viewState.center);
        state.map.getView().setResolution(viewState.resolution);
      }
      if (state.overviewControl) {
        state.overviewControl.getOverviewMap().getLayers().clear();
        state.overviewControl.getOverviewMap().getLayers().push(
          new ol.layer.Tile({
            source: baseLayer.getSource(),
          }),
        );
      }
    }

    state.baseLayerMeta = slideLayerMeta;
    state.annotationLayer = annotationLayer;
    state.annotationMeta = annotationLayerMeta || null;
    state.annotationSource = annotationLayer ? annotationLayer.getSource() : null;
    state.rasterLayers = rasterLayers;

    toggleEmptyState(null);
  }

  function createSlideProjection(baseSource, slideLayerMeta) {
    const tileGrid = baseSource.getTileGrid();
    return new ol.proj.Projection({
      code: "TIAZoomifyProjection",
      units: "pixels",
      extent: tileGrid.getExtent(),
      metersPerUnit: slideLayerMeta.mpp * 1e-6,
      getPointResolution: function (resolution) {
        return resolution;
      },
    });
  }

  function createMap(layers, baseLayer, slideLayerMeta, viewState, fitView) {
    if (state.map) {
      state.map.setTarget(null);
    }

    const tileGrid = baseLayer.getSource().getTileGrid();
    const projection = createSlideProjection(baseLayer.getSource(), slideLayerMeta);
    const view = new ol.View({
      projection: projection,
      resolutions: tileGrid.getResolutions(),
      constrainOnlyCenter: true,
      extent: tileGrid.getExtent(),
    });

    state.map = new ol.Map({
      target: "map",
      layers: layers,
      view: view,
    });

    addMapControls(baseLayer, projection);
    state.map.on("singleclick", handleFeatureClick);

    if (viewState) {
      view.setCenter(viewState.center);
      view.setResolution(viewState.resolution);
    } else if (fitView !== false) {
      view.fit(tileGrid.getExtent(), {
        size: state.map.getSize() || [800, 600],
      });
    }
  }

  function addMapControls(baseLayer, projection) {
    const scaleLineControl = new ol.control.ScaleLine({
      units: "metric",
      bar: true,
      steps: 8,
      minWidth: 180,
    });
    const overviewMapControl = new ol.control.OverviewMap({
      className: "ol-overviewmap ol-custom-overviewmap",
      layers: [
        new ol.layer.Tile({
          source: baseLayer.getSource(),
        }),
      ],
    });
    const mousePositionControl = new ol.control.MousePosition({
      coordinateFormat: function (coordinate) {
        return ol.coordinate.format(
          [coordinate[0], -coordinate[1]],
          "{x}, {y}",
          0,
        );
      },
      projection: projection,
      className: "ol-mouse-position",
      undefinedHTML: "&nbsp;",
    });
    const rotate = new ol.control.Rotate({
      autoHide: false,
      className: "ol-rotate",
    });
    const fullscreen = new ol.control.FullScreen();
    const layerSwitcher = new ol.control.LayerSwitcher();

    state.map.addControl(scaleLineControl);
    state.map.addControl(overviewMapControl);
    state.map.addControl(mousePositionControl);
    state.map.addControl(rotate);
    state.map.addControl(fullscreen);
    state.map.addControl(layerSwitcher);
    state.overviewControl = overviewMapControl;
  }

  function createTileLayer(metadata, opacity) {
    return new ol.layer.Tile({
      title: metadata.name,
      opacity: opacity,
      source: new ol.source.Zoomify({
        url: metadata.url,
        size: metadata.size,
        crossOrigin: "anonymous",
        zDirection: -1,
      }),
    });
  }

  function buildAnnotationUrl(path, metadata) {
    const url = new URL(path, window.location.origin);
    const annotationMeta = metadata || state.annotationMeta;
    if (annotationMeta && annotationMeta.name) {
      url.searchParams.set("layer_name", annotationMeta.name);
    }
    return url;
  }

  function createAnnotationLayer(metadata, baseSource, slideLayerMeta) {
    const tileGrid = baseSource.getTileGrid();
    const projection = createSlideProjection(baseSource, slideLayerMeta);

    if (metadata.vector_format === "mvt" && metadata.vector_url) {
      const params = new URLSearchParams();
      params.set("layer_name", metadata.name);
      params.set("where", JSON.stringify(state.annotationFilter || null));
      const vectorUrl = `${metadata.vector_url}?${params.toString()}`;

      return new ol.layer.VectorTile({
        title: metadata.name,
        source: new ol.source.VectorTile({
          format: new ol.format.MVT(),
          projection: projection,
          tileGrid: tileGrid,
          url: vectorUrl,
          zDirection: -1,
        }),
        style: annotationStyle,
      });
    }

    const vectorSource = new ol.source.Vector({
      format: new ol.format.GeoJSON(),
      strategy: ol.loadingstrategy.tile(tileGrid),
      loader: function (extent, resolution, projection, success, failure) {
        loadAnnotationExtent(vectorSource, extent, metadata, success, failure);
      },
    });

    return new ol.layer.Vector({
      title: metadata.name,
      source: vectorSource,
      style: annotationStyle,
    });
  }

  async function loadAnnotationExtent(source, extent, metadata, success, failure) {
    const url = buildAnnotationUrl("/tileserver/annotations/geojson", metadata);
    url.searchParams.set(
      "bounds",
      JSON.stringify(extentToSlideBounds(extent, state.baseLayerMeta.size)),
    );
    url.searchParams.set("where", JSON.stringify(state.annotationFilter || null));

    try {
      const payload = await fetchJson(url);
      const features = source.getFormat().readFeatures(payload);
      source.addFeatures(features);
      if (success) {
        success(features);
      }
    } catch (error) {
      if (failure) {
        failure();
      }
      setStatus("Failed to load vector annotations.");
      throw error;
    }
  }

  async function refreshAnnotationControls() {
    if (!state.annotationMeta) {
      elements.annotationSection.classList.add("is-disabled");
      elements.annotationPath.textContent = "No vector overlay";
      elements.colorProperty.innerHTML = "";
      elements.colorProperty.disabled = true;
      elements.applyFilter.disabled = true;
      elements.resetFilter.disabled = true;
      renderLegendEmpty("Load a SQLiteStore-backed overlay to enable vector styling.");
      renderFeatureDetails(null);
      return;
    }

    elements.annotationSection.classList.remove("is-disabled");
    elements.annotationPath.textContent = state.annotationMeta.path;
    elements.colorProperty.disabled = false;
    elements.applyFilter.disabled = false;
    elements.resetFilter.disabled = false;

    const properties = await fetchJson(
      buildAnnotationUrl("/tileserver/prop_names/all"),
    );
    renderColorPropertyOptions(properties);
    await updateLegend();
  }

  function renderColorPropertyOptions(properties) {
    const selectedValue =
      elements.colorProperty.value ||
      state.project.default_cprop ||
      (properties.includes("type") ? "type" : properties[0] || "");

    elements.colorProperty.innerHTML = "";
    properties.sort().forEach((property) => {
      const option = document.createElement("option");
      option.value = property;
      option.textContent = property;
      elements.colorProperty.appendChild(option);
    });

    if (selectedValue) {
      elements.colorProperty.value = properties.includes(selectedValue)
        ? selectedValue
        : properties[0] || "";
    }
  }

  async function updateLegend() {
    if (!state.annotationMeta) {
      return;
    }

    const property = elements.colorProperty.value;
    if (!property) {
      renderLegendEmpty("No annotation properties found on this overlay.");
      return;
    }

    if (property === "color") {
      state.annotationPropertySummary = { kind: "feature-color" };
      elements.legend.innerHTML =
        "<p>Using per-feature colors from the <code>color</code> property.</p>";
      elements.legend.classList.remove("empty");
      if (state.annotationLayer) {
        state.annotationLayer.changed();
      }
      return;
    }

    const url = buildAnnotationUrl(
      `/tileserver/prop_summary/${encodeURIComponent(property)}/all`,
    );
    url.searchParams.set("where", JSON.stringify(state.annotationFilter || null));
    state.annotationPropertySummary = await fetchJson(url);

    if (state.annotationPropertySummary.kind === "empty") {
      renderLegendEmpty("No matching annotation values for the active filter.");
      return;
    }

    if (state.annotationLayer) {
      state.annotationLayer.changed();
    }

    if (state.annotationPropertySummary.kind === "numeric") {
      renderNumericLegend(state.annotationPropertySummary);
      return;
    }

    renderCategoricalLegend(state.annotationPropertySummary.values, property);
  }

  function renderCategoricalLegend(values, property) {
    const list = document.createElement("div");
    list.className = "legend-list";

    values.slice(0, 12).forEach((value) => {
      const row = document.createElement("div");
      row.className = "legend-row";

      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = colorForCategory(property, value);

      const label = document.createElement("span");
      label.textContent = value;

      row.appendChild(swatch);
      row.appendChild(label);
      list.appendChild(row);
    });

    elements.legend.innerHTML = "";
    elements.legend.classList.remove("empty");
    elements.legend.appendChild(list);

    if (values.length > 12) {
      const more = document.createElement("p");
      more.textContent = `Showing 12 of ${values.length} categories.`;
      elements.legend.appendChild(more);
    }
  }

  function renderNumericLegend(summary) {
    const minColor = palette[1];
    const maxColor = palette[3];
    elements.legend.innerHTML = `
      <div class="gradient-swatch" style="background: linear-gradient(90deg, ${minColor}, ${maxColor});"></div>
      <div class="gradient-scale">
        <span>${formatValue(summary.min)}</span>
        <span>${formatValue(summary.max)}</span>
      </div>
    `;
    elements.legend.classList.remove("empty");
  }

  function renderLegendEmpty(message) {
    elements.legend.textContent = message;
    elements.legend.classList.add("empty");
  }

  function annotationStyle(feature) {
    const color = featureColor(feature);
    const rgb = toRgb(color);
    const strokeColor = toRgba(rgb, Math.min(1, Number(elements.annotationOpacity.value) + 0.2));
    const fillColor = toRgba(rgb, Number(elements.annotationOpacity.value) * 0.28);
    const geometryType = feature.getGeometry().getType();

    if (geometryType.includes("Point")) {
      return new ol.style.Style({
        image: new ol.style.Circle({
          radius: 4,
          fill: new ol.style.Fill({ color: fillColor }),
          stroke: new ol.style.Stroke({ color: strokeColor, width: 1.2 }),
        }),
      });
    }

    if (geometryType.includes("LineString")) {
      return new ol.style.Style({
        stroke: new ol.style.Stroke({
          color: strokeColor,
          width: 2,
        }),
      });
    }

    return new ol.style.Style({
      fill: new ol.style.Fill({
        color: fillColor,
      }),
      stroke: new ol.style.Stroke({
        color: strokeColor,
        width: 1.4,
      }),
    });
  }

  function featureColor(feature) {
    const property = elements.colorProperty.value;
    if (!property) {
      return palette[0];
    }

    const value = feature.get(property);
    if (property === "color") {
      return normaliseFeatureColor(value) || palette[0];
    }

    if (state.annotationPropertySummary && state.annotationPropertySummary.kind === "numeric") {
      return interpolateColor(
        state.annotationPropertySummary.min,
        state.annotationPropertySummary.max,
        Number(value),
      );
    }

    return colorForCategory(property, value);
  }

  function colorForCategory(property, value) {
    const key = String(value);
    if (!state.categoryColorCache[property]) {
      state.categoryColorCache[property] = {};
    }

    const configuredColor =
      property === "type" && state.project.color_dict ? state.project.color_dict[key] : null;
    if (configuredColor) {
      return normaliseFeatureColor(configuredColor) || palette[0];
    }

    if (!state.categoryColorCache[property][key]) {
      const index = Object.keys(state.categoryColorCache[property]).length % palette.length;
      state.categoryColorCache[property][key] = palette[index];
    }
    return state.categoryColorCache[property][key];
  }

  function interpolateColor(min, max, value) {
    if (!Number.isFinite(value) || min === max) {
      return palette[0];
    }

    const start = toRgb(palette[1]);
    const end = toRgb(palette[3]);
    const ratio = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return `rgb(${Math.round(start[0] + (end[0] - start[0]) * ratio)}, ${Math.round(
      start[1] + (end[1] - start[1]) * ratio,
    )}, ${Math.round(start[2] + (end[2] - start[2]) * ratio)})`;
  }

  function normaliseFeatureColor(value) {
    if (Array.isArray(value)) {
      return colourFromSequence(value);
    }
    if (typeof value === "string") {
      if (value.startsWith("#") || value.startsWith("rgb")) {
        return value;
      }
      try {
        return normaliseFeatureColor(JSON.parse(value));
      } catch (_error) {
        if (value.includes(",")) {
          return colourFromSequence(value.split(",").map(Number));
        }
      }
    }
    if (value && typeof value === "object" && ["r", "g", "b"].every((key) => key in value)) {
      return colourFromSequence([value.r, value.g, value.b]);
    }
    return null;
  }

  function colourFromSequence(sequence) {
    if (sequence.length < 3) {
      return null;
    }
    const usesFloatRange = sequence.every((component) => Number(component) <= 1);
    const scale = usesFloatRange ? 255 : 1;
    return `rgb(${Math.round(Number(sequence[0]) * scale)}, ${Math.round(
      Number(sequence[1]) * scale,
    )}, ${Math.round(Number(sequence[2]) * scale)})`;
  }

  function handleFeatureClick(event) {
    if (!state.annotationLayer) {
      return;
    }

    const feature = state.map.forEachFeatureAtPixel(
      event.pixel,
      function (candidate) {
        return candidate;
      },
      {
        layerFilter: function (layer) {
          return layer === state.annotationLayer;
        },
      },
    );
    renderFeatureDetails(feature);
  }

  function renderFeatureDetails(feature) {
    if (!feature) {
      elements.featureDetails.textContent = "Click an annotation feature to inspect its properties.";
      elements.featureDetails.classList.add("empty");
      return;
    }

    const table = document.createElement("table");
    table.className = "feature-table";
    const tbody = document.createElement("tbody");

    Object.entries(feature.getProperties())
      .filter(function ([key]) {
        return key !== "geometry";
      })
      .forEach(function ([key, value]) {
        const row = document.createElement("tr");
        const header = document.createElement("td");
        const content = document.createElement("td");
        header.textContent = key;
        content.textContent = formatValue(value);
        row.appendChild(header);
        row.appendChild(content);
        tbody.appendChild(row);
      });

    table.appendChild(tbody);
    elements.featureDetails.innerHTML = "";
    elements.featureDetails.classList.remove("empty");
    elements.featureDetails.appendChild(table);
  }

  function renderSlideInfo(info) {
    const values = [
      ["Path", info.file_path],
      ["Dimensions", Array.isArray(info.slide_dimensions) ? info.slide_dimensions.join(" x ") : info.slide_dimensions],
      ["MPP", Array.isArray(info.mpp) ? info.mpp.join(", ") : info.mpp],
      ["Axes", info.axes || "YX"],
    ];
    elements.slideInfo.innerHTML = "";
    values.forEach(function ([term, description]) {
      if (description === undefined || description === null) {
        return;
      }
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = String(description);
      elements.slideInfo.appendChild(dt);
      elements.slideInfo.appendChild(dd);
    });
  }

  function renderSlideInfoFromLayer(layer) {
    if (!layer) {
      return;
    }
    renderSlideInfo({
      file_path: layer.path,
      slide_dimensions: layer.size,
      mpp: layer.mpp,
    });
  }

  function populateSlideSelect(slides) {
    elements.slideSelect.innerHTML = "";
    elements.slideCount.textContent = `${slides.length} available`;
    slides.forEach((slide) => {
      const option = document.createElement("option");
      option.value = slide.path;
      option.textContent = slide.label;
      elements.slideSelect.appendChild(option);
    });
  }

  function extentToSlideBounds(extent, slideSize) {
    const minX = Math.max(0, Math.floor(extent[0]));
    const maxX = Math.min(slideSize[0], Math.ceil(extent[2]));
    const minY = Math.max(0, Math.floor(-extent[3]));
    const maxY = Math.min(slideSize[1], Math.ceil(-extent[1]));
    return [minX, minY, maxX, maxY];
  }

  function toRgb(color) {
    if (Array.isArray(color)) {
      return color;
    }
    const candidate = normaliseFeatureColor(color) || color;
    if (candidate.startsWith("#")) {
      const clean = candidate.replace("#", "");
      const hex = clean.length === 3 ? clean.split("").map((char) => char + char).join("") : clean;
      return [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16),
      ];
    }
    const match = candidate.match(/\d+(\.\d+)?/g);
    if (!match) {
      return [249, 115, 22];
    }
    return match.slice(0, 3).map((component) => Math.round(Number(component)));
  }

  function toRgba(rgb, alpha) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }

  function formatValue(value) {
    if (value === null || value === undefined) {
      return "";
    }
    if (typeof value === "number") {
      return Number.isInteger(value) ? String(value) : value.toFixed(4);
    }
    if (Array.isArray(value) || typeof value === "object") {
      return JSON.stringify(value);
    }
    return String(value);
  }

  function toggleEmptyState(message) {
    if (message) {
      elements.emptyState.textContent = message;
      elements.emptyState.classList.remove("hidden");
      return;
    }
    elements.emptyState.classList.add("hidden");
  }

  function setStatus(message) {
    elements.statusText.textContent = message;
  }

  async function fetchJson(input) {
    const response = await fetch(input, {
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    return response.json();
  }

  async function putForm(url, data) {
    const body = new FormData();
    Object.entries(data || {}).forEach(function ([key, value]) {
      body.append(key, value);
    });

    const response = await fetch(url, {
      method: "PUT",
      body: body,
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    return response.text();
  }

  init().catch(function (error) {
    console.error(error);
    toggleEmptyState("Failed to initialise the viewer.");
    setStatus("Viewer initialisation failed.");
  });
})();
