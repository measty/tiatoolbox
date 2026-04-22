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

  const DENSE_POLYGON_STROKE_ZOOM_OFFSET = 2;

  const state = {
    annotationFilter: "",
    annotationControlsRequestId: 0,
    annotationLegendRequestId: 0,
    annotationLayer: null,
    annotationMetadataCache: new Map(),
    annotationMetadataRequests: new Map(),
    annotationMeta: null,
    annotationPropertySummary: null,
    annotationSelectedProperty: config.project?.default_cprop || "",
    annotationSource: null,
    annotationStyleCache: new Map(),
    annotationTileColorProperty: null,
    baseLayerMeta: null,
    baseResolutions: null,
    categoryColorCache: {},
    featureDetailsRequestId: 0,
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
      triggerAnnotationControlsRefresh();
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
      state.annotationSelectedProperty = elements.colorProperty.value || "";
      state.annotationPropertySummary = null;
      resetAnnotationStyleCache();
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
      resetAnnotationStyleCache();
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
    const renderState = renderLayers(layers, options);

    const slideInfo = await fetchJson("/tileserver/slide");
    state.slideInfo = slideInfo;
    renderSlideInfo(slideInfo);

    if (renderState.annotationContextChanged) {
      triggerAnnotationControlsRefresh();
    }
  }

  function renderLayers(layerMetadata, options) {
    const previousAnnotationContext = annotationContextKey(state.annotationMeta);
    const slideLayerMeta = layerMetadata.find((layer) => layer.kind === "slide");
    if (!slideLayerMeta) {
      toggleEmptyState("No slide is currently loaded.");
      return { annotationContextChanged: previousAnnotationContext !== null };
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
    state.baseResolutions = baseLayer.getSource().getTileGrid().getResolutions();
    state.annotationLayer = annotationLayer;
    state.annotationMeta = annotationLayerMeta || null;
    state.annotationPropertySummary = null;
    state.annotationSource = annotationLayer ? annotationLayer.getSource() : null;
    state.annotationTileColorProperty = null;
    state.rasterLayers = rasterLayers;
    state.featureDetailsRequestId += 1;
    resetAnnotationStyleCache();
    clearFeatureDetails();

    toggleEmptyState(null);

    const nextAnnotationContext = annotationContextKey(state.annotationMeta);
    const annotationContextChanged = previousAnnotationContext !== nextAnnotationContext;
    if (annotationContextChanged) {
      if (state.annotationMeta) {
        prepareAnnotationControlsLoading(state.annotationMeta);
      } else {
        disableAnnotationControls();
      }
    }

    return {
      annotationContextChanged: annotationContextChanged,
    };
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

  function annotationRepresentations(metadata) {
    if (Array.isArray(metadata.vector_representations) && metadata.vector_representations.length > 0) {
      return metadata.vector_representations;
    }

    if (metadata.vector_format === "mvt" && metadata.vector_url) {
      return [
        {
          id: metadata.default_vector_representation || "default",
          min_zoom: 0,
          vector_format: metadata.vector_format,
          vector_url: metadata.vector_url,
        },
      ];
    }

    return [];
  }

  function selectAnnotationRepresentation(metadata, zoom) {
    const representations = annotationRepresentations(metadata);
    if (representations.length === 0) {
      return null;
    }

    return (
      representations.find((representation) => {
        const minZoom = representation.min_zoom ?? 0;
        const maxZoom = representation.max_zoom ?? Number.POSITIVE_INFINITY;
        return zoom >= minZoom && zoom <= maxZoom;
      }) || representations[representations.length - 1]
    );
  }

  function zoomForResolution(resolution) {
    const resolutions = state.baseResolutions || [];
    if (!Number.isFinite(resolution) || resolutions.length === 0) {
      return null;
    }

    let nearestZoom = 0;
    let nearestDelta = Number.POSITIVE_INFINITY;

    resolutions.forEach((candidateResolution, index) => {
      const delta = Math.abs(candidateResolution - resolution);
      if (delta < nearestDelta) {
        nearestDelta = delta;
        nearestZoom = index;
      }
    });

    return nearestZoom;
  }

  function polygonStrokeMinZoom(metadata) {
    if (!metadata) {
      return null;
    }

    const fullRepresentation = annotationRepresentations(metadata).find(
      (representation) => representation.id === "full",
    );
    if (!fullRepresentation) {
      return null;
    }

    const fullGeometryMinZoom = Number(fullRepresentation.min_zoom || 0);
    const maxZoom = Math.max(0, (state.baseResolutions || []).length - 1);
    return Math.min(
      maxZoom,
      fullGeometryMinZoom + DENSE_POLYGON_STROKE_ZOOM_OFFSET,
    );
  }

  function shouldRenderPolygonStroke(resolution) {
    const zoom = zoomForResolution(resolution);
    if (zoom === null || !state.annotationMeta) {
      return true;
    }

    const representation = selectAnnotationRepresentation(state.annotationMeta, zoom);
    if (!representation) {
      return true;
    }

    if (representation.id && representation.id !== "full") {
      return false;
    }

    const strokeMinZoom = polygonStrokeMinZoom(state.annotationMeta);
    if (strokeMinZoom === null) {
      return true;
    }

    return zoom >= strokeMinZoom;
  }

  function createAnnotationLayer(metadata, baseSource, slideLayerMeta) {
    const tileGrid = baseSource.getTileGrid();
    const projection = createSlideProjection(baseSource, slideLayerMeta);

    if (metadata.vector_format === "mvt" && metadata.vector_url) {
      return new ol.layer.VectorTile({
        title: metadata.name,
        source: new ol.source.VectorTile({
          format: new ol.format.MVT(),
          projection: projection,
          tileGrid: tileGrid,
          tileUrlFunction: function (tileCoord) {
            if (!tileCoord) {
              return undefined;
            }

            const representation = selectAnnotationRepresentation(metadata, tileCoord[0]);
            if (!representation || !representation.vector_url) {
              return undefined;
            }

            const params = new URLSearchParams();
            params.set("layer_name", metadata.name);
            params.set("where", JSON.stringify(state.annotationFilter || null));
            params.set("rev", String(metadata.vector_revision || 0));
            params.set("cprop", elements.colorProperty.value || "");

            const vectorUrl = representation.vector_url
              .replace("{z}", tileCoord[0])
              .replace("{x}", tileCoord[1])
              .replace("{y}", tileCoord[2]);
            return `${vectorUrl}?${params.toString()}`;
          },
          zDirection: -1,
        }),
        style: annotationStyle,
      });
    }

    if (metadata.geojson_policy && metadata.geojson_policy.allowed === false) {
      setStatus(
        metadata.geojson_policy.message ||
          "Large SQLite-backed overlays require MVT tiles for normal viewing.",
      );
      return new ol.layer.Vector({
        title: metadata.name,
        source: new ol.source.Vector({
          features: [],
          format: new ol.format.GeoJSON(),
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
      const response = await fetch(url, {
        credentials: "same-origin",
      });
      if (!response.ok) {
        let message = "Failed to load vector annotations.";
        try {
          const payload = await response.json();
          if (payload && payload.message) {
            message = payload.message;
          }
        } catch (parseError) {
          message = `Request failed: ${response.status}`;
        }
        throw new Error(message);
      }

      const payload = await response.json();
      const features = source.getFormat().readFeatures(payload);
      source.addFeatures(features);
      if (success) {
        success(features);
      }
    } catch (error) {
      if (failure) {
        failure();
      }
      setStatus(error && error.message ? error.message : "Failed to load vector annotations.");
      throw error;
    }
  }

  function annotationContextKey(metadata) {
    if (!metadata) {
      return null;
    }

    return JSON.stringify({
      name: metadata.name || "",
      path: metadata.path || "",
      revision: metadata.vector_revision || 0,
    });
  }

  function annotationMetadataCacheKey(kind, metadata, extra) {
    return JSON.stringify({
      kind: kind,
      annotation: annotationContextKey(metadata),
      extra: extra || null,
    });
  }

  function fetchCachedAnnotationJson(cacheKey, input) {
    if (state.annotationMetadataCache.has(cacheKey)) {
      return Promise.resolve(state.annotationMetadataCache.get(cacheKey));
    }

    if (state.annotationMetadataRequests.has(cacheKey)) {
      return state.annotationMetadataRequests.get(cacheKey);
    }

    const request = fetchJson(input)
      .then((payload) => {
        state.annotationMetadataCache.set(cacheKey, payload);
        return payload;
      })
      .finally(() => {
        state.annotationMetadataRequests.delete(cacheKey);
      });

    state.annotationMetadataRequests.set(cacheKey, request);
    return request;
  }

  function disableAnnotationControls() {
    elements.annotationSection.classList.add("is-disabled");
    elements.annotationPath.textContent = "No vector overlay";
    elements.colorProperty.innerHTML = "";
    elements.colorProperty.disabled = true;
    elements.applyFilter.disabled = true;
    elements.resetFilter.disabled = true;
    renderLegendEmpty("Load a SQLiteStore-backed overlay to enable vector styling.");
    renderFeatureDetails(null);
  }

  function prepareAnnotationControlsLoading(metadata) {
    elements.annotationSection.classList.remove("is-disabled");
    elements.annotationPath.textContent = metadata.path;
    elements.colorProperty.innerHTML = "";

    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Loading properties…";
    elements.colorProperty.appendChild(option);
    elements.colorProperty.disabled = true;
    elements.applyFilter.disabled = false;
    elements.resetFilter.disabled = false;

    renderLegendEmpty("Loading annotation metadata…");
  }

  function triggerAnnotationControlsRefresh() {
    refreshAnnotationControls().catch((error) => {
      console.error(error);
    });
  }

  async function refreshAnnotationControls() {
    if (!state.annotationMeta) {
      disableAnnotationControls();
      return;
    }

    const requestId = ++state.annotationControlsRequestId;
    const annotationMeta = state.annotationMeta;
    const annotationContext = annotationContextKey(annotationMeta);

    elements.annotationSection.classList.remove("is-disabled");
    elements.annotationPath.textContent = annotationMeta.path;
    elements.applyFilter.disabled = false;
    elements.resetFilter.disabled = false;

    const propertiesCacheKey = annotationMetadataCacheKey(
      "properties",
      annotationMeta,
    );
    const cachedProperties = state.annotationMetadataCache.get(propertiesCacheKey);
    if (cachedProperties) {
      renderColorPropertyOptions(cachedProperties);
      elements.colorProperty.disabled = false;
      void updateLegend();
      return;
    }

    prepareAnnotationControlsLoading(annotationMeta);

    try {
      const properties = await fetchCachedAnnotationJson(
        propertiesCacheKey,
        buildAnnotationUrl("/tileserver/prop_names/all", annotationMeta),
      );
      if (
        requestId !== state.annotationControlsRequestId ||
        annotationContext !== annotationContextKey(state.annotationMeta)
      ) {
        return;
      }

      renderColorPropertyOptions(properties);
      elements.colorProperty.disabled = false;
      void updateLegend();
    } catch (error) {
      if (
        requestId !== state.annotationControlsRequestId ||
        annotationContext !== annotationContextKey(state.annotationMeta)
      ) {
        return;
      }

      elements.colorProperty.innerHTML = "";
      elements.colorProperty.disabled = true;
      renderLegendEmpty("Annotation metadata could not be loaded.");
      throw error;
    }
  }

  function renderColorPropertyOptions(properties) {
    const selectedValue =
      state.annotationSelectedProperty ||
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

    state.annotationSelectedProperty = elements.colorProperty.value || "";
  }

  function refreshAnnotationTilesForCurrentProperty() {
    const property = elements.colorProperty.value || "";
    if (
      state.annotationSource &&
      typeof state.annotationSource.refresh === "function" &&
      state.annotationTileColorProperty !== property
    ) {
      state.annotationTileColorProperty = property;
      state.annotationSource.refresh();
    }
  }

  async function updateLegend() {
    if (!state.annotationMeta) {
      return;
    }

    const requestId = ++state.annotationLegendRequestId;
    const annotationMeta = state.annotationMeta;
    const annotationContext = annotationContextKey(annotationMeta);
    const property = elements.colorProperty.value;
    if (!property) {
      state.annotationPropertySummary = null;
      renderLegendEmpty("No annotation properties found on this overlay.");
      return;
    }

    state.annotationSelectedProperty = property;
    refreshAnnotationTilesForCurrentProperty();

    if (property === "color") {
      if (
        requestId !== state.annotationLegendRequestId ||
        annotationContext !== annotationContextKey(state.annotationMeta) ||
        property !== elements.colorProperty.value
      ) {
        return;
      }

      state.annotationPropertySummary = { kind: "feature-color" };
      elements.legend.innerHTML =
        "<p>Using per-feature colors from the <code>color</code> property.</p>";
      elements.legend.classList.remove("empty");
      if (state.annotationLayer) {
        state.annotationLayer.changed();
      }
      return;
    }

    const filterExpression = state.annotationFilter || null;
    const summaryCacheKey = annotationMetadataCacheKey(
      "property-summary",
      annotationMeta,
      {
        filter: filterExpression,
        property: property,
      },
    );
    if (!state.annotationMetadataCache.has(summaryCacheKey)) {
      renderLegendEmpty(`Loading legend for ${property}…`);
    }

    try {
      const url = buildAnnotationUrl(
        `/tileserver/prop_summary/${encodeURIComponent(property)}/all`,
        annotationMeta,
      );
      url.searchParams.set("where", JSON.stringify(filterExpression));
      const summary = await fetchCachedAnnotationJson(summaryCacheKey, url);

      if (
        requestId !== state.annotationLegendRequestId ||
        annotationContext !== annotationContextKey(state.annotationMeta) ||
        property !== elements.colorProperty.value ||
        filterExpression !== (state.annotationFilter || null)
      ) {
        return;
      }

      state.annotationPropertySummary = summary;
    } catch (error) {
      if (
        requestId !== state.annotationLegendRequestId ||
        annotationContext !== annotationContextKey(state.annotationMeta) ||
        property !== elements.colorProperty.value ||
        filterExpression !== (state.annotationFilter || null)
      ) {
        return;
      }

      state.annotationPropertySummary = null;
      renderLegendEmpty("Annotation legend could not be loaded.");
      console.error(error);
      return;
    }

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

  function annotationStyle(feature, resolution) {
    const color = featureColor(feature);
    const rgb = toRgb(color);
    const opacity = Number(elements.annotationOpacity.value);
    const strokeAlpha = Math.min(1, opacity + 0.2);
    const fillAlpha = opacity * 0.28;
    const geometryType = feature.getGeometry().getType();
    const polygonStrokeEnabled =
      !geometryType.includes("Polygon") || shouldRenderPolygonStroke(resolution);
    const cacheKey = `${geometryType}:${rgb.join(",")}:${strokeAlpha.toFixed(3)}:${fillAlpha.toFixed(3)}:${polygonStrokeEnabled ? "stroke" : "fill"}`;

    if (state.annotationStyleCache.has(cacheKey)) {
      return state.annotationStyleCache.get(cacheKey);
    }

    let style;
    if (geometryType.includes("Point")) {
      style = new ol.style.Style({
        image: new ol.style.Circle({
          radius: 4,
          fill: new ol.style.Fill({ color: toRgba(rgb, fillAlpha) }),
          stroke: new ol.style.Stroke({ color: toRgba(rgb, strokeAlpha), width: 1.2 }),
        }),
      });
      state.annotationStyleCache.set(cacheKey, style);
      return style;
    }

    if (geometryType.includes("LineString")) {
      style = new ol.style.Style({
        stroke: new ol.style.Stroke({
          color: toRgba(rgb, strokeAlpha),
          width: 2,
        }),
      });
      state.annotationStyleCache.set(cacheKey, style);
      return style;
    }

    style = new ol.style.Style({
      fill: new ol.style.Fill({
        color: toRgba(rgb, fillAlpha),
      }),
      stroke: polygonStrokeEnabled
        ? new ol.style.Stroke({
            color: toRgba(rgb, strokeAlpha),
            width: 1.4,
          })
        : undefined,
    });
    state.annotationStyleCache.set(cacheKey, style);
    return style;
  }

  function resetAnnotationStyleCache() {
    state.annotationStyleCache.clear();
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

  async function handleFeatureClick(event) {
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
    await renderFeatureDetails(feature);
  }

  function clearFeatureDetails() {
    elements.featureDetails.textContent = "Click an annotation feature to inspect its properties.";
    elements.featureDetails.classList.add("empty");
  }

  function featurePropertiesForDisplay(feature) {
    return Object.fromEntries(
      Object.entries(feature.getProperties()).filter(function ([key]) {
        return key !== "geometry";
      }),
    );
  }

  function renderFeatureDetailsTable(properties, message) {
    const table = document.createElement("table");
    table.className = "feature-table";
    const tbody = document.createElement("tbody");

    Object.entries(properties).forEach(function ([key, value]) {
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

    if (message) {
      const note = document.createElement("p");
      note.textContent = message;
      elements.featureDetails.appendChild(note);
    }
  }

  async function renderFeatureDetails(feature) {
    state.featureDetailsRequestId += 1;
    const requestId = state.featureDetailsRequestId;

    if (!feature) {
      clearFeatureDetails();
      return;
    }

    const projectedProperties = featurePropertiesForDisplay(feature);
    const featureId = projectedProperties.id;

    if (!featureId || !state.annotationMeta) {
      renderFeatureDetailsTable(projectedProperties);
      return;
    }

    renderFeatureDetailsTable(projectedProperties, "Loading full annotation details…");

    try {
      const url = buildAnnotationUrl(
        state.annotationMeta.detail_url || "/tileserver/annotations/detail",
      );
      url.searchParams.set("key", featureId);
      const payload = await fetchJson(url);
      if (requestId !== state.featureDetailsRequestId) {
        return;
      }
      const fullProperties = Object.assign(
        { id: payload.id || featureId },
        payload.properties || {},
      );
      renderFeatureDetailsTable(fullProperties);
    } catch (_error) {
      if (requestId !== state.featureDetailsRequestId) {
        return;
      }
      renderFeatureDetailsTable(projectedProperties, "Full annotation details could not be loaded.");
    }
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

  if (typeof window !== "undefined" && window.__TIA_VISUALIZER_ENABLE_TEST_HOOKS__) {
    window.__TIA_VISUALIZER_TEST_HOOKS__ = {
      annotationStyle: annotationStyle,
      annotationContextKey: annotationContextKey,
      annotationMetadataCacheKey: annotationMetadataCacheKey,
      createAnnotationLayer: createAnnotationLayer,
      elements: elements,
      fetchCachedAnnotationJson: fetchCachedAnnotationJson,
      loadAnnotationExtent: loadAnnotationExtent,
      prepareAnnotationControlsLoading: prepareAnnotationControlsLoading,
      refreshAnnotationControls: refreshAnnotationControls,
      renderColorPropertyOptions: renderColorPropertyOptions,
      shouldRenderPolygonStroke: shouldRenderPolygonStroke,
      state: state,
      syncLayers: syncLayers,
      updateLegend: updateLegend,
      zoomForResolution: zoomForResolution,
    };
  }

  init().catch(function (error) {
    console.error(error);
    toggleEmptyState("Failed to initialise the viewer.");
    setStatus("Viewer initialisation failed.");
  });
})();
