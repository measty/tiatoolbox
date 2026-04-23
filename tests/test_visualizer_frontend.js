const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const VISUALIZER_PATH = path.join(
  '/media/mark-eastwood/Work/tiatoolbox',
  'tiatoolbox/data/visualization/static/js/visualizer.js',
);

class FakeClassList {
  constructor(initial = []) {
    this.values = new Set(initial);
  }

  add(...names) {
    names.forEach((name) => this.values.add(name));
  }

  remove(...names) {
    names.forEach((name) => this.values.delete(name));
  }

  contains(name) {
    return this.values.has(name);
  }
}

class FakeElement {
  constructor(tagName = 'div', id = null) {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.value = '';
    this.disabled = false;
    this.children = [];
    this.listeners = new Map();
    this.className = '';
    this.classList = new FakeClassList();
    this.style = {};
    this._innerHTML = '';
    this._textContent = '';
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this.children = [];
    if (value === '') {
      this._textContent = '';
    }
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(value) {
    this._textContent = String(value);
    this._innerHTML = '';
    this.children = [];
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  addEventListener(type, handler) {
    this.listeners.set(type, handler);
  }
}

class FakeCollection {
  constructor(items = []) {
    this.items = items.slice();
  }

  clear() {
    this.items = [];
  }

  push(item) {
    this.items.push(item);
  }
}

class FakeTileGrid {
  getExtent() {
    return [0, 0, 1000, 1000];
  }

  getResolutions() {
    return [4, 2, 1];
  }
}

class FakeTileSource {
  constructor(config) {
    this.config = config;
    this.tileGrid = new FakeTileGrid();
  }

  getTileGrid() {
    return this.tileGrid;
  }
}

class FakeTileLayer {
  constructor(config) {
    this.config = config;
    this.source = config.source;
    this.opacity = config.opacity;
  }

  getSource() {
    return this.source;
  }

  setOpacity(opacity) {
    this.opacity = opacity;
  }

  changed() {}
}

class FakeVectorTileSource {
  constructor(config) {
    this.config = config;
    this.refreshCount = 0;
  }

  refresh() {
    this.refreshCount += 1;
  }
}

class FakeVectorLayer {
  constructor(config) {
    this.config = config;
    this.source = config.source;
    this.changedCount = 0;
  }

  getSource() {
    return this.source;
  }

  changed() {
    this.changedCount += 1;
  }
}

class FakeView {
  constructor(config) {
    this.center = null;
    this.resolution = (config.resolutions || [1])[0];
    this.extent = config.extent;
  }

  getCenter() {
    return this.center;
  }

  getResolution() {
    return this.resolution;
  }

  setCenter(center) {
    this.center = center;
  }

  setResolution(resolution) {
    this.resolution = resolution;
  }

  fit(extent) {
    this.extent = extent;
  }
}

class FakeMap {
  constructor(config) {
    this.layers = new FakeCollection(config.layers || []);
    this.view = config.view;
    this.controls = [];
  }

  addControl(control) {
    this.controls.push(control);
  }

  on() {}

  getView() {
    return this.view;
  }

  getLayers() {
    return this.layers;
  }

  getSize() {
    return [800, 600];
  }

  setTarget() {}

  forEachFeatureAtPixel() {
    return null;
  }
}

class FakeOverviewMapControl {
  constructor() {
    this.layers = new FakeCollection();
  }

  getOverviewMap() {
    return {
      getLayers: () => this.layers,
    };
  }
}

class FakeFormData {
  constructor() {
    this.entries = [];
  }

  append(key, value) {
    this.entries.push([key, value]);
  }
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function response(payload) {
  return {
    ok: true,
    async json() {
      return payload;
    },
    async text() {
      return typeof payload === 'string' ? payload : JSON.stringify(payload);
    },
  };
}

class FakeOlStyle {
  constructor(config = {}) {
    this.config = config;
    Object.assign(this, config);
  }
}

function createOlStub() {
  return {
    layer: {
      Tile: FakeTileLayer,
      VectorTile: FakeVectorLayer,
      Vector: FakeVectorLayer,
    },
    source: {
      Zoomify: FakeTileSource,
      VectorTile: FakeVectorTileSource,
      Vector: class {
        constructor(config) {
          this.config = config;
        }

        getFormat() {
          return {
            readFeatures(payload) {
              return payload.features || [];
            },
          };
        }

        addFeatures() {}

        refresh() {}
      },
    },
    format: {
      MVT: class {},
      GeoJSON: class {},
    },
    style: {
      Style: FakeOlStyle,
      Circle: FakeOlStyle,
      Fill: FakeOlStyle,
      Stroke: FakeOlStyle,
    },
    loadingstrategy: {
      tile() {
        return null;
      },
    },
    proj: {
      Projection: class {
        constructor(config) {
          this.config = config;
        }
      },
    },
    View: FakeView,
    Map: FakeMap,
    control: {
      ScaleLine: class {},
      OverviewMap: FakeOverviewMapControl,
      MousePosition: class {},
      Rotate: class {},
      FullScreen: class {},
      LayerSwitcher: class {},
    },
    coordinate: {
      format() {
        return '';
      },
    },
  };
}

function createEnvironment(fetchImpl, { withOl = false } = {}) {
  const ids = {
    'add-overlay': 'button',
    'annotation-opacity': 'input',
    'annotation-path': 'span',
    'annotation-section': 'section',
    'apply-filter': 'button',
    'clear-overlays': 'button',
    'color-property': 'select',
    'empty-state': 'div',
    'feature-details': 'div',
    'filter-expression': 'textarea',
    legend: 'div',
    map: 'div',
    'overlay-count': 'span',
    'overlay-opacity': 'input',
    'overlay-select': 'select',
    'reset-filter': 'button',
    'slide-count': 'span',
    'slide-info': 'dl',
    'slide-select': 'select',
    'status-text': 'span',
    'viewer-config': 'script',
  };

  const elements = Object.fromEntries(
    Object.entries(ids).map(([id, tag]) => [id, new FakeElement(tag, id)]),
  );
  elements['annotation-section'].classList.add('is-disabled');
  elements.legend.classList.add('empty');
  elements['viewer-config'].textContent = JSON.stringify({
    default_session_id: 'test-session',
    initial_layers: [],
    project: { slides: [] },
  });
  elements['overlay-opacity'].value = '0.7';
  elements['annotation-opacity'].value = '0.8';

  const document = {
    getElementById(id) {
      return elements[id];
    },
    createElement(tagName) {
      return new FakeElement(tagName);
    },
  };

  const window = {
    document,
    location: { origin: 'http://localhost' },
    __TIA_VISUALIZER_ENABLE_TEST_HOOKS__: true,
  };
  window.window = window;

  const context = vm.createContext({
    console,
    document,
    fetch: fetchImpl,
    FormData: FakeFormData,
    Promise,
    setTimeout,
    clearTimeout,
    URL,
    URLSearchParams,
    window,
    ol: withOl ? createOlStub() : undefined,
  });

  const code = fs.readFileSync(VISUALIZER_PATH, 'utf8');
  vm.runInContext(code, context, { filename: VISUALIZER_PATH });

  return {
    elements,
    hooks: window.__TIA_VISUALIZER_TEST_HOOKS__,
  };
}

test('refreshAnnotationControls dedupes in-flight property-name requests', async () => {
  const propertyFetch = deferred();
  let propertyCalls = 0;

  const { hooks, elements } = createEnvironment(async (input) => {
    const url = String(input);
    if (url.includes('/tileserver/prop_names/all')) {
      propertyCalls += 1;
      const payload = await propertyFetch.promise;
      return response(payload);
    }
    if (url.includes('/tileserver/prop_summary/')) {
      return response({ kind: 'categorical', values: ['tumour'] });
    }
    throw new Error(`Unexpected fetch: ${url}`);
  });

  hooks.state.annotationMeta = {
    name: 'overlay',
    path: '/tmp/overlay.db',
    vector_revision: 3,
  };

  const first = hooks.refreshAnnotationControls();
  const second = hooks.refreshAnnotationControls();
  propertyFetch.resolve(['score', 'type']);

  await Promise.all([first, second]);

  assert.equal(propertyCalls, 1);
  assert.equal(elements['color-property'].disabled, false);
  assert.equal(elements['color-property'].value, 'type');
});

test('updateLegend keeps the latest summary when earlier requests finish later', async () => {
  const firstSummary = deferred();
  const secondSummary = deferred();
  let summaryCalls = 0;

  const { hooks, elements } = createEnvironment(async (input) => {
    const url = new URL(String(input), 'http://localhost');
    if (url.pathname.includes('/tileserver/prop_summary/')) {
      summaryCalls += 1;
      const where = url.searchParams.get('where');
      const payload = where === 'null' ? await firstSummary.promise : await secondSummary.promise;
      return response(payload);
    }
    throw new Error(`Unexpected fetch: ${String(input)}`);
  });

  hooks.state.annotationMeta = {
    name: 'overlay',
    path: '/tmp/overlay.db',
    vector_revision: 5,
  };
  hooks.renderColorPropertyOptions(['score']);
  elements['color-property'].value = 'score';

  hooks.state.annotationFilter = '';
  const staleRequest = hooks.updateLegend();

  hooks.state.annotationFilter = "props['score'] > 0.5";
  const latestRequest = hooks.updateLegend();

  secondSummary.resolve({ kind: 'numeric', min: 0.5, max: 1.0 });
  await latestRequest;

  firstSummary.resolve({ kind: 'numeric', min: 0, max: 10 });
  await staleRequest;

  assert.equal(summaryCalls, 2);
  assert.equal(hooks.state.annotationPropertySummary.min, 0.5);
  assert.equal(hooks.state.annotationPropertySummary.max, 1.0);
  assert.match(elements.legend.innerHTML, /gradient-swatch/);
});

test('annotationStyle skips dense polygon strokes until higher zooms', async () => {
  const { hooks } = createEnvironment(async () => response([]), { withOl: true });

  hooks.state.baseResolutions = [4, 2, 1, 0.5, 0.25];
  hooks.state.annotationMeta = {
    name: 'overlay',
    path: '/tmp/overlay.db',
    vector_format: 'mvt',
    vector_representations: [
      { id: 'overview', min_zoom: 0, max_zoom: 1, geometry_type: 'polygon' },
      { id: 'full', min_zoom: 2, geometry_type: 'mixed' },
    ],
  };

  const feature = {
    getGeometry() {
      return {
        getType() {
          return 'Polygon';
        },
      };
    },
  };

  const lowZoomStyle = hooks.annotationStyle(feature, 1);
  const highZoomStyle = hooks.annotationStyle(feature, 0.25);

  assert.equal(hooks.shouldRenderPolygonStroke(4), false);
  assert.equal(lowZoomStyle.stroke, undefined);
  assert.notEqual(highZoomStyle.stroke, undefined);
  assert.equal(highZoomStyle.stroke.width, 1.4);
});

test('annotationStyle caches fill-only and stroked polygon variants separately', async () => {
  const { hooks } = createEnvironment(async () => response([]), { withOl: true });

  hooks.state.baseResolutions = [4, 2, 1, 0.5, 0.25];
  hooks.state.annotationMeta = {
    name: 'overlay',
    path: '/tmp/overlay.db',
    vector_format: 'mvt',
    vector_representations: [
      { id: 'overview', min_zoom: 0, max_zoom: 1, geometry_type: 'polygon' },
      { id: 'full', min_zoom: 2, geometry_type: 'mixed' },
    ],
  };

  const feature = {
    getGeometry() {
      return {
        getType() {
          return 'Polygon';
        },
      };
    },
  };

  const fillOnlyStyle = hooks.annotationStyle(feature, 1);
  const strokedStyle = hooks.annotationStyle(feature, 0.25);

  assert.notEqual(fillOnlyStyle, strokedStyle);
  assert.equal(hooks.state.annotationStyleCache.size, 2);
});

test('annotationStyle makes overview density cells much gentler until clusters get dense', async () => {
  const { hooks } = createEnvironment(async () => response([]), { withOl: true });

  hooks.state.baseResolutions = [4, 2, 1, 0.5, 0.25];
  hooks.state.annotationMeta = {
    name: 'overlay',
    path: '/tmp/overlay.db',
    vector_format: 'mvt',
    vector_representations: [
      { id: 'overview', min_zoom: 0, max_zoom: 1, geometry_type: 'polygon' },
      { id: 'full', min_zoom: 2, geometry_type: 'mixed' },
    ],
  };

  function makeFeature(count) {
    return {
      get(name) {
        if (name === 'count') {
          return count;
        }
        if (name === 'overview_kind') {
          return 'density';
        }
        return undefined;
      },
      getGeometry() {
        return {
          getType() {
            return 'Polygon';
          },
        };
      },
    };
  }

  const sparseStyle = hooks.annotationStyle(makeFeature(1), 2);
  const denseStyle = hooks.annotationStyle(makeFeature(63), 2);

  const sparseComponents = sparseStyle.fill.color.match(/\d+(?:\.\d+)?/g).map(Number);
  const denseComponents = denseStyle.fill.color.match(/\d+(?:\.\d+)?/g).map(Number);

  assert.notEqual(sparseStyle.fill.color, denseStyle.fill.color);
  assert.ok(sparseComponents[3] < 0.04);
  assert.ok(denseComponents[3] > 0.16);
  assert.ok(denseComponents[3] < 0.18);
  assert.ok(sparseComponents[0] > denseComponents[0]);
  assert.ok(sparseComponents[1] > denseComponents[1]);
  assert.ok(sparseComponents[2] > denseComponents[2]);
  assert.equal(sparseStyle.stroke, undefined);
  assert.equal(denseStyle.stroke, undefined);
});

test('annotationStyle keeps overview geometry more visible than density cells', async () => {
  const { hooks } = createEnvironment(async () => response([]), { withOl: true });

  hooks.state.baseResolutions = [4, 2, 1, 0.5, 0.25];
  hooks.state.annotationMeta = {
    name: 'overlay',
    path: '/tmp/overlay.db',
    vector_format: 'mvt',
    vector_representations: [
      { id: 'overview', min_zoom: 0, max_zoom: 1, geometry_type: 'polygon' },
      { id: 'full', min_zoom: 2, geometry_type: 'mixed' },
    ],
  };

  function makeFeature(kind, count) {
    return {
      get(name) {
        if (name === 'overview_kind') {
          return kind;
        }
        if (name === 'count') {
          return count;
        }
        return undefined;
      },
      getGeometry() {
        return {
          getType() {
            return 'Polygon';
          },
        };
      },
    };
  }

  const densityStyle = hooks.annotationStyle(makeFeature('density', 1), 2);
  const geometryStyle = hooks.annotationStyle(makeFeature('geometry'), 2);

  const densityComponents = densityStyle.fill.color.match(/\d+(?:\.\d+)?/g).map(Number);
  const geometryComponents = geometryStyle.fill.color.match(/\d+(?:\.\d+)?/g).map(Number);

  assert.ok(geometryComponents[3] > densityComponents[3]);
  assert.ok(geometryComponents[3] < 0.2);
  assert.ok(geometryComponents[0] < densityComponents[0]);
  assert.ok(geometryComponents[1] < densityComponents[1]);
  assert.ok(geometryComponents[2] < densityComponents[2]);
  assert.equal(geometryStyle.stroke, undefined);
});

test('createAnnotationLayer refuses unsafe GeoJSON fallback for large overlays', async () => {
  const { hooks, elements } = createEnvironment(async () => response([]), { withOl: true });

  const layer = hooks.createAnnotationLayer(
    {
      name: 'overlay',
      path: '/tmp/overlay.db',
      geojson_policy: {
        allowed: false,
        debug_only: true,
        message: 'Large SQLite-backed overlays stay on the MVT path during normal viewing.',
      },
    },
    new FakeTileSource({}),
    { size: [1000, 1000], mpp: 0.25 },
  );

  assert.equal(
    elements['status-text'].textContent,
    'Large SQLite-backed overlays stay on the MVT path during normal viewing.',
  );
  assert.equal(typeof layer.getSource().config.loader, 'undefined');
});

test('syncLayers resolves before annotation metadata refresh completes', async () => {
  const propertyFetch = deferred();
  let propertyCalls = 0;

  const { hooks } = createEnvironment(async (input) => {
    const url = new URL(String(input), 'http://localhost');
    if (url.pathname === '/tileserver/layers') {
      return response([
        {
          kind: 'slide',
          name: 'slide',
          path: '/tmp/slide.svs',
          size: [1000, 1000],
          mpp: 0.25,
          url: 'http://localhost/slide',
        },
        {
          kind: 'annotation',
          name: 'overlay',
          path: '/tmp/overlay.db',
          vector_format: 'mvt',
          vector_url: 'http://localhost/vector/{z}/{x}/{y}',
          vector_revision: 9,
        },
      ]);
    }
    if (url.pathname === '/tileserver/slide') {
      return response({
        file_path: '/tmp/slide.svs',
        slide_dimensions: [1000, 1000],
        mpp: [0.25, 0.25],
      });
    }
    if (url.pathname === '/tileserver/prop_names/all') {
      propertyCalls += 1;
      const payload = await propertyFetch.promise;
      return response(payload);
    }
    if (url.pathname.includes('/tileserver/prop_summary/')) {
      return response({ kind: 'categorical', values: ['tumour'] });
    }
    throw new Error(`Unexpected fetch: ${String(input)}`);
  }, { withOl: true });

  let syncResolved = false;
  const syncPromise = hooks.syncLayers({ fit: true }).then(() => {
    syncResolved = true;
  });

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(syncResolved, true);
  assert.equal(propertyCalls, 1);

  propertyFetch.resolve(['type']);
  await syncPromise;
});
