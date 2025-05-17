function init() {
    var dataElement = document.getElementById('map');
    var layersData = JSON.parse(dataElement.dataset.layers);
    var layers = layersData.map(function (layer) {
        var source = new ol.source.Zoomify({
            url: layer.url,
            size: layer.size,
            crossOrigin: 'anonymous',
            zDirection: -1,
        });
        return new ol.layer.Tile({ title: layer.name, source: source });
    });
    var resolutions = layers[0].getSource().getTileGrid().getResolutions();
    var extent = layers[0].getSource().getTileGrid().getExtent();
    var projection = new ol.proj.Projection({
        code: 'Zoomify',
        units: 'pixels',
        extent: extent,
        metersPerUnit: layersData[0].mpp * 1e-6,
        getPointResolution: function(r){ return r; }
    });
    var view = new ol.View({
        projection: projection,
        resolutions: resolutions,
        constrainOnlyCenter: true,
    });
    var vectorSource = new ol.source.Vector();
    var vectorLayer = new ol.layer.Vector({ source: vectorSource });
    layers.push(vectorLayer);
    var map = new ol.Map({ target: 'map', layers: layers, view: view });
    map.getView().fit(extent);

    setupUI();

    fetchAnnotations(vectorSource, map.getView().calculateExtent());
    map.on('moveend', function() {
        var b = map.getView().calculateExtent();
        fetchAnnotations(vectorSource, b);
    });
}

function fetchAnnotations(source, bounds) {
    var params = new URLSearchParams();
    params.append('bounds', JSON.stringify(bounds));
    params.append('where', JSON.stringify(null));
    fetch('/tileserver/vector_annotations?' + params.toString())
        .then(function(r){ return r.json(); })
        .then(function(data){
            var format = new ol.format.GeoJSON();
            var feats = format.readFeatures(data, {featureProjection: 'Zoomify'});
            source.clear();
            source.addFeatures(feats);
        });
}

document.addEventListener('DOMContentLoaded', init);

function setupUI() {
    var slideSelect = document.getElementById('slideSelect');
    if (slideSelect) {
        slideSelect.addEventListener('change', function() {
            var val = slideSelect.value;
            if (!val) { return; }
            fetch('/tileserver/slide', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'slide_path=' + encodeURIComponent(val)
            }).then(function(){ location.reload(); });
        });
    }

    var overlaySelect = document.getElementById('overlaySelect');
    if (overlaySelect) {
        overlaySelect.addEventListener('change', function() {
            var val = overlaySelect.value;
            if (!val) { return; }
            fetch('/tileserver/overlay', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'overlay_path=' + encodeURIComponent(val)
            }).then(function(){ location.reload(); });
        });
    }
}
