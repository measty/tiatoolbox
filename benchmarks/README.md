# Benchmarks

## Viewer Responsiveness Backend Harness

`viewer_responsiveness.py` measures the TileServer backend requests used by the
OpenLayers viewer for slide and annotation-overlay loading. It emits structured
JSON covering session setup, slide and overlay load requests, annotation
metadata refreshes, representative cold and warm MVT tile timings, and a
backend-only `first_visible_overlay_proxy`.

Run it from the repository root with one or more named cases:

```bash
python benchmarks/viewer_responsiveness.py \
  --case tcga_large \
    /media/mark-eastwood/Work/TTB_vis_test/slides/TCGA-SC-A6LN-01Z-00-DX1.svs \
    /media/mark-eastwood/Work/TTB_vis_test/overlays/TCGA-SC-A6LN-01Z-00-DX1_saved_anns.db \
  --warm-runs 3 \
  --sample-points 3 \
  --json-out /tmp/viewer-bench.json
```

Repeat `--case NAME SLIDE OVERLAY` to benchmark multiple datasets in one JSON
report. If `--tile` is omitted, the harness samples evenly spaced tiles across
the advertised annotation representations and benchmarks a dense tile for each.

The harness is backend-first. It does not launch a browser, so frontend vector
decode, style, render, and paint timing are not captured.
