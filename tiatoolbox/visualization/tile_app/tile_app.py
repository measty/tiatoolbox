from flask_cors import CORS

from tiatoolbox.visualization.tileserver import TileServer

app = TileServer(
    title="Testing TileServer",
    layers={
        # "slide": wsi[0],
    },
)
CORS(app, send_wildcard=True)
# app.run(host="0.0.0.0", threaded=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=False)
