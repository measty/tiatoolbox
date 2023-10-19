from flask_cors import CORS

from tiatoolbox.visualization.tileserver import TileServer

app = TileServer(
    title="Testing TileServer",
    layers={},
)
CORS(app, send_wildcard=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True)
