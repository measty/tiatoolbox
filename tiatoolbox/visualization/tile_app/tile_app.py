from tiatoolbox.visualization.tileserver import TileServer
from flask_cors import CORS

def run_app():

    app = TileServer(
        title="Testing TileServer",
        layers={
            #"slide": wsi[0],
        },
        state={},
    )
    CORS(app, send_wildcard=True)
    #app.run(host="0.0.0.0", threaded=False)
    app.run(threaded=False)

if __name__ == '__main__':
    run_app()