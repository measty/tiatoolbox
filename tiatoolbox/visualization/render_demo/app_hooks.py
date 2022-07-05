import sys
import requests

def on_session_destroyed(session_context):
    # If present, this function executes when the server closes a session.
    fname = r'/tiatoolbox/app_data/slides/TCGA-SC-A6LN-01Z-00-DX1.svs'
    fname = fname='-*-'.join(fname.split('/'))
    resp = requests.get(f'http://127.0.0.1:5000/changeslide/slide/{fname}')