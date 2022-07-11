import sys
import urllib
import requests
from pathlib import PureWindowsPath

def make_safe_name(name):
    return urllib.parse.quote(str(PureWindowsPath(name)), safe="")

def on_session_destroyed(session_context):
    # If present, this function executes when the server closes session.
    fname = r"/app_data/slides/TCGA-SC-A6LN-01Z-00-DX1.svs"
    fname = make_safe_name(fname)
    resp = requests.get(f"http://127.0.0.1:5000/changeslide/slide/{fname}")
