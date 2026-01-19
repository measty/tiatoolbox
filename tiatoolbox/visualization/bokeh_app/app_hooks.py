"""Hooks to be executed upon specific events in bokeh app."""

import os
import sys
from contextlib import suppress

import requests
from bokeh.application.application import SessionContext

PORT = os.environ.get("TIATOOLBOX_TILESERVER_PORT", "5000")


def on_session_destroyed(session_context: SessionContext) -> None:
    """Hook to be executed when a session is destroyed."""
    user = session_context.request.arguments["user"]
    host = os.environ.get("HOST2")
    port = os.environ.get("PORT") or "7300"
    print(host)
    if host is None:
        host = "127.0.0.1"
        sys.exit()
    with suppress(requests.exceptions.ReadTimeout):
        requests.get(
            f"http://{host}:{PORT}/tileserver/reset/{user}",
            timeout=5,
        )
    sys.exit()
