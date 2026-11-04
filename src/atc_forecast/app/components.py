"""Embed the interactive HTML components with their data.

Each component is a self-contained page in web/ that reads `window.PAYLOAD`. The shared
stylesheet and helpers are inlined, the payload is injected as JSON, and the result is
rendered in an iframe, so animation runs entirely in the browser and never waits on a
Streamlit rerun.
"""

import json
import re
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components

from .theme import colors, label_url, theme_name, tile_url

WEB = Path(__file__).parent / "web"
_NAN = re.compile(r"\bNaN\b|-?Infinity\b")


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)


def to_json(payload) -> str:
    """JSON with NaN and infinities as null, which the browser can parse; and with `</`
    escaped, so no string in the payload can close the script tag it is embedded in."""
    text = json.dumps(payload, cls=_Enc, separators=(",", ":"))
    return _NAN.sub("null", text).replace("</", "<\\/")


def render(name: str, payload: dict, height: int):
    theme = theme_name()
    payload = {**payload, "theme": theme, "colors": colors(theme), "tile_url": tile_url(theme),
               "label_url": label_url(theme)}
    page = (WEB / f"{name}.html").read_text()
    page = (page.replace("/*__BASE_CSS__*/", (WEB / "base.css").read_text())
                .replace("/*__BASE_JS__*/", (WEB / "base.js").read_text())
                .replace("/*__PAYLOAD__*/", "window.PAYLOAD = " + to_json(payload) + ";"))
    components.html(page, height=height, scrolling=False)
