"""One colour system for every view, light and dark.

Conditions take categorical slots in a FIXED order (belief, point, oracle), which is the
order validated colourblind-safe for all pairs; "no instruction" is the neutral baseline
and wears gray rather than a hue. The recorded truth wears ink. A colour always means the
same condition on every page.
"""

PALETTE = {
    "light": {
        "surface": "#fcfcfb", "surface2": "#f3f2ef", "border": "#e2e1dc",
        "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984", "grid": "#ecebe7",
        "belief": "#2a78d6", "point": "#eb6834", "oracle": "#1baf7a", "none": "#8a8984",
        "truth": "#0b0b0b", "past": "#52514e", "accent": "#6250d6",
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "div_neg": "#e34948", "div_mid": "#f0efec", "div_pos": "#2a78d6",
        "tiles": "Light",
    },
    "dark": {
        "surface": "#1a1a19", "surface2": "#242422", "border": "#383835",
        "text": "#f0efec", "text2": "#c3c2b7", "muted": "#8f8e86", "grid": "#2c2c2a",
        "belief": "#3987e5", "point": "#d95926", "oracle": "#199e70", "none": "#8f8e86",
        "truth": "#f0efec", "past": "#c3c2b7", "accent": "#9085e9",
        "seq": ["#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "div_neg": "#e66767", "div_mid": "#383835", "div_pos": "#3987e5",
        "tiles": "Dark",
    },
}
CONDITION_ORDER = ("belief", "point", "oracle", "none")


def theme_name() -> str:
    """The active Streamlit theme, light unless it says otherwise."""
    try:
        import streamlit as st
        return "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
    except Exception:
        return "light"


def colors(name: str | None = None) -> dict:
    return PALETTE[name or theme_name()]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


_ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_{}_Gray_{}/MapServer/tile/{{z}}/{{y}}/{{x}}"


def tile_url(name: str | None = None) -> str:
    """Esri's gray canvas: a basemap designed to sit underneath data, needing no key."""
    return _ESRI.format(colors(name)["tiles"], "Base")


def label_url(name: str | None = None) -> str:
    """The matching place-name layer, drawn above the data so labels stay readable."""
    return _ESRI.format(colors(name)["tiles"], "Reference")


TILE_ATTRIBUTION = "Esri, HERE, Garmin, © OpenStreetMap contributors"
