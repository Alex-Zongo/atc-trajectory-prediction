"""Static figures for the forecast page.

A forecast here is four-dimensional (east, north, altitude, and speed along the path),
and no single projection shows all of it honestly. The page therefore splits it along the
lines the problem already has:

  plan view      where the aircraft goes, on the real airspace (airports, runways, navaids)
  axis profiles  one panel per axis a controller commands (heading, altitude, speed),
                 against time, with the commanded target drawn on the panel it commands.
                 This is where a clearance and its execution can be compared directly.
  3-D view       the joint geometry, with ground speed as an optional colour channel
"""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .forecast import LABELS
from .geo import rel_to_latlon
from .theme import CONDITION_ORDER, TILE_ATTRIBUTION, colors, label_url, rgba, tile_url


def _base_layout(c, **kw):
    """The shared layout; keyword arguments override it, with `legend` merged in."""
    legend = {"bgcolor": "rgba(0,0,0,0)", "font": dict(color=c["text2"]), **kw.pop("legend", {})}
    base = dict(paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
                font=dict(family="Inter, system-ui, sans-serif", size=12, color=c["text2"]),
                margin=dict(l=8, r=8, t=8, b=8), hoverlabel=dict(bgcolor=c["surface2"],
                                                                  font_color=c["text"]),
                legend=legend)
    return {**base, **kw}


def _joined(arrs):
    """Many polylines as one trace, separated by None. Far cheaper than a trace each."""
    out = []
    for a in arrs:
        out.extend(list(a) + [None])
    return out


def _zoom_for(lat_span, lon_span, lat0):
    span = max(lon_span * math.cos(math.radians(lat0)), lat_span, 1e-3) * 1.6
    return float(np.clip(math.log2(360.0 / span) - 0.6, 5.0, 14.0))


def plan_map(view, conds, feats, tile_opacity=0.55, n_lines=40, theme=None):
    """The plan view on a basemap. `view` holds past/truth (physical units) and the onset
    lat/lon; `conds` maps condition -> {east_nm, north_nm} arrays [N, H]."""
    c = colors(theme)
    lat0, lon0 = view["lat0"], view["lon0"]
    nm = 1852.0 / 10_000.0                      # nm to the normalised position unit

    def ll(e, n):
        return rel_to_latlon(np.asarray(e) * nm, np.asarray(n) * nm, lat0, lon0)

    fig = go.Figure()
    if len(feats["runways"]):
        lats = _joined([[r.lat1, r.lat2] for r in feats["runways"].itertuples()])
        lons = _joined([[r.lon1, r.lon2] for r in feats["runways"].itertuples()])
        fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines", hoverinfo="skip",
                                    line=dict(width=3, color=c["text2"]), name="runways",
                                    showlegend=False))
    ap = feats["airports"]
    if len(ap):
        big = ap["type"].isin(["large_airport", "medium_airport"])
        lab = ap["iata_code"].fillna(ap["ident"]).where(big, "")
        fig.add_trace(go.Scattermap(
            lat=ap["lat"], lon=ap["lon"], mode="markers+text", text=lab,
            textposition="top right", textfont=dict(size=11, color=c["text2"]),
            marker=dict(size=np.where(big, 9, 5), color=c["text2"], opacity=0.75),
            customdata=np.stack([ap["ident"], ap["name"]], -1),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<extra></extra>",
            name="airports", showlegend=False))
    nv = feats["navaids"]
    if len(nv):
        fig.add_trace(go.Scattermap(
            lat=nv["lat"], lon=nv["lon"], mode="markers+text", text=nv["ident"],
            textposition="bottom center", textfont=dict(size=9, color=c["muted"]),
            marker=dict(size=7, color=c["accent"], opacity=0.6),
            customdata=np.stack([nv["type"], nv["name"]], -1),
            hovertemplate="%{text} %{customdata[0]} · %{customdata[1]}<extra></extra>",
            name="navaids", showlegend=False))

    rng = np.random.default_rng(0)
    for cond in CONDITION_ORDER:
        if cond not in conds:
            continue
        e, n = conds[cond]["east_nm"], conds[cond]["north_nm"]
        pick = rng.choice(len(e), size=min(n_lines, len(e)), replace=False)
        la, lo = ll(e[pick], n[pick])
        fig.add_trace(go.Scattermap(lat=_joined(la), lon=_joined(lo), mode="lines",
                                    line=dict(width=1.2, color=rgba(c[cond], 0.32)),
                                    hoverinfo="skip", legendgroup=cond, showlegend=False))
        mla, mlo = ll(np.median(e, 0), np.median(n, 0))
        fig.add_trace(go.Scattermap(lat=mla, lon=mlo, mode="lines", name=LABELS[cond],
                                    line=dict(width=3, color=c[cond]), legendgroup=cond,
                                    hoverinfo="skip"))
    pla, plo = ll(view["past"]["east_nm"], view["past"]["north_nm"])
    fig.add_trace(go.Scattermap(lat=pla, lon=plo, mode="lines+markers", name="observed past",
                                line=dict(width=2, color=c["past"]),
                                marker=dict(size=4, color=c["past"]), hoverinfo="skip"))
    m = view["truth"]["mask"]
    tla, tlo = ll(view["truth"]["east_nm"][m], view["truth"]["north_nm"][m])
    fig.add_trace(go.Scattermap(lat=tla, lon=tlo, mode="lines", name="recorded future",
                                line=dict(width=3, color=c["truth"]), hoverinfo="skip"))
    fig.add_trace(go.Scattermap(lat=[lat0], lon=[lon0], mode="markers", name="clearance onset",
                                marker=dict(size=12, color=c["truth"]),
                                hovertemplate="clearance onset<extra></extra>"))

    allla = np.concatenate([pla, tla, *[ll(d["east_nm"], d["north_nm"])[0].ravel()
                                        for d in conds.values()]])
    alllo = np.concatenate([plo, tlo, *[ll(d["east_nm"], d["north_nm"])[1].ravel()
                                        for d in conds.values()]])
    zoom = _zoom_for(float(np.ptp(allla)), float(np.ptp(alllo)), lat0)
    fig.update_layout(**_base_layout(c, height=520, legend=dict(
        orientation="h", y=1.0, x=0.0, yanchor="bottom", bgcolor=rgba(c["surface"], 0.85))),
        map=dict(
        style="white-bg", zoom=zoom,
        center=dict(lat=float((allla.min() + allla.max()) / 2),
                    lon=float((alllo.min() + alllo.max()) / 2)),
        layers=[dict(sourcetype="raster", source=[tile_url(theme)], below="traces",
                     sourceattribution=TILE_ATTRIBUTION, opacity=tile_opacity),
                dict(sourcetype="raster", source=[label_url(theme)],
                     opacity=min(1.0, tile_opacity + 0.25))]))
    return fig


def _unwrap_to(h, ref):
    """Headings unwrapped around a reference, so a fan crossing north stays one fan."""
    return ((np.asarray(h) - ref + 180.0) % 360.0) - 180.0 + ref


def axis_profiles(view, conds, targets, samples_targets, show_lines=False, theme=None):
    """Heading, altitude and ground speed against time, one panel per commanded axis.

    `targets` is the issued clearance per axis (None where no clause); `samples_targets`
    maps axis -> (belief draws, MAP value) from Stage 1, drawn as a rug past the horizon so
    the commanded value and the flown value can be read on the same scale.
    """
    c = colors(theme)
    past, truth = view["past"], view["truth"]
    ref = float(past["hdg_deg"][-1])
    rows = [("hdg_deg", "Heading (°)", "lateral"), ("alt_ft", "Altitude (ft)", "vertical"),
            ("gs_kt", "Ground speed (kt)", "speed")]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[r[1] for r in rows])
    H = truth["mask"].shape[0]
    t = np.arange(H, dtype=float)
    for ri, (key, _, axis) in enumerate(rows, start=1):
        tf = (lambda a: _unwrap_to(a, ref)) if key == "hdg_deg" else (lambda a: a)
        for cond in CONDITION_ORDER:
            if cond not in conds:
                continue
            a = tf(conds[cond][key])
            q10, q25, q50, q75, q90 = np.percentile(a, [10, 25, 50, 75, 90], axis=0)
            for lo, hi, al in ((q10, q90, 0.12), (q25, q75, 0.22)):
                fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]],
                                         fill="toself", fillcolor=rgba(c[cond], al),
                                         line=dict(width=0), hoverinfo="skip",
                                         legendgroup=cond, showlegend=False), ri, 1)
            if show_lines:
                sel = a[:: max(1, len(a) // 24)]
                fig.add_trace(go.Scatter(x=_joined([t] * len(sel)), y=_joined(sel), mode="lines",
                                         line=dict(width=0.8, color=rgba(c[cond], 0.35)),
                                         hoverinfo="skip", legendgroup=cond, showlegend=False),
                              ri, 1)
            fig.add_trace(go.Scatter(x=t, y=q50, mode="lines", line=dict(width=2, color=c[cond]),
                                     name=LABELS[cond], legendgroup=cond, showlegend=ri == 1,
                                     hovertemplate=LABELS[cond] + " median %{y:.0f}<extra></extra>"),
                          ri, 1)
        fig.add_trace(go.Scatter(x=past["t_s"], y=tf(past[key]), mode="lines+markers",
                                 line=dict(width=2, color=c["past"], dash="dot"),
                                 marker=dict(size=4), name="observed past", showlegend=ri == 1,
                                 legendgroup="past", hovertemplate="past %{y:.0f}<extra></extra>"),
                      ri, 1)
        m = truth["mask"]
        y = np.where(m, tf(truth[key]), np.nan)
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(width=2.5, color=c["truth"]),
                                 name="recorded future", showlegend=ri == 1, legendgroup="truth",
                                 hovertemplate="recorded %{y:.0f}<extra></extra>"), ri, 1)
        tgt = targets.get(axis)
        if tgt is not None:
            v = tf(np.array([tgt]))[0]
            fig.add_trace(go.Scatter(x=[0, H + 4], y=[v, v], mode="lines",
                                     line=dict(width=1.5, color=c["oracle"], dash="dash"),
                                     name="issued target", showlegend=ri == 1,
                                     legendgroup="issued",
                                     hovertemplate=f"issued {tgt:.0f}<extra></extra>"), ri, 1)
        draws, map_v = samples_targets.get(axis, (None, None))
        if draws is not None and len(draws):
            d = tf(np.asarray(draws))
            fig.add_trace(go.Scatter(x=np.full(len(d), H + 3.0), y=d, mode="markers",
                                     marker=dict(symbol="line-ew-open", size=14,
                                                 color=c["belief"], line=dict(width=2)),
                                     name="Stage-1 draws", showlegend=ri == 1,
                                     legendgroup="draws",
                                     hovertemplate="Stage-1 draw %{y:.0f}<extra></extra>"), ri, 1)
        if map_v is not None:
            fig.add_trace(go.Scatter(x=[H + 3.0], y=tf(np.array([map_v])), mode="markers",
                                     marker=dict(symbol="diamond", size=10, color=c["point"],
                                                 line=dict(width=2, color=c["surface"])),
                                     name="Stage-1 MAP", showlegend=ri == 1, legendgroup="map",
                                     hovertemplate="Stage-1 MAP %{y:.0f}<extra></extra>"), ri, 1)
        fig.add_vline(x=0, line=dict(width=1, color=c["muted"], dash="dot"), row=ri, col=1)
    fig.update_layout(**_base_layout(c, height=640, margin=dict(l=8, r=8, t=92, b=8),
                                     legend=dict(orientation="h", y=1.07, x=0, yanchor="bottom")))
    fig.update_xaxes(gridcolor=c["grid"], zeroline=False, linecolor=c["border"],
                     range=[min(past["t_s"].min(), -5) - 1, H + 6])
    fig.update_xaxes(title_text="seconds from clearance onset", row=3, col=1)
    fig.update_yaxes(gridcolor=c["grid"], zeroline=False, linecolor=c["border"])
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=c["text"]), x=0, xanchor="left")
    return fig


def view_3d(view, conds, color_by="condition", z_exag=1.0, n_lines=30, theme=None):
    """East, north, altitude. With `color_by="speed"` every path is coloured by its own
    ground speed, which is the fourth dimension a clearance can command."""
    c = colors(theme)
    fig = go.Figure()
    rng = np.random.default_rng(0)
    past, truth = view["past"], view["truth"]
    m = truth["mask"]
    all_gs = [truth["gs_kt"][m], past["gs_kt"]] + [d["gs_kt"].ravel() for d in conds.values()]
    gmin, gmax = (float(np.percentile(np.concatenate(all_gs), p)) for p in (2, 98))
    scale = [[i / (len(c["seq"]) - 1), col] for i, col in enumerate(c["seq"])]
    floor = float(min(past["alt_ft"].min(), truth["alt_ft"][m].min(),
                      *[d["alt_ft"].min() for d in conds.values()]))

    for cond in CONDITION_ORDER:
        if cond not in conds:
            continue
        d = conds[cond]
        pick = rng.choice(len(d["east_nm"]), size=min(n_lines, len(d["east_nm"])), replace=False)
        # the separator points have no coordinates, so their colour is never drawn
        gs_line = [gmin if v is None else v for v in _joined(d["gs_kt"][pick])]
        line = (dict(width=2, color=gs_line, colorscale=scale,
                     cmin=gmin, cmax=gmax)
                if color_by == "speed" else dict(width=2, color=rgba(c[cond], 0.4)))
        fig.add_trace(go.Scatter3d(x=_joined(d["east_nm"][pick]), y=_joined(d["north_nm"][pick]),
                                   z=_joined(d["alt_ft"][pick]), mode="lines", line=line,
                                   name=LABELS[cond], hoverinfo="skip"))
        med = [np.median(d[k], 0) for k in ("east_nm", "north_nm")]
        fig.add_trace(go.Scatter3d(x=med[0], y=med[1], z=np.full(len(med[0]), floor),
                                   mode="lines", line=dict(width=2, color=rgba(c[cond], 0.5)),
                                   showlegend=False, hoverinfo="skip"))
    tl = (dict(width=7, color=truth["gs_kt"][m], colorscale=scale, cmin=gmin, cmax=gmax,
               colorbar=dict(title="GS kt", thickness=10, len=0.6, x=0.98))
          if color_by == "speed" else dict(width=7, color=c["truth"]))
    fig.add_trace(go.Scatter3d(x=truth["east_nm"][m], y=truth["north_nm"][m], z=truth["alt_ft"][m],
                               mode="lines", line=tl, name="recorded future",
                               customdata=np.stack([truth["gs_kt"][m], truth["t_s"][m]], -1),
                               hovertemplate="t+%{customdata[1]:.0f}s · %{z:.0f} ft · "
                                             "%{customdata[0]:.0f} kt<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=truth["east_nm"][m], y=truth["north_nm"][m],
                               z=np.full(int(m.sum()), floor), mode="lines",
                               line=dict(width=3, color=rgba(c["truth"], 0.35)),
                               name="ground track (shadow)", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=past["east_nm"], y=past["north_nm"], z=past["alt_ft"],
                               mode="lines+markers", name="observed past",
                               line=dict(width=4, color=c["past"]), marker=dict(size=2),
                               hoverinfo="skip"))
    xs = np.concatenate([past["east_nm"], truth["east_nm"][m]])
    ys = np.concatenate([past["north_nm"], truth["north_nm"][m]])
    span = max(float(np.ptp(xs)), float(np.ptp(ys)), 0.5)
    ax = dict(gridcolor=c["grid"], backgroundcolor=c["surface"], showbackground=True,
              color=c["text2"], zerolinecolor=c["border"])
    fig.update_layout(**_base_layout(c, height=560, legend=dict(
        orientation="h", y=1.0, x=0, yanchor="bottom")), scene=dict(
        xaxis=dict(title="east (nm)", **ax), yaxis=dict(title="north (nm)", **ax),
        zaxis=dict(title="altitude (ft)", **ax), aspectmode="manual",
        aspectratio=dict(x=max(float(np.ptp(xs)) / span, 0.3),
                         y=max(float(np.ptp(ys)) / span, 0.3), z=0.45 * z_exag),
        camera=dict(eye=dict(x=-1.3, y=-1.5, z=0.9))))
    return fig


def relative_plan(view, conds, n_lines=40, theme=None):
    """Plan view in nm about the onset, for when no raw ADS-B is available to anchor a map."""
    c = colors(theme)
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for cond in CONDITION_ORDER:
        if cond not in conds:
            continue
        d = conds[cond]
        pick = rng.choice(len(d["east_nm"]), size=min(n_lines, len(d["east_nm"])), replace=False)
        fig.add_trace(go.Scatter(x=_joined(d["east_nm"][pick]), y=_joined(d["north_nm"][pick]),
                                 mode="lines", line=dict(width=1, color=rgba(c[cond], 0.3)),
                                 name=LABELS[cond], hoverinfo="skip"))
    m = view["truth"]["mask"]
    fig.add_trace(go.Scatter(x=view["past"]["east_nm"], y=view["past"]["north_nm"],
                             mode="lines+markers", line=dict(color=c["past"], width=2),
                             marker=dict(size=4), name="observed past"))
    fig.add_trace(go.Scatter(x=view["truth"]["east_nm"][m], y=view["truth"]["north_nm"][m],
                             mode="lines", line=dict(color=c["truth"], width=3),
                             name="recorded future"))
    fig.update_layout(**_base_layout(c, height=520))
    fig.update_xaxes(title_text="east (nm)", gridcolor=c["grid"], zeroline=False)
    fig.update_yaxes(title_text="north (nm)", gridcolor=c["grid"], zeroline=False,
                     scaleanchor="x", scaleratio=1)
    return fig
