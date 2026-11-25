{
"use strict";
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (x, d = 0) => (x == null || !isFinite(x)) ? "–" : (Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : Number(x)).toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d});
const COND = {
  composed: {label: "Composed over the belief", color: "#1240e0"},
  most_probable: {label: "Most probable instruction", color: "#7c3aed"},
  label: {label: "Semantic label (reference)", color: "#0f7a4a", dash: "5 3"},
  adsb_only: {label: "ADS-B only", color: "#8a93a8"},
  transcript: {label: "Transcript-conditioned flow", color: "#0e7490"},
  kinematic: {label: "Kinematic rule", color: "#ea580c", dash: "6 3"},
};
const CONF_STEPS = [1, 0.6, 0.3, 0.1];
const MAX_DRAW = 36;

const S = {
  variants: null, data: null,
  sel: {sbm: "final", ctp: "flow_pca", sampler: "euler12", K: 12, S: 6, seed: 0},
  live: {scene: 0, t: -30, playing: false, results: {}, cur: null, last: 0, extra: []},
  tr: {case: 0, window: null, words: [], unknown: [], conds: ["composed", "most_probable", "adsb_only", "label"], res: null},
  cmp: {a: null, b: null, res: {}},
};

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("on");
  clearTimeout(toast.h);
  toast.h = setTimeout(() => t.classList.remove("on"), 3500);
}

const STATIC = document.body.classList.contains("static");
const DEFAULTS_STATIC = {sampler: "euler12", K: 12, S: 6, seed: 0};
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`not available in the static version (${url})`);
  return r.json();
}
async function staticApi(path, body) {
  /* the static build: precomputed files instead of the model service */
  if (path === "/api/variants") return getJSON("data/variants.json");
  if (path === "/api/cases") return getJSON("data/cases.json");
  for (const [k, v] of Object.entries(DEFAULTS_STATIC)) if (body[k] !== undefined && body[k] !== v) throw new Error("The static version has only the default sampler and sample counts.");
  if (path === "/api/predict") {
    const ci = S.tr.case;
    const [r, ref] = await Promise.all([getJSON(`data/try/${ci}/${body.sbm}__${body.ctp}.json`), getJSON(`data/try/${ci}/ref.json`)]);
    const all = {...r.predictions, ...ref.predictions}, want = body.conditions || Object.keys(all);
    return {...r, predictions: Object.fromEntries(want.filter((k) => all[k]).map((k) => [k, all[k]]))};
  }
  if (path === "/api/scene") {
    if (body._tx === undefined || body._tx >= scene().transmissions.length) throw new Error("Typed transmissions need the model service: run the demo locally.");
    if (body.sbm !== "final" || body.ctp !== "flow_pca") toast("The live sector in the static version uses the main models (final SBM, PCA flow).");
    return getJSON(`data/scene/${body._scene}/${body._tx}_${body.routing ? 1 : 0}.json`);
  }
  throw new Error(`unknown request ${path}`);
}
async function api(path, body) {
  if (STATIC) return staticApi(path, body || {});
  const r = await fetch(path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {});
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* keep status text */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

function sel() {
  return {sbm: S.sel.sbm, ctp: S.sel.ctp, sampler: S.sel.sampler, K: S.sel.K, S: S.sel.S, seed: S.sel.seed};
}

/* ------------------------------------------------------------------ drawing helpers */
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function size(svg) { return [svg.clientWidth || 600, svg.clientHeight || 360]; }
function niceStep(span, n = 6) {
  const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (raw <= m * p) return m * p;
  return 10 * p;
}
const ZOOM = {}, REDRAW = {};
function zkey(svg) { return svg.dataset.zoom || svg.id; }
function frame(svg, bounds, equal, pad = {l: 46, r: 14, t: 14, b: 34}) {
  const [W, H] = size(svg);
  let [x0, x1, y0, y1] = ZOOM[zkey(svg)] || bounds;
  if (x1 - x0 < 1e-9) { x0 -= 1; x1 += 1; }
  if (y1 - y0 < 1e-9) { y0 -= 1; y1 += 1; }
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  let sx = iw / (x1 - x0), sy = ih / (y1 - y0);
  if (equal) {
    const s = Math.min(sx, sy);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    x0 = cx - iw / s / 2; x1 = cx + iw / s / 2; y0 = cy - ih / s / 2; y1 = cy + ih / s / 2;
    sx = sy = s;
  }
  const f = {W, H, pad, x0, x1, y0, y1, X: (x) => pad.l + (x - x0) * sx, Y: (y) => pad.t + (y1 - y) * sy,
    ix: (px) => x0 + (px - pad.l) / sx, iy: (py) => y1 - (py - pad.t) / sy};
  svg._f = f;
  return f;
}
/* scroll to zoom about the pointer, drag to pan, double-click to reset; plots that share
   data-zoom move together (the two sides of the comparison) */
let zoomFrame = 0;
function redraw(key) {
  cancelAnimationFrame(zoomFrame);
  zoomFrame = requestAnimationFrame(() => { const fn = REDRAW[key]; if (fn) fn(); });
}
function enableZoom(svg, fn) {
  const key = zkey(svg);
  REDRAW[key] = fn;
  svg.style.touchAction = "none";
  svg.style.cursor = "grab";
  svg.addEventListener("wheel", (e) => {
    const f = svg._f;
    if (!f) return;
    e.preventDefault();
    const r = svg.getBoundingClientRect(), mx = f.ix(e.clientX - r.left), my = f.iy(e.clientY - r.top);
    const k = Math.exp(Math.max(-60, Math.min(60, e.deltaY)) * 0.004);
    ZOOM[key] = [mx + (f.x0 - mx) * k, mx + (f.x1 - mx) * k, my + (f.y0 - my) * k, my + (f.y1 - my) * k];
    redraw(key);
  }, {passive: false});
  let drag = null;
  svg.addEventListener("pointerdown", (e) => {
    const f = svg._f;
    if (!f || e.button !== 0) return;
    drag = {x: e.clientX, y: e.clientY, f, b: [f.x0, f.x1, f.y0, f.y1]};
    svg.setPointerCapture(e.pointerId);
    svg.style.cursor = "grabbing";
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = drag.f.ix(e.clientX) - drag.f.ix(drag.x), dy = drag.f.iy(e.clientY) - drag.f.iy(drag.y);
    ZOOM[key] = [drag.b[0] - dx, drag.b[1] - dx, drag.b[2] - dy, drag.b[3] - dy];
    redraw(key);
  });
  const end = () => { drag = null; svg.style.cursor = "grab"; };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("dblclick", () => { delete ZOOM[key]; redraw(key); });
}
function resetZoom(...keys) { for (const k of keys) delete ZOOM[k]; }
function axes(svg, f, xl, yl, narrow) {
  const g = el("g", {}, svg);
  const sx = niceStep(f.x1 - f.x0, narrow ? 4 : 7), sy = niceStep(f.y1 - f.y0, narrow ? 4 : 6);
  for (let x = Math.ceil(f.x0 / sx) * sx; x <= f.x1; x += sx) {
    el("line", {x1: f.X(x), x2: f.X(x), y1: f.pad.t, y2: f.H - f.pad.b, stroke: "var(--line)", "stroke-width": 1}, g);
    el("text", {x: f.X(x), y: f.H - f.pad.b + 16, "text-anchor": "middle"}, g).textContent = fmt(x, sx < 1 ? 1 : 0);
  }
  for (let y = Math.ceil(f.y0 / sy) * sy; y <= f.y1; y += sy) {
    el("line", {x1: f.pad.l, x2: f.W - f.pad.r, y1: f.Y(y), y2: f.Y(y), stroke: "var(--line)", "stroke-width": 1}, g);
    el("text", {x: f.pad.l - 6, y: f.Y(y) + 4, "text-anchor": "end"}, g).textContent = fmt(y, sy < 1 ? 1 : 0);
  }
  el("text", {x: f.W - f.pad.r, y: f.H - 6, "text-anchor": "end"}, g).textContent = xl;
  el("text", {x: 4, y: f.pad.t - 2 > 10 ? f.pad.t - 2 : 12}, g).textContent = yl;
  return g;
}
function line(g, f, pts, attrs) {
  if (pts.length < 2) return null;
  return el("polyline", {points: pts.map((p) => `${f.X(p[0]).toFixed(1)},${f.Y(p[1]).toFixed(1)}`).join(" "), fill: "none", ...attrs}, g);
}
function pick(n, max) {
  if (n <= max) return [...Array(n).keys()];
  const out = [];
  for (let i = 0; i < max; i++) out.push(Math.floor(i * n / max));
  return out;
}
function bars(id, rows, alt) {
  $(id).innerHTML = rows.length ? rows.map((r) => `<div class="bar-row"><span class="lab" title="${esc(r.title || r.label)}">${esc(r.label)}</span><span class="trk"><span class="fill ${alt && r.alt ? "alt" : ""}" style="width:${Math.max(1, 100 * Math.min(1, r.p))}%"></span></span><span class="val">${r.text != null ? esc(r.text) : fmt(100 * r.p, r.p < 0.1 ? 1 : 0) + "%"}</span></div>`).join("") : '<p class="note">None.</p>';
}

/* ------------------------------------------------------------------ probability display */
const LEADS = [{t: 20, i: 10}, {t: 40, i: 20}, {t: 60, i: 30}];
function displayMode() { return $("sel-display").value; }
function kde(pts) {
  /* sample-point kernel density, isotropic Gaussian kernels whose width follows the local
     spacing (distance to the 8th neighbour), so separate branches keep separate modes */
  const n = pts.length, k = Math.min(n - 1, 8), floor = 0.03;
  const h = pts.map((p) => {
    const d = pts.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])).sort((a, b) => a - b);
    return Math.max(floor, 0.7 * (d[k] || floor));
  });
  const dens = (x, y) => {
    let s = 0;
    for (let j = 0; j < n; j++) { const u = (x - pts[j][0]) / h[j], v = (y - pts[j][1]) / h[j]; s += Math.exp(-0.5 * (u * u + v * v)) / (h[j] * h[j]); }
    return s / (n * 2 * Math.PI);
  };
  const at = pts.map((p) => dens(p[0], p[1])).sort((a, b) => b - a);
  const thr = (lv) => at[Math.max(0, Math.ceil(lv * n) - 1)];
  return {dens, thr, hmax: Math.max(...h)};
}
function contourSegs(G, xs, ys, t) {
  /* marching squares on grid G[j][i] (y index j, x index i), level t -> segments */
  const segs = [];
  const cross = (x0, y0, v0, x1, y1, v1) => { const f = (t - v0) / (v1 - v0); return [x0 + f * (x1 - x0), y0 + f * (y1 - y0)]; };
  for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
    const c = [[xs[i], ys[j], G[j][i]], [xs[i + 1], ys[j], G[j][i + 1]], [xs[i + 1], ys[j + 1], G[j + 1][i + 1]], [xs[i], ys[j + 1], G[j + 1][i]]];
    const pts = [];
    for (let e = 0; e < 4; e++) { const a = c[e], b = c[(e + 1) % 4]; if ((a[2] >= t) !== (b[2] >= t)) pts.push(cross(a[0], a[1], a[2], b[0], b[1], b[2])); }
    if (pts.length === 2) segs.push(pts);
    else if (pts.length === 4) segs.push([pts[0], pts[1]], [pts[2], pts[3]]);
  }
  return segs;
}
const REGIONS = new WeakMap();
function regionGrid(pts) {
  const K = kde(pts), N = 64, pad = 3 * K.hmax;
  const xs0 = Math.min(...pts.map((p) => p[0])) - pad, xs1 = Math.max(...pts.map((p) => p[0])) + pad;
  const ys0 = Math.min(...pts.map((p) => p[1])) - pad, ys1 = Math.max(...pts.map((p) => p[1])) + pad;
  const xs = [...Array(N)].map((_, i) => xs0 + (xs1 - xs0) * i / (N - 1)), ys = [...Array(N)].map((_, j) => ys0 + (ys1 - ys0) * j / (N - 1));
  return {K, N, xs, ys, G: ys.map((y) => xs.map((x) => K.dens(x, y))), t50: K.thr(0.5), t90: K.thr(0.9),
    dx: (xs1 - xs0) / (N - 1), dy: (ys1 - ys0) / (N - 1)};
}
function drawRegions(g, f, pts, color, truth, cacheObj, lead) {
  /* 50 % and 90 % highest-density regions of pts [[x, y]]; returns [in50, in90] for truth.
     The density grid is computed once per prediction and lead time, and reused when zooming. */
  let m = cacheObj ? REGIONS.get(cacheObj) : null;
  if (cacheObj && !m) { m = new Map(); REGIONS.set(cacheObj, m); }
  const R = (m && m.get(lead)) || regionGrid(pts);
  if (m) m.set(lead, R);
  const {N, xs, ys, G, t50, t90, dx, dy, K} = R;
  for (const [t, op, lv] of [[t90, 0.13, 90], [t50, 0.2, 50]]) {
    const cells = el("g", {opacity: op}, g);
    el("title", {}, cells).textContent = `${lv}% region at +${lead} s`;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) if (G[j][i] >= t) {
      el("rect", {x: f.X(xs[i] - dx / 2), y: f.Y(ys[j] + dy / 2), width: Math.abs(f.X(xs[i] + dx / 2) - f.X(xs[i] - dx / 2)) + 0.6, height: Math.abs(f.Y(ys[j] - dy / 2) - f.Y(ys[j] + dy / 2)) + 0.6, fill: color, "shape-rendering": "crispEdges"}, cells);
    }
  }
  for (const [t, dash] of [[t90, "3 3"], [t50, null]]) {
    const d = contourSegs(G, xs, ys, t).map((q) => `M${f.X(q[0][0]).toFixed(1)},${f.Y(q[0][1]).toFixed(1)}L${f.X(q[1][0]).toFixed(1)},${f.Y(q[1][1]).toFixed(1)}`).join("");
    if (d) el("path", {d, fill: "none", stroke: color, "stroke-width": 1.1, ...(dash ? {"stroke-dasharray": dash} : {})}, g);
  }
  return truth ? [K.dens(truth[0], truth[1]) >= t50, K.dens(truth[0], truth[1]) >= t90] : null;
}
function branchBadges(g, f, ends, names, color, origin = [0, 0]) {
  /* numbered badges at the ends of the main instruction branches; the names and shares go
     into a legend outside the plot. Returns [{n, name, share}]. */
  const by = {};
  names.forEach((nm, j) => { (by[nm] = by[nm] || []).push(ends[j]); });
  const total = names.length;
  const labs = Object.entries(by).map(([name, e]) => ({name, share: e.length / total, x: e.reduce((a2, p) => a2 + p[0], 0) / e.length, y: e.reduce((a2, p) => a2 + p[1], 0) / e.length}))
    .filter((q) => q.share >= 0.08).sort((p1, p2) => p2.share - p1.share).slice(0, 4);
  const ox = f.X(origin[0]), oy = f.Y(origin[1]), placed = [];
  labs.forEach((q, i) => {
    const ex = f.X(q.x), ey = f.Y(q.y), d = Math.hypot(ex - ox, ey - oy) || 1;
    let bx = ex + 14 * (ex - ox) / d, by2 = ey + 14 * (ey - oy) / d;
    for (const p of placed) if (Math.hypot(bx - p[0], by2 - p[1]) < 18) by2 = p[1] + 18;
    placed.push([bx, by2]);
    el("line", {x1: ex, y1: ey, x2: bx, y2: by2, stroke: color, "stroke-width": 1}, g);
    el("circle", {cx: bx, cy: by2, r: 8.5, fill: color, stroke: "var(--bg)", "stroke-width": 1.5}, g);
    el("text", {x: bx, y: by2 + 3.8, "text-anchor": "middle", style: "fill:#fff;stroke:none;font-size:11px;font-weight:700"}, g).textContent = String(i + 1);
    q.n = i + 1;
  });
  return labs;
}
function badgeLegend(id, labs, color, who) {
  const elx = $(id);
  if (!elx) return;
  elx.innerHTML = labs && labs.length ? (who ? `<span class="who">${esc(who)}</span>` : "") + labs.map((q) => `<span class="br"><span class="badge" style="background:${color}">${q.n}</span>${esc(q.name)} · ${Math.round(100 * q.share)}%</span>`).join("") : "";
}
function covText(cov) {
  if (!cov) return "";
  const m = (k) => cov.map((c) => (c[k] ? "✓" : "✗")).join(" ");
  return `Flown position inside the predicted region at +20, +40 and +60 s. 50%: ${m(0)} · 90%: ${m(1)}`;
}
function quantile(sorted, q) { const i = q * (sorted.length - 1), a = Math.floor(i), b = Math.ceil(i); return sorted[a] + (sorted[b] - sorted[a]) * (i - a); }

/* ------------------------------------------------------------------ model selectors */
function setupModels() {
  const v = S.variants;
  const fill = (id, entries, val) => {
    $(id).innerHTML = entries.map(([k, e]) => `<option value="${esc(k)}">${esc(e.label || e)}</option>`).join("");
    $(id).value = val;
  };
  try { Object.assign(S.sel, JSON.parse(localStorage.getItem("atc-demo-sel") || "{}")); } catch (e) { /* no storage */ }
  if (STATIC) Object.assign(S.sel, DEFAULTS_STATIC);
  if (!v.sbm[S.sel.sbm]) S.sel.sbm = v.defaults.sbm;
  if (!v.ctp[S.sel.ctp]) S.sel.ctp = v.defaults.ctp;
  if (!v.samplers[S.sel.sampler]) S.sel.sampler = v.defaults.sampler;
  fill("sel-sbm", Object.entries(v.sbm), S.sel.sbm);
  fill("sel-ctp", Object.entries(v.ctp), S.sel.ctp);
  fill("sel-sampler", Object.entries(v.samplers), S.sel.sampler);
  $("sel-k").value = S.sel.K; $("sel-s").value = S.sel.S; $("sel-seed").value = S.sel.seed;
  const changed = () => {
    S.sel.sbm = $("sel-sbm").value; S.sel.ctp = $("sel-ctp").value; S.sel.sampler = $("sel-sampler").value;
    S.sel.K = Math.max(1, Math.min(48, +$("sel-k").value || 12));
    S.sel.S = Math.max(1, Math.min(12, +$("sel-s").value || 6));
    S.sel.seed = Math.max(0, Math.min(10000, +$("sel-seed").value || 0));
    try { localStorage.setItem("atc-demo-sel", JSON.stringify(S.sel)); } catch (e) { /* no storage */ }
    note();
    S.live.results = {}; S.live.cur = null; renderLive(); liveSide();
    if (view() === "try") runTry();
  };
  for (const id of ["sel-sbm", "sel-ctp", "sel-sampler", "sel-k", "sel-s", "sel-seed"]) $(id).addEventListener("change", changed);
  try { const d = localStorage.getItem("atc-demo-display"); if (d) $("sel-display").value = d; } catch (e) { /* no storage */ }
  $("sel-display").addEventListener("change", () => {
    try { localStorage.setItem("atc-demo-display", $("sel-display").value); } catch (e) { /* no storage */ }
    const v = view();
    if (v === "live") renderLive();
    if (v === "try") drawTry();
    if (v === "compare") runCmp();
  });
  note();
}
function note() {
  const a = S.variants.sbm[S.sel.sbm], b = S.variants.ctp[S.sel.ctp];
  $("model-note").textContent = `${a.note} (${a.where}) · ${b.note} (${b.where})`;
}

/* ------------------------------------------------------------------ live sector */
function scene() { return S.data.scenes[S.live.scene]; }
function win(i) { return S.data.windows[i]; }
function txs() { return scene().transmissions.concat(S.live.extra); }

function sceneSetup() {
  $("scene-sel").innerHTML = S.data.scenes.map((s, i) => `<option value="${i}">${esc(s.id)} · ${s.aircraft.length} aircraft</option>`).join("");
  const go = (i) => {
    S.live.scene = (i + S.data.scenes.length) % S.data.scenes.length;
    $("scene-sel").value = S.live.scene;
    S.live.results = {}; S.live.cur = null; S.live.extra = [];
    resetZoom("live-map");
    const ts = scene().aircraft.map((a) => a.t);
    S.live.tmin = Math.min(...ts) - 30; S.live.tmax = Math.max(...ts) + 60;
    $("clock-range").min = S.live.tmin; $("clock-range").max = S.live.tmax;
    $("custom-for").innerHTML = '<option value="">not specified</option>' + scene().aircraft.map((a) => `<option value="${a.window}">${esc(win(a.window).callsign)}</option>`).join("");
    renderLive(); liveSide();
    /* open mid-use: the first transmission addressed to an aircraft in the scene, already run */
    const all = txs(), first = all.findIndex((x) => x.recipient != null);
    setClock(first >= 0 ? all[first].t + 20 : S.live.tmin);
    if (first >= 0) process(first);
  };
  enableZoom($("live-map"), renderLive);
  $("scene-sel").addEventListener("change", () => go(+$("scene-sel").value));
  $("scene-prev").addEventListener("click", () => go(S.live.scene - 1));
  $("scene-next").addEventListener("click", () => go(S.live.scene + 1));
  for (const id of ["live-norm", "live-route"]) $(id).addEventListener("change", () => { S.live.results = {}; S.live.cur = null; renderLive(); liveSide(); });
  $("live-fans").addEventListener("change", renderLive);
  $("clock-play").addEventListener("click", () => { S.live.playing = !S.live.playing; $("clock-play").textContent = S.live.playing ? "Pause" : "Play"; if (S.live.playing) { if (S.live.t >= S.live.tmax) setClock(S.live.tmin); S.live.last = performance.now(); requestAnimationFrame(tick); } });
  $("clock-range").addEventListener("input", () => setClock(+$("clock-range").value));
  $("custom-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = $("custom-text").value.trim();
    if (!text) return;
    let t = parseFloat($("custom-t").value);
    if (!isFinite(t)) t = S.live.t;
    t = Math.max(S.live.tmin, Math.min(S.live.tmax, t));
    const f = $("custom-for").value;
    S.live.extra.push({t, recipient: f === "" ? null : +f, custom: true, intended: f !== "",
      message: {words: text.split(/\s+/).slice(0, 60).map((w) => ({w, c: 1})), label: "typed"}});
    $("custom-text").value = "";
    setClock(t);
    process(txs().length - 1);
  });
  go(0);
}

function setClock(t) {
  S.live.t = Math.max(S.live.tmin, Math.min(S.live.tmax, t));
  $("clock-range").value = S.live.t;
  $("clock-t").textContent = `${S.live.t >= 0 ? "+" : ""}${S.live.t.toFixed(0)} s`;
  if (document.activeElement !== $("custom-t")) $("custom-t").value = Math.round(S.live.t);
  drawMovers();
}

function tick(now) {
  if (!S.live.playing) return;
  const dt = (now - S.live.last) / 1000 * +$("clock-speed").value;
  S.live.last = now;
  const prev = S.live.t;
  setClock(prev + dt);
  txs().forEach((tx, i) => { if (!tx.custom && tx.t > prev && tx.t <= S.live.t && !S.live.results[i]) process(i); });
  if (S.live.t >= S.live.tmax) { S.live.playing = false; $("clock-play").textContent = "Play"; return; }
  requestAnimationFrame(tick);
}

async function process(i) {
  const tx = txs()[i];
  S.live.results[i] = "pending";
  liveSide();
  try {
    const res = await api("/api/scene", {...sel(), windows: scene().aircraft.map((a) => a.window), words: tx.message.words,
      mode: $("live-norm").checked ? "normalized" : "independent", routing: $("live-route").checked, _tx: i, _scene: S.live.scene});
    S.live.results[i] = res; S.live.cur = i;
    $("live-lat").textContent = STATIC ? "precomputed" : `last prediction ${res.ms} ms for ${res.aircraft.length} aircraft`;
  } catch (e) {
    delete S.live.results[i];
    toast(e.message);
  }
  renderLive(); liveSide();
}

function acPos(a, u) {
  /* km position of aircraft a at its own time u (s from its prediction time) */
  const h = a.hist_km, w = win(a.window);
  if (u <= 0) {
    const x = (u + 30) / 2, j = Math.max(0, Math.min(h.length - 2, Math.floor(x))), f = Math.max(0, Math.min(1, x - j));
    return [h[j][0] + f * (h[j + 1][0] - h[j][0]), h[j][1] + f * (h[j + 1][1] - h[j][1])];
  }
  const p0 = h[h.length - 1], fl = w.flown_m, x = Math.min(u, 60) / 2, j = Math.min(fl.length - 2, Math.floor(x)), f = Math.min(1, x - j);
  return [p0[0] + (fl[j][0] + f * (fl[j + 1][0] - fl[j][0])) / 1000, p0[1] + (fl[j][1] + f * (fl[j + 1][1] - fl[j][1])) / 1000];
}

function renderLive() {
  const svg = $("live-map");
  if (!S.data || !svg) return;
  clear(svg);
  const sc = scene();
  let xs = [], ys = [];
  for (const a of sc.aircraft) {
    const p0 = a.hist_km[a.hist_km.length - 1];
    for (const p of a.hist_km) { xs.push(p[0]); ys.push(p[1]); }
    for (const p of win(a.window).flown_m) { xs.push(p0[0] + p[0] / 1000); ys.push(p0[1] + p[1] / 1000); }
  }
  const padk = 3;
  const f = frame(svg, [Math.min(...xs) - padk, Math.max(...xs) + padk, Math.min(...ys) - padk, Math.max(...ys) + padk], true);
  S.live.f = f;
  axes(svg, f, "east (km)", "north (km)", f.W < 520);
  const clip = el("clipPath", {id: "mapclip"}, el("defs", {}, svg));
  el("rect", {x: f.pad.l, y: f.pad.t, width: f.W - f.pad.l - f.pad.r, height: f.H - f.pad.t - f.pad.b}, clip);
  const fans = el("g", {"clip-path": "url(#mapclip)"}, svg);
  const res = S.live.results[S.live.cur];
  const mode = displayMode();
  let cov = null, covWho = null, shown = false;
  if (res && res !== "pending") {
    const all = $("live-fans").checked;
    const top = [...res.aircraft].sort((x, y) => y.p_addressed - x.p_addressed)[0];
    for (const r of res.aircraft) {
      const a = sc.aircraft.find((x) => x.window === r.window);
      if (!a) continue;
      const addressed = r.p_addressed >= 0.5;
      if (!addressed && !all) continue;
      const p0 = a.hist_km[a.hist_km.length - 1];
      const km = (p) => [p0[0] + p[0] / 1000, p0[1] + p[1] / 1000];
      const color = addressed ? "#1240e0" : "#8a93a8";
      if (mode !== "prob") {
        for (const k of pick(r.tracks.length, MAX_DRAW)) {
          line(fans, f, r.tracks[k].map(km), {stroke: color, "stroke-opacity": mode === "both" ? (addressed ? 0.2 : 0.15) : (addressed ? 0.35 : 0.25), "stroke-width": 1.2});
        }
      }
      if (mode !== "samples") {
        const flown = win(a.window).flown_m;
        const c = LEADS.map((L) => drawRegions(fans, f, r.tracks.map((t) => km(t[L.i])), color, km(flown[L.i]), r.tracks, L.t));
        if (r === top) { cov = c; covWho = r.callsign; }
      }
      if (r === top && addressed) {
        const labs = branchBadges(fans, f, r.tracks.map((t) => km(t[30])), r.group_of_track.map((k) => r.branch_of_sample[k]), color, p0);
        badgeLegend("live-branches", labs, color, r.callsign);
        shown = true;
      }
    }
  }
  $("live-cov").textContent = cov ? `${covWho}: ${covText(cov)}` : "";
  if (!shown) badgeLegend("live-branches", null);
  const stat = el("g", {}, svg);
  for (const a of sc.aircraft) line(stat, f, a.hist_km.map((p) => [p[0], p[1]]), {stroke: "var(--mut)", "stroke-width": 2, "stroke-opacity": 0.6});
  S.live.movers = el("g", {}, svg);
  drawMovers();
}

function drawMovers() {
  const g = S.live.movers, f = S.live.f;
  if (!g || !f) return;
  while (g.firstChild) g.removeChild(g.firstChild);
  const sc = scene(), t = S.live.t, res = S.live.results[S.live.cur];
  for (const a of sc.aircraft) {
    const u = t - a.t, w = win(a.window);
    if (u > 0) {
      const p0 = a.hist_km[a.hist_km.length - 1], n = Math.min(w.flown_m.length, Math.floor(Math.min(u, 60) / 2) + 1);
      const pts = w.flown_m.slice(0, n).map((p) => [p0[0] + p[0] / 1000, p0[1] + p[1] / 1000]);
      pts.push(acPos(a, u));
      line(g, f, pts, {stroke: "var(--ink)", "stroke-width": 1.6, "stroke-dasharray": "4 3"});
    }
    const live = u >= -30 && u <= 60;
    const [x, y] = acPos(a, Math.max(-30, Math.min(60, u)));
    const r = res && res !== "pending" ? res.aircraft.find((q) => q.window === a.window) : null;
    const hot = r && r.p_addressed >= 0.5;
    el("circle", {cx: f.X(x), cy: f.Y(y), r: hot ? 6 : 4.5, fill: hot ? "#1240e0" : "var(--ink)", "fill-opacity": live ? 1 : 0.35}, g);
    el("text", {x: f.X(x) + 8, y: f.Y(y) - 7, fill: "var(--ink)", "font-size": 12}, g).textContent = w.callsign;
  }
}

function liveSide() {
  if (!S.data) return;
  const sc = scene(), all = txs();
  const order = all.map((tx, i) => [tx, i]).sort((a, b) => a[0].t - b[0].t);
  $("feed").innerHTML = order.map(([tx, i]) => {
    const r = S.live.results[i];
    const words = tx.message.words.map((w) => w.w).join(" ");
    const who = tx.custom ? (tx.recipient == null ? "typed, recipient not specified" : `typed, meant for ${win(tx.recipient).callsign}`)
      : (tx.recipient == null ? "for an aircraft outside the scene" : `for ${win(tx.recipient).callsign}`);
    let out = "";
    if (r === "pending") out = "running…";
    else if (r) {
      const top = [...r.aircraft].sort((a, b) => b.p_addressed - a.p_addressed)[0];
      out = `model: ${top.p_addressed >= 0.5 ? top.callsign + " " + fmt(100 * top.p_addressed) + "%" : "no aircraft above 50%"}`;
    }
    return `<li data-i="${i}" class="${r && r !== "pending" ? "done" : ""} ${S.live.cur === i ? "cur" : ""}"><span class="t">${tx.t >= 0 ? "+" : ""}${fmt(tx.t, 0)} s</span><span class="txt">${esc(words)}</span><span class="res">${esc(who)}${out ? " · " + esc(out) : ""}</span></li>`;
  }).join("");
  for (const li of $("feed").querySelectorAll("li")) li.addEventListener("click", () => {
    const i = +li.dataset.i, tx = all[i];
    setClock(tx.t);
    if (S.live.results[i] && S.live.results[i] !== "pending") { S.live.cur = i; renderLive(); liveSide(); } else process(i);
  });
  const res = S.live.results[S.live.cur];
  if (!res || res === "pending") {
    $("recipients").innerHTML = '<p class="note">Play the scene or click a transmission.</p>';
    $("live-belief").innerHTML = ""; $("live-scores").innerHTML = "";
    return;
  }
  const tx = all[S.live.cur];
  const rows = [...res.aircraft].sort((a, b) => b.p_addressed - a.p_addressed).map((a) => ({label: a.callsign + (tx.recipient === a.window ? "  (addressed)" : ""), p: a.p_addressed}));
  if (res.p_outside != null) rows.push({label: "an aircraft outside the scene" + (tx.recipient == null && !tx.custom ? "  (addressed)" : ""), p: res.p_outside, alt: true});
  else rows.push({label: "each aircraft judged alone: the shares need not sum to 100%", p: 0, text: ""});
  bars("recipients", rows, true);
  const top = [...res.aircraft].sort((a, b) => b.p_addressed - a.p_addressed)[0];
  const K = top.samples.reduce((s, g) => s + g.n, 0);
  bars("live-belief", top.samples.map((g) => ({label: g.text, p: g.n / K, text: `${g.n}/${K}`})));
  $("live-scores").innerHTML = `<div class="scroll"><table><thead><tr><th>Aircraft</th><th>P(addressed)</th><th>ES (m)</th><th>ADE (m)</th><th>Routed</th></tr></thead><tbody>${res.aircraft.map((a) => `<tr><td>${esc(a.callsign)}</td><td>${fmt(100 * a.p_addressed, 1)}%</td><td>${fmt(a.scores.es_h_m)}</td><td>${fmt(a.scores.ade_h_m)}</td><td>${a.routed}</td></tr>`).join("")}</tbody></table></div><p class="legend">Energy score (ES) and average displacement error (ADE) against each aircraft's flown 60 s, horizontal, in metres; lower is better. Routed: samples sent to the ADS-B-only predictor. Each aircraft is predicted from its own window time.</p>`;
}

/* ------------------------------------------------------------------ try a transmission */
function trySetup() {
  $("case-sel").innerHTML = S.data.cases.map((c, i) => `<option value="${i}">${esc(c.title)}</option>`).join("");
  $("case-sel").addEventListener("change", () => loadCase(+$("case-sel").value));
  /* reference predictors whose checkpoint is not installed are not offered */
  for (const k of ["adsb_only", "transcript"]) if (!S.variants.reference[k]) { delete COND[k]; S.tr.conds = S.tr.conds.filter((c) => c !== k); }
  $("conds").innerHTML = Object.entries(COND).map(([k, c]) => `<label><input type="checkbox" value="${k}" ${S.tr.conds.includes(k) ? "checked" : ""}><span class="sw" style="background:${c.color}"></span>${esc(c.label)}</label>`).join("");
  for (const cb of $("conds").querySelectorAll("input")) cb.addEventListener("change", () => { S.tr.conds = [...$("conds").querySelectorAll("input:checked")].map((x) => x.value); runTry(); });
  $("edit-text").addEventListener("click", () => {
    const ta = $("text-area");
    if (ta.classList.contains("hidden")) { ta.value = S.tr.words.map((w) => w.w).join(" "); ta.classList.remove("hidden"); $("edit-text").textContent = "Apply text"; }
    else { S.tr.words = ta.value.trim().split(/\s+/).filter(Boolean).slice(0, 60).map((w) => ({w, c: 1})); ta.classList.add("hidden"); $("edit-text").textContent = "Edit as text"; editor(); runTry(); }
  });
  $("reset-text").addEventListener("click", () => loadCase(S.tr.case));
  enableZoom($("try-plan"), drawTry);
  enableZoom($("try-alt"), drawTry);
  loadCase(0);
}
function loadCase(i, msg) {
  S.tr.case = i;
  resetZoom("try-plan", "try-alt", "cmp-plan", "cmp-alt");
  const c = S.data.cases[i];
  S.tr.window = c.window;
  const m = msg || c.message;
  S.tr.words = m.words.map((w) => ({...w}));
  S.tr.variant = m.variant;
  const w = win(c.window);
  $("case-ac").textContent = `${w.callsign} · track ${fmt(w.trk_deg)}°, ${fmt(w.alt_ft)} ft, ${fmt(w.gs_kt)} kt · semantic label of this flight segment: ${w.label}`;
  $("msg-variants").innerHTML = w.messages.map((x, j) => `<button type="button" data-j="${j}" class="${x.row === m.row ? "on" : ""}" title="${esc(x.said)}">${esc({own: "Own instruction", maintain: "Maintain", own_other: "No motion", other: "Another aircraft", swap: "Callsign contrast"}[x.variant] || x.variant)}</button>`).join("") + (c.message.row !== undefined && !w.messages.some((x) => x.row === c.message.row) ? '<button type="button" data-j="-1" class="on">Case transcript</button>' : "");
  for (const b of $("msg-variants").querySelectorAll("button")) b.addEventListener("click", () => { const j = +b.dataset.j; loadCase(i, j < 0 ? c.message : w.messages[j]); });
  S.tr.unknown = [];
  editor();
  runTry();
}
function editor() {
  const unk = new Set(S.tr.unknown);
  $("editor").innerHTML = S.tr.words.map((w, i) => {
    const c = w.c == null ? 1 : w.c;
    return `<button type="button" class="word ${unk.has(w.w.toLowerCase()) ? "unk" : ""}" data-i="${i}" style="background:color-mix(in srgb, var(--am) ${Math.round((1 - c) * 60)}%, var(--bg))" title="confidence ${c.toFixed(2)}${unk.has(w.w.toLowerCase()) ? ", outside the SBM vocabulary" : ""}">${esc(w.w)}</button>`;
  }).join("") || '<span class="note">Empty transmission.</span>';
  for (const b of $("editor").querySelectorAll(".word")) b.addEventListener("click", (ev) => {
    if (STATIC) { toast("The transcript is fixed in the static version: run the demo locally to edit it."); return; }
    const i = +b.dataset.i;
    if (ev.shiftKey) S.tr.words.splice(i, 1);
    else {
      const c = S.tr.words[i].c == null ? 1 : S.tr.words[i].c;
      const j = CONF_STEPS.findIndex((x) => Math.abs(x - c) < 0.05);
      S.tr.words[i].c = CONF_STEPS[(j + 1) % CONF_STEPS.length];
    }
    editor(); runTry();
  });
}
function runTry() {
  clearTimeout(runTry.h);
  runTry.h = setTimeout(async () => {
    if (view() !== "try" || S.tr.window == null) return;
    const id = (runTry.id = (runTry.id || 0) + 1);
    try {
      const res = await api("/api/predict", {...sel(), window: S.tr.window, words: S.tr.words, conditions: S.tr.conds});
      if (id !== runTry.id) return;
      S.tr.res = res;
      S.tr.unknown = res.input.unknown;
      editor(); drawTry();
    } catch (e) { toast(e.message); }
    runCmp.stale = true;
  }, 250);
}
function trackUp(trk) {
  const psi = trk * Math.PI / 180, s = Math.sin(psi), c = Math.cos(psi);
  return (p) => [p[0] * c - p[1] * s, p[0] * s + p[1] * c].map((v) => v / 1000);   // right, along (km)
}
function plotPrediction(planId, altId, res, bounds, covId, branchId) {
  const w = win(res.window), tu = trackUp(w.trk_deg);
  const plan = $(planId), alt = $(altId);
  clear(plan); clear(alt);
  const mode = displayMode();
  const series = Object.entries(res.predictions);
  const primary = res.predictions.composed ? "composed" : (series[0] || [null])[0];
  const b = bounds || predBounds(res);
  const f = frame(plan, b.plan, true);
  axes(plan, f, "right of track (km)", "along track (km)", f.W < 420);
  const clipId = `clip-${planId}`;
  const clip = el("clipPath", {id: clipId}, el("defs", {}, plan));
  el("rect", {x: f.pad.l, y: f.pad.t, width: f.W - f.pad.l - f.pad.r, height: f.H - f.pad.t - f.pad.b}, clip);
  const g = el("g", {"clip-path": `url(#${clipId})`}, plan);
  const sty = (c, n) => ({stroke: c.color, "stroke-width": n > 1 ? 1.1 : 2, ...(c.dash ? {"stroke-dasharray": c.dash} : {})});
  for (const [k, p] of series) {
    const c = COND[k] || {color: "#444"};
    const isP = k === primary, many = p.tracks.length > 1;
    if (mode === "samples" || (mode === "both" && many) || !many) {
      for (const j of pick(p.tracks.length, MAX_DRAW)) line(g, f, p.tracks[j].map(tu), {...sty(c, p.tracks.length), "stroke-opacity": !many ? 1 : (mode === "both" ? 0.18 : 0.35)});
    } else if (!isP) {           /* probability mode: other conditions as their pointwise median path */
      const med = p.tracks[0].map((_, i) => { const xs = p.tracks.map((t) => tu(t[i])); return [0, 1].map((d) => quantile(xs.map((x) => x[d]).sort((u, v) => u - v), 0.5)); });
      line(g, f, med, {...sty(c, 2), "stroke-width": 2, "stroke-opacity": 0.9});
    }
  }
  let cov = null;
  const pp = res.predictions[primary];
  if (mode !== "samples" && pp && pp.tracks.length > 4) {
    const color = (COND[primary] || {color: "#1240e0"}).color;
    cov = LEADS.map((L) => drawRegions(g, f, pp.tracks.map((t) => tu(t[L.i])), color, tu(res.flown[L.i]), pp.tracks, L.t));
  }
  let labs = null;
  if (primary === "composed" && res.branch_of_sample && pp) {
    labs = branchBadges(g, f, pp.tracks.map((t) => tu(t[30])), pp.group_of_track.map((k) => res.branch_of_sample[k]), (COND.composed).color);
  }
  if (branchId) badgeLegend(branchId, labs, COND.composed.color);
  line(g, f, res.flown.map(tu), {stroke: "var(--ink)", "stroke-width": 2.4});
  for (const L of LEADS) { const q = tu(res.flown[L.i]); el("circle", {cx: f.X(q[0]), cy: f.Y(q[1]), r: 3.2, fill: "var(--bg)", stroke: "var(--ink)", "stroke-width": 1.6}, g); }
  el("circle", {cx: f.X(0), cy: f.Y(0), r: 4, fill: "var(--ink)"}, g);
  const fa = frame(alt, [0, 60, b.alt[0], b.alt[1]], false);
  axes(alt, fa, "time from the transmission (s)", "altitude change (ft)", fa.W < 420);
  const ga = el("g", {}, alt);
  const ts = (tr) => tr.map((p, i) => [2 * i, p[2] * 3.28084]);
  for (const [k, p] of series) {
    const c = COND[k] || {color: "#444"};
    const many = p.tracks.length > 1;
    if (mode !== "samples" && many) {
      const Z = p.tracks[0].map((_, i) => p.tracks.map((t) => t[i][2] * 3.28084).sort((u, v) => u - v));
      const q = (lo) => Z.map((z, i) => [2 * i, quantile(z, lo)]);
      if (k === primary) {
        for (const [lo, hi, op] of [[0.05, 0.95, 0.16], [0.25, 0.75, 0.28]]) {
          const up = q(hi), dn = q(lo).reverse();
          el("polygon", {points: up.concat(dn).map((pt) => `${fa.X(pt[0]).toFixed(1)},${fa.Y(pt[1]).toFixed(1)}`).join(" "), fill: c.color, "fill-opacity": op, stroke: "none"}, ga);
        }
      }
      line(ga, fa, q(0.5), {...sty(c, 2), "stroke-width": k === primary ? 2 : 1.6});
      if (mode === "both") for (const j of pick(p.tracks.length, MAX_DRAW)) line(ga, fa, ts(p.tracks[j]), {...sty(c, 2), "stroke-width": 0.8, "stroke-opacity": 0.15});
    } else {
      for (const j of pick(p.tracks.length, MAX_DRAW)) line(ga, fa, ts(p.tracks[j]), {...sty(c, p.tracks.length), "stroke-opacity": many ? 0.35 : 1});
    }
  }
  line(ga, fa, ts(res.flown), {stroke: "var(--ink)", "stroke-width": 2.4});
  if (covId) $(covId).textContent = cov ? covText(cov) : "";
}
function predBounds(...rs) {
  const tuAll = [], alts = [];
  for (const res of rs) {
    if (!res) continue;
    const tu = trackUp(win(res.window).trk_deg);
    const add = (tr) => { for (const p of tr) { tuAll.push(tu(p)); alts.push(p[2] * 3.28084); } };
    add(res.flown);
    for (const p of Object.values(res.predictions)) for (const j of pick(p.tracks.length, MAX_DRAW)) add(p.tracks[j]);
  }
  const q = (a, lo, hi) => { const s = [...a].sort((x, y) => x - y); return [s[Math.floor(lo * (s.length - 1))], s[Math.ceil(hi * (s.length - 1))]]; };
  const [x0, x1] = q(tuAll.map((p) => p[0]), 0.01, 0.99), [y0, y1] = q(tuAll.map((p) => p[1]), 0.01, 0.99), [a0, a1] = q(alts, 0.01, 0.99);
  const pd = 0.6;
  return {plan: [Math.min(x0, 0) - pd, Math.max(x1, 0) + pd, Math.min(y0, 0) - pd, Math.max(y1, 0) + pd], alt: [Math.min(a0, -200) - 100, Math.max(a1, 200) + 100]};
}
const METRICS = [
  {key: "es_h_m", head: "ES horiz. (m)", d: 0, better: "low"},
  {key: "es_v_ft", head: "ES vert. (ft)", d: 0, better: "low"},
  {key: "ade_h_m", head: "ADE (m)", d: 0, better: "low"},
  {key: "cov90", head: "90% coverage", d: 2, better: "near0.9"},
];
const LEGEND = '<p class="legend">Green marks the best value in each column, excluding the semantic label, a reference that is not available in operation. Energy score (ES): distance from the predicted trajectories to the flown one, accounting for their spread; lower is better. ADE: error of the mean predicted path; lower is better. 90% coverage: share of flown positions inside the central 90% band; best near 0.90, lower means overconfident. One case says little about average accuracy.</p>';
function bestOf(rows, m) {
  /* index of the best row for metric m, rows: [{name, scores}] */
  let best = -1, bv = Infinity;
  rows.forEach((r, i) => {
    const v = r.scores[m.key];
    if (v == null || r.name === "label") return;
    const score = m.better === "low" ? v : Math.abs(v - 0.9);
    if (score < bv - 1e-9) { bv = score; best = i; }
  });
  return best;
}
function scoreTable(res) {
  const rows = Object.entries(res.predictions).map(([name, p]) => ({name, scores: p.scores}));
  const best = METRICS.map((m) => bestOf(rows, m));
  return `<div class="scroll"><table><thead><tr><th>Prediction</th>${METRICS.map((m) => `<th>${m.head}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) => `<tr><td><span class="sw" style="background:${(COND[r.name] || {}).color || "#444"}"></span> ${esc((COND[r.name] || {label: r.name}).label)}</td>${METRICS.map((m, j) => `<td class="${best[j] === i && rows.length > 1 ? "best" : ""}">${r.scores[m.key] == null ? "–" : fmt(r.scores[m.key], m.d)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>${LEGEND}`;
}
function drawTry() {
  const res = S.tr.res;
  if (!res) return;
  plotPrediction("try-plan", "try-alt", res, null, "try-cov", "try-branches");
  $("try-scores").innerHTML = scoreTable(res);
  bars("try-readings", res.readings.map((r) => ({label: r.text, p: r.p})));
  if (!res.readings_exact) $("try-readings").insertAdjacentHTML("beforeend", '<p class="note">This SBM has no beam search: frequencies of 48 draws.</p>');
  bars("try-status", [["own_motion", "this aircraft, motion instruction"], ["own_other", "this aircraft, no motion instruction"], ["other", "another aircraft"]].map(([k, l]) => ({label: l, p: res.status[k]})));
  const K = res.samples.reduce((s, g) => s + g.n, 0);
  bars("try-samples", res.samples.map((g) => ({label: g.text, p: g.n / K, text: `${g.n}/${K}`})));
  $("try-info").textContent = `${res.input.n_tokens} tokens${res.input.n_unknown ? `, ${res.input.n_unknown} outside this SBM's vocabulary (${res.input.unknown.join(", ")})` : ""}${STATIC ? "" : ` · ${res.ms} ms`}`;
}

/* ------------------------------------------------------------------ compare */
function cmpSetup() {
  const v = S.variants;
  const opts = (t, val) => Object.entries(t).map(([k, e]) => `<option value="${esc(k)}" ${k === val ? "selected" : ""}>${esc(e.label || e)}</option>`).join("");
  S.cmp.a = {sbm: "final", ctp: "flow_pca", sampler: "euler12"};
  S.cmp.b = {sbm: "final", ctp: v.ctp.mixture_pca ? "mixture_pca" : Object.keys(v.ctp)[1], sampler: "euler12"};
  $("cmp").innerHTML = ["a", "b"].map((s) => `<div class="panel"><div class="cmp-sel"><select id="cmp-${s}-sbm" aria-label="SBM">${opts(v.sbm, S.cmp[s].sbm)}</select><select id="cmp-${s}-ctp" aria-label="Trajectory predictor">${opts(v.ctp, S.cmp[s].ctp)}</select><select id="cmp-${s}-sampler" aria-label="Sampler">${opts(v.samplers, S.cmp[s].sampler)}</select></div><div class="figrow"><div class="fig"><svg id="cmp-${s}-plan" data-zoom="cmp-plan"></svg></div><div class="fig"><svg id="cmp-${s}-alt" data-zoom="cmp-alt"></svg></div></div><div class="branches" id="cmp-${s}-branches"></div><p class="legend" id="cmp-${s}-cov"></p><div id="cmp-${s}-scores"></div></div>`).join("");
  for (const s of ["a", "b"]) for (const k of ["sbm", "ctp", "sampler"]) $(`cmp-${s}-${k}`).addEventListener("change", () => { S.cmp[s][k] = $(`cmp-${s}-${k}`).value; runCmp(); });
  for (const s of ["a", "b"]) { enableZoom($(`cmp-${s}-plan`), drawCmp); enableZoom($(`cmp-${s}-alt`), drawCmp); }
}
function drawCmp() {
  const L = S.cmp.last;
  if (!L) return;
  for (const [sd, r] of [["a", L.a], ["b", L.b]]) plotPrediction(`cmp-${sd}-plan`, `cmp-${sd}-alt`, r, L.bd, `cmp-${sd}-cov`, `cmp-${sd}-branches`);
}
async function runCmp() {
  if (view() !== "compare" || S.tr.window == null) return;
  try {
    const go = (s) => api("/api/predict", {...sel(), ...S.cmp[s], window: S.tr.window, words: S.tr.words, conditions: ["composed", "most_probable"]});
    const a = await go("a"), b = await go("b");
    const bd = predBounds(a, b);
    S.cmp.last = {a, b, bd};
    drawCmp();
    const v = S.variants, name = (s) => `${v.sbm[S.cmp[s].sbm].label} + ${v.ctp[S.cmp[s].ctp].label}, ${v.samplers[S.cmp[s].sampler]}`;
    const rows = [["A, composed", a.predictions.composed], ["B, composed", b.predictions.composed], ["A, most probable", a.predictions.most_probable], ["B, most probable", b.predictions.most_probable]].map(([n, p]) => ({name: n, scores: p.scores}));
    const best = METRICS.map((m) => bestOf(rows, m));
    $("cmp-summary").innerHTML = `<div class="panel" style="margin-top:16px"><h3>Scores on this case</h3><p class="note">A: ${esc(name("a"))}<br>B: ${esc(name("b"))}</p><div class="scroll"><table><thead><tr><th>Prediction</th>${METRICS.map((m) => `<th>${m.head}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) => `<tr><td>${esc(r.name)}</td>${METRICS.map((m, j) => `<td class="${best[j] === i ? "best" : ""}">${r.scores[m.key] == null ? "–" : fmt(r.scores[m.key], m.d)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>${LEGEND.replace(", excluding the semantic label, a reference that is not available in operation", "")}</div>`;
  } catch (e) { toast(e.message); }
}

/* ------------------------------------------------------------------ views */
function view() { const h = location.hash.slice(1); return ["live", "try", "compare", "about"].includes(h) ? h : "live"; }
function show() {
  const v = view();
  for (const s of document.querySelectorAll(".view")) s.classList.toggle("on", s.id === `view-${v}`);
  for (const a of document.querySelectorAll(".tabs a")) a.classList.toggle("on", a.dataset.view === v);
  if (!S.data) return;
  if (v === "live") { renderLive(); }
  if (v === "try") { if (S.tr.res) drawTry(); else runTry(); }
  if (v === "compare") runCmp();
}
window.addEventListener("hashchange", show);
const ro = new ResizeObserver(() => { if (!S.data) return; const v = view(); if (v === "live") renderLive(); if (v === "try") drawTry(); });
for (const id of ["live-map", "try-plan"]) ro.observe($(id));

(async function start() {
  try {
    [S.variants, S.data] = await Promise.all([api("/api/variants"), api("/api/cases")]);
  } catch (e) { toast("The model service is not reachable: " + e.message); return; }
  setupModels(); sceneSetup(); trySetup(); cmpSetup(); show();
})();
}
