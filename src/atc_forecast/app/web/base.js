// Shared helpers for every embedded component. `P` is the payload Python injected.
const P = window.PAYLOAD;
document.documentElement.dataset.theme = P.theme || "light";
const C = P.colors;                         // the palette from app/theme.py

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
};

// Inverse of introspect.q8: base64 uint8 back to floats, with a strided accessor.
function deq(q) {
  const bin = atob(q.b64), n = bin.length, out = new Float32Array(n), s = (q.hi - q.lo) / 255;
  for (let i = 0; i < n; i++) out[i] = q.lo + bin.charCodeAt(i) * s;
  const shape = q.shape, strides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  return { data: out, shape, lo: q.lo, hi: q.hi,
    at: (...ix) => out[ix.reduce((a, v, i) => a + v * strides[i], 0)],
    slice2: (...lead) => {               // the trailing 2-D matrix at a leading index
      const r = shape[shape.length - 2], c = shape[shape.length - 1];
      const off = lead.reduce((a, v, i) => a + v * strides[i], 0);
      return Array.from({ length: r }, (_, i) => out.subarray(off + i * c, off + (i + 1) * c));
    },
    row: (...lead) => {
      const c = shape[shape.length - 1], off = lead.reduce((a, v, i) => a + v * strides[i], 0);
      return out.subarray(off, off + c);
    } };
}

const tip = (() => {
  const t = el("div", { class: "tip" });
  document.body.append(t);
  return {
    show(html, ev) {
      t.innerHTML = html; t.classList.add("show");
      const w = t.offsetWidth, h = t.offsetHeight;
      let x = ev.clientX + 14, y = ev.clientY + 14;
      if (x + w > innerWidth - 6) x = ev.clientX - w - 14;
      if (y + h > innerHeight - 6) y = ev.clientY - h - 14;
      t.style.left = x + "px"; t.style.top = y + "px";
    },
    hide() { t.classList.remove("show"); },
  };
})();

const fmt = {
  n: (v, d = 0) => v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }),
  hdg: v => v == null ? "–" : String(Math.round(((v % 360) + 360) % 360) || 360).padStart(3, "0") + "°",
  alt: v => v == null ? "–" : (v >= 18000 ? "FL" + Math.round(v / 100) : Math.round(v).toLocaleString() + " ft"),
  pct: v => (100 * v).toFixed(0) + "%",
  t: s => (s < 0 ? "−" : "") + (Math.abs(s) >= 60 ? Math.floor(Math.abs(s) / 60) + ":" + String(Math.floor(Math.abs(s) % 60)).padStart(2, "0") : Math.abs(s).toFixed(0) + "s"),
};

const hexRgb = (h, a = 255) => { h = h.replace("#", ""); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16), a]; };
const rgba = (h, a) => { const [r, g, b] = hexRgb(h); return `rgba(${r},${g},${b},${a})`; };

// A sequential colour for a magnitude in [0, 1], interpolated through the palette's ramp.
function seqColor(v) {
  const s = C.seq, x = Math.max(0, Math.min(1, v)) * (s.length - 1), i = Math.min(Math.floor(x), s.length - 2), f = x - i;
  const a = hexRgb(s[i]), b = hexRgb(s[i + 1]);
  return `rgb(${a.map((c, k) => Math.round(c + (b[k] - c) * f)).slice(0, 3).join(",")})`;
}
// Diverging colour for a value in [-1, 1]: two hues through a neutral midpoint.
function divColor(v) {
  const t = Math.max(-1, Math.min(1, v)), m = hexRgb(C.div_mid), e = hexRgb(t < 0 ? C.div_neg : C.div_pos), f = Math.abs(t);
  return `rgb(${m.map((c, k) => Math.round(c + (e[k] - c) * f)).slice(0, 3).join(",")})`;
}

// requestAnimationFrame tween with an ease, cancellable by starting another on the same key.
const _tw = {};
function tween(key, ms, fn, done) {
  const id = (_tw[key] = (_tw[key] || 0) + 1), t0 = performance.now();
  const ease = x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  const step = now => {
    if (_tw[key] !== id) return;
    const x = Math.min(1, (now - t0) / ms);
    fn(ease(x));
    if (x < 1) requestAnimationFrame(step); else done && done();
  };
  requestAnimationFrame(step);
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
