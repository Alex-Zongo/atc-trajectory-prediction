"""Look at the v2 dataset: figures and numbers that show what the models will see.

    python scripts/inspect_v2.py --data data/v2

Writes data/v2/figures/*.png and data/v2/figures/stats.json:

  01_examples          raw ADS-B against the 1 Hz history and smoothed future, plan + profile
  02_rejections        why requested windows were dropped
  03_noise_floor       raw future positions around the smoothed future, by lead time: the
                       measurement noise no model can predict (diagnosis hypothesis H1)
  04_target_scale      per-step std of the residual targets, the flow's loss scale
  05_basis_error       position error at 60 s from truncating the PCA basis, by components
  06_unannounced       share of futures with a later maneuver onset, by lead time
  07_delay_alignment   cumulative turn after t0 for lateral maneuvers, by delay
  08_atoms             instructed change relative to the present, per slot
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from atc_forecast.data import atoms as A  # noqa: E402
from atc_forecast.data.frames import cumulative_turn  # noqa: E402
from atc_forecast.training.data_v2 import WindowStore  # noqa: E402
from atc_forecast.units import M_TO_FT  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def examples(ws, out, rng):
    ev = ws.split("val")
    man = ev[ws.is_man[ev]]
    pick = rng.choice(man, 6, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5), gridspec_kw={"height_ratios": [1.4, 1]})
    for j, i in enumerate(pick):
        h = ws.hist[i]
        fp = np.cumsum(ws.fut_vel[i], 0)
        raw = ws.raw_future[ws.meta["raw_row"].iloc[i]]
        raw = raw[~np.isnan(raw[:, 0])]
        ax = axes[0, j]
        ax.plot(h[:, 1] * 1000, h[:, 0] * 1000, color=INK2, lw=1.5, label="history (filtered)")
        ax.plot(fp[:, 1], fp[:, 0], color=BLUE, label="future (smoothed)")
        ax.scatter(raw[:, 2], raw[:, 1], s=8, color=ORANGE, zorder=3, label="raw ADS-B")
        ax.set_aspect("equal", "datalim")
        m = ws.meta.iloc[i]
        z = ws.z_true[i]
        ax.set_title(" ".join(A.describe(s, z[k + 1]) for k, s in
                              enumerate(("lateral", "vertical", "speed"))
                              if z[k + 1] != A.NONE) or "steady", fontsize=8)
        ax.set_xlabel("right (m)")
        if j == 0:
            ax.set_ylabel("along track (m)")
        ax = axes[1, j]
        th = np.arange(-29, 1)
        ax.plot(th, h[:, 2] * 300 * M_TO_FT, color=INK2, lw=1.5)
        ax.plot(np.arange(1, 61), fp[:, 2] * M_TO_FT, color=BLUE)
        ax.scatter(raw[:, 0], raw[:, 3] * M_TO_FT, s=8, color=ORANGE, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel(f"s from t0 (delay {m['delta']:.1f} s)")
        if j == 0:
            ax.set_ylabel("altitude vs now (ft)")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Validation windows: raw samples against the 1 Hz resampled trajectory",
                 fontweight="bold")
    save(fig, out, "01_examples")


def rejections(ws, out, stats):
    rej = ws.manifest["rejected"]
    kept = ws.manifest["n_windows"]
    items = sorted([(k, v) for k, v in rej.items() if v], key=lambda x: x[1])
    labels = [k.replace("_", " ") for k, _ in items] + ["kept"]
    vals = [v for _, v in items] + [kept]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(labels, vals, color=[INK2] * len(items) + [BLUE], height=0.6)
    tot = sum(vals)
    for y, v in enumerate(vals):
        ax.text(v, y, f"  {v:,} ({v / tot:.0%})", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_title("Requested windows: kept and dropped, by reason")
    ax.grid(axis="y", visible=False)
    save(fig, out, "02_rejections")
    stats["kept_fraction"] = kept / tot


def noise_floor(ws, out, stats):
    ev = ws.split("val")[:20000]
    rows = ws.meta["raw_row"].to_numpy()[ev]
    raw = ws.raw_future[rows]                                  # [N, 100, 4]
    fp = np.cumsum(ws.fut_vel[ev], 1)                          # [N, 60, 3]
    tq = raw[..., 0]
    ok = ~np.isnan(tq)
    k = np.clip(np.where(ok, tq, 1), 1, 60)
    # smoothed position at each raw time (linear between the 1 Hz points, with 0 at t0)
    full = np.concatenate([np.zeros_like(fp[:, :1]), fp], 1)
    lo = np.floor(k).astype(int)
    hi = np.minimum(lo + 1, 60)
    w = (k - lo)[..., None]
    gat = lambda j: np.take_along_axis(full, j[..., None].repeat(3, -1), 1)
    sm = gat(lo) * (1 - w) + gat(hi) * w
    d = raw[..., 1:] - sm
    # raw points are relative to the history's present p0, the smoothed future to its own
    # estimate of the present; that constant per-window offset is not scatter, remove it
    d = d - np.nanmean(np.where(ok[..., None], d, np.nan), 1, keepdims=True)
    bins = np.arange(0, 61, 5)
    hz, vt = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = ok & (tq > a) & (tq <= b)
        hz.append(np.median(np.hypot(d[..., 0][m], d[..., 1][m])))
        vt.append(np.median(np.abs(d[..., 2][m])) * M_TO_FT)
    c = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(c, hz, color=BLUE, marker="o", ms=4)
    ax[0].set_title("Horizontal: raw minus smoothed")
    ax[0].set_ylabel("median distance (m)")
    ax[1].plot(c, vt, color=ORANGE, marker="o", ms=4)
    ax[1].set_title("Vertical: raw minus smoothed")
    ax[1].set_ylabel("median |difference| (ft)")
    for a in ax:
        a.set_xlabel("lead time (s)")
        a.set_ylim(0, None)
    save(fig, out, "03_noise_floor")
    stats["noise_floor_bins"] = {"lead": c.tolist(), "h_m": [float(x) for x in hz],
                                 "v_ft": [float(x) for x in vt]}
    stats["noise_floor_h_m"] = float(np.median(hz))
    stats["noise_floor_v_ft"] = float(np.median(vt))


def target_scale(ws, out):
    s = ws.basis["scale"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True)
    for a, (name, col) in enumerate([("along", BLUE), ("right", ORANGE), ("vertical", AQUA)]):
        ax[a].plot(np.arange(1, 61), s[:, a], color=col)
        ax[a].set_title(f"{name} residual std")
        ax[a].set_xlabel("future step (s)")
        ax[a].set_ylim(0, None)
    ax[0].set_ylabel("m/s")
    save(fig, out, "04_target_scale")


def basis_error(ws, out, stats):
    tr = ws.split("train")
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(tr, 30000, replace=False))
    va = np.sort(rng.choice(ws.split("val"), 10000, replace=False))
    s, mean = ws.basis["scale"], ws.basis["mean"]
    R = lambda idx: (ws.fut_vel[idx] - ws.center[idx][:, None]) / s
    Xt, Xv = R(sel) - mean, R(va) - mean
    ks = [1, 2, 3, 4, 6, 8, 10, 12, 16]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    res = {}
    for a, (name, col) in enumerate([("along", BLUE), ("right", ORANGE), ("vertical", AQUA)]):
        _, _, vt = np.linalg.svd(Xt[..., a], full_matrices=False)
        err = []
        for k in ks:
            P = vt[:k]
            rec = (Xv[..., a] @ P.T) @ P
            dv = (rec - Xv[..., a]) * s[:, a]                     # m/s per step
            err.append(float(np.sqrt(np.mean(np.cumsum(dv, 1)[:, -1] ** 2))))
        unit = M_TO_FT if name == "vertical" else 1.0
        ax.plot(ks, np.array(err) * unit, color=col, marker="o", ms=4,
                label=f"{name} ({'ft' if unit != 1 else 'm'})")
        res[name] = dict(zip(ks, np.round(np.array(err) * unit, 1).tolist()))
    ax.set_yscale("log")
    ax.set_xlabel("components per axis")
    ax.set_ylabel("RMS position error at 60 s")
    ax.set_title("What truncating the basis costs (validation)")
    ax.legend()
    save(fig, out, "05_basis_error")
    stats["basis_error_60s"] = res


def unannounced(ws, out, stats):
    m = ws.meta
    lead = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(6, 3))
    for kind, col in (("maneuver", BLUE), ("steady", ORANGE)):
        x = m.loc[m["kind"] == kind, "next_onset_rel"].to_numpy()
        ax.plot(lead, [(x <= k).mean() for k in lead], color=col, label=f"{kind} windows")
    ax.set_xlabel("lead time (s)")
    ax.set_ylabel("share with a later maneuver begun")
    ax.set_ylim(0, 1)
    ax.set_title("Futures that contain an unannounced maneuver")
    ax.legend()
    save(fig, out, "06_unannounced")
    stats["share_unannounced_by_60s"] = float((m["next_onset_rel"] <= 60).mean())


def delay_alignment(ws, out):
    m = ws.meta
    lat = np.where(ws.is_man & np.isin(m["lateral"], ["L", "R"]))[0]
    lat = np.sort(np.random.default_rng(1).choice(lat, min(20000, len(lat)), replace=False))
    turn = np.abs(cumulative_turn(ws.fut_vel[lat]))
    turn = turn - turn[:, :1]
    dl = m["delta"].to_numpy()[lat]
    fig, ax = plt.subplots(figsize=(6, 3))
    for (a, b), col in zip([(2, 5), (5, 8.5), (8.5, 12)], (BLUE, ORANGE, AQUA)):
        sel = (dl >= a) & (dl < b)
        ax.plot(np.arange(1, 61), np.median(turn[sel], 0), color=col,
                label=f"delay {a:g} to {b:g} s")
    ax.set_xlabel("s after t0")
    ax.set_ylabel("median |turn so far| (deg)")
    ax.set_title("Lateral maneuvers start after the message delay")
    ax.legend()
    save(fig, out, "07_delay_alignment")


def atom_stats(ws, out, stats):
    man = np.where(ws.is_man)[0]
    st = ws.state[man]
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9))
    cons = {}
    for a, (slot, col, scale, unit) in enumerate([("lateral", BLUE, 180, "deg"),
                                                  ("vertical", ORANGE, 10000, "ft"),
                                                  ("speed", AQUA, 100, "kt")]):
        z = ws.z_true[man, a + 1]
        on = z != A.NONE
        f = A.relative_features(slot, z[on], st[on, 0], st[on, 1], st[on, 2])
        rel = f[:, 2] * 180 if slot == "lateral" else f[:, 1] * scale
        ax[a].hist(rel, bins=80, color=col)
        ax[a].set_title(f"{slot}: instructed change ({on.mean():.0%} of maneuvers)")
        ax[a].set_xlabel(unit)
        if slot != "lateral":
            cons[slot] = float(f[:, 2].mean())
    stats["direction_consistent"] = cons
    save(fig, out, "08_atoms")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/v2")
    a = ap.parse_args()
    ws = WindowStore(a.data)
    out = Path(a.data) / "figures"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    stats = {}
    examples(ws, out, rng)
    rejections(ws, out, stats)
    noise_floor(ws, out, stats)
    target_scale(ws, out)
    basis_error(ws, out, stats)
    unannounced(ws, out, stats)
    delay_alignment(ws, out)
    atom_stats(ws, out, stats)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
