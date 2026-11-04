"""Design-A targets for every window: maneuver primitives per axis, a physical kind, and
residual weights for what the primitives miss. Writes data/v2/primitives.npz.

Per window and axis (heading deg, altitude ft, ground speed kt) the smooth primitive is
fitted with a free present rate: params (v0, d, r, tau, e). Kinds: k-means (8) on the
heading, altitude and speed change at 60 s and the onset, fitted on training windows.
Residual: the track-frame velocity error of the primitive path, 2 PCA shapes per axis.

    uv run python scripts/build_primitives_v2.py --data data/v2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atc_forecast.data.primitives import (bank_velocities, curve, fit_free, grid, signals,  # noqa: E402
                                          to_bank, velocities)
from atc_forecast.models.baselines import kmeans_anchors  # noqa: E402
from atc_forecast.training.data_v2 import WindowStore  # noqa: E402

THRESH = (0.5, 5.0, 0.3)          # rate change that counts as "the axis changes": deg/s, ft/s, kt/s
N_RES = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--clean", action="store_true",
                    help="axes that keep their rate get fixed values (d = 0, r = v0, tau = 0, "
                         "no capture) instead of arbitrary fitted delays and capture times")
    ap.add_argument("--bank", action="store_true",
                    help="store the turn as bank angle (deg) instead of turn rate; the decoder "
                         "turns it back with omega = g tan(phi) / V(t), coupling turn and speed")
    ap.add_argument("--out", default="primitives.npz")
    a = ap.parse_args()
    ws = WindowStore(a.data)
    N = len(ws.meta)
    g = grid()
    P = np.zeros((N, 3, 5), np.float32)
    rms = np.zeros((N, 3), np.float32)
    t0 = time.time()
    for i in range(0, N, a.chunk):
        fv, c = np.asarray(ws.fut_vel[i:i + a.chunk]), np.asarray(ws.center[i:i + a.chunk])
        for ax, y in enumerate(signals(fv, c)):
            P[i:i + a.chunk, ax], rms[i:i + a.chunk, ax] = fit_free(y, a.chunk, g)
        print(f"{i + len(fv)}/{N} windows, {time.time() - t0:.0f} s", flush=True)
    # kinds on the physical summary
    psi60 = curve(P[:, 0])[:, -1]
    z60 = curve(P[:, 1])[:, -1]
    gs60 = curve(P[:, 2])[:, -1]
    act = np.abs(P[:, :, 2] - P[:, :, 0]) > np.array(THRESH)
    onset = np.where(act, P[:, :, 1], 60).min(1)
    F = np.stack([psi60 / 30, z60 / 1000, gs60 / 20, onset / 15], 1).astype(np.float32)
    tr = ws.split("train")
    rng = np.random.default_rng(0)
    C = kmeans_anchors(F[rng.choice(tr, min(100000, len(tr)), replace=False)], a.k, seed=0)
    kind = ((F[:, None] - C[None]) ** 2).sum(-1).argmin(1).astype(np.int64)
    # per-axis kinds: 0 keeps its present rate, 1 stops (new rate ~ 0: rolls out, levels off,
    # settles), 2 new rate positive (right, climb, accelerate), 3 negative (left, descend, slow)
    th = np.array(THRESH)
    kind_axis = np.where(~act, 0, np.where(np.abs(P[:, :, 2]) < th, 1,
                                           np.where(P[:, :, 2] > 0, 2, 3))).astype(np.int64)
    if a.clean:
        keep = kind_axis == 0
        P[..., 1] = np.where(keep, 0.0, P[..., 1])
        P[..., 2] = np.where(keep, P[..., 0], P[..., 2])
        P[..., 3] = np.where(keep, 0.0, P[..., 3])
        P[..., 4] = np.where(keep, 61.0, P[..., 4])
    if a.bank:
        P = to_bank(P, np.asarray(ws.center))
    # residual velocities of the primitive path, 2 PCA shapes per axis (train)
    res_comp, res_scale = [], []
    sub = np.sort(rng.choice(tr, min(60000, len(tr)), replace=False))

    def prim_vel(idx):
        c = np.asarray(ws.center[idx])
        if a.bank:
            return bank_velocities(P[idx], c)
        return velocities(curve(P[idx, 0]), curve(P[idx, 1]), curve(P[idx, 2]), c)

    Rs = np.asarray(ws.fut_vel[sub]) - prim_vel(sub)
    for ax in range(3):
        s = Rs[..., ax].std() + 1e-6
        U = np.linalg.svd(Rs[..., ax] / s, full_matrices=False)[2][:N_RES]
        res_comp.append(U)
        res_scale.append(s)
    res_comp, res_scale = np.stack(res_comp).astype(np.float32), np.array(res_scale, np.float32)
    coef = np.zeros((N, 3, N_RES), np.float32)
    for i in range(0, N, a.chunk):
        idx = np.arange(i, min(N, i + a.chunk))
        R = np.asarray(ws.fut_vel[idx]) - prim_vel(idx)
        for ax in range(3):
            coef[idx, ax] = (R[..., ax] / res_scale[ax]) @ res_comp[ax].T
    # standardization of the flow's state: 15 primitive numbers + 6 residual weights
    X = np.concatenate([P.reshape(N, 15), coef.reshape(N, 3 * N_RES)], 1)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    # how well the targets reproduce the futures (upper bound of any design-A model)
    va = np.sort(rng.choice(ws.split("val"), 5000, replace=False))
    fv = np.asarray(ws.fut_vel[va])
    v_p = prim_vel(va)
    v_r = v_p + np.stack([(coef[va, ax] @ res_comp[ax]) * res_scale[ax] for ax in range(3)], -1)
    ade = lambda v: float(np.median(np.linalg.norm(np.cumsum(v - fv, 1)[..., :2], axis=-1).mean(1)))
    adev = lambda v: float(np.median(np.abs(np.cumsum(v - fv, 1)[..., 2]).mean(1) * 3.28084))
    report = {"ade_h_prim": ade(v_p), "ade_h_prim_res": ade(v_r), "ade_v_prim_ft": adev(v_p),
              "ade_v_prim_res_ft": adev(v_r), "fit_rms_median": np.median(rms[va], 0).tolist(),
              "kind_frac": np.bincount(kind[tr], minlength=a.k).astype(float).__truediv__(len(tr)).tolist(),
              "kind_centers": C.tolist(), "minutes": (time.time() - t0) / 60}
    report["kind_axis_frac"] = [np.bincount(kind_axis[tr, ax], minlength=4).astype(float).__truediv__(len(tr)).tolist()
                                for ax in range(3)]
    np.savez(Path(a.data) / a.out, bank=np.float32(a.bank), params=P, rms=rms, kind=kind, kind_axis=kind_axis, centers=C,
             res_comp=res_comp, res_scale=res_scale, coef=coef, mu=mu, sd=sd)
    (Path(a.data) / Path(a.out).with_suffix(".json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
