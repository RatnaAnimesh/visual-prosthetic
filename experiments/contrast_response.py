"""Experiment 4: contrast response, and whether tuning survives contrast.

Two questions.

First, the shape of the contrast-response function.  Magnocellular relay cells
have high contrast gain and saturate early; parvocellular cells have lower
gain and keep signalling up to high contrast (Kandel Fig. 25-14 and the
parallel-pathway account).  Layer IVC-alpha and IVC-beta inherit those
signatures from their separate geniculate inputs, so the two sublayers should
have measurably different semisaturation contrasts.

Second, contrast invariance of orientation tuning.  A cell's firing rate grows
with contrast, but its *preferred orientation and tuning width* should not --
a high-contrast orthogonal grating must not drive the cell as well as a
low-contrast preferred one.  A purely feedforward threshold model tends to
broaden with contrast; keeping the width fixed is a signature of the
intracortical inhibition doing normalisation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column import Blank, ColumnConfig, CorticalColumn, DriftingGrating
from column.analysis import (
    fit_naka_rushton,
    orientation_selectivity_index,
    tuning_width_hwhh,
)

CONTRASTS = [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
LAYERS = ["L4Cb_ss", "L4Ca_ss", "L23_pyr", "L5_pyr", "L4_pv"]


def run(cfg: ColumnConfig, dur_ms: float = 400.0,
        tuning_contrasts=(0.12, 1.0), n_ori: int = 8) -> dict:
    col = CorticalColumn(cfg)
    out = {"contrasts": CONTRASTS, "layers": {}, "tuning": {}}

    # --- contrast response at each column's preferred orientation ---------
    curves = {b: [] for b in LAYERS}
    for c in CONTRASTS:
        vals = {b: [] for b in LAYERS}
        for k in range(cfg.n_columns):
            col.reset()
            col.present(Blank(), 120.0)
            t0 = col.net.t
            col.present(
                DriftingGrating(orientation_deg=float(col.col_theta[k]),
                                spatial_freq=2.6, temporal_freq=4.0,
                                contrast=c),
                dur_ms,
            )
            r = col.net.spikes.rates(col.net.n, t0 + 60, col.net.t)
            for b in LAYERS:
                vals[b].append(r[col.pops[col.key(b, k)].idx].mean())
        for b in LAYERS:
            curves[b].append(float(np.mean(vals[b])))
        print(f"  contrast {c:.2f} done", flush=True)

    for b in LAYERS:
        fit = fit_naka_rushton(np.array(CONTRASTS), np.array(curves[b]))
        out["layers"][b] = {"response": curves[b], "fit": fit}

    # --- contrast invariance of orientation tuning ------------------------
    oris = np.linspace(0, 180, n_ori, endpoint=False)
    for c in tuning_contrasts:
        aligned, osis, widths = [], [], []
        A = np.zeros((n_ori, cfg.n_columns))
        for i, o in enumerate(oris):
            col.reset()
            col.present(Blank(), 120.0)
            t0 = col.net.t
            col.present(
                DriftingGrating(orientation_deg=float(o), spatial_freq=2.6,
                                temporal_freq=4.0, contrast=c),
                dur_ms,
            )
            r = col.net.spikes.rates(col.net.n, t0 + 60, col.net.t)
            A[i] = [r[col.pops[col.key("L23_pyr", k)].idx].mean()
                    for k in range(cfg.n_columns)]
        for k in range(cfg.n_columns):
            rk = A[:, k]
            osis.append(orientation_selectivity_index(rk, oris))
            widths.append(tuning_width_hwhh(rk, oris))
            shift = int(np.argmin(np.abs(
                ((oris - col.col_theta[k] + 90) % 180) - 90)))
            aligned.append(np.roll(rk, -shift))
        out["tuning"][str(c)] = {
            "orientations": oris.tolist(),
            "aligned_curve": np.mean(aligned, axis=0).tolist(),
            "osi": float(np.mean(osis)),
            "hwhh": float(np.nanmean(widths)),
        }
        print(f"  tuning at contrast {c:.2f} done", flush=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/contrast_response.json")
    ap.add_argument("--dur", type=float, default=400.0)
    ap.add_argument("--columns", type=int, default=8)
    args = ap.parse_args()

    cfg = ColumnConfig(n_columns=args.columns)
    res = run(cfg, dur_ms=args.dur)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))

    print(f"\nwrote {path}\n")
    print(f"{'layer':10s} {'c50':>7s} {'n':>6s} {'Rmax':>8s}")
    for b, d in res["layers"].items():
        f = d["fit"]
        print(f"{b:10s} {f['c50']:7.3f} {f['n']:6.2f} {f['r_max']:8.1f}")
    print("\ncontrast invariance of orientation tuning (layer II/III):")
    for c, d in res["tuning"].items():
        print(f"  contrast {c:>4s}: OSI={d['osi']:.3f}  HWHH={d['hwhh']:.1f} deg")


if __name__ == "__main__":
    main()
