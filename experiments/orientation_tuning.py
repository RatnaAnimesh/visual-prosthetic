"""Experiment 1: orientation tuning across the layers of the column.

The measurement Hubel and Wiesel made in 1958, repeated on the model: show
drifting gratings at a range of orientations and ask how each layer's response
depends on the orientation of the contour.

The model is not told to be orientation selective.  Selectivity has to emerge
from the arrangement of the geniculate afferents -- a row of ON-centre inputs
flanked by OFF-centre ones, elongated along one axis -- and from what the
intrinsic circuitry then does with that input.
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
    circular_mean_orientation,
    orientation_selectivity_index,
    tuning_width_hwhh,
)

LAYERS = ["L4Cb_ss", "L4Ca_ss", "L4B_pyr", "L23_pyr", "L5_pyr", "L6_pyr",
          "L4_pv", "L23_pv"]


def run(cfg: ColumnConfig, n_ori: int = 12, dur_ms: float = 500.0,
        blank_ms: float = 150.0, contrast: float = 1.0,
        sf: float = 2.6, tf: float = 4.0, seed_offset: int = 0) -> dict:
    col = CorticalColumn(cfg)
    oris = np.linspace(0, 180, n_ori, endpoint=False)

    # Spontaneous baseline.
    col.reset()
    col.present(Blank(), blank_ms + dur_ms)
    spont = {
        b: np.array([
            col.net.spikes.rates(col.net.n, blank_ms, blank_ms + dur_ms)[
                col.pops[col.key(b, k)].idx].mean()
            for k in range(cfg.n_columns)
        ])
        for b in LAYERS
    }

    curves = {b: np.zeros((n_ori, cfg.n_columns)) for b in LAYERS}
    cell_curves = {b: [] for b in LAYERS}

    for i, o in enumerate(oris):
        col.reset()
        col.present(Blank(), blank_ms)
        t0 = col.net.t
        col.present(
            DriftingGrating(orientation_deg=o, spatial_freq=sf,
                            temporal_freq=tf, contrast=contrast),
            dur_ms,
        )
        # Skip the first 60 ms: the onset transient is not the tuned
        # steady-state response.
        r = col.net.spikes.rates(col.net.n, t0 + 60, col.net.t)
        for b in LAYERS:
            curves[b][i] = [r[col.pops[col.key(b, k)].idx].mean()
                            for k in range(cfg.n_columns)]
            cell_curves[b].append(r[col.group(b)])
        print(f"  orientation {o:6.1f} deg done", flush=True)

    out = {
        "orientations": oris.tolist(),
        "column_theta": col.col_theta.tolist(),
        "layers": {},
    }

    for b in LAYERS:
        A = curves[b]
        cells = np.array(cell_curves[b])  # (n_ori, n_cells)
        prefs = col.theta_pref[col.group(b)]

        # Column-level tuning, aligned to each column's preferred orientation.
        aligned = []
        osis, widths, pref_err = [], [], []
        for k in range(cfg.n_columns):
            rk = A[:, k]
            osis.append(orientation_selectivity_index(rk, oris))
            widths.append(tuning_width_hwhh(rk, oris))
            est, _ = circular_mean_orientation(rk, oris)
            d = abs(est - col.col_theta[k]) % 180.0
            pref_err.append(min(d, 180.0 - d))
            shift = int(round(col.col_theta[k] / (180.0 / n_ori)))
            aligned.append(np.roll(rk, -shift))

        # Single-cell selectivity, using each cell's own preferred orientation.
        cell_osi = [orientation_selectivity_index(cells[:, j], oris)
                    for j in range(cells.shape[1])]

        out["layers"][b] = {
            "curve_by_column": A.tolist(),
            "aligned_curve": np.mean(aligned, axis=0).tolist(),
            "spont": spont[b].mean(),
            "osi_column": float(np.mean(osis)),
            "osi_cell_mean": float(np.nanmean(cell_osi)),
            "osi_cell": np.asarray(cell_osi).tolist(),
            "hwhh_deg": float(np.nanmean(widths)),
            "pref_error_deg": float(np.mean(pref_err)),
            "cell_prefs": prefs.tolist(),
            "cell_rates": cells.tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/orientation_tuning.json")
    ap.add_argument("--n-ori", type=int, default=12)
    ap.add_argument("--dur", type=float, default=500.0)
    ap.add_argument("--columns", type=int, default=8)
    ap.add_argument("--sf", type=float, default=2.6,
                    help="spatial frequency in cycles/deg")
    args = ap.parse_args()

    cfg = ColumnConfig(n_columns=args.columns)
    res = run(cfg, n_ori=args.n_ori, dur_ms=args.dur, sf=args.sf)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))

    print(f"\nwrote {path}\n")
    print(f"{'layer':10s} {'spont':>7s} {'OSI(col)':>9s} {'OSI(cell)':>10s} "
          f"{'HWHH':>7s} {'pref err':>9s}")
    for b, d in res["layers"].items():
        print(f"{b:10s} {d['spont']:7.2f} {d['osi_column']:9.3f} "
              f"{d['osi_cell_mean']:10.3f} {d['hwhh_deg']:7.1f} "
              f"{d['pref_error_deg']:9.1f}")


if __name__ == "__main__":
    main()
