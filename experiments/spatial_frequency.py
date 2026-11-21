"""Experiment 6: the magnocellular and parvocellular streams separate.

Kandel Fig. 25-12 and 25-14: the parvocellular layers of the lateral
geniculate nucleus project to layer IVC-beta and the magnocellular layers to
IVC-alpha, and the two streams keep their distinct properties.  Parvocellular
cells have small receptive fields, sustained responses and low contrast gain;
magnocellular cells have larger fields, transient responses and high contrast
gain.

The prediction that follows: IVC-alpha should prefer coarser gratings than
IVC-beta, and should reach half its maximum response at a much lower contrast.
Layer IVB, which reads out IVC-alpha, should inherit the magnocellular
preference; layer II/III, driven mainly by IVC-beta, should inherit the
parvocellular one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column import Blank, ColumnConfig, CorticalColumn, DriftingGrating
from column.analysis import fit_naka_rushton

SFS = [0.4, 0.7, 1.0, 1.4, 2.0, 2.6, 3.4, 4.4]
CONTRASTS = [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
LAYERS = ["L4Ca_ss", "L4Cb_ss", "L4B_pyr", "L23_pyr", "L5_pyr"]


def _drive(col, sf, contrast, dur_ms):
    """Mean response of each layer, each column at its own preferred angle."""
    vals = {b: [] for b in LAYERS}
    for k in range(col.cfg.n_columns):
        col.reset()
        col.present(Blank(), 120.0)
        t0 = col.net.t
        col.present(
            DriftingGrating(orientation_deg=float(col.col_theta[k]),
                            spatial_freq=sf, temporal_freq=4.0,
                            contrast=contrast),
            dur_ms,
        )
        r = col.net.spikes.rates(col.net.n, t0 + 60, col.net.t)
        for b in LAYERS:
            vals[b].append(r[col.pops[col.key(b, k)].idx].mean())
    return {b: float(np.mean(v)) for b, v in vals.items()}


def run(cfg: ColumnConfig, dur_ms: float = 350.0) -> dict:
    col = CorticalColumn(cfg)
    out = {"spatial_freqs": SFS, "contrasts": CONTRASTS,
           "sf_tuning": {b: [] for b in LAYERS},
           "contrast_response": {}}

    for sf in SFS:
        d = _drive(col, sf, 1.0, dur_ms)
        for b in LAYERS:
            out["sf_tuning"][b].append(d[b])
        print(f"  sf {sf:.1f} cyc/deg done", flush=True)

    # Contrast response, each stream measured at the SF it prefers.  One
    # simulation yields every layer at once, so iterate over the distinct
    # spatial frequencies rather than over layers.
    peak_sf = {b: SFS[int(np.argmax(out["sf_tuning"][b]))] for b in LAYERS}
    out["preferred_sf"] = peak_sf
    for b in LAYERS:
        out["contrast_response"][b] = {"sf": peak_sf[b], "response": []}
    for c in CONTRASTS:
        by_sf = {}
        for sf in sorted(set(peak_sf.values())):
            by_sf[sf] = _drive(col, sf, c, dur_ms)
        for b in LAYERS:
            out["contrast_response"][b]["response"].append(
                by_sf[peak_sf[b]][b])
        print(f"  contrast {c:.2f} done", flush=True)

    for b in LAYERS:
        out["contrast_response"][b]["fit"] = fit_naka_rushton(
            np.array(CONTRASTS), np.array(out["contrast_response"][b]["response"])
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/spatial_frequency.json")
    ap.add_argument("--dur", type=float, default=350.0)
    ap.add_argument("--columns", type=int, default=8)
    args = ap.parse_args()

    cfg = ColumnConfig(n_columns=args.columns)
    res = run(cfg, dur_ms=args.dur)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))

    print(f"\nwrote {path}\n")
    print("spatial-frequency tuning (spikes/s):")
    print("  layer      " + " ".join(f"{s:6.1f}" for s in SFS))
    for b in LAYERS:
        print(f"  {b:10s} " + " ".join(
            f"{v:6.1f}" for v in res["sf_tuning"][b]))
    print("\npreferred SF and contrast response:")
    print(f"  {'layer':10s} {'pref SF':>8s} {'c50':>7s} {'n':>6s} {'Rmax':>8s}")
    for b in LAYERS:
        d = res["contrast_response"][b]
        f = d["fit"]
        print(f"  {b:10s} {d['sf']:8.1f} {f['c50']:7.3f} {f['n']:6.2f} "
              f"{f['r_max']:8.1f}")


if __name__ == "__main__":
    main()
