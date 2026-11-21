"""Experiment 3: ocular dominance across the hypercolumn.

Kandel Fig. 25-11B and 25-12: the two eyes' inputs stay segregated from the
retina through the alternating layers of the lateral geniculate nucleus and
into cortex, where they form alternating left- and right-eye bands roughly
750 micrometres wide.  The model assigns each orientation column an
ocular-dominance index from its position along that stripe cycle and scales
the two eyes' geniculate input accordingly.

The measurement: stimulate one eye at a time and compare.  A cell driven only
through the contralateral eye sits at one end of Hubel and Wiesel's
seven-group histogram, a binocularly driven cell in the middle.  The
interesting question is whether monocular segregation in layer IVC gives way
to binocular convergence in the layers above and below it, as it does in the
real cortex.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column import (Blank, ColumnConfig, CorticalColumn, Dichoptic,
                    DriftingGrating)
from column.analysis import ocular_dominance_ratio

LAYERS = ["L4Cb_ss", "L4Ca_ss", "L4B_pyr", "L23_pyr", "L5_pyr", "L6_pyr"]


def _hubel_wiesel_group(odr: float) -> int:
    """Bin the contralateral bias into the classic seven ocular-dominance
    groups: 1 = driven only by the contralateral eye, 7 = only ipsilateral."""
    edges = [-0.85, -0.5, -0.15, 0.15, 0.5, 0.85]
    return int(np.searchsorted(edges, odr) + 1)


def run(cfg: ColumnConfig, dur_ms: float = 600.0, sf: float = 1.8,
        min_response: float = 1.0) -> dict:
    """Measure eye preference layer by layer.

    ``min_response`` excludes cells that neither eye drives.  Without it a
    silent cell scores an ocular-dominance ratio of exactly zero and is
    counted as perfectly binocular, which would make any sublayer that happens
    to be off its preferred spatial frequency look binocular for entirely the
    wrong reason.
    """
    col = CorticalColumn(cfg)
    out = {"column_od_index": col.col_od.tolist(), "layers": {}}

    rates = {b: {} for b in LAYERS}
    spont = {}

    col.reset()
    col.present(Blank(), 400.0)
    for b in LAYERS:
        spont[b] = col.net.spikes.rates(col.net.n, 150, 400)[col.group(b)]

    for eye_name, eye in (("left", 0), ("right", 1)):
        for k in range(cfg.n_columns):
            grating = DriftingGrating(
                orientation_deg=float(col.col_theta[k]),
                spatial_freq=sf, temporal_freq=4.0, contrast=1.0,
            )
            stim = Dichoptic(
                left=grating if eye == 0 else Blank(),
                right=grating if eye == 1 else Blank(),
            )
            col.reset()
            col.present(Blank(), 120.0)
            t0 = col.net.t
            col.present(stim, dur_ms)
            r = col.net.spikes.rates(col.net.n, t0 + 60, col.net.t)
            for b in LAYERS:
                rates[b].setdefault(eye_name, {})[k] = r[
                    col.pops[col.key(b, k)].idx]
        print(f"  {eye_name} eye done", flush=True)

    for b in LAYERS:
        per_col_odr, groups, all_odr = [], [], []
        for k in range(cfg.n_columns):
            l = rates[b]["left"][k]
            r = rates[b]["right"][k]
            idx = col.pops[col.key(b, k)].idx
            s = col.net.spikes.rates(col.net.n, 150, 400)[idx]
            cell_odr = [
                ocular_dominance_ratio(float(li), float(ri), float(si))
                for li, ri, si in zip(l, r, s)
                if max(li - si, ri - si) >= min_response
            ]
            all_odr.extend(cell_odr)
            groups.extend(_hubel_wiesel_group(v) for v in cell_odr)
            per_col_odr.append(float(np.mean(cell_odr)) if cell_odr
                               else float("nan"))
        hist = (np.bincount(np.array(groups), minlength=8)[1:8]
                if groups else np.zeros(7, dtype=int))
        out["layers"][b] = {
            "odr_by_column": per_col_odr,
            "odr_cells": all_odr,
            "histogram": hist.tolist(),
            "n_responsive": len(all_odr),
            # Monocularity: how far the average cell sits from balanced.
            "mean_abs_odr": (float(np.mean(np.abs(all_odr)))
                             if all_odr else float("nan")),
            "frac_monocular": (float(np.mean(np.abs(all_odr) > 0.5))
                               if all_odr else float("nan")),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ocular_dominance.json")
    ap.add_argument("--dur", type=float, default=600.0)
    ap.add_argument("--columns", type=int, default=8)
    ap.add_argument("--sf", type=float, default=1.8)
    args = ap.parse_args()

    cfg = ColumnConfig(n_columns=args.columns)
    res = run(cfg, dur_ms=args.dur, sf=args.sf)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))

    print(f"\nwrote {path}\n")
    print(f"{'layer':10s} {'n resp':>7s} {'|ODI|':>7s} {'% monoc':>8s}   "
          f"histogram 1-7")
    for b, d in res["layers"].items():
        h = ' '.join(f'{x:4d}' for x in d["histogram"])
        print(f"{b:10s} {d['n_responsive']:7d} {d['mean_abs_odr']:7.3f} "
              f"{100 * d['frac_monocular']:7.0f}%   {h}")


if __name__ == "__main__":
    main()
