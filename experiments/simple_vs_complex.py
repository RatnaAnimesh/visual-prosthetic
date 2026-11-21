"""Experiment 2: simple and complex receptive fields across the layers.

A simple cell has spatially separate ON and OFF subregions, so a drifting
grating drives it in and out of its preferred phase and its firing follows the
stimulus cycle: F1/F0 > 1.  A complex cell responds to a correctly oriented
contour anywhere in its receptive field, so it fires throughout the cycle:
F1/F0 < 1.

Nothing in the model builds a complex cell directly.  Layer IVC cells are
wired to the geniculate with a definite spatial phase, and every cell draws
its phase independently.  A layer II/III pyramid pools many layer IV cells of
the same orientation but different phases, which is Hubel and Wiesel's
hierarchical proposal; whether that pooling actually produces phase invariance
is what this experiment measures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column import Blank, ColumnConfig, CorticalColumn, DriftingGrating
from column.analysis import f0_f1, per_neuron_spike_times

LAYERS = ["L4Cb_ss", "L4Ca_ss", "L4B_pyr", "L23_pyr", "L5_pyr", "L5_ib",
          "L6_pyr"]


def run(cfg: ColumnConfig, dur_ms: float = 2000.0, tf: float = 4.0,
        sf: float = 2.6) -> dict:
    col = CorticalColumn(cfg)
    out = {"tf": tf, "layers": {}}

    # Drive every column at its own preferred orientation in turn, so each
    # cell is measured with the stimulus it actually likes.
    per_layer = {b: {"f0": [], "f1": [], "ratio": []} for b in LAYERS}

    for k in range(cfg.n_columns):
        col.reset()
        col.present(Blank(), 150.0)
        t0 = col.net.t
        col.present(
            DriftingGrating(orientation_deg=float(col.col_theta[k]),
                            spatial_freq=sf, temporal_freq=tf, contrast=1.0),
            dur_ms,
        )
        t_start = t0 + 200.0  # let the response settle before measuring
        for b in LAYERS:
            idx = col.pops[col.key(b, k)].idx
            trains = per_neuron_spike_times(col.net.spikes, idx)
            for gid in idx:
                f0, f1 = f0_f1(trains[int(gid)], t_start, col.net.t, tf)
                if f0 >= 1.0:  # a ratio is meaningless for a silent cell
                    per_layer[b]["f0"].append(f0)
                    per_layer[b]["f1"].append(f1)
                    per_layer[b]["ratio"].append(f1 / f0)
        print(f"  column {k} ({col.col_theta[k]:.0f} deg) done", flush=True)

    for b in LAYERS:
        r = np.array(per_layer[b]["ratio"])
        out["layers"][b] = {
            "f0": per_layer[b]["f0"],
            "f1": per_layer[b]["f1"],
            "ratio": per_layer[b]["ratio"],
            "mean_ratio": float(np.mean(r)) if r.size else float("nan"),
            "frac_simple": float(np.mean(r > 1.0)) if r.size else float("nan"),
            "n": int(r.size),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/simple_vs_complex.json")
    ap.add_argument("--dur", type=float, default=2000.0)
    ap.add_argument("--columns", type=int, default=8)
    args = ap.parse_args()

    cfg = ColumnConfig(n_columns=args.columns)
    res = run(cfg, dur_ms=args.dur)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))

    print(f"\nwrote {path}\n")
    print(f"{'layer':10s} {'n':>5s} {'mean F1/F0':>11s} {'% simple':>9s}")
    for b, d in res["layers"].items():
        print(f"{b:10s} {d['n']:5d} {d['mean_ratio']:11.2f} "
              f"{100 * d['frac_simple']:8.0f}%")


if __name__ == "__main__":
    main()
