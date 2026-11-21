"""Experiment 7: the stimulus that started it all.

Kandel, ch. 25: "While showing an anesthetized animal slides containing a
variety of images, they recorded extracellularly from individual neurons in
the visual cortex.  As they switched from one slide to another they found a
neuron that produced a brisk train of action potentials.  The cell was
responding not to the image on the slide but to the edge of the slide as it
was moved into position."

So: sweep an oriented bar across the receptive field and record the column.
This produces the raster and the peri-stimulus time histogram that a
physiologist would actually have in front of them, rather than a summary
statistic.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column import Blank, ColumnConfig, CorticalColumn, MovingBar
from column.analysis import psth

LAYERS = ["L4Cb_ss", "L23_pyr", "L5_pyr", "L6_pyr", "L4_pv"]


def run(cfg: ColumnConfig, n_trials: int = 20, speed: float = 3.0,
        sweep_ms: float = 600.0) -> dict:
    col = CorticalColumn(cfg)
    # Column 0 is the one we record from; sweep at its preferred orientation
    # and at the orthogonal one.
    pref = float(col.col_theta[0])
    out = {"preferred_deg": pref, "sweep_ms": sweep_ms, "conditions": {}}

    for label, ori in (("preferred", pref), ("orthogonal", (pref + 90) % 180)):
        spikes_by_layer = {b: [] for b in LAYERS}
        raster = []
        for trial in range(n_trials):
            col.reset()
            col.present(Blank(), 100.0)
            t0 = col.net.t
            # A bar wide enough to fill one ON subregion drives a simple cell
            # far better than a hairline; Hubel and Wiesel's slit was matched
            # to the subregion width for the same reason.
            bar = MovingBar(
                orientation_deg=ori, width_deg=0.20, length_deg=1.6,
                contrast=1.0, speed_deg_s=speed, start_offset_deg=-0.9,
            )
            col.present(bar, sweep_ms)
            col.present(Blank(), 100.0)
            t, i = col.net.spikes.arrays()
            rel = t - t0
            for b in LAYERS:
                lo = col.pops[col.key(b, 0)].start
                hi = col.pops[col.key(b, 0)].stop
                m = (i >= lo) & (i < hi)
                spikes_by_layer[b].append(rel[m])
            # Accumulate the raster over every trial, offsetting each trial so
            # repeats of the same cell stack rather than overlap.
            pop = col.pops[col.key("L4Cb_ss", 0)]
            m = (i >= pop.start) & (i < pop.stop)
            raster.append((rel[m], (i[m] - pop.start) + trial * len(pop)))
            print(f"  {label} trial {trial} done", flush=True)

        rt = np.concatenate([r[0] for r in raster]) if raster else np.empty(0)
        ri = np.concatenate([r[1] for r in raster]) if raster else np.empty(0)
        cond = {"orientation": ori,
                "raster": [rt.tolist(), ri.astype(int).tolist()],
                "psth": {}}
        for b in LAYERS:
            allsp = np.concatenate(spikes_by_layer[b]) if spikes_by_layer[b] \
                else np.empty(0)
            centres, rate = psth(allsp, -100.0, sweep_ms + 100.0, bin_ms=10.0,
                                 n_trials=n_trials * len(
                                     col.pops[col.key(b, 0)].idx))
            cond["psth"][b] = {"t": centres.tolist(), "rate": rate.tolist()}
        out["conditions"][label] = cond
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/moving_bar.json")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--columns", type=int, default=8)
    args = ap.parse_args()

    cfg = ColumnConfig(n_columns=args.columns)
    res = run(cfg, n_trials=args.trials)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    print(f"\nwrote {path}")
    for label, cond in res["conditions"].items():
        peak = {b: max(d["rate"]) for b, d in cond["psth"].items()}
        print(f"  {label:11s} peak rates: " +
              "  ".join(f"{b}={v:.1f}" for b, v in peak.items()))


if __name__ == "__main__":
    main()
