"""Experiment 8: reading the stimulus back out of the population.

Kandel Fig. 25-17: each cell's preferred orientation is a "line label", its
firing rate is the weight of its vote, and the stimulus the population
represents is the vector sum.  Orientation is defined modulo 180 degrees, so
the angles are doubled before averaging and halved afterwards.

This runs on the output of ``orientation_tuning.py``: for every stimulus that
was presented, decode it from the population response and compare with what
was actually shown.  No fitting is involved -- the decoder is given only each
cell's preferred orientation and the spike counts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column.analysis import circular_mean_orientation


def decode(path: pathlib.Path) -> dict:
    d = json.loads(path.read_text())
    oris = np.array(d["orientations"])
    out = {"orientations": oris.tolist(), "layers": {}}

    for b, layer in d["layers"].items():
        cells = np.array(layer["cell_rates"])  # (n_ori, n_cells)
        prefs = np.array(layer["cell_prefs"])  # each cell's preferred angle
        if cells.size == 0 or np.all(cells == 0):
            continue
        est, err, res = [], [], []
        for i, o in enumerate(oris):
            r = cells[i]
            e, resultant = circular_mean_orientation(r, prefs)
            est.append(e)
            res.append(resultant)
            dd = abs(e - o) % 180.0
            err.append(min(dd, 180.0 - dd))
        out["layers"][b] = {
            "decoded": est,
            "error_deg": err,
            "resultant": res,
            "mean_error_deg": float(np.mean(err)),
            "median_error_deg": float(np.median(err)),
            "n_cells": int(cells.shape[1]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/orientation_tuning_magno.json")
    ap.add_argument("--out", default="results/population_decoding.json")
    args = ap.parse_args()

    src = pathlib.Path(args.source)
    if not src.exists():
        print(f"missing {src}; run orientation_tuning.py first")
        raise SystemExit(1)

    res = decode(src)
    pathlib.Path(args.out).write_text(json.dumps(res))
    print(f"wrote {args.out}\n")
    print(f"{'layer':10s} {'cells':>6s} {'mean err':>9s} {'median err':>11s}")
    for b, x in res["layers"].items():
        print(f"{b:10s} {x['n_cells']:6d} {x['mean_error_deg']:8.2f}° "
              f"{x['median_error_deg']:10.2f}°")


if __name__ == "__main__":
    main()
