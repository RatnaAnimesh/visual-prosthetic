"""Experiment 5: dissecting the circuit that makes the tuning.

The model's orientation selectivity has three candidate sources, and this
experiment separates them by removing one at a time -- the computational
equivalent of a pharmacological block:

  feedforward only   intracortical synapses removed entirely.  What is left is
                     the raw Hubel-Wiesel geniculate convergence.
  no inhibition      GABAergic synapses removed.  Isolates how much of the
                     sharpening is done by inhibition rather than by the
                     feedforward input.
  no untuned PV      only the broadly tuned basket-cell output is removed.
  no horizontal      the long-range connections between columns are cut, which
                     should not change tuning at the centre of the receptive
                     field but should abolish contextual effects.

If tuning is entirely feedforward, the first condition changes nothing.  If
the cortex is sharpening its input, removing inhibition should broaden the
tuning curve without abolishing the preference.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from column import Blank, ColumnConfig, CorticalColumn, DriftingGrating
from column.analysis import orientation_selectivity_index, tuning_width_hwhh

LAYERS = ["L4Cb_ss", "L23_pyr", "L5_pyr"]
MODES = ["intact", "feedforward_only", "no_inhibition", "no_pv",
         "no_horizontal"]


def _lesion(col: CorticalColumn, mode: str) -> None:
    """Zero out a class of synapses in the compiled weight matrices."""
    if mode == "intact":
        return
    for key, M in list(col.net.matrices.items()):
        cname, _ = key
        sc = col.net.syn_classes[cname]
        if (mode == "feedforward_only"
                or (mode == "no_inhibition" and sc.inhibitory)
                or (mode == "no_pv" and cname == "pv_to_any")):
            M.data[:] = 0.0
        elif mode == "no_horizontal":
            C = M.tocoo()
            cross = col.column_of[C.row] != col.column_of[C.col]
            if cross.any():
                data = C.data.copy()
                data[cross] = 0.0
                new = sp.coo_matrix(
                    (data, (C.row, C.col)), shape=C.shape).tocsr()
                new.eliminate_zeros()
                col.net.matrices[key] = new


def measure(cfg: ColumnConfig, mode: str, n_ori: int = 8,
            dur_ms: float = 400.0) -> dict:
    col = CorticalColumn(cfg)
    _lesion(col, mode)
    oris = np.linspace(0, 180, n_ori, endpoint=False)

    A = {b: np.zeros((n_ori, cfg.n_columns)) for b in LAYERS}
    for i, o in enumerate(oris):
        col.reset()
        col.present(Blank(), 120.0)
        t0 = col.net.t
        col.present(
            DriftingGrating(orientation_deg=float(o), spatial_freq=2.6,
                            temporal_freq=4.0, contrast=1.0),
            dur_ms,
        )
        r = col.net.spikes.rates(col.net.n, t0 + 60, col.net.t)
        for b in LAYERS:
            A[b][i] = [r[col.pops[col.key(b, k)].idx].mean()
                       for k in range(cfg.n_columns)]

    out = {}
    for b in LAYERS:
        osis, widths, aligned = [], [], []
        for k in range(cfg.n_columns):
            rk = A[b][:, k]
            osis.append(orientation_selectivity_index(rk, oris))
            widths.append(tuning_width_hwhh(rk, oris))
            shift = int(np.argmin(np.abs(
                ((oris - col.col_theta[k] + 90) % 180) - 90)))
            aligned.append(np.roll(rk, -shift))
        curve = np.mean(aligned, axis=0)
        out[b] = {
            "osi": float(np.mean(osis)),
            "hwhh": float(np.nanmean(widths)),
            "peak": float(curve[0]),
            "mean": float(A[b].mean()),
            "aligned_curve": curve.tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/circuit_dissection.json")
    ap.add_argument("--dur", type=float, default=400.0)
    ap.add_argument("--columns", type=int, default=8)
    args = ap.parse_args()

    cfg = ColumnConfig(n_columns=args.columns)
    res = {}
    for m in MODES:
        print(f"condition: {m}", flush=True)
        res[m] = measure(cfg, m, dur_ms=args.dur)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))

    print(f"\nwrote {path}\n")
    for b in LAYERS:
        print(f"\n{b}")
        print(f"  {'condition':18s} {'OSI':>6s} {'HWHH':>7s} {'peak':>8s} "
              f"{'mean':>8s}")
        for m in MODES:
            d = res[m][b]
            print(f"  {m:18s} {d['osi']:6.3f} {d['hwhh']:7.1f} "
                  f"{d['peak']:8.2f} {d['mean']:8.2f}")


if __name__ == "__main__":
    main()
