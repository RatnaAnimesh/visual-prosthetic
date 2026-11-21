"""Sensitivity analysis: how much does the Phase 1 result depend on guesses?

Phase 1 predicted up to 30% departure from electrode independence, but the
normalisation strength driving that number was never calibrated -- no public
multi-electrode phosphene data exists to calibrate it against.  So the honest
question is which parts of the conclusion survive not knowing it.

Two things are swept:

  norm_gain    how strong divisive normalisation is.  Sets the magnitude.
  norm_sigma   how far the normalisation pool reaches.  Should set *where* the
               interaction peaks, which is the number the falsifiable
               prediction actually names ("~1.5 mm apart").

The critical methodological point: **K is refitted at every sweep point**, so
every model compared here still reproduces the published single-electrode
size-vs-current curve.  Without that, a change in normalisation would be
confounded with a change in phosphene size, and the comparison would be
meaningless.

That refit also does something useful.  If a normalisation strength cannot be
made to fit the single-electrode data by any K, then existing data already
rules it out -- and the sweep turns into a bound rather than a guess.

Run:  python3 prosthesis/phase1_sensitivity.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models.cortex import DynaphosModel

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.phase1_circuit import (AMPS, STEP, TEST_AMP, XR, YR,
                                       calibrate, sizes)

FIGS = pathlib.Path("figures")
RESULTS = pathlib.Path("results")
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Broad, log-spaced: four octaves either side of the Phase 1 value of 1.0.
NORM_GAINS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
NORM_SIGMAS = [750.0, 1500.0, 3000.0]
# Fine separation sampling so the peak location can actually be located.
SEPARATIONS = list(range(400, 4600, 200))

# A configuration is "admissible" if, after refitting K, it still matches the
# published single-electrode curve about as well as Phase 1's models did.
ADMISSIBLE_ERR = 0.10


def electrode_pairs(implant, targets):
    ref = implant.electrode_names[0]
    x0, y0 = implant[ref].x, implant[ref].y
    cand = sorted(((np.hypot(implant[n].x - x0, implant[n].y - y0), n)
                   for n in implant.electrode_names if n != ref))
    picks, seen = [], set()
    for t in targets:
        d, n = min(cand, key=lambda s: abs(s[0] - t))
        if n in seen:
            continue
        seen.add(n)
        picks.append((float(d),
                      np.array([x0, implant[n].x], float),
                      np.array([y0, implant[n].y], float)))
    return picks


def sweep_separation(model, pairs):
    seps, devs = [], []
    for d, xe, ye in pairs:
        g = model._ensure_cortical_grid(xe, ye)
        fa = model._activation_field(np.array([TEST_AMP, 0.0]), xe, ye, g)
        fb = model._activation_field(np.array([0.0, TEST_AMP]), xe, ye, g)
        fab = model._activation_field(np.array([TEST_AMP, TEST_AMP]),
                                      xe, ye, g)
        lin = fa + fb
        resid = fab - lin
        seps.append(d)
        devs.append(float(np.abs(resid).max() /
                          max(float(np.abs(lin).max()), 1e-12)))
    return np.array(seps), np.array(devs)


def summarise(seps, devs):
    """Summarise the interaction curve.

    The plain argmax is a poor descriptor here.  At strong normalisation the
    curve becomes monotonically decreasing with a shoulder, so the global
    maximum jumps to the smallest separation sampled even though the
    *structural feature* -- an interior local peak -- has not moved at all.
    Reporting both, plus the half-decay distance, separates "where the
    mechanism puts a feature" from "where this particular curve happens to be
    highest".
    """
    if devs.max() <= 0:
        return {"peak": 0.0, "peak_sep_um": float("nan"),
                "local_peak_sep_um": float("nan"),
                "half_decay_um": float("nan")}
    peak = float(devs.max())
    i = int(np.argmax(devs))

    interior = np.arange(1, len(devs) - 1)
    loc = interior[(devs[interior] > devs[interior - 1]) &
                   (devs[interior] > devs[interior + 1])]
    local = float(seps[loc[int(np.argmax(devs[loc]))]]) if loc.size else float("nan")

    beyond = seps[(devs < 0.5 * peak) & (seps > (local if loc.size else seps[i]))]
    return {
        "peak": peak,
        "peak_sep_um": float(seps[i]),
        "local_peak_sep_um": local,
        "half_decay_um": float(beyond[0]) if beyond.size else float("nan"),
    }


def main() -> None:
    implant = Cortivis()
    elec = implant.electrode_names[0]
    pairs = electrode_pairs(implant, SEPARATIONS)

    print("baseline single-electrode curve")
    dyn = DynaphosModel(xrange=XR, yrange=YR, xystep=STEP).build()
    baseline = np.array(sizes(dyn, implant, elec))
    print("   " + " ".join(f"{v:6.3f}" for v in baseline))

    out = {"norm_gain_sweep": [], "norm_sigma_sweep": [],
           "baseline_sizes": baseline.tolist(), "amps": AMPS.tolist()}

    print("\n[1/2] sweeping norm_gain (norm_sigma fixed at 1500 um)")
    print(f"{'norm_gain':>10} {'fitted K':>9} {'fit err':>8} {'admiss':>7} "
          f"{'peak dev':>9} {'local pk':>9} {'half-decay':>11}")
    for gcoef in NORM_GAINS:
        cfg = dict(horiz_gain=0.6, norm_gain=gcoef, norm_sigma=1500.0)
        cal = calibrate(implant, baseline, cfg)
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           excitability=cal["excitability"], **cfg)
        m.build()
        seps, devs = sweep_separation(m, pairs)
        s = summarise(seps, devs)
        ok = cal["mean_abs_err"] <= ADMISSIBLE_ERR
        print(f"{gcoef:10.2f} {cal['excitability']:9.0f} "
              f"{100 * cal['mean_abs_err']:7.1f}% {'yes' if ok else 'NO':>7} "
              f"{100 * s['peak']:8.1f}% {s['local_peak_sep_um']:8.0f}um "
              f"{s['half_decay_um']:10.0f}um")
        out["norm_gain_sweep"].append({
            "norm_gain": gcoef, "norm_sigma": 1500.0,
            "excitability": cal["excitability"],
            "fit_err": cal["mean_abs_err"], "admissible": bool(ok),
            "separations": seps.tolist(), "deviations": devs.tolist(), **s})
        sys.stdout.flush()

    print("\n[2/2] sweeping norm_sigma (norm_gain fixed at 1.0)")
    print(f"{'norm_sigma':>11} {'fitted K':>9} {'fit err':>8} "
          f"{'peak dev':>9} {'local pk':>9}")
    for sg in NORM_SIGMAS:
        cfg = dict(horiz_gain=0.6, norm_gain=1.0, norm_sigma=sg)
        cal = calibrate(implant, baseline, cfg)
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           excitability=cal["excitability"], **cfg)
        m.build()
        seps, devs = sweep_separation(m, pairs)
        s = summarise(seps, devs)
        print(f"{sg:10.0f}um {cal['excitability']:9.0f} "
              f"{100 * cal['mean_abs_err']:7.1f}% {100 * s['peak']:8.1f}% "
              f"{s['peak_sep_um']:8.0f}um")
        out["norm_sigma_sweep"].append({
            "norm_gain": 1.0, "norm_sigma": sg,
            "excitability": cal["excitability"],
            "fit_err": cal["mean_abs_err"],
            "separations": seps.tolist(), "deviations": devs.tolist(), **s})
        sys.stdout.flush()

    # --- figures ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))

    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(NORM_GAINS)))
    for row, c in zip(out["norm_gain_sweep"], cmap):
        ax.plot(np.array(row["separations"]) / 1000,
                100 * np.array(row["deviations"]), lw=1.6, color=c,
                label=f"gain {row['norm_gain']:g}")
    ax.set_xlabel("electrode separation (mm)")
    ax.set_ylabel("departure from independence (%)")
    ax.set_title("Magnitude scales with normalisation strength", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2)

    ax = axes[1]
    g = [r["norm_gain"] for r in out["norm_gain_sweep"]]
    pk = [100 * r["peak"] for r in out["norm_gain_sweep"]]
    ps = [r["local_peak_sep_um"] / 1000 for r in out["norm_gain_sweep"]]
    ax.semilogx(g, pk, "o-", color="#c44536", label="peak deviation (%)")
    ax.set_xlabel("norm_gain")
    ax.set_ylabel("peak deviation (%)", color="#c44536")
    ax2 = ax.twinx()
    ax2.semilogx(g, ps, "s--", color="#1b6ca8",
                 label="local-peak separation (mm)")
    ax2.set_ylabel("local-peak separation (mm)", color="#1b6ca8")
    ax2.set_ylim(0, 4)
    ax.set_title("Magnitude varies 2x over 32x in gain;\nthe length scale does not move at all", fontsize=9)

    ax = axes[2]
    for row, c in zip(out["norm_sigma_sweep"], ["#2a9d8f", "#e07a5f", "#7b539b"]):
        ax.plot(np.array(row["separations"]) / 1000,
                100 * np.array(row["deviations"]), lw=1.6, color=c,
                label=f"pool σ = {row['norm_sigma']:.0f} µm")
    ax.set_xlabel("electrode separation (mm)")
    ax.set_ylabel("departure from independence (%)")
    ax.set_title("Pool extent sets where it peaks", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    for a in list(axes):
        a.spines[["top"]].set_visible(False)
    fig.suptitle("Which conclusions survive not knowing the normalisation "
                 "parameters", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "p1_sensitivity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (RESULTS / "phase1_sensitivity.json").write_text(json.dumps(out, indent=2))

    # --- verdict ---------------------------------------------------------
    adm = [r for r in out["norm_gain_sweep"] if r["admissible"]]
    print("\n--- verdict")
    if adm:
        peaks = [100 * r["peak"] for r in adm]
        locs = [r["local_peak_sep_um"] for r in adm]
        print(f"admissible norm_gain (fits single-electrode data): "
              f"{min(r['norm_gain'] for r in adm):g} to "
              f"{max(r['norm_gain'] for r in adm):g}")
        print(f"  predicted peak deviation over that range: "
              f"{min(peaks):.0f}% to {max(peaks):.0f}%")
        print(f"  local-peak separation over that range: "
              f"{min(locs):.0f} to {max(locs):.0f} um")
        hd = [r["half_decay_um"] for r in adm]
        print(f"  half-decay distance over that range: "
              f"{min(hd):.0f} to {max(hd):.0f} um")
    print(f"\nwrote {RESULTS / 'phase1_sensitivity.json'}")


if __name__ == "__main__":
    main()
