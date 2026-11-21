"""Phase 1: does adding V1 circuitry change what a cortical implant predicts?

The design is the point.  Both models are calibrated to reproduce the *same*
published single-electrode data -- phosphene size against current.  They then
differ only in what happens when more than one electrode is on, which is
something no one has measured.  That makes the disagreement a prediction
rather than a discrepancy.

  step 1   calibrate each configuration so its size-vs-current curve matches
           the Dynaphos baseline.  This is the Phase 1 gate; nothing that
           fails it earns the right to disagree about anything else.
  step 2   sweep electrode separation and measure departure from linear
           superposition.  The baseline scores 3.7e-9 by construction.
  step 3   look at what a single phosphene and an electrode pair actually
           look like.

Run:  python3 prosthesis/phase1_circuit.py
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
from pulse2percept.stimuli import BiphasicPulseTrain

from prosthesis.cortex_circuit import CircuitSpatial

FIGS = pathlib.Path("figures")
RESULTS = pathlib.Path("results")
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

XR, YR, STEP = (-3.2, -0.4), (-0.2, 2.0), 0.02
STIM_DUR = 100.0
AMPS = np.array([100.0, 150.0, 200.0, 300.0, 400.0, 600.0])
TEST_AMP = 300.0

CONFIGS = {
    "circuit off": dict(horiz_gain=0.0, norm_gain=0.0),
    "horizontal only": dict(horiz_gain=0.6, norm_gain=0.0),
    # The fair strawman: a compressive nonlinearity applied to the summed
    # field with no spatial pooling (norm_sigma -> 0 makes the divisor a
    # pointwise Naka-Rushton function of the local total).  It breaks
    # superposition too, so "a nonlinearity is required" does not single out
    # divisive normalisation; only the length scale does.
    "pointwise saturation": dict(horiz_gain=0.0, norm_gain=1.0,
                                 norm_sigma=0.0),
    "normalisation only": dict(horiz_gain=0.0, norm_gain=1.0),
    "full circuit": dict(horiz_gain=0.6, norm_gain=1.0),
}


def calibrated(label: str) -> dict:
    """A configuration with the excitability this script fitted for it.

    Later phases read the constant from here so that every result shares one
    calibration.  Typing it in by hand let three of them fall out of step with
    the fit (608 against a fitted 589) once activation magnitude changed.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "results"
    cal = json.loads((path / "phase1_circuit.json").read_text())
    return dict(CONFIGS[label],
                excitability=float(cal["calibration"][label]["excitability"]))


def train(amp):
    return BiphasicPulseTrain(300, float(amp), 0.17, stim_dur=STIM_DUR)


def peak_frame(model, implant, stim):
    implant.stim = stim
    d = model.predict_percept(implant).data
    k = int(np.argmax(d.reshape(-1, d.shape[-1]).max(axis=0)))
    return d[..., k]


def disc_diameter(img):
    peak = float(img.max())
    if peak <= 0:
        return 0.0
    return 2.0 * np.sqrt(int((img >= peak / 2).sum()) * STEP ** 2 / np.pi)


def sizes(model, implant, elec):
    return [disc_diameter(peak_frame(model, implant, {elec: train(a)}))
            for a in AMPS]


def calibrate(implant, baseline, cfg, k0=1019.0, iters=5):
    """Fit the excitability so this configuration matches the baseline curve.

    Phosphene size goes as sqrt(1/K), so the ratio of curves gives the update
    directly.  Every configuration has to pass through here: a model that
    predicts different single-electrode sizes has no standing to make claims
    about multi-electrode ones.
    """
    elec = implant.electrode_names[0]
    k = k0
    for _ in range(iters):
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           excitability=k, **cfg)
        m.build()
        s = np.array(sizes(m, implant, elec))
        good = (s > 0) & (baseline > 0)
        if not good.any():
            break
        ratio = float(np.median(s[good] / baseline[good]))
        if abs(ratio - 1.0) < 0.02:
            break
        k *= ratio ** 2
    m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                       excitability=k, **cfg)
    m.build()
    s = np.array(sizes(m, implant, elec))
    good = baseline > 0
    err = float(np.mean(np.abs(s[good] - baseline[good]) / baseline[good]))
    return {"excitability": float(k), "sizes": s.tolist(), "mean_abs_err": err}


def separation_sweep(implant, cal, targets=(400, 600, 800, 1200, 1600,
                                            2000, 2400, 3200, 4500)):
    """Departure from linear superposition against electrode separation."""
    ref = implant.electrode_names[0]
    x0, y0 = implant[ref].x, implant[ref].y
    cand = sorted(((np.hypot(implant[n].x - x0, implant[n].y - y0), n)
                   for n in implant.electrode_names if n != ref))

    picks, seen = [], set()
    for t in targets:
        d, n = min(cand, key=lambda s: abs(s[0] - t))
        if n not in seen:
            picks.append((d, n))
            seen.add(n)

    out = {}
    for label, cfg in CONFIGS.items():
        k = cal[label]["excitability"]
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           excitability=k, **cfg)
        m.build()
        rows = []
        for d, n in picks:
            xe = np.array([x0, implant[n].x], float)
            ye = np.array([y0, implant[n].y], float)
            g = m._ensure_cortical_grid(xe, ye)
            fa = m._activation_field(np.array([TEST_AMP, 0.0]), xe, ye, g)
            fb = m._activation_field(np.array([0.0, TEST_AMP]), xe, ye, g)
            fab = m._activation_field(np.array([TEST_AMP, TEST_AMP]),
                                      xe, ye, g)
            lin = fa + fb
            resid = fab - lin
            den = max(float(np.abs(lin).max()), 1e-12)
            i = np.unravel_index(int(np.argmax(lin)), resid.shape)
            rows.append({
                "separation_um": float(d),
                "max_dev": float(np.abs(resid).max() / den),
                "signed_dev_at_peak": float(resid[i] / den),
                "rms_dev": float(np.sqrt((resid ** 2).mean()) / den),
            })
        out[label] = rows
    return out


def figure_calibration(cal, baseline):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(AMPS, baseline, "o-", lw=2.2, color="#222222", zorder=5,
            label="Dynaphos baseline")
    for (label, d), c in zip(cal.items(),
                             ["#1b6ca8", "#2a9d8f", "#e07a5f", "#c44536"]):
        ax.plot(AMPS, d["sizes"], "--", lw=1.3, color=c,
                label=f"{label} (K={d['excitability']:.0f}, "
                      f"err {100 * d['mean_abs_err']:.0f}%)")
    ax.set_xlabel("stimulus amplitude (µA)")
    ax.set_ylabel("phosphene diameter (deg)")
    ax.set_title("Every configuration is fitted to the same\n"
                 "single-electrode data", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "p1_calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_separation(sweep):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colours = {"circuit off": "#999999", "horizontal only": "#2a9d8f",
               "pointwise saturation": "#7b539b",
               "normalisation only": "#e07a5f", "full circuit": "#c44536"}
    for label, rows in sweep.items():
        d = [r["separation_um"] / 1000 for r in rows]
        v = [100 * r["max_dev"] for r in rows]
        ax.plot(d, v, "o-", ms=4, lw=1.6, color=colours[label], label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("electrode separation on cortex (mm)")
    ax.set_ylabel("departure from independence (%)")
    ax.set_title("The baseline predicts 0% everywhere.\n"
                 "A circuit predicts this.", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "p1_separation_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_pair(implant, cal):
    """What a pair of electrodes looks like, baseline vs circuit."""
    ref = implant.electrode_names[0]
    x0, y0 = implant[ref].x, implant[ref].y
    cand = sorted(((np.hypot(implant[n].x - x0, implant[n].y - y0), n)
                   for n in implant.electrode_names if n != ref))
    d, other = min(cand, key=lambda s: abs(s[0] - 1400))

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
    for row, label in enumerate(["circuit off", "full circuit"]):
        k = cal[label]["excitability"]
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           excitability=k, **CONFIGS[label])
        m.build()
        xe = np.array([x0, implant[other].x], float)
        ye = np.array([y0, implant[other].y], float)
        g = m._ensure_cortical_grid(xe, ye)
        fa = m._activation_field(np.array([TEST_AMP, 0.0]), xe, ye, g)
        fb = m._activation_field(np.array([0.0, TEST_AMP]), xe, ye, g)
        fab = m._activation_field(np.array([TEST_AMP, TEST_AMP]), xe, ye, g)
        lin, resid = fa + fb, fab - (fa + fb)
        vmax = float(np.abs(lin).max()) or 1.0
        for col, (img, t, div) in enumerate([
            (lin, "predicted if independent", False),
            (fab, "actual (both on)", False),
            (resid, "difference", True),
        ]):
            ax = axes[row, col]
            m_ = float(np.abs(img).max()) or 1.0
            ax.imshow(img, cmap="RdBu_r" if div else "inferno",
                      vmin=-m_ if div else 0, vmax=m_ if div else vmax,
                      origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(t, fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f"Two electrodes {d:.0f} µm apart on cortex, {TEST_AMP:.0f} µA "
                 f"each\n(cortical activation fields)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "p1_pair.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return float(d)


def main() -> None:
    implant = Cortivis()
    elec = implant.electrode_names[0]

    print("[1/3] baseline + calibration (the Phase 1 gate)")
    dyn = DynaphosModel(xrange=XR, yrange=YR, xystep=STEP).build()
    baseline = np.array(sizes(dyn, implant, elec))
    print("      amps (uA)  : " + " ".join(f"{a:7.0f}" for a in AMPS))
    print("      baseline   : " + " ".join(f"{v:7.3f}" for v in baseline))

    cal = {}
    for label, cfg in CONFIGS.items():
        cal[label] = calibrate(implant, baseline, cfg)
        print(f"      {label:20s}: " +
              " ".join(f"{v:7.3f}" for v in cal[label]["sizes"]) +
              f"   K={cal[label]['excitability']:6.0f}"
              f"   err={100 * cal[label]['mean_abs_err']:4.1f}%")
    figure_calibration(cal, baseline)

    print("\n[2/3] separation sweep — departure from independence")
    sweep = separation_sweep(implant, cal)
    figure_separation(sweep)
    seps = [r["separation_um"] for r in sweep["full circuit"]]
    print("      separation (um): " + " ".join(f"{s:7.0f}" for s in seps))
    for label, rows in sweep.items():
        print(f"      {label:20s}: " +
              " ".join(f"{100 * r['max_dev']:6.2f}%" for r in rows))

    print("\n[3/3] electrode pair figure")
    d = figure_pair(implant, cal)
    print(f"      rendered a pair {d:.0f} µm apart")

    (RESULTS / "phase1_circuit.json").write_text(json.dumps({
        "amps": AMPS.tolist(), "baseline_sizes": baseline.tolist(),
        "calibration": cal, "separation_sweep": sweep,
        "test_amp": TEST_AMP}, indent=2))
    print(f"\nwrote {RESULTS / 'phase1_circuit.json'}")


if __name__ == "__main__":
    main()
