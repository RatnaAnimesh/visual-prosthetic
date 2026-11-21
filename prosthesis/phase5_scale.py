"""Is the spatial-scale mismatch a model failure or a regime difference?

The model's electrode interaction vanishes by ~4.5 mm.  Bosking et al. observe
interactions between surface electrodes at mean separations of 9.6-26 mm.
Before adjusting any parameter to close that gap -- which would be fitting to
the data the model is supposed to be tested against -- the cheaper question is
whether the difference in *stimulation regime* already accounts for it.

It might.  Those experiments used 0.3-4.0 mA through 0.5 mm subdural surface
contacts; the intracortical predictions here were computed at 300 uA.  Under
the model's own current-spread law, D = 2*sqrt(I/K), a thirteen-fold current
increase enlarges the activated radius roughly 3.6-fold, and the distance over
which two electrodes can share a normalisation pool should grow with it.

So: sweep amplitude across their range and measure how the interaction length
scale moves, changing nothing else.  If the interaction reaches 10 mm at 4 mA,
the apparent mismatch is a comparison error rather than a modelling one.

Run:  python3 prosthesis/phase5_scale.py
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

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.phase1_circuit import calibrated

FIGS = pathlib.Path("figures")
RESULTS = pathlib.Path("results")
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Bosking et al. tested 0.3-4.0 mA; the intracortical work here used 300 uA.
AMPS_UA = [300.0, 1000.0, 2000.0, 4000.0]
# Out past 26.36 mm, the largest mean separation reported in humans, so that
# the comparison there reads a measured value rather than the end of a sweep.
SEPARATIONS_UM = list(range(400, 27601, 400))


# Human reference points (Bosking et al. 2022), mean separations in mm.
HUMAN = {
    "merged, overlapping both": 4.11,
    "one phosphene (all pairs)": 9.57,
    "two phosphenes (all pairs)": 26.36,
}


def sweep(model, amp, separations, x0=20000.0, y0=-5000.0):
    """Departure from superposition against separation, at one amplitude."""
    devs = []
    for d in separations:
        xe = np.array([x0, x0 + d], float)
        ye = np.array([y0, y0], float)
        g = model._ensure_cortical_grid(xe, ye)
        fa = model._activation_field(np.array([amp, 0.0]), xe, ye, g)
        fb = model._activation_field(np.array([0.0, amp]), xe, ye, g)
        fab = model._activation_field(np.array([amp, amp]), xe, ye, g)
        lin = fa + fb
        resid = fab - lin
        devs.append(float(np.abs(resid).max() /
                          max(float(np.abs(lin).max()), 1e-12)))
    return np.array(devs)


def length_scales(seps, devs):
    """Where the interaction peaks and where it has decayed to a tenth."""
    if devs.max() <= 0:
        return dict(peak=0.0, peak_um=np.nan, decay10_um=np.nan)
    i = int(np.argmax(devs))
    beyond = seps[(seps > seps[i]) & (devs < 0.1 * devs.max())]
    return dict(peak=float(devs.max()), peak_um=float(seps[i]),
                decay10_um=float(beyond[0]) if beyond.size else np.nan)


def main() -> None:
    CFG = calibrated("full circuit")
    seps = np.array(SEPARATIONS_UM, float)
    # A margin wide enough that the sheet still contains the spread at 4 mA.
    model = CircuitSpatial(xrange=(-3.2, -0.4), yrange=(-0.2, 2.0),
                           xystep=0.1, cortex_step=100.0,
                           cortex_margin=14000.0, **CFG)
    model.build()

    print("current spread and interaction range vs amplitude")
    print(f"{'amp (uA)':>9} {'act. radius':>12} {'peak dev':>9} "
          f"{'peak at':>9} {'10% by':>9}")

    out = {"separations_um": SEPARATIONS_UM, "human_reference_mm": HUMAN,
           "by_amplitude": {}}
    curves = {}
    for amp in AMPS_UA:
        devs = sweep(model, amp, seps)
        ls = length_scales(seps, devs)
        radius = 1000.0 * np.sqrt(amp / CFG["excitability"])
        curves[amp] = devs
        out["by_amplitude"][str(amp)] = {
            "activation_radius_um": float(radius),
            "deviations": devs.tolist(), **ls}
        print(f"{amp:9.0f} {radius:11.0f}um {100 * ls['peak']:8.1f}% "
              f"{ls['peak_um'] / 1000:8.2f}mm {ls['decay10_um'] / 1000:8.2f}mm")
        sys.stdout.flush()

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(AMPS_UA)))
    for (amp, devs), c in zip(curves.items(), cmap):
        ax.plot(seps / 1000, 100 * devs, lw=1.8, color=c,
                label=f"{amp / 1000:.1f} mA")
    for label, mm in HUMAN.items():
        ax.axvline(mm, color="#c44536", ls=":", lw=1.2)
        ax.text(mm, ax.get_ylim()[1] * 0.95, f" {label}", rotation=90,
                fontsize=6, va="top", color="#c44536")
    ax.set_xlabel("electrode separation (mm)")
    ax.set_ylabel("departure from independence (%)")
    ax.set_title("Interaction range scales with stimulation current\n"
                 "(dotted: mean separations reported in humans)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, title="amplitude",
              title_fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "p5_scale.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    out["far_pool"] = far_pool_search(CFG)
    (RESULTS / "phase5_scale.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'phase5_scale.json'}")


# What a second, feedback-scale pool would need to do.  Pairs in which one of
# two phosphenes went unreported sat 13.47 mm apart on average.  Making the
# pool produce that while leaving the other two classes as reported takes
# all three of these at once.
FAR_TARGETS_MM = (4.11, 13.47, 26.36)
FAR_REQUIRE = {"4.11": (">", 0.30), "13.47": (">", 0.15), "26.36": ("<", 0.03)}


def far_pool_search(cfg, amp=4000.0, sigmas_um=(4000.0, 6000.0, 8000.0,
                                                12000.0),
                    gains=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Search far-pool width and gain (relative to the near pool) at 4 mA.

    The near pool's gain is 1, so a relative gain above one inverts the
    measured ordering, in which far surround suppression is weaker than near.
    """
    print("\nfar normalisation pool, 4 mA: interaction at "
          + ", ".join(f"{d} mm" for d in FAR_TARGETS_MM))
    seps = np.array(FAR_TARGETS_MM) * 1000.0
    rows = []
    for sig in sigmas_um:
        for gfar in gains:
            m = CircuitSpatial(xrange=(-3.2, -0.4), yrange=(-0.2, 2.0),
                               xystep=0.1, cortex_step=100.0,
                               cortex_margin=14000.0, norm_far_gain=gfar,
                               norm_far_sigma=sig, **cfg)
            m.build()
            devs = sweep(m, amp, seps)
            r = {"sigma_um": sig, "relative_gain": gfar / cfg["norm_gain"],
                 "deviations": dict(zip((f"{d:g}" for d in FAR_TARGETS_MM),
                                        map(float, devs)))}
            r["meets_all"] = all(
                (r["deviations"][k] > v) if op == ">" else
                (r["deviations"][k] < v) for k, (op, v) in FAR_REQUIRE.items())
            rows.append(r)
            print(f"  sigma {sig/1000:4.0f} mm  gain x{r['relative_gain']:<5g}"
                  + "".join(f"  {100*v:6.1f}%" for v in devs)
                  + ("   meets all three" if r["meets_all"] else ""),
                  flush=True)
    ordered = [r for r in rows if r["relative_gain"] <= 1.0]
    return {"amp_ua": amp, "requirements": FAR_REQUIRE, "rows": rows,
            "settings_meeting_all": [(r["sigma_um"], r["relative_gain"])
                                     for r in rows if r["meets_all"]],
            "max_at_13.47_if_far_weaker": max(r["deviations"]["13.47"]
                                              for r in ordered),
            "min_at_13.47_if_far_weaker": min(r["deviations"]["13.47"]
                                              for r in ordered)}


if __name__ == "__main__":
    main()
