"""The control the paper was missing, and a test that actually discriminates.

Two problems with the earlier claim that "only divisive normalisation breaks
electrode independence":

1.  It is too strong.  Section 3.1 proves a *linear* operator cannot break
    superposition, which is a definition, not a finding.  It says nothing
    about which nonlinearity is needed.  Dynaphos, Fine & Boynton and this
    model all contain nonlinearities, yet Dynaphos measures exactly zero
    interaction.  The reason is structural: Dynaphos applies its nonlinearity
    to *each electrode's own drive*, before summing, so f(I_a) + f(I_b) stays
    additive across electrodes.  This model applies its nonlinearity to the
    *combined* field.

    That means the fair strawman -- absent from the original table -- is a
    pointwise compressive nonlinearity applied to the summed field, with no
    spatial pooling.  It breaks superposition too.  Conveniently it is a limit
    of the model already here: setting norm_sigma -> 0 turns the pooled
    divisor into a pointwise Naka-Rushton saturation, num/(semi^n + g*num).

    So the discriminating variable is not "nonlinear vs linear" but the
    *length scale*.  Pointwise saturation can only act where two activations
    physically overlap, so its interaction should fall monotonically with
    separation.  A pooled divisor reaches a pooling radius beyond the overlap,
    so it should show an interior peak.  That is a measurable difference.

2.  The Bosking comparison in section 3.4 does not discriminate.  Bosking et
    al. predicted phosphene *number* by linear summation of single-electrode
    activation, so merging at short separations and two phosphenes at long
    ones is reproduced by a linear model and tests nothing.

    Their Type 2 result is different: for pairs seen as two distinct
    phosphenes, the *separation* between them was larger under concurrent
    stimulation than under independent stimulation (7.80 deg vs 5.46 deg,
    t(34) = 6.58, p < .001).  Repulsion is the interesting case, because the
    sum of two overlapping blobs has peaks that sit slightly *closer* than the
    blobs themselves -- a linear model predicts attraction or nothing, never
    repulsion.  Mutual surround suppression should push them apart.

Run:  python3 prosthesis/phase6_controls.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models.cortex import DynaphosModel

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.phase1_circuit import (AMPS, STEP, TEST_AMP, XR, YR, calibrate,
                                       peak_frame, disc_diameter, train)

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

# The fifth row.  "pointwise saturation" is norm_sigma -> 0: the divisor stops
# pooling over space and becomes a function of the local summed value alone.
CONFIGS = {
    "circuit off":          dict(horiz_gain=0.0, norm_gain=0.0),
    "horizontal only":      dict(horiz_gain=0.6, norm_gain=0.0),
    "pointwise saturation": dict(horiz_gain=0.0, norm_gain=1.0,
                                 norm_sigma=0.0),
    "normalisation only":   dict(horiz_gain=0.0, norm_gain=1.0,
                                 norm_sigma=1500.0),
    "full circuit":         dict(horiz_gain=0.6, norm_gain=1.0,
                                 norm_sigma=1500.0),
}

# Fine spacing so the interior peak (or its absence) is resolved rather than
# inferred from a handful of electrode pairs.
SEPARATIONS_UM = np.arange(300.0, 6001.0, 150.0)


def baseline_curve(implant):
    dyn = DynaphosModel(xrange=XR, yrange=YR, xystep=STEP).build()
    elec = implant.electrode_names[0]
    return np.array([disc_diameter(peak_frame(dyn, implant, {elec: train(a)}))
                     for a in AMPS])


def free_pair_sweep(model, amp, separations, x0=20000.0, y0=-5000.0):
    """Interaction vs separation using freely placed electrodes.

    The Cortivis array only offers a few dozen discrete pair separations, too
    coarse to tell a monotonic decay from one with an interior maximum.
    """
    devs = []
    for d in separations:
        xe = np.array([x0, x0 + d], float)
        ye = np.array([y0, y0], float)
        g = model._ensure_cortical_grid(xe, ye)
        fa = model._activation_field(np.array([amp, 0.0]), xe, ye, g)
        fb = model._activation_field(np.array([0.0, amp]), xe, ye, g)
        fab = model._activation_field(np.array([amp, amp]), xe, ye, g)
        lin = fa + fb
        den = max(float(np.abs(lin).max()), 1e-12)
        devs.append(float(np.abs(fab - lin).max() / den))
    return np.array(devs)


def axis_profile(field, g, x0, y0, x1, y1, n=2401, extend=0.6):
    """Sample the field along the line joining two electrodes.

    Cubic rather than linear interpolation: the peak shifts being measured are
    tens of microns, so a piecewise-linear profile over a 20 um grid would
    quantise them to the grid and report a spurious zero.
    """
    dx, dy = x1 - x0, y1 - y0
    t = np.linspace(-extend, 1.0 + extend, n)
    xs = x0 + t * dx
    ys = y0 + t * dy
    col = (xs - g["xs"][0]) / g_step(g)
    row = (ys - g["ys"][0]) / g_step(g)
    vals = map_coordinates(field, [row, col], order=3, mode="nearest")
    return t, vals


def g_step(g):
    return float(g["xs"][1] - g["xs"][0])


def _subpixel_max(t, v, i):
    """Refine a discrete maximum by fitting a parabola to its three samples."""
    a, b, c = v[i - 1], v[i], v[i + 1]
    den = a - 2.0 * b + c
    if den == 0:
        return float(t[i])
    delta = 0.5 * (a - c) / den
    return float(t[i] + delta * (t[1] - t[0]))


def two_peak_separation(t, v, sep_um):
    """Distance between the maxima on either side of the midpoint, in microns.

    ``t`` runs along the electrode axis in units of the separation, as
    ``axis_profile`` returns it, so the electrodes sit at 0 and 1.  Each
    side's maximum is refined to sub-sample precision with a parabola.
    None if either maximum sits on the midpoint or at the end of the
    profile, which is the merged case with no second peak to locate.

    An earlier version took the two largest local maxima of the whole
    profile and scaled by the sampled span instead of the separation.  The
    scale error, a factor of 2.2, cancelled in every percentage reported.
    Taking the two largest maxima did not cancel.  On a textured field it can
    put both maxima beside one electrode.
    """
    mid = int(np.searchsorted(t, 0.5))
    i = int(np.argmax(v[:mid]))
    j = mid + int(np.argmax(v[mid:]))
    if i in (0, mid - 1) or j in (mid, len(v) - 1):
        return None
    return float((_subpixel_max(t, v, j) - _subpixel_max(t, v, i)) * sep_um)


def repulsion_test(models, amp=TEST_AMP, separations=(1800.0, 2200.0, 2600.0,
                                                      3000.0, 3600.0, 4200.0,
                                                      5000.0)):
    """Does concurrent stimulation push the two perceived peaks apart?

    Compares the peak separation in the concurrent field against the peak
    separation in the linear sum of the two single-electrode fields, which is
    what an independent model predicts.
    """
    out = {}
    for label, cfg_k in models.items():
        # Re-instantiate on a finer cortical sheet: the shifts of interest are
        # tens of microns, well under the 50 um grid used elsewhere.
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           cortex_step=20.0,
                           excitability=cfg_k[1], **cfg_k[0])
        m.build()
        rows = []
        for d in separations:
            x0, y0 = 20000.0, -5000.0
            xe = np.array([x0, x0 + d], float)
            ye = np.array([y0, y0], float)
            g = m._ensure_cortical_grid(xe, ye)
            fa = m._activation_field(np.array([amp, 0.0]), xe, ye, g)
            fb = m._activation_field(np.array([0.0, amp]), xe, ye, g)
            fab = m._activation_field(np.array([amp, amp]), xe, ye, g)

            t, v_lin = axis_profile(fa + fb, g, x0, y0, x0 + d, y0)
            _, v_con = axis_profile(fab, g, x0, y0, x0 + d, y0)
            sep_lin = two_peak_separation(t, v_lin, d)
            sep_con = two_peak_separation(t, v_con, d)
            rows.append({
                "separation_um": float(d),
                "sep_independent_um": sep_lin,
                "sep_concurrent_um": sep_con,
                "shift_um": (None if (sep_lin is None or sep_con is None)
                             else float(sep_con - sep_lin)),
                "shift_pct": (None if (sep_lin is None or sep_con is None
                                       or sep_lin == 0)
                              else float(100 * (sep_con - sep_lin) / sep_lin)),
            })
        out[label] = rows
    return out


def main() -> None:
    implant = Cortivis()
    print("calibrating every configuration to the same single-electrode curve")
    base = baseline_curve(implant)

    cal, models = {}, {}
    for label, cfg in CONFIGS.items():
        c = calibrate(implant, base, cfg)
        cal[label] = {**c, "config": cfg}
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           excitability=c["excitability"], **cfg)
        m.build()
        models[label] = m
        print(f"  {label:<22} K={c['excitability']:7.1f}  "
              f"size err {100 * c['mean_abs_err']:.1f}%")
    sys.stdout.flush()

    print("\ninteraction vs separation (free electrode placement)")
    print(f"{'configuration':<22} {'peak dev':>9} {'peak at':>9} "
          f"{'at 0.3mm':>9} {'monotonic?':>11}")
    sweep = {}
    for label, m in models.items():
        devs = free_pair_sweep(m, TEST_AMP, SEPARATIONS_UM)
        i = int(np.argmax(devs))
        # "Monotonic" means the maximum sits at the smallest separation
        # tested: interaction confined to the overlap region.
        monotonic = i == 0
        sweep[label] = {
            "separations_um": SEPARATIONS_UM.tolist(),
            "deviations": devs.tolist(),
            "peak": float(devs.max()),
            "peak_sep_um": float(SEPARATIONS_UM[i]),
            "dev_at_min_sep": float(devs[0]),
            "monotonic_decay": bool(monotonic),
        }
        print(f"  {label:<20} {100 * devs.max():8.1f}% "
              f"{SEPARATIONS_UM[i] / 1000:8.2f}mm {100 * devs[0]:8.1f}% "
              f"{str(monotonic):>11}")
        sys.stdout.flush()

    print("\nperceived-separation shift, concurrent vs independent")
    print("(Bosking Type 2: concurrent separation LARGER, 7.80 vs 5.46 deg)")
    rep = repulsion_test({label: (CONFIGS[label], cal[label]["excitability"])
                          for label in CONFIGS})
    for label, rows in rep.items():
        vals = [r["shift_pct"] for r in rows if r["shift_pct"] is not None]
        if vals:
            print(f"  {label:<22} shift {np.mean(vals):+6.1f}% "
                  f"(range {min(vals):+.1f} to {max(vals):+.1f}%)")
        else:
            print(f"  {label:<22} merged at all separations tested")
    sys.stdout.flush()

    out = {"calibration": cal, "separation_sweep": sweep,
           "repulsion": rep, "test_amp_ua": TEST_AMP}
    (RESULTS / "phase6_controls.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'phase6_controls.json'}")


if __name__ == "__main__":
    main()
