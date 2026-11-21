"""Does the biophysical model reproduce what has actually been measured?

Each stage of ``cortex_ssn`` is checked against the data it was built from or,
better, against data it was *not* fitted to.

    network        Obeid & Miller's own published benchmarks: surround
                   suppression index and summation-field size for E and I
                   cells, and the growth of the summation field at low
                   contrast.  A test of the reimplementation.
    recruitment    Histed et al. 2009: sparse activation, a handful of cells
                   at ~8.5 uA, filling in rather than spreading.
                   Kumaravelu et al. 2022: median distance of activated somata
                   294 -> 369 -> 455 um at 5, 20, 50 uA.  Not fitted.
    ongoing        Butovas & Schwarz 2003: stimulation suppresses ongoing
                   activity around the electrode.  Not fitted.
    summation      Nassi et al. 2015: direct cortical drive combined with a
                   visual stimulus sums sub-additively.  Not fitted.
    inhibition     Dadarlat, Sun & Stryker 2024: inhibitory and excitatory
                   neurons are recruited in similar proportions (ratio ~1.1)
                   up to 50 uA, inhibitory ones closer to the electrode, and
                   excitatory cells beyond ~500 um are mainly suppressed.
                   The ratio calibrates the inhibitory site density; the
                   spatial pattern is a prediction.
    dual           Kim et al. 2025: two electrodes 370-850 um apart in mouse
                   V1 at 7, 13 and 16 uA.  Minimal modulation at the lowest
                   current, net enhancement at the middle one, net zero at
                   the highest, with attenuation concentrated in cells that
                   respond strongly to one electrode alone, enhancement in
                   cells that respond to neither, and modulation peaking
                   between the electrodes.  Not fitted.

Two numbers are calibrated here: the excitatory site density (against the
count of responding cells in Histed et al.) and the inhibitory one (against
Dadarlat et al.'s ratio).  Everything else is a prediction.

Run:  python3 prosthesis/v2_validate.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.special import erf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_ssn import BiophysicalCortex, SSNSheet

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

MM_PER_DEG = 0.5            # Obeid & Miller's sheet
X0, Y0 = 20000.0, -5000.0
# Neurons whose soma intersects a two-photon optical section, per mm^2:
# ~1.2e5 /mm^3 in mouse V1 layer 2/3 times an ~11 um soma.
PLANE_DENSITY = 1300.0


def contrast_drive(C):
    return 50.0 * C ** 3.5 / (11.0 ** 3.5 + C ** 3.5)


def grating_patch(sheet, i, j, size_deg, C, sigma_in_deg=0.09, sfori=20.0):
    """Feedforward input for a patch of preferred-orientation grating."""
    n, step = sheet.n, sheet.step
    idx = np.arange(n)
    off = lambda c: ((idx - c + n // 2) % n - n // 2) * step
    s = sigma_in_deg * MM_PER_DEG * 1000.0 * np.sqrt(2)
    half = size_deg * MM_PER_DEG * 1000.0 / 2

    def edge(d):
        return 0.5 * (erf((d + half) / s) - erf((d - half) / s))

    h = edge(off(i))[:, None] * edge(off(j))[None, :]
    dth = np.abs(sheet.theta - sheet.theta[i, j])
    dth = np.rad2deg(np.minimum(dth, np.pi - dth))
    return contrast_drive(C) * h * np.exp(-dth ** 2 / (2 * sfori ** 2))


def network_benchmarks(n_cells=12, seed=3):
    sheet = SSNSheet(n=75, step_um=107.0, seed=seed)
    sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0,
                      6.0, 8.0, 10.0])
    rng = np.random.default_rng(0)
    cells = [(int(a), int(b)) for a, b in rng.integers(20, 60, (n_cells, 2))]
    out = {}
    for C in (16.4, 8.0):
        si_e, si_i, sf_e, sf_i = [], [], [], []
        for i, j in cells:
            re, ri = [], []
            for L in sizes:
                h = grating_patch(sheet, i, j, L, C)
                RE, RI = sheet.run(h, h, t_ms=300.0)
                re.append(RE[i, j])
                ri.append(RI[i, j])
            re, ri = np.array(re), np.array(ri)
            si_e.append((re.max() - re[-1]) / re.max())
            si_i.append((ri.max() - ri[-1]) / ri.max())
            sf_e.append(sizes[re.argmax()])
            sf_i.append(sizes[ri.argmax()])
        out[str(C)] = {"si_e": float(np.mean(si_e)),
                       "si_i": float(np.mean(si_i)),
                       "sf_e_deg": float(np.mean(sf_e)),
                       "sf_i_deg": float(np.mean(sf_i))}
    out["published_C16.4"] = {"si_e": 0.79, "si_i": 0.27,
                              "sf_e_deg": 1.14, "sf_i_deg": 1.75}
    return out


def _median_distance(p, d):
    w, dd = p.ravel(), d.ravel()
    o = np.argsort(dd)
    cw = np.cumsum(w[o])
    return float(dd[o][np.searchsorted(cw, cw[-1] / 2)])


def recruitment_benchmarks():
    # A fine sheet: the median distance of a few hundred microns should not
    # be quantised by the grid it is measured on.
    m = BiophysicalCortex(step_um=26.75, margin_um=2600.0)
    g = m._ensure_cortical_grid(np.array([X0]), np.array([Y0]))
    X, Y = np.meshgrid(g["xs"], g["ys"])
    d = np.hypot(X - X0, Y - Y0)
    cell = (m.step / 1000.0) ** 2
    rows = []
    for amp in (5.0, 8.5, 10.0, 20.0, 25.0, 50.0, 100.0, 300.0):
        pe, pi = m.recruited([amp], [X0], [Y0], g)
        rows.append({
            "amp_ua": amp,
            "init_radius_um": m.init_radius_um(amp),
            "cells_in_plane": float((pe * PLANE_DENSITY * cell).sum()),
            "peak_fraction_e": float(pe.max()),
            "peak_fraction_i": float(pi.max()),
            "median_distance_um": _median_distance(pe, d),
        })
    return {"rows": rows,
            "histed": {"cells_at_8.5ua": 5, "note": "calibration target"},
            "kumaravelu_median_um": {"5": 294, "20": 369, "50": 455}}


def ongoing_activity():
    m = BiophysicalCortex()
    g = m._ensure_cortical_grid(np.array([X0]), np.array([Y0]))
    X, Y = np.meshgrid(g["xs"], g["ys"])
    d = np.hypot(X - X0, Y - Y0)
    rows = []
    for amp in (10.0, 50.0, 100.0, 300.0):
        m._activation_field([amp], [X0], [Y0], g)
        L = m.last
        ind = L["indirect_e"]
        sup = ind < -0.02 * L["baseline_e"].mean()
        rows.append({
            "amp_ua": amp,
            "direct_total": float(L["direct_e"].sum()),
            "indirect_total": float(ind.sum()),
            "indirect_over_direct": float(ind.sum() / L["direct_e"].sum()),
            "suppression_radius_um": float(d[sup].max()) if sup.any() else 0.0,
            "deepest_suppression_frac": float(
                (ind / L["baseline_e"]).min()),
        })
    return rows


def visual_plus_stimulation():
    """Electrode plus grating at the electrode's own site (Nassi et al.)."""
    m = BiophysicalCortex()
    g = m._ensure_cortical_grid(np.array([X0]), np.array([Y0]))
    sheet = g["sheet"]
    X, Y = np.meshgrid(g["xs"], g["ys"])
    i, j = np.unravel_index(np.argmin(np.hypot(X - X0, Y - Y0)), X.shape)
    rows = []
    for C in (6.0, 10.0, 16.4, 30.0):
        h = grating_patch(sheet, i, j, 1.5, C)
        for amp in (50.0, 150.0):
            vis = m._activation_field([0.0], [X0], [Y0], g, h, h)[i, j]
            stim = m._activation_field([amp], [X0], [Y0], g)[i, j]
            both = m._activation_field([amp], [X0], [Y0], g, h, h)[i, j]
            rows.append({"contrast": C, "amp_ua": amp, "visual": float(vis),
                         "stim": float(stim), "both": float(both),
                         "ratio_to_sum": float(both / (vis + stim))})
    return rows


def inhibitory_recruitment():
    """Recruited I/E ratio near the tip, and where each class is recruited."""
    m = BiophysicalCortex(step_um=26.75, margin_um=2600.0)
    g = m._ensure_cortical_grid(np.array([X0]), np.array([Y0]))
    X, Y = np.meshgrid(g["xs"], g["ys"])
    d = np.hypot(X - X0, Y - Y0)
    rows = []
    for amp in (10.0, 20.0, 30.0, 50.0):
        pe, pi = m.recruited([amp], [X0], [Y0], g)
        rows.append({
            "amp_ua": amp,
            "ratio_within_300um": float(pi[d < 300].mean() / pe[d < 300].mean()),
            "ratio_within_500um": float(pi[d < 500].mean() / pe[d < 500].mean()),
            "median_distance_e_um": _median_distance(pe, d),
            "median_distance_i_um": _median_distance(pi, d),
        })
    # Excitatory suppression beyond 500 um (network model, 50 uA).
    full = BiophysicalCortex()
    g2 = full._ensure_cortical_grid(np.array([X0]), np.array([Y0]))
    X2, Y2 = np.meshgrid(g2["xs"], g2["ys"])
    d2 = np.hypot(X2 - X0, Y2 - Y0)
    ev = full._activation_field([50.0], [X0], [Y0], g2)
    ring = (d2 > 500) & (d2 < 1500)
    core = d2 < 200
    return {"rows": rows,
            "measured_ratio": 1.1,
            "evoked_core_200um_hz": float(ev[core].mean()),
            "evoked_ring_500_1500um_hz": float(ev[ring].mean()),
            "fraction_of_ring_suppressed": float((ev[ring] < 0).mean())}


def dual_electrode(seps=(370.0, 485.0, 558.0, 850.0),
                   amps=(6.8, 12.7, 15.9, 30.0, 60.0, 100.0, 200.0)):
    """Kim et al. 2025: the population and its parts under paired ICMS.

    The model has population rates, not single cells, so the comparison is
    made through the two classes the paper separates.  Cells that respond
    strongly to one electrode alone are, near the tips, the directly driven
    ones; cells that respond to neither are those the network drives.  Their
    modulations (both minus the sum of each alone) are reported separately,
    with the net over an 800 x 800 um field of view centred between the
    electrodes.
    """
    m = BiophysicalCortex(pulse_hz=200.0)
    out = []
    for sep in seps:
        xe, ye = np.array([X0, X0 + sep]), np.array([Y0, Y0])
        g = m._ensure_cortical_grid(xe, ye)
        X, Y = np.meshgrid(g["xs"], g["ys"])
        fov = (np.abs(X - (X0 + sep / 2)) < 400) & (np.abs(Y - Y0) < 400)
        on_axis = np.abs(Y - Y0) < m.step / 2
        for amp in amps:
            F = {}
            for k, a in (("a", [amp, 0.0]), ("b", [0.0, amp]),
                         ("ab", [amp, amp])):
                F[k] = m._activation_field(a, xe, ye, g)
                F[k + "_d"] = m.last["direct_e"]
                F[k + "_i"] = m.last["indirect_e"]
            lin = F["a"] + F["b"]
            mod = F["ab"] - lin
            mod_d = F["ab_d"] - F["a_d"] - F["b_d"]
            mod_i = F["ab_i"] - F["a_i"] - F["b_i"]
            line = np.where(on_axis & fov, np.abs(mod), -1.0)
            peak_x = float(X.ravel()[line.argmax()] - X0)
            out.append({
                "separation_um": sep, "amp_ua": amp,
                "net_fraction": float(mod[fov].sum() / lin[fov].sum()),
                "direct_fraction": float(mod_d[fov].sum() / lin[fov].sum()),
                "network_fraction": float(mod_i[fov].sum() / lin[fov].sum()),
                "peak_offset_from_e1_um": peak_x,
            })
    return out


def main() -> None:
    out = {}

    print("network: reimplementation against Obeid & Miller's benchmarks")
    nb = out["network"] = network_benchmarks()
    pub = nb["published_C16.4"]
    a = nb["16.4"]
    print(f"  suppression index E  {a['si_e']:.2f}  (published {pub['si_e']})")
    print(f"  suppression index I  {a['si_i']:.2f}  (published {pub['si_i']})")
    print(f"  summation field  E   {a['sf_e_deg']:.2f} deg "
          f"(published {pub['sf_e_deg']})")
    print(f"  summation field  I   {a['sf_i_deg']:.2f} deg "
          f"(published {pub['sf_i_deg']})")
    print(f"  summation field E at low contrast {nb['8.0']['sf_e_deg']:.2f} deg"
          f"  (must exceed the high-contrast value)")
    sys.stdout.flush()

    print("\nrecruitment: Histed 2009 and Kumaravelu 2022")
    rb = out["recruitment"] = recruitment_benchmarks()
    print(f"  {'uA':>5} {'R_init':>7} {'cells':>7} {'peak p':>7} "
          f"{'median d':>9}")
    for r in rb["rows"]:
        print(f"  {r['amp_ua']:5.1f} {r['init_radius_um']:6.0f}um "
              f"{r['cells_in_plane']:7.1f} {r['peak_fraction_e']:7.3f} "
              f"{r['median_distance_um']:7.0f}um")
    print("  Kumaravelu median distance: 294 (5 uA), 369 (20), 455 (50) um")
    sys.stdout.flush()

    print("\nongoing activity: Butovas & Schwarz 2003")
    oa = out["ongoing"] = ongoing_activity()
    for r in oa:
        print(f"  {r['amp_ua']:5.0f} uA  indirect/direct "
              f"{r['indirect_over_direct']:+.2f}  suppressed out to "
              f"{r['suppression_radius_um']:.0f} um  deepest "
              f"{100 * r['deepest_suppression_frac']:+.0f}% of spontaneous")
    sys.stdout.flush()

    print("\nvisual + stimulation: Nassi et al. 2015")
    vs = out["visual_plus_stim"] = visual_plus_stimulation()
    for r in vs:
        print(f"  C={r['contrast']:5.1f}  {r['amp_ua']:4.0f} uA  visual "
              f"{r['visual']:6.2f}  stim {r['stim']:6.2f}  both "
              f"{r['both']:6.2f}  both/sum {r['ratio_to_sum']:.2f}")
    sys.stdout.flush()

    print("\ninhibitory recruitment: Dadarlat, Sun & Stryker 2024")
    ir = out["inhibition"] = inhibitory_recruitment()
    for r in ir["rows"]:
        print(f"  {r['amp_ua']:4.0f} uA  I/E within 300 um "
              f"{r['ratio_within_300um']:.2f}, within 500 um "
              f"{r['ratio_within_500um']:.2f}   median distance E "
              f"{r['median_distance_e_um']:.0f} um, I "
              f"{r['median_distance_i_um']:.0f} um")
    print(f"  measured ratio ~{ir['measured_ratio']}; at 50 uA evoked "
          f"{ir['evoked_core_200um_hz']:+.2f} Hz within 200 um, "
          f"{ir['evoked_ring_500_1500um_hz']:+.3f} Hz at 0.5-1.5 mm "
          f"({100 * ir['fraction_of_ring_suppressed']:.0f}% of it "
          f"suppressed)")
    sys.stdout.flush()

    print("\ndual-electrode modulation: Kim et al. 2025 "
          "(% of summed response in the field of view)")
    de = out["dual"] = dual_electrode()
    for r in de:
        print(f"  {r['separation_um']:4.0f} um {r['amp_ua']:6.1f} uA  net "
              f"{100 * r['net_fraction']:+6.2f}  directly driven "
              f"{100 * r['direct_fraction']:+6.2f}  network-driven "
              f"{100 * r['network_fraction']:+6.2f}  peak at "
              f"{r['peak_offset_from_e1_um']:.0f} um from E1")

    (RESULTS / "v2_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'v2_validation.json'}")


if __name__ == "__main__":
    main()
