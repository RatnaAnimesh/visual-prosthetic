"""Check that every number quoted in the manuscript is what the code produces.

A manuscript drifts from its codebase quietly.  This asserts the specific
numbers in paper/manuscript_v2.tex against the regenerated results, so a
stale claim fails loudly instead of surviving into submission.  Labels give
the section of the manuscript the claim appears in.

Tolerances are half a unit in the last digit the manuscript quotes, so a
claim passes only if it rounds to what the code produces.  Range claims
("between 0.32 and 0.84") are checked at both ends, and qualitative claims
("at every current", "in every variant") are checked as booleans over the
whole sweep they describe.  The tables are generated from the same results
by paper/v2_tables.py, and the last check confirms that the copies the
manuscript \\input{}s are current.

Run after prosthesis/run_all.py and paper/v2_tables.py:
    python3 paper/verify_claims.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


def load(name):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"missing {p}; run prosthesis/run_all.py first")
    return json.loads(p.read_text())


CHECKS = []


def check(label, got, want, tol, unit=""):
    ok = got is not None and abs(got - want) <= tol + 1e-12
    CHECKS.append((label, got, want, tol, unit, ok))


def true(label, cond):
    check(label, float(bool(cond)), 1.0, 0.0)


def pct1(label, frac, want):
    """A percentage quoted to one decimal."""
    check(label, 100 * frac, want, 0.05, "%")


def pct0(label, frac, want):
    """A percentage quoted as a whole number."""
    check(label, 100 * frac, want, 0.5, "%")


# ---------------------------------------------------------------------------

def structure():
    """Abstract, 2.1, 2.3 and 3.1: where the nonlinearity sits."""
    info = json.loads((ROOT / "paper" / "figures" /
                       "_figure_info.json").read_text())
    check("2.1 Dynaphos departure from superposition",
          info["dynaphos_residual"] * 1e9, 3.7, 0.05, "e-9")
    true("abstract Dynaphos below 1e-9 of... (3.7e-9 < 1e-8)",
         info["dynaphos_residual"] < 1e-8)

    d = load("phase1_circuit.json")
    sw = d["separation_sweep"]
    true("3.1 horizontal spread adds nothing (< 1e-9) at every separation",
         max(abs(r["max_dev"]) for r in sw["horizontal only"]) < 1e-9)
    true("3.1 linear current spread adds nothing",
         max(abs(r["max_dev"]) for r in sw["circuit off"]) < 1e-9)
    # ... and at every horizontal gain quoted (0.3-1.2), 400-2400 um
    from prosthesis.cortex_circuit import CircuitSpatial
    worst = 0.0
    for gain in (0.3, 0.6, 1.2):
        m = CircuitSpatial(xrange=(-3.2, -0.4), yrange=(-0.2, 2.0),
                           xystep=0.05, horiz_gain=gain, norm_gain=0.0)
        m.build()
        for sep in (400.0, 800.0, 1600.0, 2400.0):
            xe = np.array([20000.0, 20000.0 + sep])
            ye = np.array([-5000.0, -5000.0])
            g = m._ensure_cortical_grid(xe, ye)
            fa = m._activation_field(np.array([300.0, 0.0]), xe, ye, g)
            fb = m._activation_field(np.array([0.0, 300.0]), xe, ye, g)
            fab = m._activation_field(np.array([300.0, 300.0]), xe, ye, g)
            worst = max(worst, float(np.abs(fab - fa - fb).max()
                                     / np.abs(fa + fb).max()))
    true("3.1 horizontal gains 0.3-1.2, 0.4-2.4 mm: below 1e-9", worst < 1e-9)

    cal = d["calibration"]
    for k, want in (("circuit off", 2.3), ("horizontal only", 2.4),
                    ("normalisation only", 3.7), ("full circuit", 4.0),
                    ("pointwise saturation", 11.7)):
        pct1(f"2.3 calibration error, {k}", cal[k]["mean_abs_err"], want)

    full = sorted(sw["full circuit"], key=lambda r: r["separation_um"])
    pw = sorted(sw["pointwise saturation"], key=lambda r: r["separation_um"])
    pct1("3.3 pooled divisor peak 300 uA @0.4 mm", full[0]["max_dev"], 18.0)
    pct1("3.3 pointwise saturation 300 uA @0.4 mm", pw[0]["max_dev"], 43.8)
    fd = [r["max_dev"] for r in full]
    check("3.10 pooled divisor peaks at", full[int(np.argmax(fd))]
          ["separation_um"], 1600, 0, "um")
    true("3.10 pooled divisor still present at 4.5 mm",
         full[-1]["max_dev"] > 0.01 and full[-1]["separation_um"] > 4400)
    pd = [r["max_dev"] for r in pw]
    true("3.10 pointwise largest at the shortest separation",
         int(np.argmax(pd)) == 0)
    true("3.10 pointwise falls monotonically",
         all(a >= b for a, b in zip(pd, pd[1:])))
    true("3.10 pointwise reaches zero by 3.2 mm",
         all(r["max_dev"] < 5e-4 for r in pw if r["separation_um"] >= 3200))

    e = load("v2_experiments.json")
    P = {(r["amp_ua"], r["separation_um"]): r for r in e["pairs"]}
    pct1("3.1 measured parts peak 300 uA @0.4 mm",
         P[(300.0, 400.0)]["max_dev"], 30.7)
    pct1("3.1 occupancy share alone, 300 uA @0.4 mm",
         P[(300.0, 400.0)]["max_dev_direct"], 30.7)
    true("3.1 network adds facilitation from 1.2 mm on (total above "
         "occupancy share)",
         all(P[(300.0, s)]["signed_total"] > P[(300.0, s)]["signed_direct"]
             for s in (1200.0, 1600.0, 2400.0, 3200.0)))
    pct0("abstract / fig 4 caption peak at 300 uA (31%)",
         P[(300.0, 400.0)]["max_dev"], 31)


def methods_constants():
    """2.2: constants quoted for the measured-parts model."""
    from prosthesis.cortex_ssn import BiophysicalCortex
    m = BiophysicalCortex()
    check("2.2 equivalent rate of a cell driven at 300 Hz", m._rho()[0],
          27, 0.5, "Hz")
    check("2.2 initiation constant K", m.rp.K_ax, 3800, 0, "uA/mm2")
    check("2.2 sheet resolution", m.step, 107.0, 0, "um")
    check("2.2 margin", m.margin, 3200.0, 0, "um")
    g = m._ensure_cortical_grid([20000.0], [-5000.0])
    check("2.2 orientation map period", g["sheet"].p.map_period_um, 1000.0,
          0, "um")
    rE, rI, _ = m.baseline(g)
    check("2.2 spontaneous E rate", float(rE.mean()), 1.4, 0.05, "Hz")
    check("2.2 spontaneous I rate", float(rI.mean()), 5.0, 0.05, "Hz")
    from prosthesis.v2_paper import STEP
    check("2.2 visual-field readout grid", STEP, 0.02, 0, "deg")
    from prosthesis.v2_experiments import VISIBLE_HZ
    check("2.2 'doubled' threshold equals spontaneous E rate", VISIBLE_HZ,
          float(rE.mean()), 0.05, "Hz")


def validation():
    """3.2 and table 3."""
    v = load("v2_validation.json")
    nb, pub = v["network"]["16.4"], v["network"]["published_C16.4"]
    check("3.2 suppression-index gap to Obeid & Miller (max)",
          max(abs(nb["si_e"] - pub["si_e"]), abs(nb["si_i"] - pub["si_i"])),
          0.06, 0.005)
    check("3.2 summation-field gap to Obeid & Miller (max)",
          max(abs(nb["sf_e_deg"] - pub["sf_e_deg"]),
              abs(nb["sf_i_deg"] - pub["sf_i_deg"])), 0.2, 0.05, "deg")
    rec = {r["amp_ua"]: r for r in v["recruitment"]["rows"]}
    check("2.2 cells at 8.5 uA (calibration, 'about five')",
          rec[8.5]["cells_in_plane"], 5.0, 0.6)
    check("3.2 peak recruited fraction at 8.5 uA",
          rec[8.5]["peak_fraction_e"], 0.007, 0.0005)
    check("3.2 recruited cells 5 -> 50 uA (fold)",
          rec[50.0]["cells_in_plane"] / rec[5.0]["cells_in_plane"],
          31.0, 0.5, "x")
    check("3.2 median distance at 5 uA", rec[5.0]["median_distance_um"],
          349, 0.5, "um")
    check("3.2 median distance at 50 uA", rec[50.0]["median_distance_um"],
          372, 0.5, "um")
    inh = {r["amp_ua"]: r for r in v["inhibition"]["rows"]}
    check("2.2 I/E recruitment ratio within 300 um (calibration, 'about 1.1')",
          inh[50.0]["ratio_within_300um"], 1.1, 0.05)
    check("3.2 median distance I at 50 uA",
          inh[50.0]["median_distance_i_um"], 242, 0.5, "um")
    check("3.2 median distance E at 50 uA",
          inh[50.0]["median_distance_e_um"], 372, 0.5, "um")
    vs = [r["ratio_to_sum"] for r in v["visual_plus_stim"]]
    check("3.2 visual+stim both/sum, min", min(vs), 0.79, 0.005)
    check("3.2 visual+stim both/sum, max", max(vs), 0.96, 0.005)
    du = {(r["separation_um"], r["amp_ua"]): r for r in v["dual"]}
    for amp, want in ((6.8, -0.09), (12.7, -0.23), (15.9, -0.32)):
        check(f"3.2 Kim et al. net modulation at {amp} uA",
              100 * du[(485.0, amp)]["net_fraction"], want, 0.005, "%")
    true("3.2 Kim et al. network-driven cells never enhanced",
         max(r["network_fraction"] for r in v["dual"]
             if r["amp_ua"] <= 15.9) < 0)
    for amp, want in ((6.8, -0.10), (12.7, -0.25), (15.9, -0.35)):
        check(f"3.2 Kim et al. net modulation at {amp} uA, their geometry",
              100 * du[(485.0, amp)]["net_fraction_e1_view"], want, 0.005,
              "%")
    ong = {r["amp_ua"]: r for r in v["ongoing"]}
    true("3.2 stimulation suppresses ongoing firing at 100 uA",
         ong[100.0]["deepest_suppression_frac"] < 0)

    s = {r["amp_ua"]: r for r in load("v2_paper_size.json")["rows"]}
    check("3.2 half-max diameter at 100 uA", s[100.0]["halfmax_diameter_deg"],
          0.124, 0.0005, "deg")
    check("3.2 Dynaphos diameter at 100 uA", s[100.0]["dynaphos_diameter_deg"],
          0.126, 0.0005, "deg")
    true("3.2 below 100 uA the half-max diameter stays near 0.12 deg",
         all(abs(s[a]["halfmax_diameter_deg"] - 0.12) < 0.005
             for a in s if a < 100))
    check("3.2 half-max diameter at 600 uA", s[600.0]["halfmax_diameter_deg"],
          0.20, 0.005, "deg")
    check("3.2 Dynaphos diameter at 600 uA", s[600.0]["dynaphos_diameter_deg"],
          0.30, 0.005, "deg")
    true("3.2 grows more slowly than Dynaphos above 100 uA",
         all(s[a]["halfmax_diameter_deg"] < s[a]["dynaphos_diameter_deg"]
             for a in s if a > 100 and s[a]["dynaphos_diameter_deg"]))
    check("3.2 doubled-rate diameter at 50 uA", s[50.0]["visible_diameter_deg"],
          0.11, 0.005, "deg")
    check("3.2 doubled-rate diameter at 600 uA",
          s[600.0]["visible_diameter_deg"], 0.34, 0.005, "deg")
    ratio = [s[a]["visible_diameter_deg"] / s[a]["dynaphos_diameter_deg"]
             for a in s if s[a]["dynaphos_diameter_deg"]]
    check("3.2 doubled-rate / Dynaphos, min", min(ratio), 1.1, 0.05, "x")
    check("3.2 doubled-rate / Dynaphos, max", max(ratio), 1.5, 0.05, "x")
    true("3.2 the two estimates bracket the baseline at every current",
         all(s[a]["halfmax_diameter_deg"] <= s[a]["dynaphos_diameter_deg"]
             <= s[a]["visible_diameter_deg"]
             for a in s if s[a]["dynaphos_diameter_deg"]))


def magnitude():
    """Abstract, 3.3 and 3.4."""
    e = load("v2_experiments.json")
    P = {(r["amp_ua"], r["separation_um"]): r for r in e["pairs"]}
    for amp, want in ((10.0, -0.1), (50.0, -1.0), (100.0, -2.6),
                      (300.0, -8.4)):
        pct1(f"3.3 signed interaction {amp:.0f} uA @0.4 mm",
             P[(amp, 400.0)]["signed_total"], want)
    pct1("3.3 occupancy share, 100 uA @0.4 mm",
         P[(100.0, 400.0)]["signed_direct"], -2.6)
    pct1("3.3 occupancy share, 300 uA @0.4 mm",
         P[(300.0, 400.0)]["signed_direct"], -10.9)
    pct1("3.3 facilitation 300 uA @2.4 mm", P[(300.0, 2400.0)]["signed_total"],
         1.9)
    true("3.3 network facilitation at 300 uA is largest at 2.4 mm",
         max(e["pairs"], key=lambda r: r["signed_total"]
             if r["amp_ua"] == 300.0 else -1)["separation_um"] == 2400.0)
    true("3.3 sign changes between 0.8 and 1.2 mm at 300 uA",
         P[(300.0, 800.0)]["signed_total"] < 0
         < P[(300.0, 1200.0)]["signed_total"])
    amps = sorted({r["amp_ua"] for r in e["pairs"]})
    check("3.3 currents swept, lowest", amps[0], 10.0, 0, "uA")
    check("3.3 currents swept, highest", amps[-1], 600.0, 0, "uA")
    true("3.3 suppressive at <=0.8 mm at every current 10-600 uA",
         all(P[(a, s)]["signed_total"] < 0 for a in amps
             for s in (400.0, 600.0, 800.0)))
    true("3.10 no facilitatory-to-suppressive crossover at any separation",
         all(v is None for v in e["crossover_ua"].values()))
    for s in (1200.0, 1600.0):
        true(f"3.3 at {s/1000:g} mm the sign goes from - to + with current",
             P[(100.0, s)]["signed_total"] < 0 < P[(600.0, s)]["signed_total"])
    pct1("3.3 1.6 mm, 100 uA", P[(100.0, 1600.0)]["signed_total"], -0.4)
    pct1("3.3 1.6 mm, 600 uA", P[(600.0, 1600.0)]["signed_total"], 4.2)
    for a in ("300.0", "600.0"):
        true(f"3.3 monotonic decay at {float(a):.0f} uA",
             e["signature"][a]["monotonic_decay"])
        check(f"3.3 peak at the shortest separation, {float(a):.0f} uA",
              e["signature"][a]["peak_at_um"], 400.0, 0, "um")
    rng = e["interaction_range_um"]
    check("3.3 range to a tenth, min (100-600 uA)",
          min(rng.values()) / 1000, 2.4, 0.05, "mm")
    check("3.3 range to a tenth, max (100-600 uA)",
          max(rng.values()) / 1000, 2.6, 0.05, "mm")
    true("abstract 'falls to a tenth by 2.6 mm'", max(rng.values()) <= 2600)

    S = e["sensitivity"]
    col = lambda k: [h[k] for h in S.values()]
    check("3.4 nineteen variants", len(S), 19, 0)
    pct1("3.4 300 uA @0.4 mm, least suppressive",
         max(col("signed_300ua_0.4mm")), -0.9)
    pct1("3.4 300 uA @0.4 mm, most suppressive",
         min(col("signed_300ua_0.4mm")), -21.1)
    true("3.4 300 uA @0.4 mm suppressive in every variant",
         max(col("signed_300ua_0.4mm")) < 0)
    pct1("3.4 peak range, min", min(col("peak_300ua_0.4mm")), 10.7)
    pct1("3.4 peak range, max", max(col("peak_300ua_0.4mm")), 48.9)
    true("3.4 peak at the shortest separation in every variant",
         all(h["peak_300ua_0.4mm"] == max(h[f"peak_300ua_{d}mm"]
                                         for d in ("0.4", "0.8", "1.6", "2.4"))
             for h in S.values()))
    check("3.4 monotonic decay in 18 of 19", sum(col("monotonic_decay_300ua")),
          18, 0)
    b14 = S["background 14 (busier cortex)"]
    true("3.4 the exception is the busier background",
         not b14["monotonic_decay_300ua"])
    true("3.4 busier background: rise from 1.6 to 2.4 mm",
         b14["peak_300ua_2.4mm"] > b14["peak_300ua_1.6mm"])
    check("3.4 that rise is about a sixth of the main peak",
          b14["peak_300ua_2.4mm"] / b14["peak_300ua_0.4mm"], 1 / 6, 0.02)
    check("3.4 16 electrodes 300 uA, min",
          min(col("efficiency_16ch_0.4mm_300ua")), 0.32, 0.005)
    check("3.4 16 electrodes 300 uA, max",
          max(col("efficiency_16ch_0.4mm_300ua")), 0.84, 0.005)
    true("3.4 16 electrodes 300 uA below the sum in every variant",
         max(col("efficiency_16ch_0.4mm_300ua")) < 1)
    pct1("3.4 100 uA @0.4 mm, min", min(col("signed_100ua_0.4mm")), -8.2)
    pct1("3.4 100 uA @0.4 mm, max", max(col("signed_100ua_0.4mm")), 9.8)
    pct1("3.4 / 4.2 no direct inhibitory recruitment, 100 uA @0.4 mm",
         S["inhibitory sites x0 (none)"]["signed_100ua_0.4mm"], -7.6)
    pct1("3.4 / 4.2 three times the inhibitory density, 100 uA @0.4 mm",
         S["inhibitory sites x3"]["signed_100ua_0.4mm"], 9.8)
    true("3.4 inhibitory density is the largest single lever at 100 uA",
         max(S, key=lambda k: abs(S[k]["signed_100ua_0.4mm"]
                                  - S["baseline"]["signed_100ua_0.4mm"]))
         .startswith("inhibitory sites"))
    pct0("4.4 human-scaled peak 300 uA @0.4 mm",
         S["human-scaled lengths (x1.63)"]["peak_300ua_0.4mm"], 21)
    true("4.4 human-scaled turns 1.6 mm facilitation into suppression",
         S["human-scaled lengths (x1.63)"]["signed_300ua_1.6mm"] < 0
         < S["baseline"]["signed_300ua_1.6mm"])
    true("4.4 human-scaled keeps the shape",
         S["human-scaled lengths (x1.63)"]["monotonic_decay_300ua"])


def human():
    """Abstract, 3.5 and table 2."""
    b = load("v2_paper_bosking.json")
    R = {(r["amp_ua"], r["separation_um"]): r for r in b["rows"]}
    amps = sorted({r["amp_ua"] for r in b["rows"]})
    check("3.6 sweep from 0.1 mA", amps[0], 100.0, 0, "uA")
    check("3.6 sweep to 4 mA", amps[-1], 4000.0, 0, "uA")
    pct1("3.6 4.11 mm, at most 1.0% up to 600 uA",
         max(R[(a, 4110.0)]["max_dev"] for a in amps if a <= 600), 1.0)
    pct1("3.6 4.11 mm at 1 mA", R[(1000.0, 4110.0)]["max_dev"], 2.3)
    pct1("3.6 / abstract 4.11 mm at 4 mA", R[(4000.0, 4110.0)]["max_dev"], 6.0)
    true("abstract 'at most 6%' at every human separation and current",
         max(R[(a, s)]["max_dev"] for a in amps
             for s in (4110.0, 9570.0, 13470.0, 26360.0)) < 0.0605)
    true("3.6 signed at 4.11 mm facilitatory from 1 mA on",
         all(R[(a, 4110.0)]["signed_total"] > 0 for a in amps if a >= 1000)
         and all(R[(a, 4110.0)]["signed_total"] < 0 for a in amps if a < 1000))
    pct1("3.6 signed at 4.11 mm never exceeds",
         max(R[(a, 4110.0)]["signed_total"] for a in amps), 1.7)
    true("3.6 below 0.5% at 9.57 mm and beyond, every current",
         max(R[(a, s)]["max_dev"] for a in amps
             for s in (9570.0, 13470.0, 26360.0)) < 0.005)
    check("3.6 at most 0.21% at 13.47 mm",
          100 * max(R[(a, 13470.0)]["max_dev"] for a in amps), 0.21, 0.005,
          "%")
    rng = b["range_to_tenth_um"]
    check("3.6 range to a tenth at 300 uA", rng["300.0"] / 1000, 2.5, 0.05,
          "mm")
    check("3.6 range to a tenth at 4 mA", rng["4000.0"] / 1000, 4.5, 0.05,
          "mm")
    true("3.6 range grows with current from 0.3 mA",
         all(rng[f"{a:.1f}"] < rng[f"{c:.1f}"] for a, c in
             zip(amps[1:], amps[2:])))

    p5 = load("phase5_scale.json")
    sx = np.array(p5["separations_um"], float)
    dv = np.array(p5["by_amplitude"]["4000.0"]["deviations"])
    pct0("3.6 / 3.10 pooled divisor at 4 mA, 4.11 mm",
         float(np.interp(4110.0, sx, dv)), 39)
    check("3.10 pooled range to a tenth at 0.3 mA",
          p5["by_amplitude"]["300.0"]["decay10_um"] / 1000, 4.4, 0.05, "mm")
    check("3.10 pooled range to a tenth at 4 mA",
          p5["by_amplitude"]["4000.0"]["decay10_um"] / 1000, 8.8, 0.05, "mm")

    far = p5["far_pool"]
    sig = sorted({r["sigma_um"] for r in far["rows"]})
    gains = sorted({r["relative_gain"] for r in far["rows"]})
    true("3.10 far pool searched 4-12 mm wide", sig[0] == 4000 and
         sig[-1] == 12000)
    true("3.10 far pool gain from a tenth to 30 times the near pool's",
         gains[0] == 0.1 and gains[-1] == 30)
    check("3.10 far pool at", far["amp_ua"], 4000.0, 0, "uA")
    true("3.10 far pool requirements as stated",
         far["requirements"] == {"4.11": [">", 0.30], "13.47": [">", 0.15],
                                 "26.36": ["<", 0.03]})
    check("3.10 far pool settings tried", len(far["rows"]), 24, 0)
    pct1("3.10 far pool no stronger than near, 13.47 mm, min",
         far["min_at_13.47_if_far_weaker"], 1.3)
    pct1("3.10 far pool no stronger than near, 13.47 mm, max",
         far["max_at_13.47_if_far_weaker"], 3.0)
    true("3.10 exactly one setting meets all three, 8 mm at 30x",
         [tuple(x) for x in far["settings_meeting_all"]] == [(8000.0, 30.0)])
    best = next(r for r in far["rows"] if r["meets_all"])["deviations"]
    pct0("3.10 that setting at 4.11 mm", best["4.11"], 40)
    pct0("3.10 that setting at 13.47 mm", best["13.47"], 17)
    pct1("3.10 that setting at 26.36 mm", best["26.36"], 0.6)

    rp = load("v2_paper_repulsion.json")
    bio = rp["measured_parts"]
    pct1("3.6 repulsion at 600 uA, largest",
         max(r["repulsion"] for r in bio if r["amp_ua"] == 600.0), 9.3)
    pct1("3.6 repulsion at 2-4 mA, largest",
         max(r["repulsion"] for r in b["rows"] if r["amp_ua"] >= 2000), 16.8)
    far = [r for r in b["rows"] + bio if r["separation_um"] > 2400]
    true("3.6 beyond 2.4 mm, shift under 1% up to 1 mA",
         all(abs(r["repulsion"]) < 0.01 for r in far if r["amp_ua"] <= 1000))
    true("3.6 beyond 2.4 mm, shift under 1.5% at 2 and 4 mA",
         all(abs(r["repulsion"]) < 0.015 for r in far))
    true("3.6 shift under 1% at 4.11 mm and beyond, every current",
         all(abs(r["repulsion"]) < 0.01 for r in b["rows"]
             if r["separation_um"] >= 4110))
    pool = rp["pooled_divisor"]

    def extreme(rows, key, amp, fn=min):
        r = fn((r for r in rows if r["amp_ua"] == amp
                and r[key] is not None), key=lambda r: r[key])
        return r[key], r["separation_um"]

    v, at = extreme(pool, "repulsion", 100.0)
    pct1("3.6 pooled centroids pulled together at 100 uA", -v, 10.2)
    check("3.6 ... at", at / 1000, 0.8, 0, "mm")
    v, at = extreme(pool, "repulsion", 300.0)
    pct1("3.6 pooled centroids pulled together at 300 uA", -v, 6.3)
    check("3.6 ... at", at / 1000, 1.2, 0, "mm")
    pk = rp["peaks_pooled_divisor"]
    ok = [r for r in pk if r["peak_shift"] is not None]
    pct1("3.6 pooled maxima pulled together, largest",
         -min(r["peak_shift"] for r in ok), 11.6)
    true("3.6 ... where the peaks are closest, 1.8 to 2.2 mm",
         all(r["peak_shift"] < 0 for r in ok if r["amp_ua"] >= 300
             and r["separation_um"] == min(x["separation_um"] for x in ok
                                            if x["amp_ua"] == r["amp_ua"]))
         and min(ok, key=lambda r: r["peak_shift"])["separation_um"] <= 2200)
    true("3.6 pooled maxima pushed apart from 2.6 mm on (300, 600 uA)",
         all(r["peak_shift"] > 0 for r in ok if r["amp_ua"] >= 300
             and r["separation_um"] >= 2600))
    pct1("3.6 pooled maxima pushed apart at 300 uA, up to",
         extreme(pk, "peak_shift", 300.0, max)[0], 2.1)
    pct1("3.6 pooled maxima pushed apart at 600 uA, up to",
         extreme(pk, "peak_shift", 600.0, max)[0], 3.7)
    bpk = [r for r in rp["peaks_measured_parts"] if r["peak_shift"] is not None]
    check("3.6 peak estimator starts at",
          min(r["separation_um"] for r in bpk + ok) / 1000, 1.8, 0, "mm")
    big = max(bpk, key=lambda r: abs(r["peak_shift"]))
    pct1("3.6 measured parts moves the maxima by at most",
         abs(big["peak_shift"]), 3.3)
    true("3.6 ... toward each other", big["peak_shift"] < 0)
    true("3.6 ... the largest shifts at 1.8 and 2.2 mm, toward each other",
         all(r["peak_shift"] < 0 for r in bpk if r["separation_um"] <= 2200)
         and all(abs(r["peak_shift"]) < 0.01 for r in bpk
                 if r["separation_um"] > 2200))


def arrays():
    """Abstract, 3.6."""
    e = load("v2_experiments.json")
    C = {(r["amp_ua"], r["pitch_um"], r["n"]): r for r in e["channels"]}
    ns = sorted({r["n"] for r in e["channels"]})
    wide = [C[(100.0, p, n)]["efficiency"] for p in (800.0, 1600.0)
            for n in ns]
    true("3.7 100 uA, 0.8 and 1.6 mm pitch, within 5% of one to N=100",
         max(abs(1 - x) for x in wide) <= 0.05 and ns[-1] == 100)
    tight = [C[(100.0, 400.0, n)]["efficiency"] for n in ns if n >= 9]
    check("3.7 100 uA, 0.4 mm pitch, settles near 0.9",
          float(np.mean(tight)), 0.9, 0.05)
    check("3.7 300 uA, 0.4 mm, N=100, ratio to sum",
          C[(300.0, 400.0, 100)]["efficiency"], 0.46, 0.005)
    check("3.7 / abstract 300 uA, 0.4 mm, N=100, times one electrode",
          C[(300.0, 400.0, 100)]["relative_to_one"], 46, 0.5, "x")
    check("3.7 300 uA, 1.6 mm, N=100, ratio to sum",
          C[(300.0, 1600.0, 100)]["efficiency"], 1.16, 0.005)
    true("3.7 never falls as 1/N (ratio at N=100 within 25% of N=16)",
         all(C[(a, p, 100)]["efficiency"] > 0.75 * C[(a, p, 16)]["efficiency"]
             for a in (100.0, 300.0) for p in (400.0, 800.0, 1600.0)))
    ch = {(r["pitch_um"], r["n"]): r for r in e["charge"]}
    check("3.7 delta-k at 0.8 mm, largest to N=64",
          max(ch[(800.0, n)]["delta_k"] for n in (4, 16, 64)), 0.03, 0.005)
    check("3.7 amplitude to hold @0.4 mm N=16",
          ch[(400.0, 16)]["amp_ua"] / 1000, 0.78, 0.005, "mA")
    check("3.7 amplitude to hold @0.4 mm N=64",
          ch[(400.0, 64)]["amp_ua"] / 1000, 1.68, 0.005, "mA")
    check("3.7 delta-k @0.4 mm N=16", ch[(400.0, 16)]["delta_k"], 0.8, 0.05)
    check("3.7 delta-k @0.4 mm N=64", ch[(400.0, 64)]["delta_k"], 1.5, 0.05)
    check("3.7 reference current", ch[(400.0, 16)]["base_ua"], 300.0, 0,
          "uA")

    c = load("phase8_channels.json")
    rel = {p: c["ceilings"]["circuit"][p]["relative_total"][-1]
           for p in ("400.0", "800.0", "1600.0")}
    check("3.7 / 3.10 pooled N=100 @0.4 mm", rel["400.0"], 3.9, 0.05, "x")
    check("3.10 pooled N=100 @0.8 mm", rel["800.0"], 10.5, 0.05, "x")
    check("3.10 pooled N=100 @1.6 mm", rel["1600.0"], 33.4, 0.05, "x")
    amps = {r["n"]: r["amp_ua"] for r in c["charge"]["800.0"]}
    check("3.10 pooled amplitude to hold @0.8 mm N=4", amps[4] / 1000, 1.3,
          0.05, "mA")
    check("3.10 pooled amplitude to hold @0.8 mm N=16", amps[16] / 1000, 5.1,
          0.05, "mA")
    check("3.10 pooled amplitude to hold @0.8 mm N=64", amps[64] / 1000, 46,
          0.5, "mA")
    dk = {r["n"]: r.get("delta_k") for r in c["shannon"]["by_pitch"]["800.0"]}
    check("3.10 pooled delta-k @0.8 mm N=16", dk[16], 2.5, 0.05)
    check("3.10 pooled delta-k @0.8 mm N=64", dk[64], 4.4, 0.05)
    a400 = {r["n"]: r["amp_ua"] for r in c["charge"]["400.0"]}
    check("3.7 pooled amplitude @0.4 mm N=16", a400[16] / 1000, 5.5, 0.05,
          "mA")
    true("3.7 pooled @0.4 mm N=64 unreachable by 200 mA", a400[64] is None)


def sequence():
    """3.8 and the pooled divisor's dynamics in 3.9."""
    e = load("v2_experiments.json")
    seq = [x["second_over_solo"] for res in e["sequence"].values()
           for r in res for x in r["rows"]]
    check("3.8 second phosphene within 3.5% of its brightness alone",
          100 * max(abs(1 - x) for x in seq), 3.5, 0.05, "%")
    cfg = {(r["separation_um"], r["amp_ua"]) for res in e["sequence"].values()
           for r in res}
    true("3.8 at 0.8 and 1.6 mm, 100 and 300 uA",
         cfg == {(s, a) for s in (800.0, 1600.0) for a in (100.0, 300.0)})
    delays = {x["delay_ms"] for res in e["sequence"].values() for r in res
              for x in r["rows"]}
    true("3.8 delays from 0 to 300 ms", min(delays) == 0 and
         max(delays) == 300)
    true("3.8 with and without slow inhibition",
         set(e["sequence"]) == {"fast inhibition only", "30% slow inhibition"})

    q = load("phase9_sequence.json")
    check("3.8 / 3.10 pooled divisor's strongest dimming",
          q["max_predicted_dimming_pct"], 39, 0.5, "%")
    rows = q["sequence_bias"]["600.0"]
    worst = max(rows, key=lambda r: 1 - r["ratio_second_over_first"])
    check("3.10 dimming largest near", worst["delay_ms"], 25, 0, "ms")
    true("3.10 dimming gone by 400 ms",
         all(abs(1 - r["ratio_second_over_first"]) < 0.005
             for rr in q["sequence_bias"].values() for r in rr
             if r["delay_ms"] >= 400))
    true("3.10 pooled divisor never makes the later phosphene brighter",
         max(r["ratio_second_over_first"] for rr in
             q["sequence_bias"].values() for r in rr) <= 1.005)

    g = load("phase7_regimes.json")
    surf = g["surface"]

    def cross(label, sep):
        return next(r["crossover_ua"] for r in surf[label]
                    if abs(r["separation_um"] - sep) < 1)

    check("3.10 pooled crossover @0.4 mm", cross("full circuit", 400), 223,
          0.5, "uA")
    check("3.10 pooled crossover @3.2 mm", cross("full circuit", 3200), 96,
          0.5, "uA")
    check("3.10 pointwise crossover @0.4 mm",
          cross("pointwise saturation", 400), 144, 0.5, "uA")
    check("3.10 pointwise crossover @2.4 mm",
          cross("pointwise saturation", 2400), 969, 0.5, "uA")
    fc = [r["crossover_ua"] for r in surf["full circuit"]]
    true("3.10 pooled crossover falls with separation",
         all(a > b for a, b in zip(fc, fc[1:])))
    allv = [v for r in surf["full circuit"] for v in r["signed_total"]]
    pct0("3.10 pooled facilitation at the lowest drives", max(allv), 70)
    pct0("3.10 pooled suppression at the highest drives", min(allv), -38)

    sc = g["scaling"]
    wm = [r["ratio_window_mean"] for r in sc]
    on = [r["ratio_onset"] for r in sc]
    check("3.10 window-mean slope / tau", float(np.mean(wm)), 1.80, 0.005)
    check("3.10 that ratio across tau (spread)", max(wm) - min(wm), 0.0, 0.02)
    check("3.10 onset-readout slope / tau, every tau", max(abs(x - 1)
                                                          for x in on),
          0.0, 0.005)
    check("3.10 log-linearity, R^2", min(r["r2"] for r in sc), 0.994, 0.0005)
    taus = sorted(r["tau_ms"] for r in sc)
    true("3.10 tau from 25 to 200 ms", taus[0] == 25 and taus[-1] == 200)
    pr = {p["amp_ua"]: p["critical_delay_ms"]
          for p in g["dynamics"]["full circuit tau=50"]["probes"]}
    true("3.10 no delay needed at or below 120 uA, some by 160 uA",
         all(pr[a] == 0 for a in pr if a <= 120) and pr[160.0] >= 0)
    true("3.10 'below 120-160 uA' (zero at 120 uA in every condition)",
         all(p["critical_delay_ms"] == 0 for d in g["dynamics"].values()
             for p in d["probes"] if p["amp_ua"] <= 120))
    check("3.10 delay at 300 uA, tau 50 ms", pr[300.0], 121, 0.5, "ms")
    check("3.10 delay at 1000 uA, tau 50 ms", pr[1000.0], 186, 0.5, "ms")


def pooled_gain():
    """3.9: the pooled divisor's gain and shape."""
    s = load("phase1_sensitivity.json")["norm_gain_sweep"]
    gains = [r["norm_gain"] for r in s]
    check("3.10 gain range spans 32-fold", max(gains) / min(gains), 32, 0)
    pct0("3.10 peak deviation, min over gains", min(r["peak"] for r in s), 19)
    pct0("3.10 peak deviation, max over gains", max(r["peak"] for r in s), 44)
    locs = [r["local_peak_sep_um"] for r in s]
    check("3.10 interior maximum, min", min(locs) / 1000, 1.2, 0.05, "mm")
    check("3.10 interior maximum, max", max(locs) / 1000, 1.6, 0.05, "mm")
    true("3.10 interior maximum at every gain", all(x > 400 for x in locs))
    hd = [r["half_decay_um"] for r in s]
    check("3.10 half-decay, min", min(hd) / 1000, 2.2, 0.05, "mm")
    check("3.10 half-decay, max", max(hd) / 1000, 2.8, 0.05, "mm")
    true("3.10 above gain 2 the shortest separation overtakes",
         all((r["peak_sep_um"] == 400.0) == (r["norm_gain"] > 2) for r in s))
    pct0("3.10 every gain fits within 4%", max(r["fit_err"] for r in s), 4)

    p6 = load("phase6_controls.json")["separation_sweep"]
    for k in ("full circuit", "normalisation only"):
        r = p6[k]
        true(f"3.10 fine sweep 0.3-6 mm in 150 um steps ({k})",
             r["separations_um"][0] == 300 and r["separations_um"][-1] == 6000
             and r["separations_um"][1] - r["separations_um"][0] == 150)
    check("3.10 fine sweep, full circuit peak", p6["full circuit"]
          ["peak_sep_um"] / 1000, 1.5, 0, "mm")
    check("3.10 fine sweep, normalisation-only peak",
          p6["normalisation only"]["peak_sep_um"] / 1000, 1.65, 0, "mm")


def percepts():
    """2.4, 3.8."""
    p = load("v2_paper_percepts.json")
    check("2.4 / 3.9 decoder R^2, measured-parts map", p["decoder_r2"], 0.66,
          0.005)
    p2 = load("phase2_percepts.json")
    check("2.4 decoder R^2, pooled-divisor map", p2["decoder"]["r2"], 0.66,
          0.005)
    from prosthesis.phase2_percepts import Cortivis, pick_line
    imp = Cortivis()
    xy = np.array([[imp[n].x, imp[n].y] for n in pick_line(imp, 5, 1400.0)])
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1) / 1000
    check("3.9 five electrodes", len(xy), 5, 0)
    true("3.9 the group the results used is this group",
         np.allclose(p["line_separations_um"],
                     np.hypot(*(xy - xy[0]).T)))
    check("3.9 nearest neighbour in the group, min", nn.min(), 0.9, 0.05,
          "mm")
    check("3.9 nearest neighbour in the group, max", nn.max(), 1.8, 0.05,
          "mm")
    check("3.9 current", p["amp_ua"], 300.0, 0, "uA")
    pct1("3.9 / abstract percept change", p["percept_change"], 11.5)
    pct0("abstract percept change (12%)", p["percept_change"], 12)
    pct1("3.9 activation change", p["activation_change"], 18.7)
    pct1("3.9 summed activation", p["activation_signed"], 1.7)
    check("3.9 correlation with the independent prediction",
          p["percept_correlation"], 0.99, 0.005)
    pct0("3.9 RMS contrast falls by", 1 - p["percept_rms_ratio"], 6)
    q = p["pooled_divisor"]
    pct0("3.9 pooled divisor, activation change", q["activation_change"], 32)
    pct0("3.9 pooled divisor, percept change", q["percept_change"], 32)
    pct0("3.9 pooled divisor, summed activation falls by",
         -q["activation_signed"], 30)
    pct0("3.9 pooled divisor, RMS contrast falls by",
         1 - q["percept_rms_ratio"], 29)
    p2 = load("phase2_percepts.json")["pattern"]["circuit"]
    true("3.9 phase 2 and this readout agree on the pooled divisor",
         abs(p2["percept_change"] - q["percept_change"]) < 0.005)

    r = load("v2_paper_reach.json")
    check("2.5 linear fit at", r["amp_ua"], 400.0, 0, "uA")
    check("2.5 deliverable targets, worst linear fit", min(r["deliverable"]),
          0.999, 0.0005)
    check("2.5 thin shapes, worst linear fit", min(r["shapes"].values()),
          0.05, 0.005)
    check("2.5 thin shapes, best linear fit", max(r["shapes"].values()),
          0.21, 0.005)


def encoder():
    """Abstract, 2.5, 3.8, 3.9, 4.3."""
    q = load("v2_paper_encoder.json")
    m, R = q["mean"], list(q["results"].values())
    check("2.5 targets", len(R), 6, 0)
    check("2.5 electrodes", len(q["electrodes"]), 24, 0)
    check("2.5 re-optimisation iterations", q["maxiter_full"], 15, 0)
    check("3.9 fidelity programmed as independent, delivered",
          m["indep_on_full"], 0.982, 0.0005)
    check("3.9 fidelity re-optimised under the full model",
          m["full_on_full"], 0.992, 0.0005)
    pct1("3.9 fidelity lost to independence", m["fraction_lost"], 1.0)
    pct0("3.9 / abstract contrast kept", m["contrast_ratio_mismatched"], 91)
    pct0("3.9 contrast kept, lowest target",
         min(r["contrast_ratio_mismatched"] for r in R), 79)
    pct0("3.9 contrast kept, highest target",
         max(r["contrast_ratio_mismatched"] for r in R), 100)
    pct0("3.9 re-optimising for fidelity lowers contrast to",
         m["contrast_ratio_matched"], 75)
    pct0("3.9 brightness objective reaches", m["contrast_ratio_mse"], 95)
    check("3.9 ... at a fidelity of", m["fidelity_mse"], 0.984, 0.0005)
    pct0("3.9 ... with up to this much more total current",
         max(r["current_mse"] / r["current_indep"] for r in R) - 1, 41)
    check("3.9 ... and this many targets reach the 600 uA bound",
          sum(r["max_amp_mse"] >= 599 for r in R), 1, 0)
    lost = 1 - m["contrast_ratio_mismatched"]
    regained = m["contrast_ratio_mse"] - m["contrast_ratio_mismatched"]
    check("4.3 brightness objective wins back about half the loss",
          regained / lost, 0.5, 0.05)

    o = load("phase4_encoder.json")
    check("3.10 pooled, fidelity programmed as independent",
          o["mean_mismatched"], 0.988, 0.0005)
    check("3.10 pooled, fidelity programmed for the divisor",
          o["mean_matched"], 0.995, 0.0005)
    pct0("3.10 pooled, contrast kept", o["mean_contrast_ratio"], 38)
    pct0("3.10 pooled, brightness objective recovers",
         o["mean_contrast_ratio_mse"], 44)
    check("3.10 pooled, ... at a fidelity of", o["mean_fidelity_mse"], 0.949,
          0.0005)
    below = [r for r in o["results"].values() if r["max_amp_mse"] < 599]
    check("3.10 pooled, targets stopping short of the bound", len(below), 4, 0)
    true("3.10 pooled, ... with at most half the intended contrast",
         all(r["contrast_ratio_mse"] <= 0.5 for r in below))


def power():
    """4.2."""
    w = load("v2_paper_power.json")
    rows = {r["sem_pp"]: r for r in w["rows"]}
    check("4.2 power at 10 pp",
          min(rows[10.0]["power_if_measured"], rows[10.0]["power_if_pooled"]),
          0.96, 0.005)
    check("4.2 power at 20 pp",
          min(rows[20.0]["power_if_measured"], rows[20.0]["power_if_pooled"]),
          0.80, 0.005)
    check("4.2 nine separations", len(w["separations_um"]), 9, 0)
    check("4.2 tolerance, all separations", w["reduced_protocols"]["all"], 20,
          0, "pp")
    check("4.2 tolerance, three separations",
          w["reduced_protocols"]["0.4/1.6/3.2 mm"], 20, 0, "pp")
    dr = {r["drift"]: r["power_if_pooled"] for r in w["drift_confound"]}
    check("4.2 no drift", dr[0.0], 0.82, 0.005)
    check("4.2 30% drift", dr[0.3], 0.74, 0.005)
    check("4.2 70% drift", dr[0.7], 0.60, 0.005)
    check("4.2 trials per separation at s = 60", (60 / 20) ** 2, 9, 0)


def inhibition_sweep():
    """3.5 and table 5: the inhibitory density against both mouse datasets."""
    d = load("v2_paper_inhibition.json")
    rows = {r["mult"]: r for r in d["rows"]}
    cr = d["crossings"]
    kim_amps, seps = (6.8, 12.7, 15.9), (370.0, 485.0, 558.0, 850.0)
    views = ("net_fraction", "net_fraction_e1_view")

    def kim(r, sep, amp):
        return next(x for x in r["kim"]
                    if x["separation_um"] == sep and x["amp_ua"] == amp)

    check("3.5 I/E ratio at calibration (20 uA)",
          rows[1.0]["ie_ratio_300um_20ua"], 1.13, 0.005)
    check("3.5 I/E ratio at x1.1", rows[1.1]["ie_ratio_300um_20ua"], 1.24,
          0.005)
    true("3.5 network-driven cells at 485 um enhanced at every current "
         "from x1.1",
         all(kim(r, 485.0, a)["network_fraction"] > 0 for r in d["rows"]
             if r["mult"] >= 1.1 for a in kim_amps))
    true("3.5 ... and at every separation from x1.4",
         all(kim(r, s, a)["network_fraction"] > 0 for r in d["rows"]
             if r["mult"] >= 1.4 for s in seps for a in kim_amps)
         and any(kim(rows[1.3], s, a)["network_fraction"] <= 0
                 for s in seps for a in kim_amps))
    check("3.5 I/E ratio at x1.4", rows[1.4]["ie_ratio_300um_20ua"], 1.58,
          0.005)
    true("3.5 ... and suppressed at x1",
         all(kim(rows[1.0], s, a)["network_fraction"] < 0
             for s in seps for a in kim_amps))
    true("3.5 directly driven cells attenuated at every density",
         all(kim(r, s, a)["direct_fraction"] < 0 for r in d["rows"]
             for s in seps for a in kim_amps))
    k127 = [cr[f"kim_12.7ua_{s:.0f}um_{v}"] for s in seps for v in views]
    check("3.5 net turns positive at 12.7 uA, lowest density",
          min(c["mult"] for c in k127), 1.22, 0.005, "x")
    check("3.5 ... highest density", max(c["mult"] for c in k127), 1.47,
          0.005, "x")
    check("3.5 ... lowest I/E ratio", min(c["ie_ratio"] for c in k127),
          1.37, 0.005)
    check("3.5 ... highest I/E ratio", max(c["ie_ratio"] for c in k127),
          1.65, 0.005)
    true("3.5 ... at all three currents together (within 0.05x)",
         all(abs(cr[f"kim_{a}ua_{s:.0f}um_{v}"]["mult"]
                 - cr[f"kim_12.7ua_{s:.0f}um_{v}"]["mult"]) < 0.05
             for a in kim_amps for s in seps for v in views))
    c100 = cr["signed_100ua_0.4mm"]
    check("3.5 100 uA, 0.4 mm turns facilitatory at", c100["mult"], 1.47,
          0.005, "x")
    check("3.5 ... I/E ratio there", c100["ie_ratio"], 1.65, 0.005)
    true("3.5 300 uA, 0.4 mm suppressive at every density",
         cr["signed_300ua_0.4mm"]["mult"] is None
         and all(r["signed_300ua_0.4mm"] < 0 for r in d["rows"]))
    true("3.5 datasets disagree by at least a fifth in the I/E ratio",
         min(c["ie_ratio"] for c in k127)
         / rows[1.0]["ie_ratio_300um_20ua"] >= 1.2)
    ks = [r["mult"] for r in d["rows"]]
    at_bound = np.interp(min(c["mult"] for c in k127), ks,
                         [r["signed_100ua_0.4mm"] for r in d["rows"]])
    pct1("3.5 100 uA interaction at Kim's lower bound", at_bound, -1.5)
    pct1("abstract 100 uA at calibration", rows[1.0]["signed_100ua_0.4mm"],
         -2.6)
    true("3.5 net grows from 6.8 to 15.9 uA at every density, separation "
         "and view, and once positive stays positive",
         all(abs(kim(r, s, 15.9)[v]) >= abs(kim(r, s, 12.7)[v])
             >= abs(kim(r, s, 6.8)[v])
             and (kim(r, s, 12.7)[v] <= 0 or kim(r, s, 15.9)[v] > 0)
             for r in d["rows"] for s in seps for v in views))
    check("3.5 densities from half to three times", min(ks), 0.5, 0, "x")
    check("3.5 ... three times", max(ks), 3.0, 0, "x")


def awake():
    """2.3 and 3.6: the model against raw cells from awake cortex."""
    c = load("v2_awake_cells.json")
    check("2.3 excitatory cells", c["n_cells"] - c["n_inhibitory"], 6162, 0)
    check("2.3 inhibitory cells", c["n_inhibitory"], 532, 0)
    check("3.6 cells in all", c["n_cells"], 6694, 0)
    pf = c["published_fractions"]
    pct1("2.3 excitatory modulated at 40 uA", pf["40"]["e"], 49.0)
    pct1("2.3 inhibitory modulated at 40 uA", pf["40"]["i"], 54.0)
    pct1("2.3 inhibitory at 10 uA", pf["10"]["i"], 11.1)
    pct1("2.3 excitatory at 5 uA", pf["5"]["e"], 2.7)
    check("2.3 fitted site from the code's convention (mean)",
          float(np.mean([x["from_code_um"] for x in c["sites"]])), 64, 0.5, "um")

    f, prof = c["fits"], c["profiles"]["sig"]
    fm = c["field_medians"]
    check("3.2 imaged field, model I median at 50 uA", fm["calibration_i50"], 206, 0.5, "um")
    check("3.2 imaged field, model E median at 50 uA", fm["calibration_e50"], 222, 0.5, "um")
    check("3.2 imaged field, data I median at 50 uA", fm["data_i50"], 220, 0.5, "um")
    check("3.2 imaged field, data E median at 50 uA", fm["data_e50"], 222, 0.5, "um")

    def obs(a, b=0):
        k, n = prof[f"e{a}"][b]
        return k / n

    pct1("3.6 E responding within 100 um at 10 uA", obs(10), 18.4)
    pct1("3.6 ... calibration predicts", f["sig_paper"]["binned_prediction"]["e10"][0], 2.4)
    pct1("3.6 E responding within 100 um at 50 uA", obs(50), 79.6)
    pct1("3.6 ... calibration predicts", f["sig_paper"]["binned_prediction"]["e50"][0], 10.7)
    true("3.6 shortfall at every distance out to 500 um, 10-50 uA",
         all(obs(a, b) > f["sig_paper"]["binned_prediction"][f"e{a}"][b]
             for a in (10, 15, 20, 25, 30, 40, 50) for b in range(4)))
    check("3.6 AIC drop, densities refitted",
          f["sig_paper"]["aic"] - f["sig_densities"]["aic"], 33400, 50)
    check("3.6 refitted excitatory density", f["sig_densities"]["s_e"], 104, 0.5)
    check("3.6 refitted inhibitory density", f["sig_densities"]["s_i"], 79, 0.5)
    check("3.6 AIC drop, local term", f["sig_densities"]["aic"] - f["sig_local"]["aic"],
          990, 5)
    check("3.6 local reach", f["sig_local"]["L_um"], 88, 0.5, "um")
    check("3.6 E density over calibration", f["sig_local"]["s_e"] / 9, 9.1, 0.05, "x")
    check("3.6 I density over calibration", f["sig_local"]["s_i"] / 6, 12.8, 0.05, "x")
    check("3.6 short-latency E over calibration", f["early_local"]["s_e"] / 9, 6.8, 0.05, "x")
    check("3.6 short-latency I over calibration", f["early_local"]["s_i"] / 6, 7.7, 0.05, "x")
    check("3.6 short-latency core", f["early_local"]["L_um"], 90, 0.5, "um")

    r = c["site_ratio"]
    for key, want in (("sig20", (1.47, 1.00, 1.98)), ("early20", (1.23, 0.80, 1.72))):
        check(f"3.6 I/E {key}", r[key]["ratio"], want[0], 0.005)
        check(f"3.6 I/E {key} low", r[key]["ci"][0], want[1], 0.005)
        check(f"3.6 I/E {key} high", r[key]["ci"][1], want[2], 0.005)
    check("3.6 I/E sig10", r["sig10"]["ratio"], 1.72, 0.005)
    check("3.6 I/E early10", r["early10"]["ratio"], 1.94, 0.005)
    hi = [r[f"{lab}{a}"]["ratio"] for lab in ("sig", "early") for a in (40, 50)]
    check("3.6 I/E at 40-50 uA, lowest", min(hi), 0.89, 0.005)
    check("3.6 I/E at 40-50 uA, highest", max(hi), 1.13, 0.005)
    true("3.6 1.37-1.65 and the calibration's 1.13 inside both 20 uA intervals",
         all(r[k]["ci"][0] <= 1.13 and r[k]["ci"][1] >= 1.65 for k in ("sig20", "early20")))

    k = load("v2_awake_kim.json")
    by = {b["mult"]: b for b in k["by_mult"]}

    def net(m, sep, key="net_fraction"):
        return [x[key] for x in by[m]["rows"] if x["separation_um"] == sep]

    for m, ratio, want in ((1.0, 1.20, (-0.50, -1.29, -1.80)),
                           (1.25, 1.50, (0.39, 0.21, -0.13)),
                           (1.5, 1.79, (2.02, 2.44, 2.11))):
        check(f"3.6 dense x{m} ratio", by[m]["ie_ratio_20ua"], ratio, 0.005)
        for a, got, w in zip((6.8, 12.7, 15.9), net(m, 485.0), want):
            check(f"3.6 dense x{m} net at 485 um, {a} uA", 100 * got, w, 0.005, "%")
    seps = (370.0, 485.0, 558.0, 850.0)
    avg = np.mean([net(1.25, s_) for s_ in seps], axis=0)
    for a, got, w in zip((6.8, 12.7, 15.9), avg, (0.66, 0.64, 0.34)):
        check(f"3.6 dense x1.25 net over separations, {a} uA", 100 * got, w, 0.005, "%")
    true("3.6 x1.25 network-driven enhanced and direct attenuated everywhere",
         all(x["network_fraction"] > 0 and x["direct_fraction"] < 0
             for x in by[1.25]["rows"]))
    for key, want in (("direct_fraction", (-0.64, -1.54, -2.09)),
                      ("network_fraction", (2.67, 3.98, 4.20))):
        for a, got, w in zip((6.8, 12.7, 15.9), net(1.5, 485.0, key), want):
            check(f"3.6 dense x1.5 {key} at {a} uA", 100 * got, w, 0.005, "%")

    h = load("v2_awake_human.json")["rows"]

    def chi(var, add, single=73.5, sigma=100.0):
        return next(q["chi2"] for q in h if q["variant"] == var and q["additive"] == add
                    and q["single_ua"] == single and q["sigma_um"] == sigma)

    at = [q for q in h if q["single_ua"] == 73.5]
    tot = [q for q in at if q["sigma_um"] is None]
    check("3.8 total readout chi2, lowest", min(q["chi2"] for q in tot), 22, 0.5)
    check("3.8 total readout chi2, highest", max(q["chi2"] for q in tot), 32, 0.5)
    check("3.8 total readout pair ratio, lowest", min(q["ratios"]["2"] for q in tot), 0.55, 0.005)
    check("3.8 total readout pair ratio, highest", max(q["ratios"]["2"] for q in tot), 0.63, 0.005)
    check("3.8 smoothing wider than 200 um, lowest chi2",
          min(q["chi2"] for q in h if q["sigma_um"] in (400.0, 800.0)), 13, 0.5)
    check("3.8 sparse, occupancy", chi("sparse", False), 3.2, 0.05)
    check("3.8 sparse, added", chi("sparse", True), 3.8, 0.05)
    check("3.8 dense, occupancy", chi("awake_all", False), 3.5, 0.05)
    check("3.8 short-latency, occupancy", chi("awake_fast", False), 3.2, 0.05)
    check("3.8 dense, added, best readout",
          min(q["chi2"] for q in at if q["variant"] == "awake_all" and q["additive"]), 15.3, 0.05)
    check("3.8 short-latency, added, best readout",
          min(q["chi2"] for q in at if q["variant"] == "awake_fast" and q["additive"]), 8.6, 0.05)
    true("3.8 dense, added, rejected at 150 uA too",
         min(q["chi2"] for q in h if q["variant"] == "awake_all" and q["additive"]
             and q["single_ua"] == 150.0) > 15.3)
    check("3.8 dense, added, at 35 uA", chi("awake_all", True, 35.0), 2.7, 0.05)
    pair = [q["ratios"]["2"] for q in at if q["variant"] == "awake_all" and q["additive"]
            and q["sigma_um"] is not None]
    true("3.8 added overlap lets a pair match one electrode at about half the current",
         0.5 <= min(pair) and max(pair) <= 0.6)
    sys.path.insert(0, str(ROOT))
    from prosthesis.v2_awake import FERNANDEZ, WEIBULL, chi2
    check("3.8 probability summation slope", WEIBULL, 3.3, 0)
    check("3.8 probability summation chi2",
          chi2({n: n ** (-1 / WEIBULL) for n in (2, 4, 9, 16)}), 2.0, 0.05)
    check("3.8 per-electrode threshold for 16", FERNANDEZ[16][0], 28.1, 0)
    pr = load("v2_awake_pairs.json")["rows"]

    def pq(var, amp, sep, key="signed_total"):
        return next(x[key] for x in pr if x["variant"] == var and x["amp_ua"] == amp
                    and x["separation_um"] == sep)

    for amp, w, pk in ((50.0, -3.0, 28), (100.0, -10.1, 46), (300.0, -23.2, 50)):
        pct1(f"3.6 dense signed at {amp:.0f} uA, 0.4 mm", pq("awake_all", amp, 400.0), w)
        pct0(f"3.6 dense peak at {amp:.0f} uA, 0.4 mm", pq("awake_all", amp, 400.0, "max_dev"), pk)
    for amp, w, pk in ((100.0, -2.0, 10), (300.0, -7.7, 32)):
        pct1(f"3.6 sparse signed at {amp:.0f} uA, 0.4 mm, same grid", pq("sparse", amp, 400.0), w)
        pct0(f"3.6 sparse peak at {amp:.0f} uA, 0.4 mm, same grid",
             pq("sparse", amp, 400.0, "max_dev"), pk)
    seps = sorted({x["separation_um"] for x in pr})
    true("3.6 dense peak largest at the shortest separation, every current",
         all(pq("awake_all", a, 400.0, "max_dev") >= max(pq("awake_all", a, s_, "max_dev")
                                                        for s_ in seps)
             for a in (10.0, 50.0, 100.0, 300.0)))
    true("3.6 dense signed suppressive at 0.4 mm from 50 uA up",
         all(pq("awake_all", a, 400.0) < 0 for a in (50.0, 100.0, 300.0)))
    pct1("3.6 dense facilitation at 1.6 mm, 100 uA", pq("awake_all", 100.0, 1600.0), 3.6)
    pct1("3.6 dense facilitation at 2.4 mm, 300 uA", pq("awake_all", 300.0, 2400.0), 5.7)
    true("4.3 signed error under 4% at 100 uA from 1.2 mm, both models",
         max(abs(pq(v, 100.0, s_)) for v in ("sparse", "awake_all")
             for s_ in seps if s_ >= 1200.0) < 0.04)
    mp = load("v2_awake_maps.json")
    sm = mp["summary"]
    for key, mean, sd in (("sparse_100_signed_total", -2.9, 0.8),
                          ("awake_all_100_signed_total", -12.5, 2.3),
                          ("sparse_300_signed_total", -9.1, 1.6),
                          ("awake_all_300_signed_total", -23.9, 0.9)):
        pct1(f"3.6 twelve maps, {key} mean", sm[key]["mean"], mean)
        pct1(f"3.6 twelve maps, {key} sd", sm[key]["sd"], sd)
    pct1("3.3 twelve maps, sparse 100 uA lowest", sm["sparse_100_signed_total"]["min"], -4.5)
    pct1("3.3 twelve maps, sparse 100 uA highest", sm["sparse_100_signed_total"]["max"], -2.0)
    pct0("abstract twelve maps, sparse peak at 300 uA", sm["sparse_300_max_dev"]["mean"], 33)
    pct0("abstract twelve maps, dense peak at 300 uA", sm["awake_all_300_max_dev"]["mean"], 50)
    true("3.6 the paper's map is the least suppressive of the twelve, both models",
         all(next(r["signed_total"] for r in mp["rows"] if r["variant"] == v
                  and r["amp_ua"] == 100.0 and r["seed"] == 0)
             == sm[f"{v}_100_signed_total"]["max"] for v in ("sparse", "awake_all")))
    sp = load("v2_awake_spaced.json")
    for n, w in zip(("2", "4", "9", "16"), (0.81, 0.66, 0.51, 0.43)):
        check(f"4.2 probability summation, {n} electrodes", sp["probability_summation"][n], w, 0.005)

    def spaced(var, pitch, n):
        return next(q["ratios"][n] for q in sp["rows"] if q["variant"] == var
                    and q["pitch_um"] == pitch and q["sigma_um"] == 100.0)

    check("4.2 dense, 16 at 0.8 mm", spaced("awake_all", 800.0, "16"), 0.86, 0.005)
    check("4.2 dense, 16 at 1.2 mm", spaced("awake_all", 1200.0, "16"), 0.94, 0.005)
    check("4.2 sparse, 16 at 0.8 mm", spaced("sparse", 800.0, "16"), 0.81, 0.005)
    check("4.2 sparse, 16 at 1.2 mm", spaced("sparse", 1200.0, "16"), 0.95, 0.005)
    check("4.2 dense, 4 at 1.2 mm", spaced("awake_all", 1200.0, "4"), 0.96, 0.005)
    check("4.2 sparse, 4 at 1.2 mm", spaced("sparse", 1200.0, "4"), 0.97, 0.005)


def tables_current():
    """The tables the manuscript inputs are what v2_tables.py writes now."""
    sys.path.insert(0, str(ROOT / "paper"))
    import v2_tables as vt
    for name, fn in vt.TABLES.items():
        path = vt.OUT / name
        true(f"table {name} is current", path.exists()
             and path.read_text() == fn())


SECTIONS = (structure, methods_constants, validation, magnitude, human,
            arrays, sequence, pooled_gain, percepts, encoder, power,
            inhibition_sweep, awake, tables_current)


def main() -> int:
    for fn in SECTIONS:
        fn()
    width = max(len(c[0]) for c in CHECKS)
    n_bad = 0
    for label, got, want, tol, unit, ok in CHECKS:
        got_s = "missing" if got is None else f"{got:.4g}{unit}"
        print(f"{'ok  ' if ok else 'FAIL'}  {label:<{width}}  "
              f"got {got_s:>12}   paper says {want:g}{unit} (±{tol:g})")
        n_bad += not ok
    print(f"\n{len(CHECKS) - n_bad}/{len(CHECKS)} claims verified")
    if n_bad:
        print("The manuscript disagrees with the code. Fix one of them.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
