"""Drive sets the sign of the interaction, and the same ratio sets the delay.

Three of the limitations in the previous draft were really one limitation:
no facilitation, no enlargement for merged pairs, repulsion far too weak.
All three were measured at a single drive level (300 uA), and the normaliser
is not sign-stable in drive.

Write it out:

    r = A^n / (s^n + g (G_sigma * A^n))          P := g (G_sigma * A^n)

When P << s^n the divisor is nearly constant and the expansive numerator
runs the show: for n > 1 and overlapping activations, (A_a + A_b)^n exceeds
A_a^n + A_b^n, so the model is *supra-additive*.  When P >> s^n the divisor
dominates and each electrode inflates its neighbour's pool, so the model is
suppressive.  The crossing is at

    R := P / s^n = 1.

Now give the pool dynamics, tau dp/dt = -p + P.  Fire A, then B after a delay
D.  B sees A's residual pool decayed by exp(-D/tau), and B's brightness is
restored once that residual drops below the semi-saturation floor:

    P_a exp(-D/tau) << s^n   =>   D* ~ tau ln(P_a / s^n) = tau ln R.

So one dimensionless quantity -- pooled drive in units of semi-saturation --
fixes both the drive at which interaction reverses sign and the delay at
which sequential stimulation stops beating simultaneous.  The corollary is
sharp and falsifiable: for R <= 1, ln R <= 0, so D* <= 0 and *no* delay
helps, because there is no suppression to escape.  The sequential advantage
should switch on at the same drive where the interaction changes sign, and
grow only logarithmically above it.

This script measures both, and checks the delay against tau ln R.

Run:  python3 prosthesis/phase7_regimes.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_circuit import CircuitSpatial

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

XR, YR, STEP = (-3.2, -0.4), (-0.2, 2.0), 0.05

# Calibrated excitabilities come from phase1 so every configuration is still
# matched to the same single-electrode size curve.
CONFIGS = {
    "pointwise saturation": dict(horiz_gain=0.0, norm_gain=1.0,
                                 norm_sigma=0.0),
    "normalisation only":   dict(horiz_gain=0.0, norm_gain=1.0,
                                 norm_sigma=1500.0),
    "full circuit":         dict(horiz_gain=0.6, norm_gain=1.0,
                                 norm_sigma=1500.0),
}

AMPS_UA = np.array([10., 15., 20., 30., 40., 60., 80., 100., 130., 160.,
                    200., 250., 300., 400., 500., 650., 800., 1000.])
SEPARATIONS_UM = np.array([400., 800., 1200., 1600., 2000., 2400., 3200.])

X0, Y0 = 20000.0, -5000.0


def excitabilities():
    p = RESULTS / "phase1_circuit.json"
    cal = json.loads(p.read_text())["calibration"]
    return {k: cal[k]["excitability"] for k in CONFIGS if k in cal}


def build(label, k):
    m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                       excitability=k, **CONFIGS[label])
    m.build()
    return m


def signed_interaction(m, d, amp):
    """Signed departure from superposition: total and at the linear peak.

    Total is the ensemble-brightness quantity the paper's central claim is
    about; the peak value is what a size/brightness report would track.
    Positive means supra-additive (facilitation).
    """
    xe = np.array([X0, X0 + d], float)
    ye = np.array([Y0, Y0], float)
    g = m._ensure_cortical_grid(xe, ye)
    fa = m._activation_field(np.array([amp, 0.0]), xe, ye, g)
    fb = m._activation_field(np.array([0.0, amp]), xe, ye, g)
    fab = m._activation_field(np.array([amp, amp]), xe, ye, g)
    lin = fa + fb
    resid = fab - lin
    tot = float(lin.sum())
    if tot <= 0:
        return 0.0, 0.0
    i = np.unravel_index(int(np.argmax(lin)), lin.shape)
    return float(resid.sum() / tot), float(resid[i] / lin[i])


def crossing(amps, vals):
    """Amplitude at which a signed curve crosses zero, by log interpolation."""
    v = np.asarray(vals)
    for j in range(len(v) - 1):
        if v[j] > 0 >= v[j + 1]:
            f = v[j] / (v[j] - v[j + 1])
            la, lb = np.log(amps[j]), np.log(amps[j + 1])
            return float(np.exp(la + f * (lb - la)))
    return None


# ---------------------------------------------------------------------------
# Dynamics.  The pool is a leaky integrator of the pooled drive.  Because the
# stimulation schedule is piecewise constant, the ODE integrates exactly on
# each interval and there is no need for a time-stepping solver.
# ---------------------------------------------------------------------------

def drive_fields(m, d, amp):
    """Pooled drive at electrode B's peak, for each set of active electrodes.

    Only the divisor carries temporal memory: the numerator is instantaneous.
    So the sequential-versus-simultaneous question is entirely about the pool,
    and the observable is the *gain* 1/(s^n + p), not the response.  Measuring
    the response instead conflates suppression with A's activation spilling
    spatially into B's location, which at large amplitudes dominates and makes
    simultaneous stimulation look brighter than solo.

    The pool ODE is pointwise in space, so reading out at one location needs
    only the scalar drive there.
    """
    xe = np.array([X0, X0 + d], float)
    ye = np.array([Y0, Y0], float)
    g = m._ensure_cortical_grid(xe, ye)

    # A model with normalisation disabled returns `total` (direct + horizontal).
    lin_m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                           excitability=m.excitability,
                           horiz_gain=m.horiz_gain, norm_gain=0.0)
    lin_m.build()
    ta = lin_m._activation_field(np.array([amp, 0.0]), xe, ye, g)
    tb = lin_m._activation_field(np.array([0.0, amp]), xe, ye, g)

    n = m.expansive_n
    sig_px = m.norm_sigma / m.cortex_step

    def pooled(num):
        if sig_px <= 0:
            return m.norm_gain * num
        return m.norm_gain * gaussian_filter(num, sig_px, mode="nearest")

    iy, ix = np.unravel_index(int(np.argmax(tb)), tb.shape)
    P_a = float(pooled(ta ** n)[iy, ix])
    P_b = float(pooled(tb ** n)[iy, ix])
    P_ab = float(pooled((ta + tb) ** n)[iy, ix])
    return {"n": n, "s_n": float(m.norm_semi ** n),
            "P_a": P_a, "P_b": P_b, "P_ab": P_ab,
            "R": P_a / float(m.norm_semi ** n)}


def simulate_gain(df, schedule, tau_ms, dt=0.25):
    """Mean gain 1/(s^n + p) at B's location across B's stimulation windows.

    `schedule` is a list of (t_start, t_end, key) with key in {a, b, ab}.
    The pool integrates exactly on each constant-drive interval.
    """
    s_n, p, t = df["s_n"], 0.0, 0.0
    t_end = max(e for _, e, _ in schedule)
    decay = np.exp(-dt / tau_ms)
    acc, cnt = 0.0, 0
    while t < t_end:
        key = None
        for a0, a1, k in schedule:
            if a0 <= t < a1:
                key = k
                break
        P = df[f"P_{key}"] if key else 0.0
        p = P + (p - P) * decay
        if key in ("b", "ab"):
            acc += 1.0 / (s_n + p)
            cnt += 1
        t += dt
    return acc / cnt if cnt else float("nan")


def sequential_test(df, tau_ms, on_ms=50.0, delays=None, dt=0.25):
    """Gain at electrode B, sequential at various delays vs simultaneous.

    Normalised to B stimulated alone, which is the ceiling.
    """
    if delays is None:
        delays = [0., 5., 10., 20., 30., 50., 75., 100., 150., 200., 300.,
                  450., 650.]
    solo = simulate_gain(df, [(0.0, on_ms, "b")], tau_ms, dt)
    if not np.isfinite(solo) or solo <= 0:
        return None
    sim = simulate_gain(df, [(0.0, on_ms, "ab")], tau_ms, dt) / solo
    rows = []
    for D in delays:
        # A occupies [0, on]; B occupies [D, D+on].  At D = 0 they coincide.
        if D <= 0:
            rows.append({"delay_ms": 0.0, "preservation": sim})
            continue
        sched = [(0.0, min(on_ms, D), "a")]
        if D < on_ms:
            sched.append((D, on_ms, "ab"))
            sched.append((on_ms, D + on_ms, "b"))
        else:
            sched.append((D, D + on_ms, "b"))
        rows.append({"delay_ms": float(D),
                     "preservation": simulate_gain(df, sched, tau_ms) / solo})
    return {"solo": solo, "simultaneous": sim, "by_delay": rows}


def critical_delay(rows, crit=0.95):
    """Smallest delay at which B recovers `crit` of its solo brightness."""
    prev = None
    for r in rows:
        if r["preservation"] >= crit:
            if prev is None:
                return 0.0
            d0, v0 = prev["delay_ms"], prev["preservation"]
            d1, v1 = r["delay_ms"], r["preservation"]
            if v1 == v0:
                return float(d1)
            return float(d0 + (crit - v0) * (d1 - d0) / (v1 - v0))
        prev = r
    return None


def scaling_check(m, d=1600.0, amps=(160., 220., 300., 420., 600., 800.,
                                     1000.),
                  taus=(25., 50., 100., 200.), on_ratio=6.0, crit=0.95):
    """Is D* really proportional to tau, and log-linear in R'?

    The pulse duration has to be scaled with tau.  Holding on_ms fixed while
    tau varies changes on_ms/tau, and the prefactor depends on that ratio,
    which would make a genuine scaling law look like an approximate one.

    Reports the window-mean slope (what the simulation measures) beside the
    onset-readout slope solved directly from the pool ODE, which is the
    quantity the analytic derivation predicts should equal tau exactly.
    """
    cache = {a: drive_fields(m, d, a) for a in amps}
    rows = []
    for tau in taus:
        on = on_ratio * tau
        dt = tau / 100.0
        delays = list(np.linspace(0.0, 16 * tau, 80))
        xs, ys = [], []
        for a in amps:
            df = cache[a]
            res = sequential_test(df, tau, on_ms=on, delays=delays, dt=dt)
            if res is None:
                continue
            cd = critical_delay(res["by_delay"], crit)
            rp = df["P_a"] / (df["s_n"] + df["P_b"])
            if cd and cd > 0 and rp > 0:
                xs.append(np.log(rp))
                ys.append(cd)
        if len(xs) < 2:
            continue
        s, i = np.polyfit(xs, ys, 1)
        ya = np.array(ys)
        r2 = 1 - ((ya - (s * np.array(xs) + i)) ** 2).sum() / \
            ((ya - ya.mean()) ** 2).sum()

        # Onset readout, solved from the ODE rather than simulated.
        oxs, oys = [], []
        for a in amps:
            df = cache[a]
            target = df["s_n"] * (1.0 / crit - 1.0)
            peak_a = df["P_a"] * (1 - np.exp(-on / tau))
            if peak_a <= target:
                continue
            oxs.append(np.log(df["P_a"] / df["s_n"]))
            oys.append(on + tau * np.log(peak_a / target))
        os_ = float(np.polyfit(oxs, oys, 1)[0]) if len(oxs) >= 2 else float("nan")

        rows.append({"tau_ms": tau, "on_ms": on, "n_points": len(xs),
                     "slope_window_mean_ms": float(s),
                     "ratio_window_mean": float(s / tau), "r2": float(r2),
                     "slope_onset_ms": os_, "ratio_onset": float(os_ / tau)})
    return rows


def main() -> None:
    ks = excitabilities()
    models = {label: build(label, ks[label]) for label in CONFIGS if label in ks}

    # ---- 1. signed interaction over (separation x drive) -----------------
    print("signed interaction, + = facilitation, - = suppression")
    print("crossover amplitude (uA) where the sign flips:\n")
    print(f"{'configuration':<22}" +
          "".join(f"{d/1000:>8.1f}mm" for d in SEPARATIONS_UM))
    surface = {}
    for label, m in models.items():
        rows, crosses = [], []
        for d in SEPARATIONS_UM:
            tot = [signed_interaction(m, d, a)[0] for a in AMPS_UA]
            pk = [signed_interaction(m, d, a)[1] for a in AMPS_UA]
            rows.append({"separation_um": float(d),
                         "amps_ua": AMPS_UA.tolist(),
                         "signed_total": tot, "signed_peak": pk,
                         "crossover_ua": crossing(AMPS_UA, tot)})
            crosses.append(rows[-1]["crossover_ua"])
        surface[label] = rows
        cells = "".join(("     n/a" if c is None else f"{c:8.0f}")
                        for c in crosses)
        print(f"  {label:<20}{cells}")
        sys.stdout.flush()

    # ---- 2. does the crossover depend on separation? ---------------------
    print("\nspread of crossover across separations "
          "(a pooled divisor should vary, a pointwise one should not):")
    for label, rows in surface.items():
        cs = [r["crossover_ua"] for r in rows if r["crossover_ua"]]
        if len(cs) >= 2:
            print(f"  {label:<22} {min(cs):6.0f} - {max(cs):6.0f} uA  "
                  f"(ratio {max(cs)/min(cs):.2f})")
        else:
            print(f"  {label:<22} no sign change in range")

    # ---- 3. sequential vs simultaneous, and tau ln R ---------------------
    # Theory: preservation(D) = (s^n + P_b)/(s^n + P_b + P_a exp(-D/tau)), so
    # the delay needed to recover a fraction c of solo gain is
    #     D* = tau * ln( (c/(1-c)) * R' ),   R' := P_a / (s^n + P_b).
    # Slope of D* against ln R' should therefore recover tau exactly, and D*
    # goes negative -- no delay needed -- once R' drops below (1-c)/c.
    print("\nsequential stimulation: gain at B, relative to B alone")
    d_test = 1600.0
    crit = 0.95
    dyn = {}
    for label in ("normalisation only", "full circuit"):
        if label not in models:
            continue
        m = models[label]
        cross = next((r["crossover_ua"] for r in surface[label]
                      if r["separation_um"] == d_test), None)
        probes = [40., 60., 80., 120., 160., 220., 300., 420., 600., 800.,
                  1000.]
        for tau in (25.0, 50.0, 100.0):
            out = []
            print(f"\n  {label}  (sep {d_test/1000:.1f} mm, tau {tau:.0f} ms, "
                  f"criterion {crit:.2f})")
            print(f"    {'amp':>6} {'R=P_a/s^n':>10} {'Rprime':>8} "
                  f"{'simult.':>8} {'D* meas':>9} {'D* pred':>9}")
            for amp in probes:
                df = drive_fields(m, d_test, amp)
                res = sequential_test(df, tau)
                if res is None:
                    continue
                cd = critical_delay(res["by_delay"], crit)
                rprime = df["P_a"] / (df["s_n"] + df["P_b"])
                pred = (tau * np.log((crit / (1 - crit)) * rprime)
                        if rprime > 0 else float("nan"))
                out.append({"amp_ua": amp, "R": df["R"], "R_prime": rprime,
                            "simultaneous": res["simultaneous"],
                            "critical_delay_ms": cd,
                            "predicted_delay_ms": float(pred),
                            "by_delay": res["by_delay"]})
                cd_s = "     none" if cd is None else f"{cd:8.1f}ms"
                print(f"    {amp:6.0f} {df['R']:10.3f} {rprime:8.3f} "
                      f"{res['simultaneous']:8.3f} {cd_s:>9} {pred:8.1f}ms")
                sys.stdout.flush()
            # Recover tau from the slope of D* against ln R'.
            pts = [(np.log(o["R_prime"]), o["critical_delay_ms"]) for o in out
                   if o["critical_delay_ms"] and o["critical_delay_ms"] > 0
                   and o["R_prime"] > 0]
            if len(pts) >= 2:
                xs, ys = np.array([p[0] for p in pts]), np.array([p[1]
                                                                 for p in pts])
                slope, intercept = np.polyfit(xs, ys, 1)
                print(f"    slope of D* vs ln R' = {slope:.1f} ms "
                      f"(tau = {tau:.0f} ms)")
                dyn[f"{label} tau={tau:.0f}"] = {
                    "tau_ms": tau, "separation_um": d_test,
                    "crossover_ua": cross, "criterion": crit,
                    "recovered_tau_ms": float(slope),
                    "intercept_ms": float(intercept), "probes": out}
            else:
                dyn[f"{label} tau={tau:.0f}"] = {
                    "tau_ms": tau, "separation_um": d_test,
                    "crossover_ua": cross, "criterion": crit,
                    "recovered_tau_ms": None, "probes": out}

    # ---- 4. is D* really proportional to tau? ---------------------------
    print("\nscaling check: D* against ln R', pulse duration held at 6 tau")
    print(f"    {'tau':>6} {'slope':>9} {'slope/tau':>10} {'R^2':>8} "
          f"{'onset slope/tau':>16}")
    scaling = scaling_check(models["normalisation only"])
    for r in scaling:
        print(f"    {r['tau_ms']:6.0f} {r['slope_window_mean_ms']:9.1f} "
              f"{r['ratio_window_mean']:10.3f} {r['r2']:8.4f} "
              f"{r['ratio_onset']:16.4f}")
        sys.stdout.flush()

    (RESULTS / "phase7_regimes.json").write_text(json.dumps(
        {"surface": surface, "dynamics": dyn, "scaling": scaling,
         "amps_ua": AMPS_UA.tolist(),
         "separations_um": SEPARATIONS_UM.tolist()}, indent=2))
    print(f"\nwrote {RESULTS / 'phase7_regimes.json'}")


if __name__ == "__main__":
    main()
