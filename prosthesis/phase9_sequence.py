"""Two things the model owes: a test against real data, and a runnable protocol.

Part 1 confronts the temporal model with the only published human cortical
multi-electrode *brightness* measurements.  Armenta Salas et al. stimulated
Orion users with sequences of two and three electrodes and asked which
phosphene looked brighter.  A pooled divisor with dynamics makes an
unambiguous prediction here: an electrode fired later in a sequence sits in
the residual pool left by its predecessors, so it should be *dimmer*.  Their
result is mixed, and for two of three subjects it goes the other way.  This
script quantifies the model's prediction so the disagreement is on the record
with a number attached rather than as a hand-wave.

Part 2 turns section 4.1's proposed experiment into a protocol.  Time on
implanted humans is the scarcest resource in the field, so "measure at four or
five separations" is not a proposal anyone can act on.  The discriminating
observable is the *shape* of interaction against separation, and magnitude is
unconstrained, so the decision rule has to be scale-free: fit each candidate
curve with a free scale factor and take the one with lower residual.  Running
that rule over simulated experiments with realistic report noise gives the
number of trials per separation needed for 80% power.

Run:  python3 prosthesis/phase9_sequence.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.phase1_circuit import calibrated

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

XR, YR, STEP = (-3.2, -0.4), (-0.2, 2.0), 0.05
X0, Y0 = 20000.0, -5000.0

CFG = dict(horiz_gain=0.6, norm_gain=1.0, norm_sigma=1500.0)


def build():
    m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                       excitability=calibrated("full circuit")["excitability"],
                       **CFG)
    m.build()
    return m


def pair_pools(m, d, amp):
    """Pooled drive and numerator at each electrode's own peak.

    Returns one dict per electrode, in the form phase 7's simulator expects,
    so the same exact-integration machinery can read out at either site.
    """
    xe = np.array([X0, X0 + d], float)
    ye = np.array([Y0, Y0], float)
    g = m._ensure_cortical_grid(xe, ye)

    lin = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                         excitability=m.excitability,
                         horiz_gain=m.horiz_gain, norm_gain=0.0)
    lin.build()
    ta = lin._activation_field(np.array([amp, 0.0]), xe, ye, g)
    tb = lin._activation_field(np.array([0.0, amp]), xe, ye, g)

    n = m.expansive_n
    sig_px = m.norm_sigma / m.cortex_step
    pool = (lambda a: m.norm_gain * gaussian_filter(a, sig_px, mode="nearest")
            if sig_px > 0 else lambda a: m.norm_gain * a)

    Pa, Pb = pool(ta ** n), pool(tb ** n)
    Pab = pool((ta + tb) ** n)
    ia = np.unravel_index(int(np.argmax(ta)), ta.shape)
    ib = np.unravel_index(int(np.argmax(tb)), tb.shape)
    s_n = float(m.norm_semi ** n)
    at = lambda i: {"s_n": s_n, "P_a": float(Pa[i]), "P_b": float(Pb[i]),
                    "P_ab": float(Pab[i])}
    return at(ia), at(ib), float(ta[ia] ** n), float(tb[ib] ** n)


def mean_gain(df, schedule, tau_ms, keys, dt=0.25):
    """Mean of 1/(s^n + p) over the intervals whose key is in `keys`."""
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
        if key in keys:
            acc += 1.0 / (s_n + p)
            cnt += 1
        t += dt
    return acc / cnt if cnt else float("nan")


def sequence_bias(m, d=1600.0, amp=300.0, tau_ms=50.0, on_ms=50.0,
                  delays=(0., 25., 50., 100., 200., 400.)):
    """Brightness of the second phosphene relative to the first.

    Below 1 means the later electrode looks dimmer, which is what a pooled
    divisor with memory has to predict.
    """
    dfa, dfb, num_a, num_b = pair_pools(m, d, amp)
    rows = []
    for D in delays:
        if D <= 0:
            sched = [(0.0, on_ms, "ab")]
            ga = mean_gain(dfa, sched, tau_ms, ("ab",))
            gb = mean_gain(dfb, sched, tau_ms, ("ab",))
        else:
            sched = [(0.0, min(on_ms, D), "a")]
            if D < on_ms:
                sched += [(D, on_ms, "ab"), (on_ms, D + on_ms, "b")]
            else:
                sched += [(D, D + on_ms, "b")]
            ga = mean_gain(dfa, sched, tau_ms, ("a", "ab"))
            gb = mean_gain(dfb, sched, tau_ms, ("b", "ab"))
        rows.append({"delay_ms": float(D),
                     "brightness_first": float(num_a * ga),
                     "brightness_second": float(num_b * gb),
                     "ratio_second_over_first": float((num_b * gb) /
                                                      (num_a * ga))})
    return rows


# ---------------------------------------------------------------------------
# Part 2: how many trials does the discriminating experiment need?
# ---------------------------------------------------------------------------

def candidate_curves():
    """Predicted interaction against separation for the two live mechanisms."""
    d = json.loads((RESULTS / "phase1_circuit.json").read_text())
    sw = d["separation_sweep"]

    def curve(label):
        rows = sorted(sw[label], key=lambda r: r["separation_um"])
        return (np.array([r["separation_um"] for r in rows]),
                np.array([100 * r["max_dev"] for r in rows]))

    seps, pw = curve("pointwise saturation")
    _, po = curve("full circuit")
    return seps, pw, po


def fit_scale(pred, obs):
    """Least-squares scale factor and residual, so only shape is tested."""
    denom = float(np.dot(pred, pred))
    if denom <= 0:
        return 0.0, float(np.dot(obs, obs))
    s = float(np.dot(pred, obs) / denom)
    return s, float(((obs - s * pred) ** 2).sum())


def power(truth, other, sem, n_sims=4000, rng=None):
    """Fraction of simulated experiments that identify the true mechanism."""
    rng = rng or np.random.default_rng(0)
    hits = 0
    for _ in range(n_sims):
        obs = truth + rng.normal(0.0, sem, size=truth.size)
        _, r_true = fit_scale(truth, obs)
        _, r_other = fit_scale(other, obs)
        hits += r_true < r_other
    return hits / n_sims


def trials_for_power(truth, other, sd_per_trial, target=0.8, max_trials=400):
    """Trials per separation needed to reach `target` power."""
    for t in range(1, max_trials + 1):
        if power(truth, other, sd_per_trial / np.sqrt(t)) >= target:
            return t
    return None


def main() -> None:
    out = {}
    m = build()

    # ---- Part 1: sequence-position brightness bias -----------------------
    print("sequence-position brightness, second phosphene / first")
    print("model predicts < 1 (later is dimmer); Armenta Salas et al. report")
    print("later-is-brighter in 2 of 3 subjects, first-is-brighter in 1\n")
    seq = {}
    for amp in (150.0, 300.0, 600.0):
        rows = sequence_bias(m, amp=amp)
        seq[str(amp)] = rows
        cells = " ".join(f"{r['delay_ms']:.0f}ms:{r['ratio_second_over_first']:.3f}"
                         for r in rows)
        print(f"  {amp:5.0f} uA   {cells}")
        sys.stdout.flush()
    out["sequence_bias"] = seq

    worst = min(r["ratio_second_over_first"]
                for rows in seq.values() for r in rows)
    print(f"\n  model never predicts the later phosphene brighter; "
          f"strongest predicted dimming is {100*(1-worst):.1f}%")
    out["max_predicted_dimming_pct"] = float(100 * (1 - worst))

    # ---- Part 2: power analysis for the shape experiment -----------------
    seps, pw, po = candidate_curves()
    shape_corr = float(np.corrcoef(pw, po)[0, 1])
    print(f"\ndiscriminating the two mechanisms by shape "
          f"(correlation of the two predicted curves: {shape_corr:.3f})")
    print("decision rule: fit each candidate with a free scale, take lower "
          "residual\n")
    print(f"  separations tested: "
          f"{', '.join(f'{s/1000:.1f}' for s in seps)} mm")

    # Power against the standard error actually achieved per separation.
    # Quoting trials is only meaningful once the per-trial variance of the
    # paradigm is known, and that variance is not ours to assume; the tolerance
    # on the standard error is the transferable number.
    rows = []
    print(f"  {'sem (pp)':>9} {'power if pointwise':>20} "
          f"{'power if pooled':>17}")
    for sem in (10., 20., 30., 50., 75., 100., 150., 200.):
        p_pw = power(pw, po, sem)
        p_po = power(po, pw, sem)
        rows.append({"sem_pp": sem, "power_if_pointwise": p_pw,
                     "power_if_pooled": p_po})
        print(f"  {sem:9.0f} {p_pw:20.3f} {p_po:17.3f}")
        sys.stdout.flush()

    tol = None
    for r in rows:
        if min(r["power_if_pointwise"], r["power_if_pooled"]) < 0.8:
            tol = r["sem_pp"]
            break
    print(f"\n  power stays above 0.8 until the standard error per separation "
          f"reaches ~{tol:.0f} pp,")
    print(f"  which is {tol / max(po.max(), 1e-9) * 100:.0f}% of the largest "
          f"predicted effect.  With a per-trial sd of s, that is "
          f"(s/{tol:.0f})^2 trials.")
    for sd in (30., 60., 100.):
        print(f"    per-trial sd {sd:5.0f} pp -> "
              f"{max(1, int(np.ceil((sd / tol) ** 2))):3d} trials/separation")

    out["power"] = {"separations_um": seps.tolist(),
                    "curve_pointwise": pw.tolist(), "curve_pooled": po.tolist(),
                    "shape_correlation": shape_corr, "rows": rows,
                    "sem_tolerance_pp": tol}

    # A reduced protocol: can it be done with fewer separations?
    print("\nreduced protocols (sem tolerance, pp):")
    subsets = {
        "9 separations (full)": list(range(len(seps))),
        "4 separations (0.4/1.2/1.6/3.2)": [0, 3, 4, 7],
        "3 separations (0.4/1.6/3.2)": [0, 4, 7],
        "2 separations (0.4/1.6)": [0, 4],
    }
    red = {}
    for name, idx in subsets.items():
        i = np.array(idx)
        t = None
        for sem in (10., 20., 30., 50., 75., 100., 150., 200.):
            if min(power(po[i], pw[i], sem),
                   power(pw[i], po[i], sem)) < 0.8:
                t = sem
                break
        red[name] = t
        print(f"  {name:<34} {'>200' if t is None else f'{t:.0f}'}")
    out["reduced_protocols"] = red

    # A confound the protocol has to defend against.  Measuring separations in
    # increasing order while the subject adapts attenuates the later, larger
    # separations, which bends a pooled curve toward the monotonic decay that
    # is the pointwise signature.  Randomising separation order is not
    # optional.
    print("\nconfound: session drift attenuating later measurements")
    rng = np.random.default_rng(7)
    drift_rows = []
    for beta in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        ramp = 1.0 - beta * np.linspace(0.0, 1.0, po.size)
        hits = 0
        for _ in range(3000):
            obs = po * ramp + rng.normal(0.0, 20.0, size=po.size)
            _, r_po = fit_scale(po, obs)
            _, r_pw = fit_scale(pw, obs)
            hits += r_po < r_pw
        drift_rows.append({"drift": beta, "power_if_pooled": hits / 3000})
        print(f"  {beta*100:3.0f}% attenuation across the sweep -> "
              f"pooled correctly identified {hits/3000:.3f} of the time")
    out["drift_confound"] = drift_rows

    (RESULTS / "phase9_sequence.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'phase9_sequence.json'}")


if __name__ == "__main__":
    main()
