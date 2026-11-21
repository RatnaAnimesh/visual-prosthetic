"""Does adding electrodes keep adding vision, and what does it cost in charge?

Every roadmap in the field assumes more channels is monotonically better:
1024-channel arrays, denser Utah stacks, higher-count wireless implants.  Under
a pooled divisor that assumption fails in a specific and checkable way, and the
algebra says so before any simulation does.

Write the normaliser for N electrodes each contributing activation a to a
shared pool:

    r_site = a^n / (s^n + g N a^n)

Total delivered activation is N r_site = N a^n / (s^n + g N a^n).  As N grows
the pooled term swamps the semi-saturation constant and this tends to 1/g,
*independent of N*.  So ensemble brightness has a hard ceiling set by the
normalisation gain, per-channel contrast falls as 1/N, and the ceiling cannot
be raised by driving harder, because drive enters numerator and pool alike.

Two things follow that matter for hardware.

1.  There is a channel count beyond which delivered contrast per channel falls
    faster than channel count rises.  Where it sits depends on pitch, because
    pitch controls how much of the pool each electrode shares.

2.  Holding delivered brightness fixed while N grows demands more charge per
    phase.  Charge per phase and charge density are exactly the two axes of the
    Shannon safety criterion, so a perceptual nonlinearity turns into a
    hard engineering bound.  The Shannon criterion is

        log10(D) = k - log10(Q)

    with Q the charge per phase (uC) and D the charge density (uC/cm^2); k
    below about 1.5-2.0 is the range generally treated as safe for chronic
    stimulation (Shannon 1992).  Electrode area is left as a parameter and
    swept, rather than asserting a figure for any particular device.

Run:  python3 prosthesis/phase8_channels.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.phase1_circuit import calibrated
from prosthesis.percept_decoder import CorticalCode, InversePercept

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

XR, YR, STEP = (-3.2, -0.4), (-0.2, 2.0), 0.05
CTX_X, CTX_Y = (16200, 23800), (-8800, -1200)

# Calibrated in phase 1; both reproduce the same single-electrode size curve.
def configs():
    return {"independent": calibrated("circuit off"),
            "circuit": calibrated("full circuit")}

TEST_AMP = 300.0
PITCHES_UM = (400.0, 800.0, 1600.0)
COUNTS = (1, 2, 4, 9, 16, 25, 36, 49, 64, 81, 100)

# Shannon safety criterion, k = log10(D) + log10(Q).
SHANNON_K_SAFE = 1.5
# Charge per phase = amplitude x pulse width; the pulse width used throughout.
PHASE_WIDTH_S = 170e-6


def square_grid(n, pitch_um, cx=20000.0, cy=-5000.0):
    """n electrodes on the tightest square grid that holds them."""
    side = int(np.ceil(np.sqrt(n)))
    offs = (np.arange(side) - (side - 1) / 2.0) * pitch_um
    xs, ys = np.meshgrid(offs, offs)
    pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
    return pts[:, 0] + cx, pts[:, 1] + cy


def build(label):
    m = CircuitSpatial(xrange=XR, yrange=YR, xystep=STEP,
                       cortex_margin=9000.0, **configs()[label])
    m.build()
    return m


def ensemble(m, n, pitch, amp):
    """Total and per-channel delivered activation for n electrodes."""
    xe, ye = square_grid(n, pitch)
    g = m._ensure_cortical_grid(xe, ye)
    field = m._activation_field(np.full(n, amp), xe, ye, g)
    total = float(field.sum())

    # Sum of the same electrodes driven one at a time: the independent
    # prediction, and the thing a roadmap implicitly assumes.
    alone = 0.0
    for i in range(n):
        a = np.zeros(n)
        a[i] = amp
        alone += float(m._activation_field(a, xe, ye, g).sum())
    return {"total": total, "sum_of_singles": alone,
            "per_channel": total / n,
            "efficiency": total / alone if alone > 0 else float("nan")}


def total_only(m, n, pitch, amp):
    """Total delivered activation, without the N single-electrode runs.

    `ensemble` needs N+1 field evaluations because it also computes the
    sum-of-singles; the amplitude search only needs the joint field, so this
    keeps the bisection at one evaluation per step instead of N+1.
    """
    xe, ye = square_grid(n, pitch)
    g = m._ensure_cortical_grid(xe, ye)
    return float(m._activation_field(np.full(n, amp), xe, ye, g).sum())


def amp_for_target(m, n, pitch, target_per_channel, lo=10.0, hi=200000.0,
                   tol=1e-3, iters=44):
    """Amplitude needed so each channel still delivers `target_per_channel`.

    The upper bound is deliberately far beyond anything clinically usable:
    the question is not whether the target is mathematically reachable (it
    is, because the activated radius keeps growing as sqrt(I)) but what
    amplitude it demands.  Returns None only if even `hi` falls short.
    """
    def f(a):
        return total_only(m, n, pitch, a) / n

    if f(hi) < target_per_channel:
        return None
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if f(mid) < target_per_channel:
            lo = mid
        else:
            hi = mid
        if (hi - lo) / hi < tol:
            break
    return float(np.sqrt(lo * hi))


def shannon_k(amp_ua, area_cm2, width_s=PHASE_WIDTH_S):
    """k = log10(charge density) + log10(charge per phase).

    Charge per phase in uC is amplitude(uA) * width(s), since the 1e-6 taking
    uA to A and the 1e6 taking C to uC cancel exactly.
    """
    q_uc = amp_ua * width_s
    d = q_uc / area_cm2
    return float(np.log10(d) + np.log10(q_uc)), float(q_uc), float(d)


def decoded_fidelity(m, code, dec, n, pitch, amp):
    """Decoded percept contrast, relative to the independent prediction."""
    xe, ye = square_grid(n, pitch)
    g = m._ensure_cortical_grid(xe, ye)
    together = code.sample_field(
        m._activation_field(np.full(n, amp), xe, ye, g), g["xs"], g["ys"])
    alone = np.zeros(code.n_sites)
    for i in range(n):
        a = np.zeros(n)
        a[i] = amp
        alone += code.sample_field(
            m._activation_field(a, xe, ye, g), g["xs"], g["ys"])
    img_t = dec.decode(together)[0]
    img_a = dec.decode(alone)[0]
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
    return {"rms_together": rms(img_t), "rms_alone": rms(img_a),
            "contrast_ratio": rms(img_t) / max(rms(img_a), 1e-12)}


def main() -> None:
    out = {"counts": list(COUNTS), "pitches_um": list(PITCHES_UM),
           "test_amp_ua": TEST_AMP, "shannon_k_safe": SHANNON_K_SAFE,
           "phase_width_s": PHASE_WIDTH_S}

    models = {k: build(k) for k in configs()}

    # ---- 1. ensemble brightness against channel count --------------------
    print("delivered activation vs channel count (300 uA per electrode)")
    print("efficiency = delivered / sum-of-singles; 1.00 = independent\n")
    sweeps = {}
    for label, m in models.items():
        sweeps[label] = {}
        for pitch in PITCHES_UM:
            rows = []
            for n in COUNTS:
                rows.append({"n": n, **ensemble(m, n, pitch, TEST_AMP)})
            sweeps[label][str(pitch)] = rows
            if label == "circuit":
                eff = [f"{r['efficiency']:.2f}" for r in rows]
                print(f"  pitch {pitch/1000:.1f} mm  N=" +
                      " ".join(f"{n:>4}" for n in COUNTS))
                print("                efficiency " + " ".join(
                    f"{e:>4}" for e in eff))
                pc = [r["per_channel"] for r in rows]
                print("                per-chan   " + " ".join(
                    f"{p/pc[0]:4.2f}" for p in pc))
        sys.stdout.flush()
    out["sweeps"] = sweeps

    # ---- 2. is there a ceiling, and where -------------------------------
    print("\ntotal delivered activation, normalised to N=1:")
    ceilings = {}
    for label, m in models.items():
        ceilings[label] = {}
        for pitch in PITCHES_UM:
            rows = sweeps[label][str(pitch)]
            t = np.array([r["total"] for r in rows])
            rel = t / t[0]
            # Marginal return: change in total per added electrode.
            marg = np.diff(t) / np.diff(np.array(COUNTS, float))
            half = None
            for j in range(1, len(marg)):
                if marg[j] < 0.5 * marg[0]:
                    half = COUNTS[j]
                    break
            ceilings[label][str(pitch)] = {
                "relative_total": rel.tolist(),
                "marginal": marg.tolist(),
                "n_half_marginal": half,
                "total_at_100_over_N": float(t[-1] / COUNTS[-1] / t[0]),
            }
            if label == "circuit":
                print(f"  pitch {pitch/1000:.1f} mm: N=100 gives "
                      f"{rel[-1]:5.1f}x the N=1 total "
                      f"(independent would give 100x); marginal return halves "
                      f"by N={half}")
    out["ceilings"] = ceilings

    # ---- 3. charge cost of holding per-channel brightness ---------------
    print("\namplitude needed to hold per-channel brightness at the N=1 value")
    m = models["circuit"]
    charge = {}
    for pitch in PITCHES_UM:
        base = total_only(m, 1, pitch, TEST_AMP)
        rows = []
        for n in COUNTS:
            a = amp_for_target(m, n, pitch, base)
            rows.append({"n": n, "amp_ua": a,
                         "amp_ratio": None if a is None else a / TEST_AMP})
        charge[str(pitch)] = rows
        shown = ", ".join(
            f"N={r['n']}:" + ("  >200mA" if r["amp_ua"] is None
                              else f"{r['amp_ua']/1000:.2f}mA")
            for r in rows if r["n"] in (1, 4, 16, 36, 64, 100))
        print(f"  pitch {pitch/1000:.1f} mm  {shown}")
        sys.stdout.flush()
    out["charge"] = charge

    # ---- 4. what that costs in Shannon terms ----------------------------
    # k = log10(D) + log10(Q), and both D and Q are proportional to charge
    # per phase, so raising amplitude by a factor F raises k by exactly
    # 2*log10(F) whatever the electrode area or pulse width.  The *increase*
    # in k is therefore a parameter-free consequence of the model, which the
    # absolute value of k is not: Shannon's k < 1.5 was derived for
    # macroelectrodes, and microelectrode arrays are normally held to a
    # charge-density ceiling instead.  We report the shift, and give absolute
    # k only parametrically in area.
    print("\ncost of that in Shannon terms.  k = log10(D) + log10(Q), so")
    print("raising amplitude by F raises k by 2*log10(F), area-independent.")
    areas_cm2 = [1e-5, 2e-5, 5e-5, 1e-4]   # 1000 to 10000 um^2
    shannon = {}
    for pitch in PITCHES_UM:
        rows = []
        for r in charge[str(pitch)]:
            entry = {"n": r["n"], "amp_ua": r["amp_ua"],
                     "amp_ratio": r["amp_ratio"]}
            if r["amp_ratio"]:
                entry["delta_k"] = float(2.0 * np.log10(r["amp_ratio"]))
                for a_cm2 in areas_cm2:
                    k, q, d = shannon_k(r["amp_ua"], a_cm2)
                    entry[f"k_{a_cm2:g}"] = k
                    entry[f"density_{a_cm2:g}"] = d
                entry["q_uc"] = r["amp_ua"] * PHASE_WIDTH_S
            rows.append(entry)
        shannon[str(pitch)] = rows
        shown = ", ".join(
            f"N={r['n']}:+{r['delta_k']:.2f}"
            for r in rows if r.get("delta_k") and r["n"] in (4, 16, 36, 64))
        print(f"  pitch {pitch/1000:.1f} mm  delta-k  {shown}")
    out["shannon"] = {"areas_cm2": areas_cm2, "by_pitch": shannon,
                      "note": "delta_k is area-independent; absolute k uses "
                              "the macroelectrode Shannon form and is given "
                              "parametrically in area for reference only"}

    # ---- 5. decoded percept contrast ------------------------------------
    print("\ndecoded percept contrast, driven together / sum of singles")
    code = CorticalCode(CTX_X, CTX_Y)
    dec = InversePercept(code, lam=0.02, precompute=True)
    dec_rows = {}
    for label, m in models.items():
        rows = []
        for n in (1, 4, 9, 16, 25, 36):
            rows.append({"n": n,
                         **decoded_fidelity(m, code, dec, n, 800.0, TEST_AMP)})
        dec_rows[label] = rows
        print(f"  {label:<12} " + " ".join(
            f"N={r['n']}:{r['contrast_ratio']:.2f}" for r in rows))
        sys.stdout.flush()
    out["decoded"] = dec_rows

    (RESULTS / "phase8_channels.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'phase8_channels.json'}")


if __name__ == "__main__":
    main()
