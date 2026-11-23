"""The manuscript's experiments, re-run on the biophysical model.

Same questions as phases 1, 7, 8 and 9, asked of ``BiophysicalCortex`` instead
of the phenomenological ``CircuitSpatial``:

    pairs        departure from superposition against separation and current,
                 split into the part that comes from direct recruitment alone
                 (occupancy) and the part the network adds; where the sign
                 changes, and whether the two length-scale signatures of the
                 first model (peak position, drift of the crossover current)
                 come out pointwise-like or pooled-like
    extent       how the activated region grows with current, as a shape
                 (half maximum) and as an area above a fixed threshold
    channels     total delivered output against the number of electrodes
    sequence     brightness of a second phosphene against delay
    sensitivity  how far the headline numbers, and the signatures, move across
                 the parameters the literature does not pin down

Run:  python3 prosthesis/v2_experiments.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prosthesis.cortex_ssn import (BiophysicalCortex, RecruitmentParams,
                                   SSNParams, _ori_harmonics)

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

X0, Y0 = 20000.0, -5000.0
AMPS = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 450.0,
        600.0)
SEPS = (400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0, 4500.0)
# Evoked change taken as visible for the fixed-threshold extent: the
# spontaneous excitatory rate, i.e. the region where firing has doubled.
VISIBLE_HZ = 1.4
# Human V1 ocular-dominance columns are 863 um wide against 531 um in
# macaque (Adams, Sincich & Horton 2007).  Scaling every lateral length by
# that ratio is the simplest hypothesis for a human-sized sheet.
HUMAN_SCALE = 863.0 / 531.0


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------

def pair(m, sep, amp):
    """Evoked fields for A alone, B alone and both, plus the direct parts."""
    xe = np.array([X0, X0 + sep])
    ye = np.array([Y0, Y0])
    g = m._ensure_cortical_grid(xe, ye)
    out = {}
    for key, a in (("a", [amp, 0.0]), ("b", [0.0, amp]), ("ab", [amp, amp])):
        out[key] = m._activation_field(a, xe, ye, g)
        out[key + "_direct"] = m.last["direct_e"]
    return out


def interaction(f):
    """Signed and peak departure from superposition, total and direct-only."""
    lin = f["a"] + f["b"]
    res = f["ab"] - lin
    lin_d = f["a_direct"] + f["b_direct"]
    res_d = f["ab_direct"] - lin_d
    return {
        "signed_total": float(res.sum() / lin.sum()),
        "max_dev": float(np.abs(res).max() / np.abs(lin).max()),
        "signed_direct": float(res_d.sum() / lin_d.sum()),
        "max_dev_direct": float(np.abs(res_d).max() / np.abs(lin_d).max()),
    }


def pair_sweep(m, amps=AMPS, seps=SEPS):
    rows = []
    for amp in amps:
        for sep in seps:
            rows.append({"amp_ua": amp, "separation_um": sep,
                         **interaction(pair(m, sep, amp))})
    return rows


def crossovers(rows, amps=AMPS, seps=SEPS, noise=1e-3):
    """Current at which the signed interaction turns suppressive for good.

    The highest current at which it is still facilitatory by more than
    ``noise`` (0.1%, below which the sign is not meaningful), interpolated in
    log current to the next amplitude.  None if it is never robustly
    facilitatory, or never robustly suppressive.
    """
    out = {}
    for sep in seps:
        v = [next(r["signed_total"] for r in rows if r["amp_ua"] == a
                  and r["separation_um"] == sep) for a in amps]
        pos = [k for k in range(len(v)) if v[k] > noise]
        cross = None
        if pos and pos[-1] + 1 < len(v) and v[pos[-1] + 1] < -noise:
            k = pos[-1] + 1
            la, lb = np.log(amps[k - 1]), np.log(amps[k])
            t = v[k - 1] / (v[k - 1] - v[k])
            cross = float(np.exp(la + t * (lb - la)))
        out[str(sep)] = cross
    return out


def signatures(rows, amp=300.0, seps=SEPS):
    """The first model's discriminating shape, read off this one.

    Pointwise saturation: peak departure largest at the shortest separation
    and falling monotonically.  Pooled normalisation: an interior maximum.
    """
    dev = [next(r["max_dev"] for r in rows if r["amp_ua"] == amp
                and r["separation_um"] == s) for s in seps]
    k = int(np.argmax(dev))
    return {"amp_ua": amp, "peak_dev": dev,
            "peak_at_um": float(seps[k]),
            "monotonic_decay": bool(np.all(np.diff(dev) <= 1e-4))}


# ---------------------------------------------------------------------------
# Single electrode
# ---------------------------------------------------------------------------

def extent(m):
    """Size of the evoked region against current, two ways."""
    g = m._ensure_cortical_grid(np.array([X0]), np.array([Y0]))
    cell = (m.step / 1000.0) ** 2
    rows = []
    for amp in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 600.0, 1000.0):
        ev = m._activation_field([amp], [X0], [Y0], g)
        half = (ev >= ev.max() / 2).sum() * cell
        vis = (ev >= VISIBLE_HZ).sum() * cell
        rows.append({
            "amp_ua": amp,
            "halfmax_diameter_mm": float(2 * np.sqrt(half / np.pi)),
            "visible_diameter_mm": float(2 * np.sqrt(vis / np.pi)),
            "tehovnik_diameter_mm": float(2 * np.sqrt(amp / 675.0)),
            "peak_evoked_hz": float(ev.max()),
            "total_evoked": float(ev.sum()),
            "total_positive": float(np.maximum(ev, 0).sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# Many electrodes
# ---------------------------------------------------------------------------

def square_grid(n, pitch):
    side = int(np.ceil(np.sqrt(n)))
    offs = (np.arange(side) - (side - 1) / 2.0) * pitch
    gx, gy = np.meshgrid(offs, offs)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:n]
    return pts[:, 0] + X0, pts[:, 1] + Y0


def channels(m, counts=(1, 4, 9, 16, 25, 36, 64, 100),
             pitches=(400.0, 800.0, 1600.0), amps=(30.0, 100.0, 300.0),
             n_single=6, seed=0):
    """Delivered output for N electrodes, against the sum of singles."""
    rng = np.random.default_rng(seed)
    out = []
    for amp in amps:
        for pitch in pitches:
            for n in counts:
                xe, ye = square_grid(n, pitch)
                g = m._ensure_cortical_grid(xe, ye)
                together = m._activation_field(np.full(n, amp), xe, ye, g)
                tot = float(together.sum())
                direct = float(m.last["direct_e"].sum())
                # Single-electrode totals differ only through the orientation
                # map, so a sample of them estimates the sum of all N.
                pick = rng.choice(n, size=min(n, n_single), replace=False)
                singles, singles_d = [], []
                for k in pick:
                    a = np.zeros(n)
                    a[k] = amp
                    singles.append(float(m._activation_field(a, xe, ye,
                                                             g).sum()))
                    singles_d.append(float(m.last["direct_e"].sum()))
                out.append({
                    "amp_ua": amp, "pitch_um": pitch, "n": n,
                    "total": tot,
                    "relative_to_one": tot / np.mean(singles),
                    "efficiency": tot / (n * np.mean(singles)),
                    "efficiency_direct": direct / (n * np.mean(singles_d)),
                })
    return out


def charge_to_hold(m, base=300.0, counts=(4, 16, 64),
                   pitches=(400.0, 800.0), max_ua=20000.0):
    """Amplitude each of N electrodes needs to deliver one electrode's output.

    Where even max_ua falls short the row has amp_ua None and the fraction of
    the target reached at max_ua.  At 20 mA the initiation radius, 2.3 mm, stays
    inside the grid's 3.2 mm margin, so the search never reads the grid edge.

    The phenomenological model's ceiling made this explode (46 mA at N = 64).
    Delta k = 2 log10(I_N / I_1) is the rise in Shannon's k, independent of
    electrode area and pulse width because charge and charge density both
    scale with current.
    """
    out = []
    for pitch in pitches:
        for n in counts:
            xe, ye = square_grid(n, pitch)
            g = m._ensure_cortical_grid(xe, ye)
            k = n // 2                       # an interior electrode
            a = np.zeros(n)
            a[k] = base
            target = float(m._activation_field(a, xe, ye, g).sum())

            def per_channel(amp):
                return float(m._activation_field(np.full(n, amp), xe, ye,
                                                 g).sum()) / n

            lo, hi = base, base
            if per_channel(base) >= target:
                amp = base
            elif (top := per_channel(max_ua)) < target:
                out.append({"pitch_um": pitch, "n": n, "base_ua": base,
                            "amp_ua": None, "delta_k": None,
                            "fraction_at_max": top / target})
                continue
            else:
                while per_channel(hi) < target:
                    lo, hi = hi, hi * 2.0
                for _ in range(14):
                    mid = np.sqrt(lo * hi)
                    if per_channel(mid) < target:
                        lo = mid
                    else:
                        hi = mid
                amp = float(np.sqrt(lo * hi))
            out.append({"pitch_um": pitch, "n": n, "base_ua": base,
                        "amp_ua": amp,
                        "delta_k": float(2 * np.log10(amp / base))})
    return out


def interaction_range(rows, amps=(100.0, 300.0, 600.0), seps=SEPS):
    """Separation at which peak departure falls to a tenth of its 0.4 mm value.

    Linear interpolation in separation between sweep points.
    """
    out = {}
    for amp in amps:
        dev = np.array([next(r["max_dev"] for r in rows if r["amp_ua"] == amp
                             and r["separation_um"] == s) for s in seps])
        thr = dev[0] / 10.0
        k = int(np.argmax(dev < thr))
        if dev[k] >= thr:
            out[str(amp)] = None
            continue
        s0, s1, d0, d1 = seps[k - 1], seps[k], dev[k - 1], dev[k]
        out[str(amp)] = float(s0 + (d0 - thr) / (d0 - d1) * (s1 - s0))
    return out


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def sequence(m, sep, amp, delays=(0., 10., 20., 30., 50., 75., 100., 150.,
                                  200., 300.), on_ms=50.0):
    """Evoked output at B's site during B's train, relative to B alone."""
    xe = np.array([X0, X0 + sep])
    ye = np.array([Y0, Y0])
    g = m._ensure_cortical_grid(xe, ye)

    def b_mean(schedule, t0, t1):
        ts, out = m.simulate(schedule, xe, ye, g, t_end=t1)
        w = (ts > t0) & (ts <= t1)
        return float(out[w, 1].mean())

    solo = b_mean([(0.0, on_ms, (0.0, amp))], 0.0, on_ms)
    rows = []
    for D in delays:
        if D <= 0:
            sched = [(0.0, on_ms, (amp, amp))]
        elif D < on_ms:
            sched = [(0.0, D, (amp, 0.0)), (D, on_ms, (amp, amp)),
                     (on_ms, D + on_ms, (0.0, amp))]
        else:
            sched = [(0.0, on_ms, (amp, 0.0)), (D, D + on_ms, (0.0, amp))]
        rows.append({"delay_ms": float(D),
                     "second_over_solo": b_mean(sched, D, D + on_ms) / solo})
    return {"separation_um": sep, "amp_ua": amp, "solo": solo, "rows": rows}


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

HEAD_SEPS = (400.0, 800.0, 1600.0, 2400.0)


def headline(m):
    """The numbers the sensitivity sweep tracks, signatures included."""
    out = {}
    for amp in (100.0, 300.0):
        for sep in HEAD_SEPS:
            r = interaction(pair(m, sep, amp))
            out[f"signed_{amp:.0f}ua_{sep / 1000:.1f}mm"] = r["signed_total"]
            out[f"peak_{amp:.0f}ua_{sep / 1000:.1f}mm"] = r["max_dev"]
    dev = [out[f"peak_300ua_{s / 1000:.1f}mm"] for s in HEAD_SEPS]
    out["monotonic_decay_300ua"] = bool(np.all(np.diff(dev) <= 1e-4))
    xe, ye = square_grid(16, 400.0)
    g = m._ensure_cortical_grid(xe, ye)
    for amp in (100.0, 300.0):
        tot = float(m._activation_field(np.full(16, amp), xe, ye, g).sum())
        a = np.zeros(16)
        a[5] = amp
        one = float(m._activation_field(a, xe, ye, g).sum())
        out[f"efficiency_16ch_0.4mm_{amp:.0f}ua"] = tot / (16 * one)
    return out


def _no_map(p: SSNParams) -> SSNParams:
    """Orientation-averaged weights: a sheet with no orientation map."""
    cn = _ori_harmonics(p.A_near, p.B_near, p.sori_near, 1)[0]
    cf = _ori_harmonics(p.A_far, p.B_far, p.sori_far, 1)[0]
    return replace(p, A_near=cn, B_near=0.0, A_far=cf, B_far=0.0)


def _human(p: SSNParams, s: float = HUMAN_SCALE) -> SSNParams:
    """Every lateral length scaled by s, total synaptic weight unchanged.

    Rescaling the reference grid with the lengths is exactly the published
    model on a coarser grid, relabelled: each cell keeps its total input.
    """
    return replace(p, L_o=p.L_o * s, sigma_ee=p.sigma_ee * s,
                   sigma_ie=p.sigma_ie * s, sigma_i=p.sigma_i * s,
                   map_period_um=p.map_period_um * s,
                   ref_step_um=p.ref_step_um * s)


def variants():
    r, s = RecruitmentParams(), SSNParams()
    return {
        "baseline": {},
        "inhibitory sites x0 (none)": dict(
            recruit=replace(r, site_density_i=0.0)),
        "inhibitory sites x0.5": dict(
            recruit=replace(r, site_density_i=0.5 * r.site_density_i)),
        "inhibitory sites x1.5 (= excitatory)": dict(
            recruit=replace(r, site_density_i=r.site_density_e)),
        "inhibitory sites x3": dict(
            recruit=replace(r, site_density_i=3 * r.site_density_i)),
        "all sites x1/3": dict(recruit=replace(
            r, site_density_e=r.site_density_e / 3,
            site_density_i=r.site_density_i / 3)),
        "all sites x3": dict(recruit=replace(
            r, site_density_e=r.site_density_e * 3,
            site_density_i=r.site_density_i * 3)),
        "K_ax = 1292 (Stoney)": dict(recruit=replace(r, K_ax=1292.0)),
        "K_ax = 10000": dict(recruit=replace(r, K_ax=10000.0)),
        # Halving or doubling tau_rec moves the rate at which depression
        # saturates, 1 / (U tau_rec), by a factor of two either way.
        "weaker depression (tau_rec /2)": dict(recruit=replace(
            r, tau_rec_e=r.tau_rec_e / 2, tau_rec_i=r.tau_rec_i / 2)),
        "stronger depression (tau_rec x2)": dict(recruit=replace(
            r, tau_rec_e=r.tau_rec_e * 2, tau_rec_i=r.tau_rec_i * 2)),
        "pulse rate 200 Hz": dict(pulse_hz=200.0),
        "background 5 (quieter cortex)": dict(background=5.0),
        "background 14 (busier cortex)": dict(background=14.0),
        "no rate ceilings": dict(ssn=replace(s, r_max_e=1e9, r_max_i=1e9)),
        "no orientation map": dict(ssn=_no_map(s)),
        "human-scaled lengths (x1.63)": dict(ssn=_human(s),
                                             margin_um=3200.0 * HUMAN_SCALE),
        "orientation-map seed 1": dict(seed=1),
        "orientation-map seed 2": dict(seed=2),
    }


def sensitivity():
    out = {}
    for name, kw in variants().items():
        out[name] = h = headline(BiophysicalCortex(**kw))
        print(f"  {name:<38} {100*h['signed_100ua_0.4mm']:+6.1f} "
              f"{100*h['signed_300ua_0.4mm']:+6.1f} "
              f"{100*h['signed_300ua_1.6mm']:+6.1f}   "
              f"{100*h['peak_300ua_0.4mm']:5.1f} "
              f"{100*h['peak_300ua_1.6mm']:5.1f} "
              f"{'mono' if h['monotonic_decay_300ua'] else 'INTERIOR'}   "
              f"{h['efficiency_16ch_0.4mm_100ua']:5.2f} "
              f"{h['efficiency_16ch_0.4mm_300ua']:5.2f}")
        sys.stdout.flush()
    return out


# ---------------------------------------------------------------------------

def _row(rows, amp, sep):
    return next(x for x in rows if x["amp_ua"] == amp
                and x["separation_um"] == sep)


def update(parts) -> None:
    """Recompute only some parts and merge them into the existing results.

    python3 prosthesis/v2_experiments.py charge range

    Only these two parts can be rerun alone. Everything else depends on the
    pair sweep, so it is recomputed by a full run.
    """
    unknown = set(parts) - {"charge", "range"}
    if unknown:
        raise SystemExit(f"unknown part(s) {sorted(unknown)}; "
                         "choose from: charge, range")
    path = RESULTS / "v2_experiments.json"
    out = json.loads(path.read_text())
    full = BiophysicalCortex()
    if "charge" in parts:
        out["charge"] = charge_to_hold(full)
    if "range" in parts:
        out["interaction_range_um"] = interaction_range(out["pairs"])
    path.write_text(json.dumps(out, indent=2))
    for k in ("charge", "interaction_range_um"):
        if k in out:
            print(k, json.dumps(out[k], indent=1))


def main() -> None:
    if len(sys.argv) > 1:
        update(set(sys.argv[1:]))
        return
    out = {}
    full = BiophysicalCortex()

    print("pair interaction, signed total (%): full model | direct only")
    rows = out["pairs"] = pair_sweep(full)
    print("   uA " + "".join(f"{s/1000:>13.1f}mm" for s in SEPS))
    for amp in AMPS:
        cells = [f"{100*_row(rows, amp, s)['signed_total']:+6.1f}|"
                 f"{100*_row(rows, amp, s)['signed_direct']:+6.1f}"
                 for s in SEPS]
        print(f"  {amp:4.0f} " + "  ".join(cells))
    print("\npeak departure max|both - sum| / max|sum| (%), full model")
    for amp in AMPS:
        print(f"  {amp:4.0f} " + "  ".join(
            f"{100*_row(rows, amp, s)['max_dev']:6.1f}" for s in SEPS))
    out["crossover_ua"] = crossovers(rows)
    out["interaction_range_um"] = interaction_range(rows)
    print("\nseparation where peak departure falls to a tenth: " + "  ".join(
        f"{float(a):.0f}uA:{v/1000:.2f}mm" for a, v in
        out["interaction_range_um"].items() if v))
    out["signature"] = {str(a): signatures(rows, a)
                        for a in (100.0, 300.0, 600.0)}
    print("\ncrossover current (facilitatory -> suppressive), by separation")
    print("  " + "  ".join(
        f"{float(s)/1000:.1f}mm:{'-' if v is None else f'{v:.0f}uA'}"
        for s, v in out["crossover_ua"].items()))
    for a, sg in out["signature"].items():
        print(f"  peak departure at {float(a):.0f} uA largest at "
              f"{sg['peak_at_um']/1000:.1f} mm, "
              f"{'monotonic decay' if sg['monotonic_decay'] else 'interior'}")
    sys.stdout.flush()

    print("\nextent of the evoked region against current")
    out["extent"] = extent(full)
    for r in out["extent"]:
        print(f"  {r['amp_ua']:5.0f} uA  half-max "
              f"{r['halfmax_diameter_mm']:.2f} mm  above {VISIBLE_HZ} Hz "
              f"{r['visible_diameter_mm']:.2f} mm  (K=675 ball "
              f"{r['tehovnik_diameter_mm']:.2f} mm)  peak "
              f"{r['peak_evoked_hz']:5.1f} Hz  total "
              f"{r['total_evoked']:8.1f}")
    sys.stdout.flush()

    print("\nchannel count: delivered / sum of singles  (full | direct only)")
    out["channels"] = channels(full)
    for amp in (30.0, 100.0, 300.0):
        for pitch in (400.0, 800.0, 1600.0):
            cr = [r for r in out["channels"]
                  if r["amp_ua"] == amp and r["pitch_um"] == pitch]
            print(f"  {amp:4.0f} uA pitch {pitch/1000:.1f}mm  " + " ".join(
                f"N={r['n']}:{r['efficiency']:.2f}|"
                f"{r['efficiency_direct']:.2f}" for r in cr))
        sys.stdout.flush()

    print("\namplitude per electrode to hold one electrode's output (300 uA)")
    out["charge"] = charge_to_hold(full)
    for r in out["charge"]:
        print(f"  pitch {r['pitch_um']/1000:.1f}mm N={r['n']:3d}  "
              f"{r['amp_ua']:7.1f} uA   delta k {r['delta_k']:+.2f}")
    sys.stdout.flush()

    print("\nsequence: second phosphene relative to alone")
    out["sequence"] = {}
    for label, phi in (("fast inhibition only", 0.0),
                       ("30% slow inhibition", 0.3)):
        m = BiophysicalCortex(ssn=replace(SSNParams(), slow_inh_frac=phi))
        res = [sequence(m, sep, amp) for sep in (800.0, 1600.0)
               for amp in (100.0, 300.0)]
        out["sequence"][label] = res
        for r in res:
            print(f"  {label:<22} {r['separation_um']/1000:.1f}mm "
                  f"{r['amp_ua']:.0f}uA  " + " ".join(
                      f"{x['delay_ms']:.0f}:{x['second_over_solo']:.2f}"
                      for x in r["rows"]))
        sys.stdout.flush()

    print("\nsensitivity: signed % (100uA/0.4mm, 300uA/0.4mm, 300uA/1.6mm)"
          "   peak % (300uA 0.4, 1.6mm) shape   16-ch efficiency "
          "(100, 300 uA)")
    out["sensitivity"] = sensitivity()

    (RESULTS / "v2_experiments.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'v2_experiments.json'}")


if __name__ == "__main__":
    main()
