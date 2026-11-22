"""The rest of the manuscript's experiments, on the model built from measured parts.

``v2_experiments.py`` covers pairs, arrays, charge and sequences.  This script
covers what the first model did in visual-field terms, and the comparisons
with human data:

    size       phosphene diameter against current, read out through the
               implant's retinotopy exactly as the Dynaphos baseline is, and
               compared with the baseline curve the first model was
               calibrated to.  Not fitted.
    bosking    interaction at the separations Bosking et al. (2022) report,
               from 0.1 to 4 mA, and the distance over which it decays.  Above
               ~0.6 mA this pushes recruitment far past the currents it was
               measured at, and their contacts sat on the surface, so these
               rows bound the effect and do not predict it.  Each row also
               carries the repulsion measure below.
    repulsion  whether two phosphenes push apart: the separation of the two
               activity centroids, paired against summed, for this model and
               for the pooled divisor, with one estimator for both
    percepts   the percept decoder, rebuilt on this model's orientation map:
               held-out reconstruction, one electrode, five together
    encoder    optimal stimulation.  The independent forward model is the sum
               of this model's own single-electrode responses, so the two
               agree exactly whenever one electrode is on.  Programmed as
               independent, delivered to the full model, then re-optimised
               for it
    power      how much data the shape experiment needs when the candidates
               are this model and a pooled divisor
    inhibition Kim et al. (2025) as a second constraint on how many inhibitory
               cells an electrode drives directly, the parameter that sets the
               sign at 100 uA.  Each multiple of the calibrated density gives
               an I/E ratio to set against Dadarlat et al.'s 1.1

Each part writes ``results/v2_paper_<part>.json`` (and, where a figure needs
fields, ``results/v2_fields_<part>.npz``), so parts can run in parallel.

Run:  python3 prosthesis/v2_paper.py [part ...]      (default: all)
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pulse2percept.implants.cortex import Cortivis

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.cortex_ssn import BiophysicalCortex, RecruitmentParams
from prosthesis.percept_decoder import (CorticalCode, InversePercept,
                                        pink_noise, reconstruction_r2)
from prosthesis.phase2_percepts import CTX_X, CTX_Y, pick_line
from prosthesis.phase4_encoder import (AMP_MAX, contrast, fidelity,
                                       pick_electrodes)
from prosthesis.phase9_sequence import fit_scale
from prosthesis.phase9_sequence import power as shape_power
from prosthesis.v2_experiments import (SEPS, VISIBLE_HZ, X0, Y0, interaction,
                                       interaction_range, pair)
from prosthesis.v2_validate import dual_electrode

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

# Phase 1's visual-field grid, so phosphene sizes are measured the same way.
XR, YR, STEP = (-3.2, -0.4), (-0.2, 2.0), 0.02
BOSKING_SEPS = (4110.0, 9570.0, 13470.0, 26360.0)


def _save(part, data, fields=None):
    (RESULTS / f"v2_paper_{part}.json").write_text(json.dumps(data, indent=2))
    if fields:
        np.savez_compressed(RESULTS / f"v2_fields_{part}.npz", **fields)
    print(f"wrote {RESULTS / f'v2_paper_{part}.json'}", flush=True)


# ---------------------------------------------------------------------------
# Reading cortex out in the visual field
# ---------------------------------------------------------------------------

class Retinotopy:
    """Visual-field grid points and the cortical location each one reads.

    The grid and its cortical coordinates come from the same pulse2percept
    spatial model the Dynaphos comparison uses, so the two readouts differ
    only in the cortical field they sample.
    """

    def __init__(self, xrange=XR, yrange=YR, step=STEP):
        m = CircuitSpatial(xrange=xrange, yrange=yrange, xystep=step)
        m.build()
        g = m.grid["v1"]
        self.shape = np.shape(g.x)
        self.cx = np.asarray(g.x, float).ravel()
        self.cy = np.asarray(g.y, float).ravel()
        self.step = step

    def sample(self, field, g):
        xs, ys = g["xs"], g["ys"]
        h = xs[1] - xs[0]
        col = (self.cx - xs[0]) / h
        row = (self.cy - ys[0]) / h
        inside = (np.isfinite(col) & np.isfinite(row) & (col >= 0)
                  & (col <= len(xs) - 1) & (row >= 0) & (row <= len(ys) - 1))
        out = np.zeros(self.cx.size)
        if inside.any():
            out[inside] = map_coordinates(field, [row[inside], col[inside]],
                                          order=1, mode="constant")
        return out.reshape(self.shape)


def disc_diameter(img, step=STEP, level=None):
    """Diameter of the disc with the area above ``level`` (default half max)."""
    peak = float(img.max())
    if peak <= 0:
        return 0.0
    thr = peak / 2.0 if level is None else level
    return float(2.0 * np.sqrt((img >= thr).sum() * step ** 2 / np.pi))


def size_curve(m, amps=(10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0,
                        600.0)):
    """Single-electrode phosphene size against the Dynaphos baseline."""
    imp = Cortivis()
    e = imp.electrode_names[0]
    xe, ye = np.array([imp[e].x]), np.array([imp[e].y])
    g = m._ensure_cortical_grid(xe, ye)
    vf = Retinotopy()
    base = json.loads((RESULTS / "phase1_circuit.json").read_text())
    dyn = {float(a): float(d) for a, d in zip(base["amps"],
                                              base["baseline_sizes"])}
    rows, imgs = [], {}
    for a in amps:
        ev = m._activation_field([a], xe, ye, g)
        img = vf.sample(ev, g)
        rows.append({
            "amp_ua": a,
            "halfmax_diameter_deg": disc_diameter(img),
            "visible_diameter_deg": disc_diameter(img, level=VISIBLE_HZ)
            if img.max() >= VISIBLE_HZ else 0.0,
            "dynaphos_diameter_deg": dyn.get(a),
            "peak_evoked_hz": float(ev.max()),
        })
        if a in (50.0, 300.0):
            imgs[f"vf_{a:.0f}ua"] = img
        if a == 100.0:
            imgs["cortex_100ua"] = ev
            imgs["p_e_100ua"] = m.last["p_e"]
            imgs["p_i_100ua"] = m.last["p_i"]
            imgs["theta"] = g["sheet"].theta
            imgs["xs"], imgs["ys"] = g["xs"], g["ys"]
            imgs["electrode"] = np.array([xe[0], ye[0]])
    return rows, imgs


# ---------------------------------------------------------------------------
# Human comparisons
# ---------------------------------------------------------------------------

def centroid_separation(field, xs, ys, xa, xb):
    """Distance between the activity centroids either side of the midline.

    Pairs lie along x.  Only excitation counts, as a phosphene is a region
    of raised activity.  Compared against the same estimator applied to the
    sum of the single responses, so any linear model scores exactly zero:
    the overlap of two blobs pulls their centroids together in the sum and
    in the pair alike.
    """
    X = np.broadcast_to(xs[None, :], field.shape)
    w = np.maximum(field, 0.0)
    left = X < 0.5 * (xa + xb)
    wl, wr = w[left].sum(), w[~left].sum()
    if wl <= 0 or wr <= 0:
        return float("nan")
    return float((w[~left] * X[~left]).sum() / wr
                 - (w[left] * X[left]).sum() / wl)


def repulsion(f, g, sep):
    lin = f["a"] + f["b"]
    d_lin = centroid_separation(lin, g["xs"], g["ys"], X0, X0 + sep)
    d_ab = centroid_separation(f["ab"], g["xs"], g["ys"], X0, X0 + sep)
    return float(d_ab / d_lin - 1.0)


def bosking(m, amps=(100.0, 300.0, 600.0, 1000.0, 2000.0, 4000.0),
            seps=(400.0, 800.0, 1600.0, 2400.0, 3200.0, 4110.0, 6400.0,
                  9570.0, 13470.0, 26360.0)):
    rows = []
    for a in amps:
        for s in seps:
            t0 = time.time()
            f = pair(m, s, a)
            g = m._ensure_cortical_grid(np.array([X0, X0 + s]),
                                        np.array([Y0, Y0]))
            r = {"amp_ua": a, "separation_um": s, **interaction(f),
                 "repulsion": repulsion(f, g, s)}
            rows.append(r)
            print(f"  {a:6.0f} uA {s/1000:6.2f} mm  signed "
                  f"{100*r['signed_total']:+7.2f}%  peak "
                  f"{100*r['max_dev']:6.2f}%  repulsion "
                  f"{100*r['repulsion']:+6.2f}%  ({time.time()-t0:.0f}s)",
                  flush=True)
    return {"rows": rows,
            "range_to_tenth_um": interaction_range(rows, amps=amps,
                                                   seps=seps)}


def repulsion_sweep(m, amps=(100.0, 300.0, 600.0), seps=SEPS):
    rows = []
    for a in amps:
        for s in seps:
            f = pair(m, s, a)
            g = m._ensure_cortical_grid(np.array([X0, X0 + s]),
                                        np.array([Y0, Y0]))
            rows.append({"amp_ua": a, "separation_um": s,
                         "repulsion": repulsion(f, g, s)})
    return rows


def peak_shift(f, g, sep):
    """Change in the distance between the two maxima on the electrode axis.

    A cubic profile along the line joining the electrodes, as in phase 6.  It
    reads the maxima only, where the centroid estimator above reads the
    whole of each region, and the two can disagree in sign.
    """
    from prosthesis.phase6_controls import axis_profile, two_peak_separation
    t, v_lin = axis_profile(f["a"] + f["b"], g, X0, Y0, X0 + sep, Y0)
    _, v_con = axis_profile(f["ab"], g, X0, Y0, X0 + sep, Y0)
    d_lin = two_peak_separation(t, v_lin, sep)
    d_con = two_peak_separation(t, v_con, sep)
    if d_lin is None or d_con is None or d_lin <= 0:
        return None
    return float(d_con / d_lin - 1.0)


PEAK_SEPS = (1800.0, 2200.0, 2600.0, 3000.0, 3600.0, 4200.0, 5000.0)


def peak_sweep(m, amps=(100.0, 300.0, 600.0), seps=PEAK_SEPS):
    rows = []
    for a in amps:
        for s in seps:
            f = pair(m, s, a)
            g = m._ensure_cortical_grid(np.array([X0, X0 + s]),
                                        np.array([Y0, Y0]))
            rows.append({"amp_ua": a, "separation_um": s,
                         "peak_shift": peak_shift(f, g, s)})
    return rows


def pooled_repulsion(amps=(100.0, 300.0, 600.0), seps=SEPS,
                     peak_seps=PEAK_SEPS):
    """Both estimators on the pooled divisor's cortical activation.

    The peak estimator runs on a 20 um sheet, as in phase 6, because the
    shifts it measures are tens of microns.
    """
    from prosthesis.phase1_circuit import calibrated
    out = {}
    for key, step, ss in (("centroid", None, seps),
                          ("peaks", 20.0, peak_seps)):
        kw = {} if step is None else {"cortex_step": step}
        m = CircuitSpatial(xrange=XR, yrange=YR, xystep=0.05, **kw,
                           **calibrated("full circuit"))
        m.build()
        rows = []
        for a in amps:
            for s in ss:
                xe, ye = np.array([X0, X0 + s]), np.array([Y0, Y0])
                g = m._ensure_cortical_grid(xe, ye)
                f = {"a": m._activation_field(np.array([a, 0.0]), xe, ye, g),
                     "b": m._activation_field(np.array([0.0, a]), xe, ye, g),
                     "ab": m._activation_field(np.array([a, a]), xe, ye, g)}
                r = {"amp_ua": a, "separation_um": s}
                if key == "centroid":
                    r["repulsion"] = repulsion(f, g, s)
                else:
                    r["peak_shift"] = peak_shift(f, g, s)
                rows.append(r)
        out[key] = rows
    return out


# ---------------------------------------------------------------------------
# Percepts
# ---------------------------------------------------------------------------

def array_grid(m, imp):
    """One grid covering the whole array, so every layout shares one map."""
    xs = np.array([imp[k].x for k in imp.electrode_names], float)
    ys = np.array([imp[k].y for k in imp.electrode_names], float)
    return m._ensure_cortical_grid(xs, ys)


def bio_code(g, step_um=100.0):
    """Decoder sites that read this model's own orientation map."""
    th = np.rad2deg(g["sheet"].theta)
    h = g["xs"][1] - g["xs"][0]

    def theta_at(X, Y):
        i = np.clip(np.rint((Y - g["ys"][0]) / h).astype(int), 0, g["n"] - 1)
        j = np.clip(np.rint((X - g["xs"][0]) / h).astype(int), 0, g["n"] - 1)
        return th[i, j]

    return CorticalCode(CTX_X, CTX_Y, step_um=step_um, theta_at=theta_at)


def percepts(m, amp=300.0):
    imp = Cortivis()
    g = array_grid(m, imp)
    code = bio_code(g)
    iterative = InversePercept(code, lam=0.02)
    rng = np.random.default_rng(1234)
    test = pink_noise(24, code.n_px, rng)
    recon = iterative.round_trip(test)
    r2 = reconstruction_r2(test, recon)
    print(f"  decoder on this model's map: {code.n_sites} sites, held-out "
          f"R^2 = {r2:.3f}", flush=True)
    # Stimulation is read with the fixed linear operator.  The iterative
    # inverse drops silent sites, which makes it nonlinear, and a nonlinear
    # readout could manufacture interaction of its own.  This model
    # suppresses activity, so silent and negative sites are common.
    dec = InversePercept(code, lam=0.02, precompute=True)

    def at_sites(names, amps):
        xe = np.array([imp[n].x for n in names], float)
        ye = np.array([imp[n].y for n in names], float)
        return code.sample_field(m._activation_field(amps, xe, ye, g),
                                 g["xs"], g["ys"])

    e0 = imp.electrode_names[0]
    single = dec.decode(at_sites([e0], [amp]))[0]

    names = pick_line(imp, n=5, spacing_um=1400.0)
    xe = np.array([imp[n].x for n in names], float)
    ye = np.array([imp[n].y for n in names], float)
    seps = [float(np.hypot(x - xe[0], y - ye[0])) for x, y in zip(xe, ye)]
    alone = np.zeros(code.n_sites)
    for i in range(len(names)):
        a = np.zeros(len(names))
        a[i] = amp
        alone += at_sites(names, a)
    together = at_sites(names, np.full(len(names), amp))
    img_alone, img_tog = dec.decode(alone)[0], dec.decode(together)[0]
    diff = img_tog - img_alone
    out = {
        "decoder_r2": r2, "n_sites": code.n_sites, "n_px": code.n_px,
        "line_separations_um": seps, "amp_ua": amp,
        "activation_change": float(np.abs(together - alone).max()
                                   / np.abs(alone).max()),
        "percept_change": float(np.abs(diff).max() / np.abs(img_alone).max()),
        "activation_signed": float((together - alone).sum() / alone.sum()),
        "percept_rms_ratio": float(np.std(img_tog) / np.std(img_alone)),
        "percept_correlation": float(np.corrcoef(img_tog.ravel(),
                                                 img_alone.ravel())[0, 1]),
    }
    print(f"  five electrodes, {amp:.0f} uA: activation changes "
          f"{100*out['activation_change']:.1f}%, percept "
          f"{100*out['percept_change']:.1f}%", flush=True)
    pooled, pooled_fields = pooled_line(names, amp)
    out["pooled_divisor"] = pooled
    print(f"  pooled divisor, same electrodes and readout: activation "
          f"{100*pooled['activation_change']:.1f}%, percept "
          f"{100*pooled['percept_change']:.1f}%", flush=True)
    fields = {"test": test[:6], "recon": recon[:6], "single": single,
              "line_alone": img_alone, "line_together": img_tog,
              **pooled_fields}
    return out, fields


def pooled_line(names, amp):
    """The same five electrodes under the pooled divisor, for comparison.

    Read through the decoder built on that model's own orientation map, with
    the same fixed linear operator, so the two rows of the figure differ only
    in the forward model.
    """
    from prosthesis.phase1_circuit import calibrated
    imp = Cortivis()
    code = CorticalCode(CTX_X, CTX_Y)
    dec = InversePercept(code, lam=0.02, precompute=True)
    m = CircuitSpatial(xrange=XR, yrange=YR, xystep=0.05,
                       **calibrated("full circuit"))
    m.build()
    xe = np.array([imp[n].x for n in names], float)
    ye = np.array([imp[n].y for n in names], float)
    g = m._ensure_cortical_grid(xe, ye)

    def at_sites(amps):
        return code.sample_field(m._activation_field(np.asarray(amps, float),
                                                     xe, ye, g),
                                 g["xs"], g["ys"])

    alone = np.zeros(code.n_sites)
    for i in range(len(names)):
        a = np.zeros(len(names))
        a[i] = amp
        alone += at_sites(a)
    together = at_sites(np.full(len(names), amp))
    img_alone, img_tog = dec.decode(alone)[0], dec.decode(together)[0]
    out = {
        "activation_change": float(np.abs(together - alone).max()
                                   / np.abs(alone).max()),
        "percept_change": float(np.abs(img_tog - img_alone).max()
                                / np.abs(img_alone).max()),
        "activation_signed": float((together - alone).sum() / alone.sum()),
        "percept_rms_ratio": float(np.std(img_tog) / np.std(img_alone)),
    }
    return out, {"pooled_alone": img_alone, "pooled_together": img_tog}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

AMP_GRID = np.arange(0.0, AMP_MAX + 1.0, 25.0)


class Forward:
    """amplitudes -> decoded percept, for the full model and for independence.

    The independent model is the sum of this model's own single-electrode
    percepts, tabulated every 25 uA and interpolated.  It is therefore the
    best independent model there could be: right about every electrode on
    its own, wrong only in assuming they add.
    """

    def __init__(self, m, imp, names, build_table=True):
        self.m = m
        self.g = array_grid(m, imp)
        self.code = bio_code(self.g)
        self.dec = InversePercept(self.code, lam=0.02, precompute=True)
        self.xe = np.array([imp[n].x for n in names], float)
        self.ye = np.array([imp[n].y for n in names], float)
        self.n = len(names)
        self.calls = 0
        if build_table:
            self._build_table()

    def _build_table(self):
        t0 = time.time()
        npx = self.code.n_px
        self.table = np.zeros((self.n, AMP_GRID.size, npx, npx))
        for i in range(self.n):
            for k, a in enumerate(AMP_GRID[1:], 1):
                v = np.zeros(self.n)
                v[i] = a
                self.table[i, k] = self._raw(v)
        print(f"  single-electrode table: {self.n} x {AMP_GRID.size - 1} "
              f"runs in {time.time() - t0:.0f}s", flush=True)
        k300 = int(np.argmin(np.abs(AMP_GRID - 300.0)))
        self.scale = 1.0 / float(np.std(self.table[0, k300]))

    def _raw(self, amps):
        self.calls += 1
        f = self.m._activation_field(np.asarray(amps, float), self.xe,
                                     self.ye, self.g)
        a = self.code.sample_field(f, self.g["xs"], self.g["ys"])
        return self.dec.decode(a)[0]

    def full(self, amps):
        return self._raw(np.clip(amps, 0.0, AMP_MAX)) * self.scale

    def independent(self, amps):
        u = np.clip(np.asarray(amps, float), 0.0, AMP_MAX) / 25.0
        lo = np.minimum(np.floor(u).astype(int), AMP_GRID.size - 2)
        w = u - lo
        idx = np.arange(self.n)
        p = (self.table[idx, lo] * (1 - w)[:, None, None]
             + self.table[idx, lo + 1] * w[:, None, None])
        return p.sum(0) * self.scale


def linear_fit_fidelity(singles, target):
    """Fidelity of the best unconstrained linear combination of percepts.

    ``singles`` holds one single-electrode percept per electrode.  Least
    squares with target and percepts both mean-removed, since squared
    correlation ignores an offset, so any sum of the singles scores exactly
    one.  A scale for whether a target is within reach of the electrode set
    at all.  It is not a strict ceiling, because percepts are not linear in
    current.  (Phase 4's ``linear_reference`` left the percepts' means in,
    which costs a target some fidelity it could have had.)
    """
    B = np.asarray(singles, float).reshape(len(singles), -1)
    B = B - B.mean(axis=1, keepdims=True)
    t = np.ravel(target) - np.mean(target)
    w, *_ = np.linalg.lstsq(B.T, t, rcond=None)
    fit = B.T @ w
    den = float(np.dot(fit, fit) * np.dot(t, t))
    return float(np.dot(fit, t) ** 2 / den) if den > 0 else 0.0


def shape_targets(n):
    """Thin geometric shapes on an n x n percept grid."""
    c = n // 2
    shapes = {}
    shapes["horizontal bar"] = np.zeros((n, n))
    shapes["horizontal bar"][c, 2:-2] = 1.0
    shapes["vertical bar"] = shapes["horizontal bar"].T.copy()
    shapes["diagonal bar"] = np.eye(n)
    sq = np.zeros((n, n))
    sq[3, 3:-3] = sq[-4, 3:-3] = sq[3:-3, 3] = sq[3:-3, -4] = 1.0
    shapes["square outline"] = sq
    return shapes


def reach(m, amp=400.0, n_targets=6, n_active=5, seed=0):
    """Which targets a linear combination of single percepts can reach.

    The same 24 electrodes and the same deliverable targets as the encoder
    (five electrodes at 450 uA, same seed), against thin geometric shapes.
    48 runs of the network, so it is cheap next to the encoder itself.
    """
    imp = Cortivis()
    names = pick_electrodes(imp, 24)
    F = Forward(m, imp, names, build_table=False)
    eye = np.eye(F.n)
    singles = np.array([F._raw(eye[i] * amp) for i in range(F.n)])
    at450 = np.array([F._raw(eye[i] * 450.0) for i in range(F.n)])
    rng = np.random.default_rng(seed)
    deliverable = []
    for _ in range(n_targets):
        idx = rng.choice(F.n, size=n_active, replace=False)
        deliverable.append(linear_fit_fidelity(singles, at450[idx].sum(0)))
    shapes = {k: linear_fit_fidelity(singles, v)
              for k, v in shape_targets(F.code.n_px).items()}
    for i, f in enumerate(deliverable, 1):
        print(f"  deliverable target {i}: {f:.3f}", flush=True)
    for k, f in shapes.items():
        print(f"  {k:16s} {f:.3f}", flush=True)
    return {"amp_ua": amp, "electrodes": names,
            "deliverable": deliverable, "shapes": shapes}


def _optimise(fn, target, x0, objective, maxiter):
    if objective == "fidelity":
        f = lambda a: -fidelity(fn(a), target)
    else:
        f = lambda a: float(np.mean((fn(a) - target) ** 2))
    res = minimize(f, np.asarray(x0, float), method="L-BFGS-B",
                   bounds=[(0.0, AMP_MAX)] * len(x0),
                   options={"maxiter": maxiter, "eps": 5.0})
    return res.x


def encoder(m, n_targets=6, n_active=5, maxiter_full=15, seed=0):
    imp = Cortivis()
    names = pick_electrodes(imp, 24)
    F = Forward(m, imp, names)
    rng = np.random.default_rng(seed)
    rows, examples = {}, {}
    for t in range(n_targets):
        true = np.zeros(F.n)
        true[rng.choice(F.n, size=n_active, replace=False)] = 450.0
        tgt = F.independent(true)
        t0, c0 = time.time(), F.calls
        # Programmed under independence (cheap: the table).
        a_ind = _optimise(F.independent, tgt, true, "fidelity", 60)
        f_ind_ind = fidelity(F.independent(a_ind), tgt)
        # Delivered to the full model.
        p_ind_full = F.full(a_ind)
        # Re-optimised for the full model, from the independent solution.
        a_full = _optimise(F.full, tgt, a_ind, "fidelity", maxiter_full)
        p_full = F.full(a_full)
        # A brightness-seeking objective under the full model.
        a_mse = _optimise(F.full, tgt, true, "mse", maxiter_full)
        p_mse = F.full(a_mse)
        ct = contrast(tgt)
        rows[f"pattern {t + 1}"] = {
            "indep_on_indep": f_ind_ind,
            "indep_on_full": fidelity(p_ind_full, tgt),
            "full_on_full": fidelity(p_full, tgt),
            "contrast_ratio_mismatched": contrast(p_ind_full) / ct,
            "contrast_ratio_matched": contrast(p_full) / ct,
            "contrast_ratio_mse": contrast(p_mse) / ct,
            "fidelity_mse": fidelity(p_mse, tgt),
            "current_indep": float(a_ind.sum()),
            "current_mse": float(a_mse.sum()),
            "max_amp_mse": float(a_mse.max()),
        }
        r = rows[f"pattern {t + 1}"]
        print(f"  target {t + 1}: fidelity indep {r['indep_on_indep']:.3f} "
              f"-> on full {r['indep_on_full']:.3f}, re-optimised "
              f"{r['full_on_full']:.3f}; contrast {r['contrast_ratio_mismatched']:.2f}"
              f" (mse {r['contrast_ratio_mse']:.2f}); "
              f"{F.calls - c0} runs, {time.time() - t0:.0f}s", flush=True)
        if t < 3:
            examples[f"t{t}_target"] = tgt
            examples[f"t{t}_indep"] = F.independent(a_ind)
            examples[f"t{t}_mismatch"] = p_ind_full
            examples[f"t{t}_matched"] = p_full
    keys = ("indep_on_indep", "indep_on_full", "full_on_full",
            "contrast_ratio_mismatched", "contrast_ratio_matched",
            "contrast_ratio_mse", "fidelity_mse")
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
    mean["fraction_lost"] = float(np.mean([
        (r["full_on_full"] - r["indep_on_full"]) / r["full_on_full"]
        for r in rows.values()]))
    mean["mse_reached_bound"] = bool(any(r["max_amp_mse"] >= AMP_MAX - 1
                                         for r in rows.values()))
    print("  mean " + "  ".join(f"{k} {v:.3f}" for k, v in mean.items()
                                if not isinstance(v, bool)), flush=True)
    return {"electrodes": names, "results": rows, "mean": mean,
            "maxiter_full": maxiter_full}, examples


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

SEMS = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)


def power_part(m, amp=300.0):
    """Can the shape experiment tell this model from a pooled divisor?"""
    sw = json.loads((RESULTS / "phase1_circuit.json").read_text())
    rows = sorted(sw["separation_sweep"]["full circuit"],
                  key=lambda r: r["separation_um"])
    seps = np.array([r["separation_um"] for r in rows])
    pooled = np.array([100 * r["max_dev"] for r in rows])
    bio = np.array([100 * interaction(pair(m, s, amp))["max_dev"]
                    for s in seps])
    out = {"separations_um": seps.tolist(), "curve_pooled": pooled.tolist(),
           "curve_measured": bio.tolist(),
           "shape_correlation": float(np.corrcoef(bio, pooled)[0, 1])}
    table, tol = [], None
    for sem in SEMS:
        pb = shape_power(bio, pooled, sem)
        pp = shape_power(pooled, bio, sem)
        table.append({"sem_pp": sem, "power_if_measured": pb,
                      "power_if_pooled": pp})
        if tol is None and min(pb, pp) < 0.8:
            tol = sem
        print(f"  sem {sem:5.0f} pp  power if measured-parts {pb:.3f}  "
              f"if pooled {pp:.3f}", flush=True)
    out["rows"], out["sem_tolerance_pp"] = table, tol
    subsets = {"all": list(range(len(seps))),
               "0.4/1.6/3.2 mm": [int(np.argmin(np.abs(seps - s)))
                                  for s in (400, 1600, 3200)]}
    red = {}
    for name, idx in subsets.items():
        i = np.array(idx)
        t = None
        for sem in SEMS:
            if min(shape_power(bio[i], pooled[i], sem),
                   shape_power(pooled[i], bio[i], sem)) < 0.8:
                t = sem
                break
        red[name] = t
    out["reduced_protocols"] = red
    rng = np.random.default_rng(7)
    drift = []
    for beta in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        ramp = 1.0 - beta * np.linspace(0.0, 1.0, pooled.size)
        hits = 0
        for _ in range(3000):
            obs = pooled * ramp + rng.normal(0.0, 20.0, size=pooled.size)
            hits += fit_scale(pooled, obs)[1] < fit_scale(bio, obs)[1]
        drift.append({"drift": beta, "power_if_pooled": hits / 3000})
    out["drift_confound"] = drift
    return out


INH_MULTS = (0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0)
KIM_SEPS, KIM_AMPS = (370.0, 485.0, 558.0, 850.0), (6.8, 12.7, 15.9)


def crossing(xs, ys):
    """First x where ys turns from negative to non-negative, linearly
    interpolated, or None if it never does."""
    for xa, xb, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
        if ya < 0 <= yb:
            return float(xa + (xb - xa) * ya / (ya - yb))
    return None


def inhibition(mults=INH_MULTS):
    """Kim et al. and Dadarlat et al. as two constraints on one number.

    For each multiple of the calibrated inhibitory site density: the
    recruited I/E ratio within 300 um at 20 uA (Dadarlat's measure), Kim et
    al.'s net paired modulation at their three currents and four
    separations, and the signed interaction at 100 and 300 uA, 0.4 mm.  The
    crossings are the multiples where each turns from suppression to
    enhancement, with the I/E ratio there.
    """
    r = RecruitmentParams()
    rows = []
    for k in mults:
        rec = replace(r, site_density_i=k * r.site_density_i)
        fine = BiophysicalCortex(step_um=26.75, margin_um=2600.0, recruit=rec)
        g = fine._ensure_cortical_grid(np.array([X0]), np.array([Y0]))
        X, Y = np.meshgrid(g["xs"], g["ys"])
        near = np.hypot(X - X0, Y - Y0) < 300
        pe, pi = fine.recruited([20.0], [X0], [Y0], g)
        m = BiophysicalCortex(recruit=rec)
        row = {"mult": k,
               "ie_ratio_300um_20ua": float(pi[near].mean() / pe[near].mean()),
               "kim": dual_electrode(KIM_SEPS, KIM_AMPS, BiophysicalCortex(
                   pulse_hz=200.0, recruit=rec))}
        for a in (100.0, 300.0):
            row[f"signed_{a:.0f}ua_0.4mm"] = interaction(
                pair(m, 400.0, a))["signed_total"]
        rows.append(row)
        kim = {x["amp_ua"]: x for x in row["kim"]
               if x["separation_um"] == 485.0}
        print(f"  x{k:<5} I/E {row['ie_ratio_300um_20ua']:.2f}  Kim 485 um "
              + " ".join(f"{100 * kim[a]['net_fraction']:+6.2f}"
                         for a in KIM_AMPS)
              + "  E1 view " + " ".join(
                  f"{100 * kim[a]['net_fraction_e1_view']:+6.2f}"
                  for a in KIM_AMPS)
              + f"  signed 100/300 uA {100 * row['signed_100ua_0.4mm']:+6.2f}"
              f" {100 * row['signed_300ua_0.4mm']:+6.2f}", flush=True)

    ks = [x["mult"] for x in rows]
    ie = [x["ie_ratio_300um_20ua"] for x in rows]

    def at(ys):
        k = crossing(ks, ys)
        return {"mult": k,
                "ie_ratio": None if k is None else float(np.interp(k, ks, ie))}

    cross = {f"signed_{a:.0f}ua_0.4mm": at([x[f"signed_{a:.0f}ua_0.4mm"]
                                            for x in rows])
             for a in (100.0, 300.0)}
    for sep in KIM_SEPS:
        for a in KIM_AMPS:
            for view in ("net_fraction", "net_fraction_e1_view"):
                cross[f"kim_{a}ua_{sep:.0f}um_{view}"] = at(
                    [next(y[view] for y in x["kim"]
                          if y["separation_um"] == sep and y["amp_ua"] == a)
                     for x in rows])
    for name, c in cross.items():
        if "12.7" in name or name.startswith("signed"):
            print(f"  turns positive: {name:<40} "
                  + ("never" if c["mult"] is None else
                     f"x{c['mult']:.2f}  (I/E {c['ie_ratio']:.2f})"),
                  flush=True)
    return {"rows": rows, "crossings": cross, "dadarlat_ratio": 1.1}


# ---------------------------------------------------------------------------

PARTS = ("size", "bosking", "repulsion", "percepts", "reach",
         "encoder", "power", "inhibition")


def run(part):
    t0 = time.time()
    m = BiophysicalCortex()
    print(f"\n=== {part}", flush=True)
    if part == "size":
        rows, imgs = size_curve(m)
        for r in rows:
            d = r["dynaphos_diameter_deg"]
            print(f"  {r['amp_ua']:5.0f} uA  half-max "
                  f"{r['halfmax_diameter_deg']:.3f} deg  visible "
                  f"{r['visible_diameter_deg']:.3f} deg  Dynaphos "
                  f"{'-' if d is None else f'{d:.3f}'} deg", flush=True)
        _save(part, {"rows": rows}, imgs)
    elif part == "bosking":
        _save(part, bosking(m))
    elif part == "repulsion":
        pooled = pooled_repulsion()
        out = {"measured_parts": repulsion_sweep(m),
               "pooled_divisor": pooled["centroid"],
               "peaks_measured_parts": peak_sweep(m),
               "peaks_pooled_divisor": pooled["peaks"]}
        for name, rows in out.items():
            key = "peak_shift" if name.startswith("peaks") else "repulsion"
            for a in sorted({r["amp_ua"] for r in rows}):
                vals = [(r[key], r["separation_um"]) for r in rows
                        if r["amp_ua"] == a and r[key] is not None]
                lo, hi = min(vals), max(vals)
                print(f"  {name:21s} {a:4.0f} uA  from {100*lo[0]:+.2f}% "
                      f"({lo[1]/1000:.1f} mm) to {100*hi[0]:+.2f}% "
                      f"({hi[1]/1000:.1f} mm)", flush=True)
        _save(part, out)
    elif part == "percepts":
        out, fields = percepts(m)
        _save(part, out, fields)
    elif part == "reach":
        _save(part, reach(m))
    elif part == "encoder":
        out, ex = encoder(m)
        _save(part, out, ex)
    elif part == "power":
        _save(part, power_part(m))
    elif part == "inhibition":
        _save(part, inhibition())
    print(f"  ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    parts = sys.argv[1:] or list(PARTS)
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown part(s) {sorted(unknown)}; choose from "
                         f"{', '.join(PARTS)}")
    for p in parts:
        run(p)


if __name__ == "__main__":
    main()
