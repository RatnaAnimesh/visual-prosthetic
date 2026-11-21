"""Phase 4: does the forward model change how a device should be programmed?

This is the question with consequences.  Encoders for cortical prostheses are
optimised against a forward model, and every published forward model assumes
electrodes are independent.  If the cortex is not independent, those encoders
are solving the wrong problem -- but only in a way that *matters* if the
resulting stimulation is measurably worse when it meets a real cortex.

The design deliberately avoids training a neural network.  A learned encoder
confounds two things: how well the forward model matches reality, and how much
capacity the network had.  Instead, for each target image the *optimal*
amplitude vector is found directly under each forward model.  That is the best
any encoder of any architecture could do given that model, so the comparison
isolates model mismatch alone.

  a*_indep    = argmin  || decode(F_indep(a)) - target ||   (the current practice)
  a*_circuit  = argmin  || decode(F_circuit(a)) - target || (if the circuit is real)

Then the number that matters: run a*_indep through F_circuit.  That is what a
patient would actually see if their cortex has the circuit and their device was
programmed as though it did not.

Run:  python3 prosthesis/phase4_encoder.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pulse2percept.implants.cortex import Cortivis

from prosthesis.cortex_circuit import CircuitSpatial
from prosthesis.phase1_circuit import calibrated
from prosthesis.percept_decoder import CorticalCode, InversePercept

FIGS = pathlib.Path("figures")
RESULTS = pathlib.Path("results")
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CTX_X, CTX_Y = (16200, 23800), (-8800, -1200)
AMP_MAX = 600.0
N_ELECTRODES = 24

def configs():
    return {"independent": calibrated("circuit off"),
            "circuit": calibrated("full circuit")}


def make_targets(pipe, n_el, n_targets=6, n_active=5, seed=0):
    """Targets a device of this size could actually deliver.

    A first attempt used geometric shapes -- bars, a cross, a disc -- and every
    fidelity came out near zero.  That was not the optimiser failing: the
    ceiling for *any* combination of 16 electrodes, computed by unconstrained
    least squares over the per-electrode percepts, was 0.03-0.15.  Sixteen
    phosphenes cannot draw a thin bar in a 256-pixel image, and asking how much
    fidelity a modelling error costs is meaningless when there is no fidelity
    to lose.

    So targets are defined as percepts the *independent* model can actually
    produce: a handful of electrodes driven together, which is what a device
    designer would realistically aim for.  On these the ceiling is ~1.0, and
    the question becomes well posed.
    """
    rng = np.random.default_rng(seed)
    targets = {}
    for i in range(n_targets):
        amps = np.zeros(n_el)
        amps[rng.choice(n_el, size=n_active, replace=False)] = 450.0
        targets[f"pattern {i + 1}"] = (pipe.percept(amps), amps)
    return targets


def pick_electrodes(implant, n):
    """A spread-out subset, so the array covers the patch rather than a corner."""
    names = list(implant.electrode_names)
    xs = np.array([implant[k].x for k in names])
    ys = np.array([implant[k].y for k in names])
    chosen = [int(np.argmin(xs + ys))]
    while len(chosen) < n:
        d = np.min(np.stack([np.hypot(xs - xs[i], ys - ys[i])
                             for i in chosen]), axis=0)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return [names[i] for i in chosen]


class Pipeline:
    """amplitudes -> cortical activation -> sites -> decoded percept."""

    def __init__(self, cfg, code, decoder, xe, ye):
        self.model = CircuitSpatial(xrange=(-3.2, -0.4), yrange=(-0.2, 2.0),
                                    xystep=0.1, cortex_step=75.0,
                                    cortex_margin=2500.0, **cfg)
        self.model.build()
        self.code, self.decoder = code, decoder
        self.xe, self.ye = xe, ye
        self.grid = self.model._ensure_cortical_grid(xe, ye)

        # Anchor brightness to a common reference.  The two configurations
        # were calibrated in Phase 1 to give the same single-electrode
        # phosphene *size*, but nothing tied their activation *scales*
        # together -- normalisation divides by a pooled term and changes the
        # units.  Without this anchor a brightness comparison between models
        # measures an arbitrary constant.  Matching the contrast of one
        # electrode at a reference current is the natural completion of the
        # size calibration.
        self._scale = 1.0
        ref = self._raw_percept(np.eye(len(xe))[0] * 300.0)
        rms = float(np.std(ref))
        self._scale = 1.0 / rms if rms > 0 else 1.0

    def _raw_percept(self, amps):
        field = self.model._activation_field(np.asarray(amps, float),
                                             self.xe, self.ye, self.grid)
        a = self.code.sample_field(field, self.grid["xs"], self.grid["ys"])
        return self.decoder.decode(a)[0]

    def percept(self, amps):
        return self._raw_percept(amps) * self._scale


def fidelity(percept, target):
    """Squared correlation between predicted percept and target.

    Correlation rather than raw error, because the decoder's output has no
    calibrated scale: what matters for pattern fidelity is whether the right
    shape appears.

    But note precisely what this is blind to.  Squared correlation is
    invariant to overall scale, so a percept that is the right shape and half
    as bright scores identically to one at full brightness.  Since the circuit
    suppresses percepts fairly uniformly (Phase 2), this metric alone would
    hide the effect entirely -- which is why ``contrast`` is reported beside
    it.
    """
    p = percept.ravel() - percept.mean()
    t = target.ravel() - target.mean()
    den = np.linalg.norm(p) * np.linalg.norm(t)
    return float((p @ t) ** 2 / den ** 2) if den > 0 else 0.0


def contrast(percept):
    """RMS contrast of the predicted percept -- the part correlation ignores."""
    return float(np.std(percept))


def linear_reference(pipe, target, n_el, amp=400.0):
    """How well a *linear* combination of single-electrode percepts fits.

    Computed by unconstrained least squares over the per-electrode percepts at
    a fixed reference amplitude.  This is a diagnostic for whether a target is
    remotely deliverable by this electrode set -- it is what revealed that the
    first set of geometric targets was not (0.03-0.15, against ~1.0 for
    deliverable ones).

    It is *not* a strict upper bound, and an earlier version of this docstring
    wrongly said it was.  The model is nonlinear in amplitude -- the activated
    radius grows as sqrt(I) and the magnitude saturates -- so a percept
    produced at 450 uA is not a linear combination of 400 uA percepts, and
    optimisation can exceed this figure slightly.  Treat it as a scale, not a
    ceiling.
    """
    B = np.array([pipe.percept(np.eye(n_el)[i] * amp).ravel()
                  for i in range(n_el)])
    t = target.ravel() - target.mean()
    w, *_ = np.linalg.lstsq(B.T, t, rcond=None)
    fit = B.T @ w
    den = (np.linalg.norm(fit) ** 2) * (np.linalg.norm(t) ** 2)
    return float((fit @ t) ** 2 / den) if den > 0 else 0.0


def optimise_mse(pipe, target, n_el, x0_hint=None):
    """Optimise a *scale-sensitive* objective (plain squared error).

    Needed because the correlation objective is blind to brightness, so it
    never even attempts to compensate for suppression -- it finishes using
    less total current than it started with.  Asking whether the brightness
    loss is recoverable requires an objective that cares about brightness.
    """
    x0 = (np.asarray(x0_hint, float) if x0_hint is not None
          else np.full(n_el, 250.0))
    res = minimize(lambda a: float(np.mean((pipe.percept(a) - target) ** 2)),
                   x0, method="L-BFGS-B", bounds=[(0.0, AMP_MAX)] * n_el,
                   options={"maxiter": 60, "eps": 5.0})
    return res.x


def optimise(pipe, target, n_el, seed=0, restarts=2, x0_hint=None):
    """Best amplitudes for this target under this forward model."""
    rng = np.random.default_rng(seed)
    best, best_a = -1.0, None
    starts = [np.full(n_el, 250.0)]
    if x0_hint is not None:
        starts.insert(0, np.asarray(x0_hint, float))
    for r in range(restarts):
        x0 = (starts[r] if r < len(starts)
              else rng.uniform(50, AMP_MAX, n_el))
        res = minimize(lambda a: -fidelity(pipe.percept(a), target), x0,
                       method="L-BFGS-B",
                       bounds=[(0.0, AMP_MAX)] * n_el,
                       options={"maxiter": 60, "eps": 5.0})
        if -res.fun > best:
            best, best_a = -res.fun, res.x
    return best_a, best


def main() -> None:
    implant = Cortivis()
    names = pick_electrodes(implant, N_ELECTRODES)
    xe = np.array([implant[k].x for k in names], float)
    ye = np.array([implant[k].y for k in names], float)

    code = CorticalCode(CTX_X, CTX_Y)
    # Precomputing the inverse is exact here and about eight times faster.
    # Stimulation leaves every site with positive activation (the Gaussian
    # tails never reach zero), so the active set is everything and the fixed
    # operator agrees with the iterative one to correlation 1.0000.  The
    # cheaper operator would be wrong for *natural* images, where
    # rectification silences half the sites -- which is why the decoder was
    # validated with the iterative version.
    decoder = InversePercept(code, lam=0.02, precompute=True)

    pipes = {k: Pipeline(cfg, code, decoder, xe, ye)
             for k, cfg in configs().items()}
    targets = make_targets(pipes["independent"], N_ELECTRODES)

    print(f"{N_ELECTRODES} electrodes, {code.n_px}x{code.n_px} percepts, "
          f"amplitudes 0-{AMP_MAX:.0f} uA")
    print("targets are percepts the independent model can actually deliver\n")
    print(f"{'target':12s} {'lin.ref':>8s} {'matched':>9s} {'mismatched':>11s} "
          f"{'lost':>8s} {'contrast':>9s} {'c(bright)':>9s} {'f(bright)':>9s}")

    rows = {}
    for name, (tgt, true_amps) in targets.items():
        ceiling = linear_reference(pipes["circuit"], tgt, N_ELECTRODES)
        a_ind, f_ind_ind = optimise(pipes["independent"], tgt, N_ELECTRODES,
                                    x0_hint=true_amps)
        a_cir, f_cir_cir = optimise(pipes["circuit"], tgt, N_ELECTRODES,
                                    x0_hint=true_amps)

        # The number that matters: programmed for an independent cortex,
        # delivered to one with a circuit.
        p_ind_cir = pipes["circuit"].percept(a_ind)
        f_ind_cir = fidelity(p_ind_cir, tgt)
        lost = (f_cir_cir - f_ind_cir) / max(f_cir_cir, 1e-12)

        # Brightness, which the correlation metric cannot see.
        c_target = contrast(tgt)
        c_mismatch = contrast(p_ind_cir)
        c_matched = contrast(pipes["circuit"].percept(a_cir))

        # Can brightness be recovered at all, given an objective that wants
        # to?  This is the test of whether the shortfall is a programming
        # error or a hard limit.
        a_mse = optimise_mse(pipes["circuit"], tgt, N_ELECTRODES,
                             x0_hint=true_amps)
        p_mse = pipes["circuit"].percept(a_mse)
        c_mse = contrast(p_mse)
        f_mse = fidelity(p_mse, tgt)

        rows[name] = {
            "ceiling": ceiling,
            "indep_on_indep": f_ind_ind,
            "circuit_on_circuit": f_cir_cir,
            "indep_on_circuit": f_ind_cir,
            "fraction_lost": float(lost),
            "contrast_target": c_target,
            "contrast_mismatched": c_mismatch,
            "contrast_matched": c_matched,
            "contrast_ratio_mismatched": float(c_mismatch / max(c_target, 1e-12)),
            "contrast_ratio_mse": float(c_mse / max(c_target, 1e-12)),
            "fidelity_mse": f_mse,
            "current_mse": float(a_mse.sum()),
            "current_indep": float(a_ind.sum()),
            "amps_indep": a_ind.tolist(),
            "amps_circuit": a_cir.tolist(),
            "amps_mse": a_mse.tolist(),
            "max_amp_mse": float(a_mse.max()),
        }
        print(f"{name:12s} {ceiling:8.3f} {f_cir_cir:9.3f} {f_ind_cir:11.3f} "
              f"{100 * lost:7.1f}% {c_mismatch / max(c_target, 1e-12):9.2f}"
              f" {c_mse / max(c_target, 1e-12):9.2f} {f_mse:8.3f}")

    mean_ceiling = float(np.mean([r["ceiling"] for r in rows.values()]))
    mean_lost = float(np.mean([r["fraction_lost"] for r in rows.values()]))
    mean_matched = float(np.mean([r["circuit_on_circuit"]
                                  for r in rows.values()]))
    mean_mismatched = float(np.mean([r["indep_on_circuit"]
                                     for r in rows.values()]))
    mean_contrast = float(np.mean([r["contrast_ratio_mismatched"]
                                   for r in rows.values()]))
    mean_c_mse = float(np.mean([r["contrast_ratio_mse"] for r in rows.values()]))
    mean_f_mse = float(np.mean([r["fidelity_mse"] for r in rows.values()]))
    print(f"\n{'mean':12s} {mean_ceiling:8.3f} {mean_matched:9.3f} "
          f"{mean_mismatched:11.3f} {100 * mean_lost:7.1f}% "
          f"{mean_contrast:9.2f} {mean_c_mse:9.2f} {mean_f_mse:9.3f}")
    print("\n  'lost'      pattern fidelity forgone by using the wrong model")
    print("  'contrast'  percept RMS / target RMS -- what correlation cannot see")
    print("  'c(bright)' contrast after re-optimising a brightness-sensitive")
    print("              objective; 'f(bright)' the pattern fidelity that costs")

    # --- figure ----------------------------------------------------------
    n = len(targets)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.4))
    for j, (name, (tgt, _)) in enumerate(targets.items()):
        r = rows[name]
        a_ind = np.array(r["amps_indep"])
        a_cir = np.array(r["amps_circuit"])
        panels = [
            (tgt, "target"),
            (pipes["independent"].percept(a_ind),
             f"programmed & delivered\nindependent ({r['indep_on_indep']:.2f})"),
            (pipes["circuit"].percept(a_ind),
             f"same programming,\ncircuit cortex ({r['indep_on_circuit']:.2f})"),
            (pipes["circuit"].percept(a_cir),
             f"programmed for circuit\n({r['circuit_on_circuit']:.2f})"),
        ]
        for i, (img, title) in enumerate(panels):
            ax = axes[i, j]
            m = float(np.abs(img).max()) or 1.0
            ax.imshow(img, cmap="gray" if i == 0 else "RdBu_r",
                      vmin=0 if i == 0 else -m, vmax=1 if i == 0 else m,
                      origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(title.split("\n")[0], fontsize=6)
    fig.suptitle("Optimal stimulation under each forward model, and what "
                 "happens when they disagree", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "p4_encoder.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (RESULTS / "phase4_encoder.json").write_text(json.dumps({
        "n_electrodes": N_ELECTRODES, "amp_max": AMP_MAX,
        "electrodes": names, "targets": list(targets),
        "results": rows, "mean_fraction_lost": mean_lost,
        "mean_matched": mean_matched,
        "mean_mismatched": mean_mismatched,
        "mean_ceiling": mean_ceiling,
        "mean_contrast_ratio": mean_contrast,
        "mean_contrast_ratio_mse": mean_c_mse,
        "mean_fidelity_mse": mean_f_mse,
        "mse_reached_bound": bool(any(r["max_amp_mse"] >= AMP_MAX - 1
                                      for r in rows.values()))}, indent=2))
    print(f"\nwrote {RESULTS / 'phase4_encoder.json'}")


if __name__ == "__main__":
    main()
